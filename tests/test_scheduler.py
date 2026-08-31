"""Verification for ContinuousBatchScheduler. Test 1 is the headline gate:
everything the scheduler exists for is worthless if admission changes tokens.
Test 5 verifies the one property that actually distinguishes continuous
batching from static batching — a freed slot is reused the same step, not
the next one.
"""

import dataclasses
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402
from server.engine import Engine  # noqa: E402
from server.scheduler import ContinuousBatchScheduler  # noqa: E402
from server.types import Request  # noqa: E402
from tests.conftest import make_request  # noqa: E402

MAX_NEW_TOKENS = 12  # kept small so the suite stays fast on CPU


@contextmanager
def capture_scheduler_logits(engine, scheduler: ContinuousBatchScheduler):
    """Like conftest's capture_forward_logits, but resolves each captured row
    to a request_id via scheduler._running AT CALL TIME — necessary here
    because, under slot pressure, the same slot_id is reused by different
    requests over the run, so a slot_id -> request_id map built only from the
    final finished list would misattribute logits captured while an earlier
    occupant held that slot."""
    captures: list[list[tuple[str | None, torch.Tensor]]] = []
    original_forward = engine.model.forward

    def wrapped(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        slot_ids = engine.cache._slot_ids.tolist()
        live = {sid: seq.request_id for sid, seq in scheduler._running.items()}
        captures.append([(live.get(sid), row.clone()) for sid, row in zip(slot_ids, out.logits[:, -1, :])])
        return out

    engine.model.forward = wrapped
    try:
        yield captures
    finally:
        engine.model.forward = original_forward


def test_staggered_admission_matches_singles(small_slots_engine):
    engine = small_slots_engine  # max_slots=3
    tok = engine.tokenizer
    prompts = ["Hi.", "What is 2+2?", "Explain what a hash table is.", "Explain how photosynthesis works."]
    requests = [make_request(tok, p, MAX_NEW_TOKENS, f"stag-{i}") for i, p in enumerate(prompts)]

    scheduler = ContinuousBatchScheduler(engine)
    with capture_scheduler_logits(engine, scheduler) as captures:
        finished = []
        for req in requests:
            scheduler.submit(req)
            for _ in range(2):  # run a couple of steps before the next arrives
                if scheduler.has_work():
                    finished.extend(scheduler.step())
        while scheduler.has_work():
            finished.extend(scheduler.step())

    by_id = {s.request_id: s for s in finished}
    assert len(by_id) == 4

    logits_by_id: dict[str, list[torch.Tensor]] = defaultdict(list)
    for entries in captures:
        for rid, row in entries:
            if rid is not None:
                logits_by_id[rid].append(row)

    overall_max_delta = 0.0
    for req, prompt in zip(requests, prompts):
        solo = engine.generate([make_request(tok, prompt, MAX_NEW_TOKENS, f"solo-{req.request_id}")])[0]
        sched_seq = by_id[req.request_id]
        assert sched_seq.output_token_ids == solo.output_token_ids, f"{req.request_id} mismatch"

        with capture_scheduler_logits(engine, scheduler) as solo_capture_unused:
            pass  # scheduler is idle here; only used to keep the helper signature uniform
        sched_logits = torch.stack(logits_by_id[req.request_id], dim=0)

        # recompute the solo run's logits directly for the delta comparison
        solo_seq_ids = torch.tensor(req.prompt_token_ids).unsqueeze(0)
        with torch.no_grad():
            single_logits = []
            out = engine.model(solo_seq_ids, use_cache=True)
            single_logits.append(out.logits[0, -1, :])
            past = out.past_key_values
            cur = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
            for step_i in range(1, len(solo.output_token_ids)):
                pos = torch.tensor([[len(req.prompt_token_ids) + step_i - 1]])
                out = engine.model(cur, past_key_values=past, position_ids=pos, use_cache=True)
                past = out.past_key_values
                single_logits.append(out.logits[0, -1, :])
                cur = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
        solo_logits = torch.stack(single_logits, dim=0)

        n = min(sched_logits.shape[0], solo_logits.shape[0])
        delta = (sched_logits[:n] - solo_logits[:n]).abs().max().item()
        overall_max_delta = max(overall_max_delta, delta)
        print(f"  {req.request_id}: tokens_match=True max_logit_delta={delta:.3e}")

    print(f"test_staggered_admission_matches_singles: overall max logit delta = {overall_max_delta:.3e}")


def test_admission_under_slot_pressure(small_slots_engine):
    engine = small_slots_engine
    tok = engine.tokenizer
    prompts = [
        "Hi.", "Yes.", "No.", "Say ok.", "What is 2+2?", "Name a color.",
        "Is water wet?", "What year is it?", "Define gravity.", "What is your name?",
        "Thanks!", "Hello!",
    ]
    assert len(prompts) == 12
    requests = [make_request(tok, p, MAX_NEW_TOKENS, f"pressure-{i}") for i, p in enumerate(prompts)]

    scheduler = ContinuousBatchScheduler(engine)
    for req in requests:
        scheduler.submit(req)

    high_water = 0
    finished = []
    while scheduler.has_work():
        high_water = max(high_water, len(scheduler._waiting))
        finished.extend(scheduler.step())

    assert len(finished) == 12
    assert high_water > 0, "waiting queue was never observed non-empty — queueing path not exercised"
    print(f"test_admission_under_slot_pressure: waiting-queue high-water mark = {high_water}")

    by_id = {s.request_id: s for s in finished}
    for req, prompt in zip(requests, prompts):
        solo = engine.generate([make_request(tok, prompt, MAX_NEW_TOKENS, f"solo2-{req.request_id}")])[0]
        assert by_id[req.request_id].output_token_ids == solo.output_token_ids, f"{req.request_id} mismatch"


def test_no_slot_leak(small_slots_engine):
    engine = small_slots_engine
    tok = engine.tokenizer
    max_slots = engine.cache.max_slots
    scheduler = ContinuousBatchScheduler(engine)

    reqs_1 = [make_request(tok, p, MAX_NEW_TOKENS, f"leak1-{i}") for i, p in enumerate(["Hi.", "Yes.", "What is 2+2?"])]
    finished_1 = scheduler.generate(reqs_1)
    assert engine.cache.num_free_slots() == max_slots
    assert all(len(s.output_token_ids) > 0 for s in finished_1)

    reqs_2 = [make_request(tok, p, MAX_NEW_TOKENS, f"leak2-{i}") for i, p in enumerate(["No.", "Say ok.", "Name a color."])]
    finished_2 = scheduler.generate(reqs_2)
    assert engine.cache.num_free_slots() == max_slots
    assert all(len(s.output_token_ids) > 0 for s in finished_2)


def _long_token_ids(tok, n: int) -> list[int]:
    """Local, scaling variant of conftest's token_ids_of_length: that helper's
    fixed 5x filler repeat only tokenizes to ~96 tokens, too short for the
    150-token prompt this test needs (n // 10 + 5 repeats comfortably covers
    any n used here; the assert below verifies it regardless of the guess)."""
    unit = "The quick brown fox jumps over the lazy dog and then runs away into the deep dark forest. "
    ids = tok.encode(unit * (n // 10 + 5))
    assert len(ids) >= n, f"filler text only tokenized to {len(ids)} tokens, need >= {n}"
    return ids[:n]


def test_token_budget_respected(shared_engine):
    engine = shared_engine
    tok = engine.tokenizer
    budget = 100

    long_ids = _long_token_ids(tok, 150)
    long_req = Request(request_id="long", prompt_token_ids=long_ids, max_new_tokens=MAX_NEW_TOKENS, arrival_time=time.time())
    short_reqs = [make_request(tok, p, MAX_NEW_TOKENS, f"short-{i}") for i, p in enumerate(["Hi.", "Yes.", "Say ok.", "What is 2+2?"])]

    print(f"\ntest_token_budget_respected: batch_token_budget={budget}")
    print(f"  long   prompt tokens: {len(long_req.prompt_token_ids)}")
    for r in short_reqs:
        print(f"  {r.request_id} prompt tokens: {len(r.prompt_token_ids)}")

    assert len(long_req.prompt_token_ids) > budget, (
        f"test precondition failed: long prompt has {len(long_req.prompt_token_ids)} tokens, "
        f"which must exceed batch_token_budget={budget} for this test to exercise the cap at all"
    )

    scheduler = ContinuousBatchScheduler(engine, batch_token_budget=budget)

    prefill_batches: list[list[str]] = []
    original_prefill = engine.prefill

    def spy_prefill(seqs):
        prefill_batches.append([s.request_id for s in seqs])
        return original_prefill(seqs)

    engine.prefill = spy_prefill
    try:
        scheduler.submit(long_req)
        for r in short_reqs:
            scheduler.submit(r)
        finished = []
        while scheduler.has_work():
            finished.extend(scheduler.step())
    finally:
        engine.prefill = original_prefill

    assert len(finished) == 5
    long_batch = next(b for b in prefill_batches if "long" in b)
    assert long_batch == ["long"], f"the 150-token prompt shared a prefill pass with {long_batch}, budget={budget} was not respected"

    by_id = {s.request_id: s for s in finished}
    solo_long = engine.generate(
        [Request(request_id="solo-long", prompt_token_ids=_long_token_ids(tok, 150), max_new_tokens=MAX_NEW_TOKENS, arrival_time=time.time())]
    )[0]
    assert by_id["long"].output_token_ids == solo_long.output_token_ids


def test_retire_frees_slot_same_step():
    config = dataclasses.replace(load_config(), max_slots=1)
    engine = Engine(config)
    tok = engine.tokenizer
    scheduler = ContinuousBatchScheduler(engine)

    req_a = make_request(tok, "Hi.", 3, "A")
    req_b = make_request(tok, "Yes.", 3, "B")

    # submit() ONLY enqueues into _waiting — it does not touch the cache or
    # allocate a slot. Only step()'s ADMIT phase does that. So both A and B
    # sit in _waiting after these two calls; do not assume otherwise here or
    # anywhere else this scheduler is used.
    scheduler.submit(req_a)
    scheduler.submit(req_b)
    assert len(scheduler._waiting) == 2

    scheduler.step()  # one step: ADMIT can only take A (max_slots=1), then prefill+decode it
    assert any(seq.request_id == "A" for seq in scheduler._running.values()), "A must be admitted after one step()"
    assert any(req.request_id == "B" for req in scheduler._waiting), "B must still be waiting behind A (max_slots=1)"
    assert len(scheduler._waiting) == 1

    finished: list = []
    step_index = 1  # the step() call above was step 1
    retire_step_index = None
    b_admitted_same_step = False
    while retire_step_index is None:
        step_index += 1
        step_finished = scheduler.step()
        finished.extend(step_finished)
        if any(s.request_id == "A" for s in step_finished):
            retire_step_index = step_index
            b_running = [seq for seq in scheduler._running.values() if seq.request_id == "B"]
            # "admitted AND prefilled" this same step, not merely admitted:
            # a prefilled sequence already has at least one output token.
            b_admitted_same_step = bool(b_running) and len(b_running[0].output_token_ids) >= 1

    print(f"\ntest_retire_frees_slot_same_step: A retired on step {retire_step_index}; B admitted+prefilled on step {retire_step_index}: {b_admitted_same_step}")
    assert retire_step_index is not None, "A never retired"
    assert b_admitted_same_step, f"B was not admitted and prefilled on step {retire_step_index}, the same step that retired A"

    while scheduler.has_work():
        finished.extend(scheduler.step())
    assert {s.request_id for s in finished} == {"A", "B"}

    by_id = {s.request_id: s for s in finished}
    solo_b = engine.generate([make_request(tok, "Yes.", 3, "solo-B")])[0]
    assert by_id["B"].output_token_ids == solo_b.output_token_ids
