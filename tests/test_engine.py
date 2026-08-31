"""Verification for Engine. Tests 1 and 2 are the real gates: everything the
scheduler will eventually build on top of engine.py depends on batching being
provably identical to running requests one at a time.

Pure argmax throughout — no sampling, no repetition penalty, no temperature.
"""

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402
from server.engine import Engine  # noqa: E402
from server.types import Request, RequestMetrics, SequenceState  # noqa: E402
from tests.conftest import (  # noqa: E402
    NUM_NEW_TOKENS,
    assert_batch_matches_singles,
    make_request,
    token_ids_of_length,
)


def run_generate_with_prefill_hook(engine, requests, after_prefill):
    """Mirrors Engine.generate() exactly, but calls `after_prefill(engine, seqs)`
    right after prefill — used only to inspect cache.pad_len before free_slot()
    zeroes it out at the end of the normal generate() path."""
    seqs = [
        SequenceState(
            request_id=req.request_id,
            prompt_token_ids=req.prompt_token_ids,
            max_new_tokens=req.max_new_tokens,
            metrics=RequestMetrics(arrival_time=req.arrival_time),
        )
        for req in requests
    ]
    engine.prefill(seqs)
    after_prefill(engine, seqs)
    active = [s for s in seqs if not s.finished]
    while active:
        engine.decode(active)
        active = [s for s in active if not s.finished]
    for s in seqs:
        engine.cache.free_slot(s.slot_id)
    return seqs


# --- tests ---


def test_left_padding_is_exercised(shared_engine):
    engine = shared_engine
    tok = engine.tokenizer
    lengths = [4, 20, 60]
    requests = [
        Request(
            request_id=f"pad-{i}", prompt_token_ids=token_ids_of_length(tok, n), max_new_tokens=NUM_NEW_TOKENS,
            arrival_time=time.time(),
        )
        for i, n in enumerate(lengths)
    ]

    pad_lens: dict[str, int] = {}

    def check_padding(engine, seqs):
        for seq in seqs:
            pad_lens[seq.request_id] = int(engine.cache.pad_len[seq.slot_id].item())

    run_generate_with_prefill_hook(engine, requests, check_padding)

    print(f"\ntest_left_padding_is_exercised pad_len by request: {pad_lens}")
    positive = [v for v in pad_lens.values() if v > 0]
    assert len(positive) >= 2, f"expected at least two sequences with pad_len > 0, got {pad_lens}"

    assert_batch_matches_singles(engine, requests, label="left-padding")


def test_batch_matches_singles_20_prompts(twenty_prompt_stats):
    # twenty_prompt_stats (see conftest.py) already ran the batch-vs-singles
    # comparison and asserted token identity for all 20 prompts; this test just
    # reports the headline number without re-running the ~20s comparison.
    stats = twenty_prompt_stats
    print(
        f"\ntest_batch_matches_singles_20_prompts: 20/20 matched, "
        f"worst max logit delta = {stats.max_abs_logit_delta:.3e}"
    )


def test_dirty_slot_with_mixed_lengths():
    config = dataclasses.replace(load_config(), max_slots=2)
    engine = Engine(config)
    tok = engine.tokenizer

    # dirty both slots with a first batch, then let generate() free them without zeroing
    first_requests = [
        make_request(tok, "Hi.", NUM_NEW_TOKENS, "first-0"),
        make_request(tok, "What is the capital of France?", NUM_NEW_TOKENS, "first-1"),
    ]
    engine.generate(first_requests)
    assert engine.cache.num_free_slots() == 2

    # staggered admission via submit()/step(), landing in the same two dirty
    # slots (max_slots=2, so there is nowhere else for them to go): prompt C is
    # admitted and decoded a step before D, so by the time D is prefilled and
    # they are decoded jointly, C's cache_len is strictly ahead of D's — this is
    # the "window exceeds the shorter sequence's frontier" case generate()'s
    # all-together padded prefill can never produce on its own.
    prompt_c = "Hi."
    prompt_d = "Explain how neural networks learn from data using backpropagation."
    req_c = make_request(tok, prompt_c, NUM_NEW_TOKENS, "second-c")
    req_d = make_request(tok, prompt_d, NUM_NEW_TOKENS, "second-d")

    engine.submit(req_c)
    engine.step()  # prefills C alone, plus one decode step
    engine.submit(req_d)

    finished = []
    while engine.has_work():
        finished.extend(engine.step())

    by_id = {seq.request_id: seq for seq in finished}
    assert set(by_id) == {"second-c", "second-d"}

    fresh_engine = Engine(config)
    solo_c = fresh_engine.generate([make_request(fresh_engine.tokenizer, prompt_c, NUM_NEW_TOKENS, "solo-c")])[0]
    solo_d = fresh_engine.generate([make_request(fresh_engine.tokenizer, prompt_d, NUM_NEW_TOKENS, "solo-d")])[0]

    assert by_id["second-c"].output_token_ids == solo_c.output_token_ids
    assert by_id["second-d"].output_token_ids == solo_d.output_token_ids


def test_eos_stops_mid_batch(shared_engine):
    engine = shared_engine
    tok = engine.tokenizer

    fast_req = make_request(tok, "Hi.", NUM_NEW_TOKENS, "fast")
    slow_reqs = [
        make_request(tok, "Describe the process of making a cup of coffee step by step.", NUM_NEW_TOKENS, "slow-1"),
        make_request(tok, "Explain how neural networks learn from data using backpropagation.", NUM_NEW_TOKENS, "slow-2"),
    ]

    solo_fast = engine.generate([fast_req])[0]
    solo_slow = {r.request_id: engine.generate([r])[0] for r in slow_reqs}
    assert solo_fast.finish_reason == "stop", "test premise requires the fast prompt to hit EOS before max_new_tokens"
    assert all(s.finish_reason == "length" for s in solo_slow.values()), (
        "test premise requires the slow prompts to run to max_new_tokens"
    )

    batch_seqs = engine.generate([fast_req, *slow_reqs])
    by_id = {s.request_id: s for s in batch_seqs}

    fast_batched = by_id["fast"]
    assert fast_batched.finish_reason == "stop"
    assert fast_batched.output_token_ids == solo_fast.output_token_ids, (
        "fast sequence's output differs when batched, or kept growing after it hit EOS"
    )
    for req in slow_reqs:
        assert by_id[req.request_id].output_token_ids == solo_slow[req.request_id].output_token_ids, (
            f"{req.request_id} affected by batching with the fast sequence that finished early"
        )

    # eos_token_ids is a LIST — a hit on either id must stop the sequence
    eos_ids = list(engine.eos_token_ids)
    assert len(eos_ids) >= 2, "expected Qwen's generation_config to expose multiple eos ids"
    for eos_id in eos_ids:
        probe = SequenceState(
            request_id="probe", prompt_token_ids=[0], max_new_tokens=NUM_NEW_TOKENS,
            metrics=RequestMetrics(arrival_time=time.time()),
        )
        probe.output_token_ids.append(eos_id)
        engine._mark_if_finished(probe, eos_id, time.time())
        assert probe.finished and probe.finish_reason == "stop", f"eos id {eos_id} did not stop the sequence"


def test_metrics_are_populated(shared_engine):
    engine = shared_engine
    tok = engine.tokenizer
    requests = [make_request(tok, p, NUM_NEW_TOKENS, f"m-{i}") for i, p in enumerate(["Hi.", "What is 2+2?", "Define gravity."])]
    seqs = engine.generate(requests)

    for seq in seqs:
        m = seq.metrics
        assert m.prefill_start_time is not None
        assert m.first_token_time is not None
        assert m.finish_time is not None
        assert m.ttft is not None and m.ttft > 0
        assert m.first_token_time <= m.finish_time


def test_slots_are_released(shared_engine):
    engine = shared_engine
    tok = engine.tokenizer
    max_slots = engine.cache.max_slots

    requests_1 = [make_request(tok, p, NUM_NEW_TOKENS, f"r1-{i}") for i, p in enumerate(["Hi.", "Yes.", "What is 2+2?"])]
    seqs_1 = engine.generate(requests_1)
    assert engine.cache.num_free_slots() == max_slots
    assert all(len(s.output_token_ids) > 0 for s in seqs_1)

    requests_2 = [make_request(tok, p, NUM_NEW_TOKENS, f"r2-{i}") for i, p in enumerate(["No.", "Name a color.", "Say ok."])]
    seqs_2 = engine.generate(requests_2)
    assert engine.cache.num_free_slots() == max_slots
    assert all(len(s.output_token_ids) > 0 for s in seqs_2)
