"""Shared fixtures/helpers for the engine and numerics test suites.

`twenty_prompt_stats` is session-scoped so the expensive batch-vs-singles run
over the 20 CV prompts happens exactly once per test session, even though both
test_engine.py and test_numerics.py need its result.
"""

import dataclasses
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402
from server.engine import Engine  # noqa: E402
from server.types import Request  # noqa: E402

NUM_NEW_TOKENS = 20

TWENTY_PROMPTS = [
    "Hi.",
    "Hello!",
    "Yes.",
    "No.",
    "Thanks!",
    "What is 2+2?",
    "What is the capital of France?",
    "Say ok.",
    "Name a color.",
    "Is water wet?",
    "What year is it?",
    "Define gravity.",
    "What is your name?",
    "Explain photosynthesis briefly.",
    "Write a short paragraph about why the sky appears blue to a human observer.",
    "List three benefits of regular exercise for cardiovascular health.",
    "Describe the process of making a cup of coffee step by step.",
    "What are the main differences between Python and JavaScript as programming languages?",
    "Summarize the plot of a typical detective mystery novel in a few sentences.",
    "Explain how neural networks learn from data using backpropagation.",
]


@pytest.fixture(scope="session")
def shared_engine():
    return Engine(load_config())


@pytest.fixture(scope="session")
def small_slots_engine():
    """A dedicated Engine (own model load) with max_slots=3 — max_slots is
    baked into the cache at construction time, so tests that need real slot
    pressure can't just reuse shared_engine's default (much larger) cache."""
    return Engine(dataclasses.replace(load_config(), max_slots=3))


def make_request(tok, prompt: str, max_new_tokens: int, request_id: str) -> Request:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )["input_ids"][0].tolist()
    return Request(request_id=request_id, prompt_token_ids=ids, max_new_tokens=max_new_tokens, arrival_time=time.time())


def token_ids_of_length(tok, n: int) -> list[int]:
    filler = "The quick brown fox jumps over the lazy dog and then runs away into the deep dark forest. " * 5
    ids = tok.encode(filler)
    assert len(ids) >= n
    return ids[:n]


@contextmanager
def capture_forward_logits(engine):
    """Records (slot_ids, last-position logits) for every real model forward
    call made during the `with` block, without touching engine.py."""
    captures: list[tuple[list[int], torch.Tensor]] = []
    original_forward = engine.model.forward

    def wrapped(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        captures.append((engine.cache._slot_ids.tolist(), out.logits[:, -1, :].detach().clone()))
        return out

    engine.model.forward = wrapped
    try:
        yield captures
    finally:
        engine.model.forward = original_forward


@dataclass
class TightestStep:
    """The single decode step, across everything compared, with the smallest
    argmax margin in the single-sequence reference run — i.e. the step that
    would flip first if numerical noise grew (e.g. under float16)."""

    prompt_index: int
    request_id: str
    step_index: int
    margin: float
    delta: float  # batch-vs-single logit delta at this same step
    ratio: float  # margin / delta at this same step
    top1_token_id: int
    top2_token_id: int


@dataclass
class BatchVsSingleStats:
    max_abs_logit_delta: float
    min_argmax_margin: float
    mean_argmax_margin: float
    num_steps_compared: int
    tightest_ratio: float  # margin / delta at the tightest step
    tightest_step: TightestStep


def assert_batch_matches_singles(engine, requests: list[Request], label: str = "") -> BatchVsSingleStats:
    """Runs `requests` as one batch and, separately, one at a time. Asserts
    token-identical output. Also measures, from the single-sequence reference
    run at every decode step, argmax_margin = top1_logit - top2_logit, and
    returns a stats object comparing that margin against the batch-vs-single
    logit noise — the thing that actually determines whether token identity
    across batch shapes is guaranteed or merely lucky so far."""
    with capture_forward_logits(engine) as batch_captures:
        batch_seqs = engine.generate(list(requests))

    slot_to_index = {seq.slot_id: i for i, seq in enumerate(batch_seqs)}
    batch_logits_by_index: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(requests))}
    for slot_ids, logits in batch_captures:
        for row, slot_id in enumerate(slot_ids):
            idx = slot_to_index.get(slot_id)
            if idx is not None:
                batch_logits_by_index[idx].append(logits[row].clone())

    overall_max_delta = 0.0
    all_margins: list[float] = []
    tightest: TightestStep | None = None

    for i, req in enumerate(requests):
        with capture_forward_logits(engine) as single_captures:
            single_seq = engine.generate([req])[0]
        single_logits = torch.stack([logits[0] for _, logits in single_captures], dim=0)

        assert batch_seqs[i].output_token_ids == single_seq.output_token_ids, (
            f"{label} prompt {i} (request_id={req.request_id!r}) mismatch: "
            f"batch={batch_seqs[i].output_token_ids} single={single_seq.output_token_ids}"
        )

        batch_logits_i = torch.stack(batch_logits_by_index[i], dim=0)
        n = min(batch_logits_i.shape[0], single_logits.shape[0])
        per_step_delta = (batch_logits_i[:n] - single_logits[:n]).abs().max(dim=-1).values
        delta = per_step_delta.max().item()
        overall_max_delta = max(overall_max_delta, delta)
        print(f"  [{label}] prompt {i} (id={req.request_id}): tokens_match=True max_logit_delta={delta:.3e}")

        top2 = torch.topk(single_logits[:n], k=2, dim=-1)
        margins = top2.values[:, 0] - top2.values[:, 1]
        for step_index in range(n):
            margin = margins[step_index].item()
            all_margins.append(margin)
            if tightest is None or margin < tightest.margin:
                step_delta = per_step_delta[step_index].item()
                tightest = TightestStep(
                    prompt_index=i,
                    request_id=req.request_id,
                    step_index=step_index,
                    margin=margin,
                    delta=step_delta,
                    ratio=(margin / step_delta if step_delta > 0 else float("inf")),
                    top1_token_id=int(top2.indices[step_index, 0].item()),
                    top2_token_id=int(top2.indices[step_index, 1].item()),
                )

    return BatchVsSingleStats(
        max_abs_logit_delta=overall_max_delta,
        min_argmax_margin=min(all_margins),
        mean_argmax_margin=sum(all_margins) / len(all_margins),
        num_steps_compared=len(all_margins),
        tightest_ratio=tightest.ratio,
        tightest_step=tightest,
    )


@pytest.fixture(scope="session")
def twenty_prompt_stats(shared_engine) -> BatchVsSingleStats:
    requests = [make_request(shared_engine.tokenizer, p, NUM_NEW_TOKENS, f"cv-{i}") for i, p in enumerate(TWENTY_PROMPTS)]
    return assert_batch_matches_singles(shared_engine, requests, label="cv20")


@pytest.fixture
def app_client(shared_engine):
    """A TestClient wired to the session's shared_engine, so hitting the API
    in tests never triggers a second model load."""
    from fastapi.testclient import TestClient

    from server.app import app

    app.state.config = shared_engine.config
    app.state.engine = shared_engine
    with TestClient(app) as client:
        yield client
