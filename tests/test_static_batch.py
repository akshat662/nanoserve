"""Verification for StaticBatchEngine — the naive baseline bench.py will later
measure against. Test (b) is the one that must show real waste; test (c) is the
control that proves the metric isn't just always positive.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.static_batch import StaticBatchEngine  # noqa: E402
from server.types import Request  # noqa: E402
from tests.conftest import NUM_NEW_TOKENS, TWENTY_PROMPTS, make_request, token_ids_of_length  # noqa: E402


def test_static_batch_matches_engine_generate(shared_engine):
    tok = shared_engine.tokenizer
    requests = [make_request(tok, p, NUM_NEW_TOKENS, f"sb-{i}") for i, p in enumerate(TWENTY_PROMPTS[:8])]

    engine_seqs = {s.request_id: s for s in shared_engine.generate(list(requests))}
    static_seqs, _waste = StaticBatchEngine(shared_engine).generate(list(requests))

    for seq in static_seqs:
        assert seq.output_token_ids == engine_seqs[seq.request_id].output_token_ids, (
            f"static batch output differs from Engine.generate() for {seq.request_id}"
        )


def test_waste_with_mismatched_lengths(shared_engine):
    tok = shared_engine.tokenizer
    now = time.time()
    # a 4-token prompt that stops (via its own small max_new_tokens) long before
    # a 60-token prompt that runs to its own, much larger max_new_tokens
    short_req = Request(request_id="short", prompt_token_ids=token_ids_of_length(tok, 4), max_new_tokens=3, arrival_time=now)
    long_req = Request(request_id="long", prompt_token_ids=token_ids_of_length(tok, 60), max_new_tokens=12, arrival_time=now)

    seqs, waste = StaticBatchEngine(shared_engine).generate([short_req, long_req])
    by_id = {s.request_id: s for s in seqs}
    assert len(by_id["short"].output_token_ids) == 3
    assert len(by_id["long"].output_token_ids) == 12

    print(f"\ntest_waste_with_mismatched_lengths:\n{waste}")
    assert waste.prefill_waste > 0, "padding the 4-token prompt up to 60 must show up as prefill waste"
    assert waste.decode_waste > 0, "the short sequence riding idle for 9 extra rounds must show up as decode waste"


def test_no_waste_when_uniform(shared_engine):
    tok = shared_engine.tokenizer
    now = time.time()
    prompt_ids = token_ids_of_length(tok, 20)
    requests = [
        Request(request_id=f"uniform-{i}", prompt_token_ids=prompt_ids, max_new_tokens=8, arrival_time=now)
        for i in range(3)
    ]

    seqs, waste = StaticBatchEngine(shared_engine).generate(requests)
    assert all(s.finish_reason == "length" for s in seqs), (
        "test premise requires every sequence to run to max_new_tokens, not stop early"
    )

    print(f"\ntest_no_waste_when_uniform:\n{waste}")
    assert waste.prefill_waste == 0
    assert waste.decode_waste == 0
