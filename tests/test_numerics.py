"""Measures whether the batch-vs-single logit noise is small enough that
argmax decisions never flip — rather than assuming it because 20/20 prompts
happened to match in test_engine.py. See CLAUDE.md's "Verified environment
facts" for the two distinct correctness numbers this produces and what they
each mean.
"""


def test_argmax_margin_exceeds_numerical_noise(twenty_prompt_stats, shared_engine):
    # twenty_prompt_stats (see conftest.py) is session-scoped and shared with
    # test_engine.py's test_batch_matches_singles_20_prompts — this reuses that
    # run instead of paying for the ~20s comparison a second time.
    stats = twenty_prompt_stats
    tok = shared_engine.tokenizer
    tightest = stats.tightest_step

    assert stats.min_argmax_margin > stats.max_abs_logit_delta, (
        f"argmax margin ({stats.min_argmax_margin:.3e}) does not clear the batch-vs-single "
        f"logit noise ({stats.max_abs_logit_delta:.3e}) at prompt {tightest.prompt_index} "
        f"(id={tightest.request_id}), step {tightest.step_index} — token identity across "
        "batch shapes would be luck-dependent here, not guaranteed. Do not raise this "
        "threshold to make it pass; report the offending step instead."
    )

    safety_ratio = stats.min_argmax_margin / stats.max_abs_logit_delta
    top1_str = tok.decode([tightest.top1_token_id])
    top2_str = tok.decode([tightest.top2_token_id])

    print()
    print(f"max abs logit delta      : {stats.max_abs_logit_delta:.3e}")
    print(f"min argmax margin        : {stats.min_argmax_margin:.3e}")
    print(f"mean argmax margin       : {stats.mean_argmax_margin:.3e}")
    print(f"safety ratio (Y / X)     : {safety_ratio:.1f}")
    print(f"decode steps compared    : {stats.num_steps_compared}")
    print("prompts compared         : 20")
    print(
        f"tightest step            : prompt {tightest.prompt_index} (id={tightest.request_id}), "
        f"step {tightest.step_index}, margin={tightest.margin:.3e}, "
        f"top1={top1_str!r} top2={top2_str!r}"
    )

    if tightest.margin < 10 * stats.max_abs_logit_delta:
        print(
            f"WARNING: tightest step (prompt {tightest.prompt_index}, step {tightest.step_index}) "
            f"has margin {tightest.margin:.3e}, under 10x the observed noise floor "
            f"({stats.max_abs_logit_delta:.3e}) — this is the step that would flip first "
            "under float16, where this noise grows by roughly three orders of magnitude."
        )
