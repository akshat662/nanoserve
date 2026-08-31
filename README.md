# nanoserve

## Quickstart

## Correctness

Every decode path in this project is checked against `model.generate()`, not assumed
correct because it runs without crashing. A single sequence decoded through
`SlotKVCache`, one token at a time with an explicit absolute `position_ids` tensor and
an explicit 4D attention mask, is bit-identical to the same prompt decoded through a
plain `transformers.DynamicCache`: max abs logit delta `0.000e+00` across 30 tokens
(`tests/test_kv_cache.py::test_single_sequence_matches_gate0`). That is the strongest
claim in the codebase, and it is the one every other test builds on.

Batched output is a different, weaker, and equally real claim: **token-identical, not
bit-identical**. Running 20 varied prompts together in one batch versus one at a time
produces a max absolute logit delta of `1.822e-04` — three orders of magnitude larger
than the single-sequence number above. This is not a bug in the cache or the mask; it
is the matmul reduction order the backend kernel picks for a given batch shape, which
changes in float32 exactly as it does in float16. Padding, slot layout, and batch size
all change that shape. The two deltas measure different things and must not be quoted
interchangeably.

The reason 20/20 prompts still matched token-for-token isn't luck we happened not to
get unlucky on — it's measured. At every decode step across those 20 prompts (330
steps total), we recorded the single-sequence reference run's argmax margin (top-1
logit minus top-2 logit) and compared it against the batch-vs-single delta at that same
step. The tightest step anywhere in the run — prompt 10, step 12, choosing between
` ability` and ` capability` — still had a margin of `3.027e-02`, against a worst-case
delta of `1.822e-04`: a safety ratio of **166.2x**. Every other step had more headroom
than that. This is why float32 is a hard requirement for correctness work here (see
CLAUDE.md): float16 would shrink that 166.2x margin by roughly three orders of
magnitude, and the tightest step above would no longer be safe.

## Results

## How it works

## Limitations

## Roadmap
