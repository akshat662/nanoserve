# nanoserve

A minimal LLM inference server that manages the KV cache explicitly and batches
concurrent requests at the iteration level, instead of calling `model.generate()`
per request. Roughly, a small reimplementation of what vLLM does — built to
understand serving internals, not to replace them.

OpenAI-compatible endpoint, a static-batching baseline to measure against, and
correctness verified token-by-token against HuggingFace's own decode path.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
print(client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain KV caching in one paragraph."}],
).choices[0].message.content)
```

Nothing about that call is special-cased. The official client points at the server
and works.

## Quickstart

```bash
git clone https://github.com/akshat662/nanoserve.git
cd nanoserve
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # see note below if you have a GPU

make serve                            # uvicorn server.app:app
make test                             # 25 tests
make bench                            # benchmark table
```

`requirements.txt` pins a CPU build of torch. On a CUDA box, install torch from
your platform's index first, then the rest.

Engine selection lives in `configs/default.yaml` (`engine: engine | static |
scheduler`). Everything else — slot count, max sequence length, token budget —
is in the same file.

## Results

Measured on a single T4, `Qwen/Qwen2.5-0.5B-Instruct`, float32, greedy decoding.
`torch.cuda.synchronize()` precedes every timestamp; 2–3 warmup iterations are
discarded; latencies are percentiles, not means. TTFT is instrumented inside the
engine, because a non-streaming HTTP response cannot expose it to the client.

**Staggered arrivals, short prompt / long generation** — 32 requests over 8s, 4 slots,
`max_new_tokens=128`:

| engine | output tok/s | TTFT p50 | TTFT p95 | e2e p50 |
|---|---|---|---|---|
| static | 79.7 | 14680 ms | 27912 ms | 18302 ms |
| **scheduler** | **108.3** | **6072 ms** | **16416 ms** | **10281 ms** |
| | 1.36× | 2.42× | 1.70× | 1.78× |

**Staggered arrivals, long prompt / short generation** — 16 requests over 4s, 4 slots,
`max_new_tokens=16`:

| engine | output tok/s | TTFT p50 | TTFT p95 | e2e p50 |
|---|---|---|---|---|
| static | 46.3 | 798 ms | 1219 ms | 1309 ms |
| **scheduler** | **54.4** | **117 ms** | **179 ms** | **842 ms** |
| | 1.18× | 6.82× | 6.80× | 1.55× |

**Burst arrivals** (all 32 requests at t=0, 8 slots) — included because it is the
case where continuous batching helps *least*: 158.6 → 177.0 tok/s (1.12×), TTFT
p50 8414 → 5977 ms (1.41×). When every request arrives simultaneously there is
little for admission scheduling to exploit.

**Waste in the static baseline**, measured directly rather than estimated:

| workload | prefill waste (padding) | decode waste (head-of-line) |
|---|---|---|
| short prompt / long gen | 50.0–51.5% | 18.9–22.6% |
| long prompt / short gen | 15.7% | 6.1% |

The two are tracked separately because they have different causes and different
fixes. Padding waste is a function of prompt-length spread; head-of-line waste is
a function of how long finished sequences sit in a batch waiting for their
slowest neighbour.

The throughput gain is modest; the latency gain is not. That is the honest shape
of this result, and it matches the mechanism: continuous batching does not make
the GPU faster, it stops requests from waiting for a batch boundary.

One finding worth stating plainly, because it contradicted my own prediction: the
scheduler's advantage tracks **slot turnover rate**, not generation length. Short
generations retire quickly, free slots quickly, and hand scheduling the most
opportunities to act — which is why the long-prompt/short-generation workload
showed the largest TTFT improvement, not the smallest.

## Correctness

Three separate claims, verified separately.

**Single-sequence cache equivalence.** The slot-based cache produces logits
bit-identical to HuggingFace's `DynamicCache` on the same prompt — max absolute
delta `0.000e+00`.

**Batch equivalence.** A batch of N produces token-identical output to the same N
prompts run one at a time. Max absolute logit delta here is `1.357e-04`, not zero,
because changing batch shape changes the reduction order the backend kernel picks —
in float32 as much as float16. This is expected, not a defect.

**That the tokens match is not luck.** Across 20 prompts and 330 decode steps, the
minimum argmax margin (top-1 minus top-2 logit) was `3.027e-02` — a **223×** safety
factor over the noise floor. The tightest step was a genuine semantic tie
(`' ability'` vs `' capability'`), not a numerical artifact. The margin is identical
on CPU and GPU because it is a property of the model; only the noise floor is
hardware-dependent, which is why the same measurement gives 166× on CPU and 223×
on a T4.

float32 throughout. float16 would shrink that safety ratio by roughly three orders
of magnitude, which is why correctness work here does not use it.

Staggered admission is verified the same way: a request admitted mid-run produces
output token-identical to the same request run alone. 25 tests, green on both CPU
and T4.

## How it works

**KV cache.** Generating token *n* requires attending over the keys and values of
tokens 1..*n*−1. Recomputing them each step is quadratic, so they are cached. For
this model the cache costs 12,288 bytes per token in fp16 — `2 × 24 layers ×
2 KV heads × 64 head_dim × 2 bytes`. The KV-head count is what matters: Qwen2.5
uses grouped-query attention with 2 KV heads against 14 attention heads, so sizing
the cache by attention heads overstates it seven-fold.

The cache here is a `transformers.Cache` subclass over preallocated per-layer
tensors of shape `[max_slots, kv_heads, max_seq_len, head_dim]`. Nothing is ever
`torch.cat`'d or reallocated. Each step scatters new keys and values into per-slot
write indices with a single vectorised indexing operation, and returns the whole
window; a 4D additive attention mask excludes left-padding, stale positions, and
anything past a sequence's own frontier. Because the mask does that work, a freed
slot needs no zeroing before reuse — verified by a test that deliberately reuses a
dirty slot.

The central idea is that **cache index and RoPE position are different numbers.** A
sequence admitted mid-run occupies cache indices that have nothing to do with its
own token positions. Keeping those decoupled is what makes mid-run admission
possible at all, and it is also exactly the constraint that motivates paged
attention.

**What static batching wastes.** Collect requests, pad them to the longest, run
until the last one finishes. Two costs follow. Every prompt shorter than the longest
burns compute on padding — measured at up to 51.5%. And every sequence that hits EOS
early keeps occupying its slot, processed and discarded, until its slowest neighbour
finishes — measured at up to 22.6% of decode steps. That second cost is head-of-line
blocking, and it also means a newly arrived request waits for the entire current
batch before it can start.

**Continuous batching.** Schedule per iteration instead of per batch. Each step:
retire finished sequences and free their slots; admit waiting requests into any free
slots, capped by a token budget so one very long prompt cannot stall the step;
run one prefill-only forward pass over the newly admitted; run one decode forward
pass over everything active. Two forward passes on admission steps, one otherwise.

Prefill and decode are deliberately *not* fused into a single pass — mixing a
multi-token prefill with single-token decodes needs ragged attention, which is a V2
concern. This is roughly what vLLM did before chunked prefill.

The engine executes batches; the scheduler decides what goes in them. Neither
contains the other's logic.

## Limitations

**Preallocation wastes cache.** Each slot reserves `max_seq_len` regardless of how
much it uses. At 32 slots × 1024 tokens that is 384 MB in fp16, 768 MB in fp32 —
and a 50-token request holds the same reservation as a 1000-token one. Paged block
allocation is the fix, and it is first on the roadmap.

**The concurrency ceiling here is arithmetic, not observed.** At 0.5B with 2 KV
heads on a 16 GB T4, the cache never came close to exhausting memory, so no OOM
was measured. The ceiling is a projection at larger model scale, and it should be
read as one.

**Padding dominates at this scale.** In the short-prompt workload, half of all
prefill tokens were padding, which compresses how much scheduling can visibly
contribute. The three-way ladder in `bench.py` — static (no retirement, no
admission), engine (retirement only), scheduler (both) — was built to separate
those effects, but padding waste sits upstream of all three.

**Measurement scope.** Percentiles come from 32–64 requests per configuration, and
the correctness margin from 330 decode steps. These are real measurements at a real
sample size, not a large one.

**Scope.** float32 only, greedy decoding only, single GPU, non-streaming. No
sampling, quantization, prefix caching, eviction, cancellation, or metrics endpoint.
The static baseline simulates slot occupancy with shadow sequences; folding that
into the engine's decode path would remove around 30 lines.

## Roadmap

Paged KV cache with block tables and eviction · prefix caching for shared system
prompts · speculative decoding · SSE streaming and request cancellation · chunked
prefill · INT8/INT4 quantization as a throughput–quality comparison · a Triton
paged-attention kernel · priority scheduling with per-tenant SLOs.

## Why not just use vLLM

In production, you would. The point was understanding what it does.