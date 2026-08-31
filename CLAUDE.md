# nanoserve — project constraints

These constraints are load-bearing. Do not relax them without explicit instruction.

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Device-agnostic ALWAYS. Never hardcode .cuda(). Device comes from config; default
  "cuda" if torch.cuda.is_available() else "cpu". Development happens on CPU,
  benchmarking on a T4.
- dtype: float32 for all correctness work. FP16 changes matmul reduction order with
  batch shape and flips argmax on near-ties, producing fake test failures.
- attn_implementation must be "sdpa" or "eager", never flash_attention_2 — we build
  our own 4D attention masks and FA2 will not accept them.
- Qwen2.5 uses grouped-query attention: KV cache is sized by num_key_value_heads,
  NOT num_attention_heads.
- V1 non-goals, do not build: paged KV cache/block tables, cache eviction, SSE
  streaming, request cancellation, /metrics, backpressure, Docker, auth, UI,
  quantization, multi-GPU, sampling (greedy only), custom CUDA/Triton kernels.
- Target size: ~700 lines total across 7 source files. Resist scope growth.
- Every file must have a stated verification criterion that is run before moving on.

## Verified environment facts (Gate 0, 31 Aug 2026)

- transformers 5.16.1, torch 2.13.0. Do NOT assume the transformers 4.x Cache API.
- `past_key_values` is a `DynamicCache` object, never a legacy tuple.
  `cache_position` is accepted but absorbed via **kwargs, not a named forward param.
- `eos_token_id` is a LIST: [151645, 151643]. Finish detection must be
  `token_id in eos_token_ids`, never `token_id == eos_token_id`.
- sdpa accepts an explicit 4D additive attention mask of shape [B, 1, q_len, kv_len]
  (0.0 attendable, torch.finfo(dtype).min masked). Verified logit delta exactly 0.0
  against the default mask. Build masks ourselves; do not fall back to eager.
- Qwen's shipped `generation_config` bakes in `repetition_penalty=1.1`, and it stays
  active under `do_sample=False` because it is a logits processor, not a sampling
  warper. Our engine does PURE ARGMAX with no penalty. Therefore any comparison
  against `model.generate()` MUST pass `repetition_penalty=1.0`, or it will not match.
- Verified KV cache arithmetic: 2 * 24 layers * 2 kv_heads * 64 head_dim = 12288
  bytes/token fp16, 24576 fp32. At 32 slots x 1024 max_seq_len: 384 MB fp16,
  768 MB fp32. These are the numbers that go in the README, not estimates.
- Two distinct correctness numbers exist and must never be conflated: single-sequence
  vs DynamicCache = 0.000e+00 (bit-identical), and batch vs single = ~1.8e-04 (kernel
  reduction-order noise from a different matmul batch shape, expected, not a defect).
- Token identity across batch shapes holds because argmax margin exceeds that noise by
  a measured factor of 166.2x (min argmax margin 3.027e-02 over max abs logit delta
  1.822e-04, across 330 decode steps over 20 prompts). This is measured, not assumed —
  see tests/test_numerics.py.
- float16 is not safe for correctness work: it would shrink that safety ratio by
  roughly three orders of magnitude.

## Timing rules (learned the hard way)

- CUDA is asynchronous. A timestamp taken before the GPU work has actually completed
  measures nothing. Every timestamp in this repo must come AFTER a real sync point.
- `.item()` and `.tolist()` are sync points; a bare tensor assignment is not. engine.py
  originally captured `time.time()` before the `.item()` calls that pulled results off
  the device, which meant every TTFT and finish_time on a GPU would have been recorded
  early. Fixed by materializing results via `.tolist()` FIRST, then timestamping. This
  was invisible on CPU and would never have surfaced in the test suite.
- Rule for all future work: in any function that records a timestamp marking the
  COMPLETION of GPU work (first_token_time, finish_time, and anything like them), the
  line immediately before it must be either a `.tolist()`/`.item()` materialization or
  an explicit `torch.cuda.synchronize()` guarded by `torch.cuda.is_available()`. No
  exceptions, including in scheduler.py when it lands. The one narrow non-exception: a
  START-of-work timestamp (e.g. prefill_start_time, captured before any forward pass is
  enqueued) has nothing to sync against yet and isn't covered by this rule — don't treat
  that as a loophole to skip syncing an actual completion timestamp.
- bench.py's `now()` helper is the canonical implementation. Do not write ad-hoc
  `time.perf_counter()` calls elsewhere in timed paths; import or mirror `now()`.
- Second bug from the same session, also recorded: `submit()` does not consume a slot —
  only `prefill()` inside `step()` does. Any caller checking `num_free_slots()` must
  sample it once per scheduling iteration, not once per submission, or it will
  over-admit and hit "no free slot".
