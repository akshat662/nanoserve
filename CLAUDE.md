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
