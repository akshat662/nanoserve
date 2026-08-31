"""Gate 0 — correctness spike for nanoserve.

Proves a hand-rolled, explicit-KV-cache decode loop reproduces model.generate()
exactly (Gate 0A), then proves the same loop still matches when the default
attention mask is replaced by an explicit 4D additive mask we build ourselves
(Gate 0B), since Phase 2 builds its own masks for continuous batching.

INTERPRETING FAILURE:
A max logit delta around 1e-6 or smaller with a token mismatch means a
floating-point tie-flip, not a bug. A delta of 1e-2 or larger means a real
bug — most likely position_ids not absolute, or the mask wrong.

NOTE ON THE REFERENCE CALL: Qwen2.5-0.5B-Instruct ships a generation_config
with repetition_penalty=1.1 baked in. That penalty is a LogitsProcessor, not
a sampling warper, so it stays active even with do_sample=False and will
silently make generate() diverge from a plain greedy argmax loop. The
reference call below passes repetition_penalty=1.0 to neutralize it — this
is not optional, it was required to get Gate 0A to pass during development.
"""

import inspect
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, StaticCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402

PROMPTS = [
    "Explain photosynthesis briefly.",
    "Write a short paragraph about why the sky appears blue to a human observer.",
    "Hi.",
]
NUM_NEW_TOKENS = 30
CLAUDE_MD_PATH = Path(__file__).resolve().parent.parent / "CLAUDE.md"


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def eos_ids_from(generation_config) -> list[int]:
    eos = generation_config.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return list(eos)


def suppress_eos(logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Mirror generate()'s MinNewTokensLength processor: we always request
    min_new_tokens == max_new_tokens, so EOS is suppressed for every step."""
    logits = logits.clone()
    if eos_ids:
        logits[:, eos_ids] = -float("inf")
    return logits


def build_chat_ids(tok, prompt: str, device: str) -> torch.Tensor:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)


def reference_generate(model, ids: torch.Tensor, num_new_tokens: int):
    common = dict(
        max_new_tokens=num_new_tokens,
        min_new_tokens=num_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        return_dict_in_generate=True,
    )
    try:
        out = model.generate(ids, output_logits=True, **common)
        return out, out.logits, "output_logits"
    except TypeError:
        out = model.generate(ids, output_scores=True, **common)
        return out, out.scores, "output_scores (output_logits unsupported on this transformers version)"


def causal_4d_mask(q_len: int, kv_len: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    neg = torch.finfo(dtype).min
    m = torch.full((q_len, kv_len), neg, dtype=dtype, device=device)
    m = torch.triu(m, diagonal=kv_len - q_len + 1)
    return m.unsqueeze(0).unsqueeze(0)  # [batch=1, 1, q_len, kv_len]


def decode_4d_mask(kv_len: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.zeros((1, 1, 1, kv_len), dtype=dtype, device=device)


def hand_rolled_decode(
    model,
    ids: torch.Tensor,
    num_new_tokens: int,
    eos_ids: list[int],
    use_explicit_mask: bool = False,
    dtype: torch.dtype | None = None,
):
    """Prefill with use_cache=True, then decode num_new_tokens - 1 more steps.
    Each decode step passes only the new token, past_key_values, and an
    explicit absolute position_ids tensor. Optionally builds its own 4D
    additive attention mask instead of relying on the model's default."""
    device = ids.device
    prompt_len = ids.shape[1]

    with torch.no_grad():
        forward_kwargs = {}
        if use_explicit_mask:
            forward_kwargs["attention_mask"] = causal_4d_mask(prompt_len, prompt_len, dtype, device)
        out = model(ids, use_cache=True, **forward_kwargs)
        past = out.past_key_values
        raw_logits = out.logits[:, -1, :]
        next_tok = torch.argmax(suppress_eos(raw_logits, eos_ids), dim=-1)
        tokens = [next_tok.item()]
        logits = [raw_logits[0].clone()]
        cur = next_tok.unsqueeze(0)

        for i in range(1, num_new_tokens):
            pos = torch.tensor([[prompt_len + i - 1]], device=device)
            forward_kwargs = {}
            if use_explicit_mask:
                kv_len = prompt_len + i
                forward_kwargs["attention_mask"] = decode_4d_mask(kv_len, dtype, device)
            out = model(cur, past_key_values=past, position_ids=pos, use_cache=True, **forward_kwargs)
            past = out.past_key_values
            raw_logits = out.logits[:, -1, :]
            next_tok = torch.argmax(suppress_eos(raw_logits, eos_ids), dim=-1)
            tokens.append(next_tok.item())
            logits.append(raw_logits[0].clone())
            cur = next_tok.unsqueeze(0)

    return tokens, torch.stack(logits, dim=0)


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def record_eager_fallback_in_claude_md() -> None:
    note = (
        "\n- Gate 0B finding (scratch/gate0.py): sdpa rejected an explicit 4D "
        'additive attention mask; use attn_implementation="eager" for Phase 2\'s '
        "hand-built masks.\n"
    )
    with open(CLAUDE_MD_PATH, "a") as f:
        f.write(note)


def main() -> int:
    config = load_config()

    section("SECTION 1: ENVIRONMENT")
    print(f"transformers.__version__: {transformers.__version__}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"resolved device: {config.device}")
    print(f"resolved dtype: {config.dtype}")

    tok = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=config.dtype, attn_implementation=config.attn_implementation
    )
    model = model.to(config.device).eval()

    eos_config_value = model.generation_config.eos_token_id
    print(f"model.generation_config.eos_token_id: {eos_config_value!r} (type={type(eos_config_value).__name__})")
    eos_ids = eos_ids_from(model.generation_config)

    section("SECTION 2: CACHE API RECONNAISSANCE")
    sample_ids = build_chat_ids(tok, PROMPTS[0], config.device)
    with torch.no_grad():
        recon_out = model(sample_ids, use_cache=True)
    pkv = recon_out.past_key_values
    is_legacy_tuple = isinstance(pkv, tuple)
    print(f"type(out.past_key_values).__name__: {type(pkv).__name__}")
    print(f"legacy tuple: {is_legacy_tuple}")
    if not is_legacy_tuple:
        public_methods = sorted(m for m in dir(pkv) if not m.startswith("_"))
        print(f"Cache object public methods: {public_methods}")

    try:
        from transformers import Cache as _Cache  # noqa: F401
        from transformers import StaticCache as _StaticCache  # noqa: F401

        cache_imports_ok = True
    except ImportError:
        cache_imports_ok = False
    print(f"`from transformers import Cache, StaticCache` both succeed: {cache_imports_ok}")

    forward_sig = inspect.signature(model.forward)
    accepts_cache_position_param = "cache_position" in forward_sig.parameters
    print(f"model.forward signature: {forward_sig}")
    print(f"'cache_position' is a named parameter: {accepts_cache_position_param}")
    if not accepts_cache_position_param:
        print("  (it is absorbed by **kwargs typed as Unpack[TransformersKwargs] instead)")

    section("SECTION 3: GATE 0A — HAND-ROLLED DECODE MATCHES generate()")
    gate_0a_pass = True
    gate_0a_tokens: dict[str, list[int]] = {}
    gate_0a_logits: dict[str, torch.Tensor] = {}

    for prompt in PROMPTS:
        ids = build_chat_ids(tok, prompt, config.device)
        prompt_len = ids.shape[1]

        ref_out, ref_raw, logit_source = reference_generate(model, ids, NUM_NEW_TOKENS)
        ref_tokens = ref_out.sequences[0, prompt_len:].tolist()
        ref_logits = torch.stack(ref_raw, dim=0).squeeze(1)

        hand_tokens, hand_logits = hand_rolled_decode(model, ids, NUM_NEW_TOKENS, eos_ids)

        match = ref_tokens == hand_tokens
        mismatch_idx = first_mismatch(ref_tokens, hand_tokens)
        max_delta = (ref_logits - hand_logits).abs().max().item()
        if not match:
            gate_0a_pass = False

        gate_0a_tokens[prompt] = hand_tokens
        gate_0a_logits[prompt] = hand_logits

        print(f"--- prompt: {prompt!r} ---")
        print(f"  reference logit source: {logit_source}")
        print(f"  prompt length (tokens): {prompt_len}")
        mismatch_note = "" if match else f" (first mismatch at index {mismatch_idx})"
        print(f"  token match: {match}{mismatch_note}")
        print(f"  max abs logit delta: {max_delta:.3e}")
        print(f"  reference text: {tok.decode(ref_tokens)!r}")
        print(f"  hand-loop text: {tok.decode(hand_tokens)!r}")

    section("SECTION 4: GATE 0B — EXPLICIT 4D ATTENTION MASK")
    gate_0b_prompt = PROMPTS[0]
    ids = build_chat_ids(tok, gate_0b_prompt, config.device)
    attn_impl_used = config.attn_implementation
    gate_0b_pass = True

    try:
        tokens_0b, logits_0b = hand_rolled_decode(
            model, ids, NUM_NEW_TOKENS, eos_ids, use_explicit_mask=True, dtype=config.dtype
        )
    except RuntimeError as exc:
        print(f"{attn_impl_used} rejected the 4D mask: {exc}")
        print("retrying once with attn_implementation=\"eager\"")
        eager_model = (
            AutoModelForCausalLM.from_pretrained(config.model_id, dtype=config.dtype, attn_implementation="eager")
            .to(config.device)
            .eval()
        )
        tokens_0b, logits_0b = hand_rolled_decode(
            eager_model, ids, NUM_NEW_TOKENS, eos_ids, use_explicit_mask=True, dtype=config.dtype
        )
        attn_impl_used = "eager"
        record_eager_fallback_in_claude_md()
        print("recorded eager fallback in CLAUDE.md")

    print(f"attn_implementation that handled the explicit 4D mask: {attn_impl_used}")

    reference_0a_tokens = gate_0a_tokens[gate_0b_prompt]
    try:
        assert tokens_0b == reference_0a_tokens, (
            f"Gate 0B tokens diverged from Gate 0A at prompt {gate_0b_prompt!r}: "
            f"first mismatch at index {first_mismatch(reference_0a_tokens, tokens_0b)}"
        )
    except AssertionError as exc:
        print(f"ASSERTION FAILED: {exc}")
        gate_0b_pass = False

    gate_0b_delta = (gate_0a_logits[gate_0b_prompt] - logits_0b).abs().max().item()
    print(f"tokens identical to Gate 0A: {tokens_0b == reference_0a_tokens}")
    print(f"max abs logit delta (0A vs 0B): {gate_0b_delta:.3e}")

    section("FINAL")
    print(f"GATE 0A: {'PASS' if gate_0a_pass else 'FAIL'}")
    print(f"GATE 0B: {'PASS' if gate_0b_pass else 'FAIL'}")

    return 0 if (gate_0a_pass and gate_0b_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
