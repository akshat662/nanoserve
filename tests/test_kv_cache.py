"""Verification for SlotKVCache. Test 3 (test_single_sequence_matches_gate0) and
test 4 (test_isolation_between_slots) are the real gates: everything else in the
project depends on this file's cache producing bit-for-bit correct attention.

repetition_penalty is never applied anywhere here — pure argmax only, per CLAUDE.md.
"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_config  # noqa: E402
from server.kv_cache import SlotKVCache  # noqa: E402

NUM_NEW_TOKENS = 30
MAX_SLOTS = 8
MAX_SEQ_LEN = 128

PROMPT_A = "Hi."
PROMPT_B = "Write a short paragraph about why the sky appears blue to a human observer."


def eos_ids_from(generation_config) -> list[int]:
    eos = generation_config.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return list(eos)


def suppress_eos(logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    logits = logits.clone()
    if eos_ids:
        logits[:, eos_ids] = -float("inf")
    return logits


def build_chat_ids(tok, prompt: str, device: str) -> torch.Tensor:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    return enc["input_ids"].to(device)


@pytest.fixture(scope="module")
def env():
    config = load_config()
    tok = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=config.dtype, attn_implementation=config.attn_implementation
    )
    model = model.to(config.device).eval()
    eos_ids = eos_ids_from(model.generation_config)
    hf_config = AutoConfig.from_pretrained(config.model_id)
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    return {
        "config": config,
        "tok": tok,
        "model": model,
        "eos_ids": eos_ids,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "head_dim": head_dim,
    }


def make_cache(env, max_slots: int = MAX_SLOTS, max_seq_len: int = MAX_SEQ_LEN) -> SlotKVCache:
    return SlotKVCache(
        num_hidden_layers=env["num_hidden_layers"],
        num_key_value_heads=env["num_key_value_heads"],
        head_dim=env["head_dim"],
        max_slots=max_slots,
        max_seq_len=max_seq_len,
        dtype=env["config"].dtype,
        device=env["config"].device,
    )


# --- shared driver: prefill each sequence alone, then decode all of them jointly ---


def forward_step(model, cache, slot_ids, write_start, q_len, input_ids, position_ids):
    window = int((write_start + q_len).max())
    cache.set_step(slot_ids, write_start, q_len)
    mask = cache.build_attention_mask(slot_ids, write_start, q_len, window)
    with torch.no_grad():
        out = model(input_ids, attention_mask=mask, position_ids=position_ids, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :]


def run_sequences_with_slot_cache(model, cache, sequences, num_new_tokens, eos_ids, device):
    """sequences: list of {"slot_id": int, "ids": LongTensor[1, prompt_len]}.
    Returns {slot_id: (tokens, logits[num_new_tokens, vocab])}."""
    slot_id_list = [s["slot_id"] for s in sequences]
    tokens = {sid: [] for sid in slot_id_list}
    logits_hist = {sid: [] for sid in slot_id_list}
    next_input = {}
    next_position = {}

    for s in sequences:
        slot_id, ids = s["slot_id"], s["ids"]
        q_len = ids.shape[1]
        slot_ids_t = torch.tensor([slot_id], device=device)
        write_start = cache.cache_len[slot_ids_t].clone()
        position_ids = write_start[:, None] + torch.arange(q_len, device=device)[None, :]
        raw_logits = forward_step(model, cache, slot_ids_t, write_start, q_len, ids, position_ids)
        next_tok = torch.argmax(suppress_eos(raw_logits, eos_ids), dim=-1)
        tokens[slot_id].append(next_tok.item())
        logits_hist[slot_id].append(raw_logits[0].clone())
        next_input[slot_id] = next_tok.unsqueeze(0)
        next_position[slot_id] = q_len

    for _ in range(1, num_new_tokens):
        slot_ids_t = torch.tensor(slot_id_list, device=device)
        write_start = cache.cache_len[slot_ids_t].clone()
        cur_input = torch.cat([next_input[sid] for sid in slot_id_list], dim=0)
        position_ids = torch.tensor([[next_position[sid]] for sid in slot_id_list], device=device)
        raw_logits = forward_step(model, cache, slot_ids_t, write_start, 1, cur_input, position_ids)
        next_tok = torch.argmax(suppress_eos(raw_logits, eos_ids), dim=-1)
        for i, sid in enumerate(slot_id_list):
            tokens[sid].append(next_tok[i].item())
            logits_hist[sid].append(raw_logits[i].clone())
            next_input[sid] = next_tok[i].view(1, 1)
            next_position[sid] += 1

    return {sid: (tokens[sid], torch.stack(logits_hist[sid], dim=0)) for sid in slot_id_list}


def decode_with_dynamic_cache(model, ids, num_new_tokens, eos_ids):
    """Plain DynamicCache hand loop — the Gate 0A reference path."""
    prompt_len = ids.shape[1]
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        raw = out.logits[:, -1, :]
        next_tok = torch.argmax(suppress_eos(raw, eos_ids), dim=-1)
        tokens = [next_tok.item()]
        logits = [raw[0].clone()]
        cur = next_tok.unsqueeze(0)
        for i in range(1, num_new_tokens):
            pos = torch.tensor([[prompt_len + i - 1]], device=ids.device)
            out = model(cur, past_key_values=past, position_ids=pos, use_cache=True)
            past = out.past_key_values
            raw = out.logits[:, -1, :]
            next_tok = torch.argmax(suppress_eos(raw, eos_ids), dim=-1)
            tokens.append(next_tok.item())
            logits.append(raw[0].clone())
            cur = next_tok.unsqueeze(0)
    return tokens, torch.stack(logits, dim=0)


# --- tests ---


def test_slot_allocation():
    cache = SlotKVCache(
        num_hidden_layers=2, num_key_value_heads=2, head_dim=8, max_slots=4, max_seq_len=16, dtype=torch.float32,
        device="cpu",
    )
    allocated = [cache.allocate_slot() for _ in range(4)]
    assert sorted(allocated) == [0, 1, 2, 3]
    assert cache.allocate_slot() is None

    freed_id = allocated[2]
    cache.free_slot(freed_id)
    assert cache.num_free_slots() == 1
    assert cache.allocate_slot() == freed_id


def test_cache_len_increments_once_per_step():
    num_layers = 24
    cache = SlotKVCache(
        num_hidden_layers=num_layers, num_key_value_heads=2, head_dim=64, max_slots=4, max_seq_len=32,
        dtype=torch.float32, device="cpu",
    )
    slot_id = cache.allocate_slot()
    slot_ids = torch.tensor([slot_id])

    def run_step(q_len: int, write_start_val: int) -> None:
        write_start = torch.tensor([write_start_val])
        cache.set_step(slot_ids, write_start, q_len)
        for layer_idx in range(num_layers):
            k = torch.randn(1, 2, q_len, 64)
            v = torch.randn(1, 2, q_len, 64)
            cache.update(k, v, layer_idx)

    run_step(q_len=7, write_start_val=0)
    assert cache.cache_len[slot_id].item() == 7

    run_step(q_len=1, write_start_val=7)
    run_step(q_len=1, write_start_val=8)
    run_step(q_len=1, write_start_val=9)

    assert cache.cache_len[slot_id].item() == 10, "cache_len must increment once per step, not once per layer"


def test_single_sequence_matches_gate0(env):
    model, tok, device, eos_ids = env["model"], env["tok"], env["config"].device, env["eos_ids"]
    ids = build_chat_ids(tok, PROMPT_B, device)

    ref_tokens, ref_logits = decode_with_dynamic_cache(model, ids, NUM_NEW_TOKENS, eos_ids)

    cache = make_cache(env)
    slot_id = cache.allocate_slot()
    result = run_sequences_with_slot_cache(
        model, cache, [{"slot_id": slot_id, "ids": ids}], NUM_NEW_TOKENS, eos_ids, device
    )
    hand_tokens, hand_logits = result[slot_id]

    max_delta = (ref_logits - hand_logits).abs().max().item()
    print(f"\ntest_single_sequence_matches_gate0 max abs logit delta: {max_delta:.3e}")

    assert hand_tokens == ref_tokens


def test_isolation_between_slots(env):
    model, tok, device, eos_ids = env["model"], env["tok"], env["config"].device, env["eos_ids"]
    ids_a = build_chat_ids(tok, PROMPT_A, device)
    ids_b = build_chat_ids(tok, PROMPT_B, device)
    assert ids_a.shape[1] != ids_b.shape[1], "A and B must have different prompt lengths for this test to be meaningful"

    # A alone, in a fresh cache, as the ground truth for A in isolation.
    solo_cache = make_cache(env)
    solo_slot = solo_cache.allocate_slot()
    solo_result = run_sequences_with_slot_cache(
        model, solo_cache, [{"slot_id": solo_slot, "ids": ids_a}], NUM_NEW_TOKENS, eos_ids, device
    )
    solo_tokens_a, _ = solo_result[solo_slot]

    # A and B in two different, non-adjacent slots, batched together so the shared
    # decode window extends past A's own frontier into A's still-unwritten (zero)
    # cache tail. allocate_slot() order is an implementation detail (it's a LIFO
    # free list), so grab several slots and just use two distinct, separated ones.
    batched_cache = make_cache(env)
    allocated = [batched_cache.allocate_slot() for _ in range(6)]
    slot_a, slot_b = allocated[0], allocated[5]
    assert slot_a != slot_b

    batched_result = run_sequences_with_slot_cache(
        model,
        batched_cache,
        [{"slot_id": slot_a, "ids": ids_a}, {"slot_id": slot_b, "ids": ids_b}],
        NUM_NEW_TOKENS,
        eos_ids,
        device,
    )
    batched_tokens_a, _ = batched_result[slot_a]

    assert batched_tokens_a == solo_tokens_a, "sequence A's output changed when batched with a longer neighbour — mask is leaking into A's unwritten cache tail"


def test_slot_reuse_is_clean(env):
    model, tok, device, eos_ids = env["model"], env["tok"], env["config"].device, env["eos_ids"]
    ids_a = build_chat_ids(tok, PROMPT_A, device)
    ids_b = build_chat_ids(tok, PROMPT_B, device)

    shared_cache = make_cache(env)
    slot = shared_cache.allocate_slot()
    run_sequences_with_slot_cache(model, shared_cache, [{"slot_id": slot, "ids": ids_a}], NUM_NEW_TOKENS, eos_ids, device)

    shared_cache.free_slot(slot)  # does NOT zero the tensors
    assert shared_cache.allocate_slot() == slot
    reused_result = run_sequences_with_slot_cache(
        model, shared_cache, [{"slot_id": slot, "ids": ids_b}], NUM_NEW_TOKENS, eos_ids, device
    )
    reused_tokens_b, _ = reused_result[slot]

    fresh_cache = make_cache(env)
    fresh_slot = fresh_cache.allocate_slot()
    fresh_result = run_sequences_with_slot_cache(
        model, fresh_cache, [{"slot_id": fresh_slot, "ids": ids_b}], NUM_NEW_TOKENS, eos_ids, device
    )
    fresh_tokens_b, _ = fresh_result[fresh_slot]

    assert reused_tokens_b == fresh_tokens_b, "reusing a dirty slot changed the output — mask fails to hide stale data"
