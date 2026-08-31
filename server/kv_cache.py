"""Preallocated, slot-based KV cache for continuous batching.

CENTRAL IDEA: cache index and RoPE position are DIFFERENT NUMBERS. A slot's
write index (cache_len) is purely about where in that slot's preallocated
storage the next token's K/V lands. A sequence's RoPE position_ids come from
its own token count since it started. These coincide only for a sequence that
has occupied its slot since cache index 0. A sequence admitted mid-run (or
whose slot was reused) can occupy cache indices that do not equal its own
token positions — e.g. a freshly admitted sequence dropped into a slot last
used by a 40-token sequence starts writing at cache index 0 with RoPE
position 0, while a sequence that has been resident since the start of the
batch continues writing at cache index 40 for its 41st token. The two numbers
must never be conflated or derived from one another; that is what
`write_start` (cache index) and `position_ids` (RoPE position, tracked
separately by the caller as SequenceState.next_position) are for.

Verified against transformers==5.16.1 (see CLAUDE.md "Verified environment
facts"): `past_key_values.update(key_states, value_states, layer_idx)` is the
only Cache method the real Qwen2 forward pass calls when we always supply
explicit absolute position_ids and an explicit 4D attention mask — both
get_seq_length() (modeling_qwen2.py) and get_mask_sizes()/get_query_offset()
(masking_utils.py) are gated behind branches that our usage never takes.
Cache.__init__ is not called: it requires either `layers` or
`layer_class_to_replicate`, machinery this cache does not use at all.
"""

import torch
from transformers import Cache


class SlotKVCache(Cache):
    def __init__(
        self,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        max_slots: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        # Deliberately not calling super().__init__(): Cache.__init__ demands
        # `layers` or `layer_class_to_replicate`, the per-layer CacheLayerMixin
        # abstraction this cache replaces entirely with flat preallocated tensors.
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        store_shape = (max_slots, num_key_value_heads, max_seq_len, head_dim)
        self.k_store = [torch.zeros(store_shape, dtype=dtype, device=device) for _ in range(num_hidden_layers)]
        self.v_store = [torch.zeros(store_shape, dtype=dtype, device=device) for _ in range(num_hidden_layers)]

        self.cache_len = torch.zeros(max_slots, dtype=torch.long, device=device)
        self.pad_len = torch.zeros(max_slots, dtype=torch.long, device=device)

        self._free_slots = list(range(max_slots))
        self._slot_ids: torch.Tensor | None = None
        self._write_start: torch.Tensor | None = None
        self._q_len: int | None = None

    def __len__(self) -> int:
        return self.num_hidden_layers

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_slots={self.max_slots}, free={self.num_free_slots()})"

    # --- slot lifecycle ---

    def allocate_slot(self) -> int | None:
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free_slot(self, slot_id: int) -> None:
        self.cache_len[slot_id] = 0
        self.pad_len[slot_id] = 0
        self._free_slots.append(slot_id)

    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def memory_bytes(self) -> int:
        per_store_tensor = self.k_store[0].numel() * self.k_store[0].element_size()
        return 2 * self.num_hidden_layers * per_store_tensor

    # --- per-step contract ---

    def set_step(self, slot_ids: torch.Tensor, write_start: torch.Tensor, q_len: int) -> None:
        self._slot_ids = slot_ids
        self._write_start = write_start
        self._q_len = q_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if args or kwargs:
            raise NotImplementedError(
                f"SlotKVCache.update got unexpected extra args={args} kwargs={list(kwargs)}; "
                "recon of transformers==5.16.1 showed Qwen2Attention calls update() with exactly "
                "(key_states, value_states, layer_idx) and nothing else — this cache does not "
                "guess at behavior for a call shape it has not been verified against."
            )
        if self._slot_ids is None:
            raise RuntimeError("update() called before set_step(); every forward pass must call set_step() first")

        slot_ids = self._slot_ids
        write_start = self._write_start
        q_len = self._q_len
        batch_size = slot_ids.shape[0]

        if key_states.shape[0] != batch_size or key_states.shape[2] != q_len:
            raise ValueError(
                f"key_states shape {tuple(key_states.shape)} does not match set_step's "
                f"batch_size={batch_size}, q_len={q_len}"
            )

        k_store = self.k_store[layer_idx]
        v_store = self.v_store[layer_idx]

        if q_len == 1:
            k_store[slot_ids, :, write_start, :] = key_states[:, :, 0, :]
            v_store[slot_ids, :, write_start, :] = value_states[:, :, 0, :]
        else:
            cache_idx = write_start[:, None] + torch.arange(q_len, device=self.device)[None, :]  # [B, q_len]
            slot_idx = slot_ids[:, None].expand(batch_size, q_len)  # [B, q_len]
            k_store[slot_idx, :, cache_idx, :] = key_states.permute(0, 2, 1, 3)
            v_store[slot_idx, :, cache_idx, :] = value_states.permute(0, 2, 1, 3)

        if layer_idx == self.num_hidden_layers - 1:
            self.cache_len[slot_ids] = write_start + q_len

        window = int((write_start + q_len).max())
        return k_store[slot_ids, :, :window, :], v_store[slot_ids, :, :window, :]

    def build_attention_mask(
        self,
        slot_ids: torch.Tensor,
        write_start: torch.Tensor,
        q_len: int,
        window: int,
    ) -> torch.Tensor:
        """Additive float mask, shape [B, 1, q_len, window]. A query at window
        row r (absolute cache index write_start[i] + r) may attend to cache
        index c only when pad_len[slot] <= c <= write_start[i] + r: left
        padding excluded, causal within the sequence, and everything at or
        beyond this sequence's own frontier excluded (so a shorter sequence
        never attends into a longer batch-mate's still-being-written tail)."""
        pad = self.pad_len[slot_ids].view(-1, 1, 1)  # [B, 1, 1]
        query_abs_pos = (write_start[:, None] + torch.arange(q_len, device=self.device)[None, :]).view(
            -1, q_len, 1
        )  # [B, q_len, 1]
        col = torch.arange(window, device=self.device).view(1, 1, -1)  # [1, 1, window]

        attendable = (col >= pad) & (col <= query_abs_pos)  # [B, q_len, window]

        neg = torch.finfo(self.dtype).min
        mask = torch.full((slot_ids.shape[0], q_len, window), neg, dtype=self.dtype, device=self.device)
        mask = mask.masked_fill(attendable, 0.0)
        return mask.unsqueeze(1)

    # --- methods the real forward path never reaches under our usage (see module docstring) ---

    def get_seq_length(self, layer_idx: int = 0) -> int:
        raise NotImplementedError(
            "get_seq_length() is ambiguous for a multi-slot cache (there is no single sequence "
            "length) and is only ever called by modeling_qwen2.py when position_ids is None. "
            "This engine always supplies explicit absolute position_ids; read SequenceState.cache_len "
            "for a given slot instead."
        )

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        raise NotImplementedError(
            "get_mask_sizes() is only reached by masking_utils._preprocess_mask_arguments() when the "
            "attention_mask is not already a 4D tensor. This engine always builds its own 4D mask via "
            "build_attention_mask(), so this path should be unreachable — treat hitting it as a bug."
        )

    def get_query_offset(self, layer_idx: int = 0) -> int:
        raise NotImplementedError(
            "get_query_offset() is only reached by masking_utils._preprocess_mask_arguments() when the "
            "attention_mask is not already a 4D tensor, which this engine never does — see get_mask_sizes()."
        )
