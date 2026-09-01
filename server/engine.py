"""Batched prefill + decode execution on top of SlotKVCache.

This is a thin executor: it runs the batches it is given. It does not decide
which requests go into a batch or when — that is scheduler.py's job (not yet
built). Greedy argmax only; no sampling, no repetition penalty, no temperature.

Left-padded prefill batches are the other place cache_len and RoPE position
diverge in this project (see kv_cache.py's module docstring for the general
idea): every sequence in a joint prefill batch ends up with the same physical
cache_len (the padded batch width), while each keeps its own, usually smaller,
next_position (its real prompt length) — a short prompt's next generated
token still gets RoPE position len(prompt), not the padded width.
"""

import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from server.config import ServerConfig
from server.kv_cache import SlotKVCache
from server.types import Request, RequestMetrics, SequenceState


class Engine:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config.model_id, dtype=config.dtype, attn_implementation=config.attn_implementation
            )
            .to(config.device)
            .eval()
        )

        hf_config = AutoConfig.from_pretrained(config.model_id)
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        self.cache = SlotKVCache(
            num_hidden_layers=hf_config.num_hidden_layers,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            max_slots=config.max_slots,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            device=config.device,
        )

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = []
        elif isinstance(eos, int):
            eos = [eos]
        self.eos_token_ids = set(eos)

        self._pending: list[SequenceState] = []
        self._active: list[SequenceState] = []

    def _assert_device(self, **tensors: torch.Tensor) -> None:
        """Fails fast, at the boundary where tensors enter the model, instead
        of deep inside an embedding lookup with a confusing device-mismatch
        traceback. See CLAUDE.md: every tensor built in this repo must take
        its device explicitly — the silent default is CPU, which is why this
        class of bug passes the whole suite locally and only shows up on GPU."""
        model_device = next(self.model.parameters()).device
        for name, tensor in tensors.items():
            if tensor.device != model_device:
                raise AssertionError(
                    f"{name} is on device {tensor.device!s}, but the model is on {model_device!s}. "
                    "Every tensor fed into the model or cache must be built with device=self.device."
                )

    def _mark_if_finished(self, seq: SequenceState, token_id: int, now: float) -> None:
        # TIMING RULE (CLAUDE.md "Timing rules"): `now` must already be a
        # post-sync timestamp handed in by the caller — this method performs
        # no GPU synchronization of its own, so it cannot fix a bad `now`.
        if token_id in self.eos_token_ids:
            seq.finished = True
            seq.finish_reason = "stop"
            seq.metrics.finish_time = now
        elif len(seq.output_token_ids) >= seq.max_new_tokens:
            seq.finished = True
            seq.finish_reason = "length"
            seq.metrics.finish_time = now

    def prefill(self, seqs: list[SequenceState]) -> None:
        if not seqs:
            return

        for seq in seqs:
            if seq.slot_id is None:
                slot_id = self.cache.allocate_slot()
                if slot_id is None:
                    raise RuntimeError(f"no free slot for request {seq.request_id!r}; caller over-admitted")
                seq.slot_id = slot_id

        lengths = [len(seq.prompt_token_ids) for seq in seqs]
        max_len = max(lengths)
        batch_size = len(seqs)

        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        attention_mask_2d = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        for i, (seq, length) in enumerate(zip(seqs, lengths)):
            input_ids[i, max_len - length :] = torch.tensor(seq.prompt_token_ids, dtype=torch.long, device=self.device)
            attention_mask_2d[i, max_len - length :] = 1

        slot_ids = torch.tensor([seq.slot_id for seq in seqs], dtype=torch.long, device=self.device)
        pad_lens = torch.tensor([max_len - length for length in lengths], dtype=torch.long, device=self.device)
        self.cache.pad_len[slot_ids] = pad_lens

        position_ids = (attention_mask_2d.cumsum(-1) - 1).clamp(min=0)
        write_start = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        q_len = max_len
        window = q_len  # write_start is 0 for every sequence in a prefill batch

        # TIMING RULE (CLAUDE.md "Timing rules"): this is a START-of-work
        # timestamp, captured before any forward pass is enqueued, so there is
        # nothing to sync against yet — it is the one *_time field exempt from
        # the "materialize-then-timestamp" rule below. Do not use this as a
        # precedent for skipping the sync before a COMPLETION timestamp.
        now = time.time()
        for seq in seqs:
            seq.metrics.prefill_start_time = now

        self.cache.set_step(slot_ids, write_start, q_len)
        mask = self.cache.build_attention_mask(slot_ids, write_start, q_len, window)
        self._assert_device(
            input_ids=input_ids, attention_mask=mask, position_ids=position_ids, slot_ids=slot_ids, write_start=write_start
        )
        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=mask, position_ids=position_ids, past_key_values=self.cache, use_cache=True
            )

        assert bool(attention_mask_2d[:, -1].all()), (
            "left-padding invariant violated: every sequence's last real token must land in the "
            "final column, but at least one row has padding there"
        )
        # TIMING RULE (CLAUDE.md "Timing rules"): first_token_time is a
        # COMPLETION timestamp — it must never be captured before the GPU work
        # it represents has actually finished. .tolist() forces the
        # device-to-host sync in one shot; only call time.time() AFTER it
        # returns. Reordering these two lines silently breaks TTFT on CUDA
        # while looking completely correct on CPU.
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()
        now = time.time()
        for i, (seq, length) in enumerate(zip(seqs, lengths)):
            seq.metrics.first_token_time = now
            seq.cache_len = int(self.cache.cache_len[seq.slot_id].item())
            seq.next_position = length
            token_id = next_tokens[i]
            seq.output_token_ids.append(token_id)
            self._mark_if_finished(seq, token_id, now)

    def decode(self, seqs: list[SequenceState]) -> None:
        if not seqs:
            return
        assert all(not seq.finished for seq in seqs), "decode() must only be called with unfinished sequences"

        slot_ids = torch.tensor([seq.slot_id for seq in seqs], dtype=torch.long, device=self.device)
        write_start = torch.tensor([seq.cache_len for seq in seqs], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[seq.next_position] for seq in seqs], dtype=torch.long, device=self.device)
        input_ids = torch.tensor([[seq.output_token_ids[-1]] for seq in seqs], dtype=torch.long, device=self.device)

        q_len = 1
        window = int((write_start + q_len).max())
        self.cache.set_step(slot_ids, write_start, q_len)
        mask = self.cache.build_attention_mask(slot_ids, write_start, q_len, window)
        self._assert_device(
            input_ids=input_ids, attention_mask=mask, position_ids=position_ids, slot_ids=slot_ids, write_start=write_start
        )
        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=mask, position_ids=position_ids, past_key_values=self.cache, use_cache=True
            )
        # TIMING RULE (CLAUDE.md "Timing rules"): same completion-timestamp
        # requirement as prefill() above — materialize via .tolist() FIRST,
        # then read time.time(), never the reverse. `now` here also becomes
        # finish_time via _mark_if_finished(), so this is the only sync point
        # for that field too.
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()
        now = time.time()
        for i, seq in enumerate(seqs):
            token_id = next_tokens[i]
            seq.output_token_ids.append(token_id)
            seq.cache_len = int(self.cache.cache_len[seq.slot_id].item())
            seq.next_position += 1
            self._mark_if_finished(seq, token_id, now)

    def generate(self, requests: list[Request]) -> list[SequenceState]:
        """Static path: every request prefilled together, no admission mid-run."""
        seqs = [
            SequenceState(
                request_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                max_new_tokens=req.max_new_tokens,
                metrics=RequestMetrics(arrival_time=req.arrival_time),
            )
            for req in requests
        ]

        self.prefill(seqs)
        active = [seq for seq in seqs if not seq.finished]
        while active:
            self.decode(active)
            active = [seq for seq in active if not seq.finished]

        for seq in seqs:
            self.cache.free_slot(seq.slot_id)
        return seqs

    def submit(self, request: Request) -> None:
        self._pending.append(
            SequenceState(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                max_new_tokens=request.max_new_tokens,
                metrics=RequestMetrics(arrival_time=request.arrival_time),
            )
        )

    def has_work(self) -> bool:
        return bool(self._pending) or bool(self._active)

    def step(self) -> list[SequenceState]:
        if self._pending:
            newly_admitted, self._pending = self._pending, []
            self.prefill(newly_admitted)
            self._active.extend(newly_admitted)

        unfinished = [seq for seq in self._active if not seq.finished]
        if unfinished:
            self.decode(unfinished)

        finished = [seq for seq in self._active if seq.finished]
        for seq in finished:
            self.cache.free_slot(seq.slot_id)
        self._active = [seq for seq in self._active if not seq.finished]
        return finished
