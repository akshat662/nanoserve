"""Baseline static-batching engine: everyone admitted together, everyone decoded
in lockstep until the slowest sequence finishes. Wraps an existing Engine and
adds nothing but waste instrumentation.

Two wastes are tracked separately (different causes, different fixes):
prefill_waste is left-padding; decode_waste is head-of-line blocking — a
finished sequence keeps occupying its slot and riding through the forward
pass, output discarded, because nothing shrinks the batch until all are done.
"""

from dataclasses import asdict, dataclass

from server.engine import Engine
from server.types import Request, RequestMetrics, SequenceState


@dataclass
class WasteStats:
    prefill_useful: int
    prefill_total: int
    prefill_waste: int
    decode_useful: int
    decode_total: int
    decode_waste: int
    total_waste_frac: float

    @classmethod
    def compute(cls, prefill_useful: int, prefill_total: int, decode_useful: int, decode_total: int) -> "WasteStats":
        prefill_waste = prefill_total - prefill_useful
        decode_waste = decode_total - decode_useful
        total_waste_frac = (prefill_waste + decode_waste) / (prefill_total + decode_total)
        return cls(prefill_useful, prefill_total, prefill_waste, decode_useful, decode_total, decode_waste, total_waste_frac)

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"prefill: {self.prefill_useful}/{self.prefill_total} useful, {self.prefill_waste} wasted (left-padding)\n"
            f"decode:  {self.decode_useful}/{self.decode_total} useful, {self.decode_waste} wasted (head-of-line blocking)\n"
            f"total waste fraction: {self.total_waste_frac:.1%}"
        )


class StaticBatchEngine:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.cache = engine.cache
        self.tokenizer = engine.tokenizer
        self._pending: list[Request] = []
        self._waste_totals = {"prefill_useful": 0, "prefill_total": 0, "decode_useful": 0, "decode_total": 0}

    def generate(self, requests: list[Request]) -> tuple[list[SequenceState], WasteStats]:
        seqs = [
            SequenceState(
                request_id=req.request_id, prompt_token_ids=req.prompt_token_ids,
                max_new_tokens=req.max_new_tokens, metrics=RequestMetrics(arrival_time=req.arrival_time),
            )
            for req in requests
        ]
        prefill_useful = sum(len(s.prompt_token_ids) for s in seqs)
        self.engine.prefill(seqs)
        prefill_total = len(seqs) * max(len(s.prompt_token_ids) for s in seqs)

        num_decode_steps = 1  # every sequence's first token comes from prefill's own argmax
        while not all(s.finished for s in seqs):
            active = [s for s in seqs if not s.finished]
            finished_now = [s for s in seqs if s.finished]
            shadows = [self._shadow(s) for s in finished_now]
            self.engine.decode(active + shadows)  # finished ones ride along; their result is discarded below
            for s, shadow in zip(finished_now, shadows):
                s.cache_len, s.next_position = shadow.cache_len, shadow.next_position
            num_decode_steps += 1

        for s in seqs:
            self.engine.cache.free_slot(s.slot_id)

        waste = WasteStats.compute(
            prefill_useful, prefill_total,
            sum(len(s.output_token_ids) for s in seqs), len(seqs) * num_decode_steps,
        )
        return seqs, waste

    @staticmethod
    def _shadow(seq: SequenceState) -> SequenceState:
        """Throwaway stand-in for a finished sequence: same slot and position,
        so it pays the real wasted forward-pass cost without mutating the real
        SequenceState, whose output is already final."""
        return SequenceState(
            request_id=seq.request_id, prompt_token_ids=seq.prompt_token_ids, max_new_tokens=seq.max_new_tokens,
            metrics=RequestMetrics(arrival_time=0.0), slot_id=seq.slot_id, cache_len=seq.cache_len,
            next_position=seq.next_position, output_token_ids=list(seq.output_token_ids),
        )

    def submit(self, request: Request) -> None:
        self._pending.append(request)

    def has_work(self) -> bool:
        return bool(self._pending)

    def step(self) -> list[SequenceState]:
        """Drains whatever is currently pending and runs it as one static
        batch to completion — a request that arrives mid-batch simply waits
        for the next step(). Each call's waste accumulates into
        cumulative_waste() since a single step() only covers one round under
        concurrency higher than max_slots; call reset_waste() to start over."""
        if not self._pending:
            return []
        batch, self._pending = self._pending, []
        seqs, waste = self.generate(batch)
        for key in self._waste_totals:
            self._waste_totals[key] += getattr(waste, key)
        return seqs

    def cumulative_waste(self) -> WasteStats:
        return WasteStats.compute(**self._waste_totals)

    def reset_waste(self) -> None:
        self._waste_totals = dict.fromkeys(self._waste_totals, 0)
