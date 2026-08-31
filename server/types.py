"""Shared data contracts for nanoserve. No logic beyond derived timing properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_time: float


@dataclass
class RequestMetrics:
    arrival_time: float
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        """End-to-end latency from arrival to finish."""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def tpot(self, num_output_tokens: int) -> float | None:
        """Time per output token, from first token to finish, averaged over
        the tokens generated after the first (num_output_tokens - 1).

        num_output_tokens is not stored on RequestMetrics itself — it lives on
        SequenceState.output_token_ids — so the caller passes it in.
        """
        if self.first_token_time is None or self.finish_time is None:
            return None
        if num_output_tokens <= 1:
            return None
        return (self.finish_time - self.first_token_time) / (num_output_tokens - 1)


@dataclass
class SequenceState:
    """cache_len vs next_position are deliberately separate numbers.

    cache_len is a physical index: how many KV slots along the sequence
    dimension of this sequence's slot are populated. next_position is a
    logical index: the absolute RoPE position id the next generated token
    will receive. They usually move together, but a sequence admitted
    mid-run (or one whose cache is compacted/relocated) can occupy cache
    indices that do not equal its own token positions — so the two must
    never be conflated or derived from one another.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    metrics: RequestMetrics
    slot_id: int | None = None
    cache_len: int = 0
    next_position: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None


class EngineProtocol(Protocol):
    def submit(self, request: Request) -> None: ...

    def step(self) -> list[SequenceState]:
        """Advance the batch by one iteration. Returns sequences that FINISHED this step."""
        ...

    def has_work(self) -> bool: ...

    def generate(self, requests: list[Request]) -> list[SequenceState]:
        """Blocking convenience loop over step(); used by tests and bench.py."""
        ...
