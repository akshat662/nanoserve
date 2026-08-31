"""Admission scheduler for continuous batching, on top of Engine's prefill()/
decode() execution primitives.

RESPONSIBILITY BOUNDARY: Engine EXECUTES the batches it is given; it has no
admission policy of its own (its own submit()/step() prefill whatever is
pending unconditionally, which is why bench.py had to do its own external
free-slot bookkeeping to avoid "no free slot" crashes). This file DECIDES
what goes into each batch — waiting queue, slot admission, token budget — and
calls only engine.prefill()/engine.decode() to run it, never engine.submit()/
engine.step(). No model or cache mechanics live here; admission just calls the
cache's own public allocate_slot()/num_free_slots(), the same way
Engine.prefill() already does internally.

Two forward passes on an admission step, one otherwise — NEVER fused. Mixing
a q_len>1 prefill with q_len==1 decodes in a single pass needs ragged/varlen
attention, an explicit V2 non-goal (this project hand-builds 4D masks; see
kv_cache.py). This is exactly the two-pass structure vLLM used before chunked
prefill merged the two.
"""

from collections import deque

from server.engine import Engine
from server.types import Request, RequestMetrics, SequenceState


class ContinuousBatchScheduler:
    def __init__(self, engine: Engine, batch_token_budget: int | None = None) -> None:
        self.engine = engine
        self.cache = engine.cache
        self.tokenizer = engine.tokenizer
        self.batch_token_budget = batch_token_budget if batch_token_budget is not None else engine.config.batch_token_budget
        self._waiting: deque[SequenceState] = deque()
        self._running: dict[int, SequenceState] = {}

    def submit(self, request: Request) -> None:
        self._waiting.append(
            SequenceState(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                max_new_tokens=request.max_new_tokens,
                metrics=RequestMetrics(arrival_time=request.arrival_time),
            )
        )

    def has_work(self) -> bool:
        return bool(self._waiting) or bool(self._running)

    def step(self) -> list[SequenceState]:
        # 1. RETIRE — sequences finished by the PREVIOUS step's decode() call
        # (this step's own decode() runs last, so anything IT finishes is
        # retired at the start of the NEXT step() call, not this one).
        finished_this_step = []
        for slot_id, seq in list(self._running.items()):
            if seq.finished:
                self.cache.free_slot(slot_id)
                del self._running[slot_id]
                finished_this_step.append(seq)

        # 2. ADMIT — respecting both free slots and batch_token_budget. The
        # first admission of a round is unconditional even if it alone
        # exceeds budget, so one giant prompt gets its own round rather than
        # stalling forever behind a budget it can never share with company.
        newly_admitted: list[SequenceState] = []
        admitted_tokens = 0
        while self._waiting and self.cache.num_free_slots() > 0:
            seq = self._waiting[0]
            prompt_len = len(seq.prompt_token_ids)
            if newly_admitted and admitted_tokens + prompt_len > self.batch_token_budget:
                break
            self._waiting.popleft()
            seq.slot_id = self.cache.allocate_slot()
            newly_admitted.append(seq)
            admitted_tokens += prompt_len
            self._running[seq.slot_id] = seq

        # 3. PREFILL — one prefill-only pass over just the new admissions.
        if newly_admitted:
            self.engine.prefill(newly_admitted)

        # 4. DECODE — one decode pass over everyone still active, including
        # the sequences prefilled above.
        active = [seq for seq in self._running.values() if not seq.finished]
        if active:
            self.engine.decode(active)

        return finished_this_step

    def generate(self, requests: list[Request]) -> list[SequenceState]:
        for req in requests:
            self.submit(req)
        finished: list[SequenceState] = []
        while self.has_work():
            finished.extend(self.step())
        return finished
