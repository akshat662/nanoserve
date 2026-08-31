"""Benchmark harness comparing engines under load. These are the numbers that
go in the README and on a CV, so every timestamp goes through now() and every
latency metric is a percentile, never a mean.

CLOCK NOTE: now() (perf_counter + a CUDA sync when available) is used for every
wall-clock/throughput measurement bench.py makes on its own. The one exception
is Request.arrival_time, which must be on the SAME clock Engine uses internally
for first_token_time/finish_time (time.time()) or ttft/e2e_latency subtraction
across the two would be comparing different epochs entirely. arrival_time is
set to each request's INTENDED arrival per the workload's arrival pattern, not
to whenever bench.py actually got around to calling submit() for it — a request
that had to wait for a free slot must show that wait in its measured latency,
not have it silently absorbed by a late arrival_time.
"""

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from server.config import ServerConfig, load_config
from server.engine import Engine
from server.static_batch import StaticBatchEngine
from server.types import Request

SEED = 42

WORKLOADS = {
    "short_prompt_long_gen": {"prompt_len": 20, "max_new_tokens": 128},
    "long_prompt_short_gen": {"prompt_len": 300, "max_new_tokens": 16},
}

_FILLER_WORDS = (
    "the quick brown fox jumps over lazy dog system engine cache token model batch "
    "request server memory vector gradient learning network language context sequence "
    "attention transformer query value layer weight matrix scalar tensor device thread"
).split()


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def build_requests(tokenizer, prompt_len: int, max_new_tokens: int, n: int, tag: str) -> list[Request]:
    requests = []
    for i in range(n):
        rng = random.Random(f"{SEED}:{tag}:{i}")
        text = " ".join(rng.choice(_FILLER_WORDS) for _ in range(prompt_len * 2))
        ids = tokenizer.encode(text)
        if len(ids) < prompt_len:
            ids = ids * (prompt_len // max(len(ids), 1) + 1)
        requests.append(
            Request(request_id=f"{tag}-{i}", prompt_token_ids=ids[:prompt_len], max_new_tokens=max_new_tokens, arrival_time=0.0)
        )
    return requests


def run_workload(engine, requests: list[Request], stagger: float) -> tuple[list, int, float]:
    """Admits `requests` respecting the engine's free-slot capacity and the
    given arrival stagger, driving submit()/step() until everyone finishes.
    Returns (finished sequences, num requests that had to wait for a slot,
    wall-clock seconds for the whole run)."""
    n = len(requests)
    offsets = [i * stagger / n for i in range(n)] if stagger > 0 else [0.0] * n
    order = sorted(range(n), key=lambda i: offsets[i])

    wall_clock_start = now()
    schedule_start = time.time()  # matches Engine's internal clock — see module docstring
    for req, offset in zip(requests, offsets):
        req.arrival_time = schedule_start + offset

    idx = 0
    waited_ids: set[str] = set()
    finished: list = []

    while idx < n or engine.has_work():
        elapsed = time.time() - schedule_start
        # submit() doesn't consume a slot itself — only prefill() (inside the
        # next step()) does — so free-slot count must be sampled ONCE per
        # iteration and tracked locally, or a tight loop here would admit more
        # requests into _pending than there is real capacity for.
        free = engine.cache.num_free_slots()
        admitted_this_round = 0
        while idx < n and offsets[order[idx]] <= elapsed and admitted_this_round < free:
            engine.submit(requests[order[idx]])
            idx += 1
            admitted_this_round += 1
        j = idx
        while j < n and offsets[order[j]] <= elapsed:
            waited_ids.add(requests[order[j]].request_id)
            j += 1

        if engine.has_work():
            finished.extend(engine.step())
        elif idx < n:
            time.sleep(0.0005)

    wall_clock = now() - wall_clock_start
    return finished, len(waited_ids), wall_clock


@dataclasses.dataclass
class CellResult:
    engine_name: str
    workload_name: str
    concurrency: int
    output_tokens_per_sec: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    prefill_waste_frac: float | None
    decode_waste_frac: float | None
    num_waited: int
    per_request: list[dict]


def measure_cell(engine_name, engine_obj, workload_name, prompt_len, max_new_tokens, concurrency, stagger, warmups) -> CellResult:
    tag = f"{workload_name}-{engine_name}-c{concurrency}"
    for w in range(warmups):
        reqs = build_requests(engine_obj.tokenizer, prompt_len, max_new_tokens, concurrency, f"{tag}-warmup{w}")
        run_workload(engine_obj, reqs, stagger)

    if hasattr(engine_obj, "reset_waste"):
        engine_obj.reset_waste()

    reqs = build_requests(engine_obj.tokenizer, prompt_len, max_new_tokens, concurrency, tag)
    finished, num_waited, wall_clock = run_workload(engine_obj, reqs, stagger)

    output_tokens = sum(len(s.output_token_ids) for s in finished)
    ttft_ms = [s.metrics.ttft * 1000 for s in finished]
    e2e_ms = [s.metrics.e2e_latency * 1000 for s in finished]

    prefill_waste_frac = decode_waste_frac = None
    if hasattr(engine_obj, "cumulative_waste"):
        waste = engine_obj.cumulative_waste()
        prefill_waste_frac = waste.prefill_waste / waste.prefill_total if waste.prefill_total else 0.0
        decode_waste_frac = waste.decode_waste / waste.decode_total if waste.decode_total else 0.0

    per_request = [
        {
            "request_id": s.request_id,
            "prompt_tokens": len(s.prompt_token_ids),
            "output_tokens": len(s.output_token_ids),
            "ttft_ms": s.metrics.ttft * 1000,
            "e2e_ms": s.metrics.e2e_latency * 1000,
            "finish_reason": s.finish_reason,
        }
        for s in finished
    ]

    return CellResult(
        engine_name=engine_name,
        workload_name=workload_name,
        concurrency=concurrency,
        output_tokens_per_sec=output_tokens / wall_clock,
        ttft_p50_ms=percentile(ttft_ms, 50),
        ttft_p95_ms=percentile(ttft_ms, 95),
        e2e_p50_ms=percentile(e2e_ms, 50),
        e2e_p95_ms=percentile(e2e_ms, 95),
        prefill_waste_frac=prefill_waste_frac,
        decode_waste_frac=decode_waste_frac,
        num_waited=num_waited,
        per_request=per_request,
    )


def _fmt(x, spec="{:.1f}"):
    return spec.format(x) if x is not None else ""


def print_markdown_table(config: ServerConfig, workload_name: str, max_new_tokens: int, warmups: int, cells: list[CellResult]) -> None:
    workload = WORKLOADS[workload_name]
    print()
    print(
        f"### {workload_name} (prompt≈{workload['prompt_len']} tok, max_new_tokens={max_new_tokens}) — "
        f"model={config.model_id}, device={config.device}, dtype={config.dtype}, "
        f"max_slots={config.max_slots}, max_seq_len={config.max_seq_len}, warmups={warmups}"
    )
    print()
    header = ["engine", "concurrency", "output_tok/s", "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms", "prefill_waste", "decode_waste", "num_waited"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for c in cells:
        row = [
            c.engine_name, str(c.concurrency), _fmt(c.output_tokens_per_sec),
            _fmt(c.ttft_p50_ms), _fmt(c.ttft_p95_ms), _fmt(c.e2e_p50_ms), _fmt(c.e2e_p95_ms),
            _fmt(c.prefill_waste_frac, "{:.1%}"), _fmt(c.decode_waste_frac, "{:.1%}"), str(c.num_waited),
        ]
        print("| " + " | ".join(row) + " |")


def print_speedup_line(cells: list[CellResult], concurrencies: list[int]) -> None:
    by_key = {(c.engine_name, c.concurrency): c for c in cells}
    for conc in concurrencies:
        static_cell, engine_cell = by_key.get(("static", conc)), by_key.get(("engine", conc))
        if static_cell and engine_cell:
            ratio = static_cell.output_tokens_per_sec / engine_cell.output_tokens_per_sec
            print(
                f"speedup @ concurrency {conc}: static={static_cell.output_tokens_per_sec:.1f} tok/s, "
                f"engine={engine_cell.output_tokens_per_sec:.1f} tok/s, static/engine ratio={ratio:.2f}x"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanoserve benchmark harness")
    p.add_argument("--engines", default="static,engine", help="comma-separated: static,engine,scheduler")
    p.add_argument("--concurrency", default="1,8,32", help="comma-separated concurrency levels")
    p.add_argument("--workload", choices=list(WORKLOADS), default=None, help="default: run both workloads")
    p.add_argument("--max-slots", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=None, help="override every workload's own max_new_tokens")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--stagger", type=float, default=0.0, help="seconds over which arrivals are spread uniformly")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.device:
        config = dataclasses.replace(config, device=args.device)
    config = dataclasses.replace(config, max_slots=args.max_slots)

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    engine_names = [x.strip() for x in args.engines.split(",")]
    workload_names = [args.workload] if args.workload else list(WORKLOADS)

    if config.max_slots >= max(concurrencies):
        print("!" * 78)
        print(
            f"WARNING: --max-slots={config.max_slots} >= max concurrency {max(concurrencies)}. "
            "No request will ever need to queue, so this run UNDERSTATES continuous "
            "batching's benefit. Pass a smaller --max-slots or a higher --concurrency."
        )
        print("!" * 78)

    print(f"Loading {config.model_id} on {config.device} ({config.dtype})...")
    base_engine = Engine(config)
    engine_objs = {}
    for name in engine_names:
        if name == "scheduler":
            print("NOTE: engine 'scheduler' skipped — scheduler.py does not exist yet.")
        elif name == "static":
            engine_objs[name] = StaticBatchEngine(base_engine)
        elif name == "engine":
            engine_objs[name] = base_engine
        else:
            raise ValueError(f"unknown engine {name!r}")

    all_results: list[CellResult] = []
    for workload_name in workload_names:
        workload = WORKLOADS[workload_name]
        max_new_tokens = args.max_new_tokens or workload["max_new_tokens"]
        cells = []
        for concurrency in concurrencies:
            for engine_name in engine_names:
                if engine_name not in engine_objs:
                    continue
                cell = measure_cell(
                    engine_name, engine_objs[engine_name], workload_name, workload["prompt_len"],
                    max_new_tokens, concurrency, args.stagger, args.warmups,
                )
                cells.append(cell)
                all_results.append(cell)
        print_markdown_table(config, workload_name, max_new_tokens, args.warmups, cells)
        print_speedup_line(cells, concurrencies)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": {
                        "model_id": config.model_id, "device": config.device, "dtype": str(config.dtype),
                        "max_slots": config.max_slots, "max_seq_len": config.max_seq_len,
                        "warmups": args.warmups, "stagger": args.stagger,
                    },
                    "results": [dataclasses.asdict(c) for c in all_results],
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
