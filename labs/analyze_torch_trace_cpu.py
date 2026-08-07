#!/usr/bin/env python3
"""Compute CPU-side inclusive/self wall time from a PyTorch Chrome trace.

This reproduces the useful *self* concept (exclusive of nested recorded events)
without pretending that a blocking CUDA API's self wall time is active CPU time.
Use perf or Nsys CPU samples for true on-CPU self time.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_graph"}
SYNC_RE = re.compile(
    r"^(?:cuda(?:Device|Event|Stream|Thread)Synchronize|"
    r"cu(?:Ctx|Event|Stream)Synchronize)(?:_v\d+)?$"
)
SUBMIT_RE = re.compile(
    r"^(?:cudaGraphLaunch|cudaLaunchKernel(?:ExC)?|cudaMemcpyAsync|"
    r"cudaMemsetAsync|cuGraphLaunch|cuLaunchKernel).*$"
)


@dataclass
class CpuEvent:
    name: str
    category: str
    pid: int
    tid: int
    start_us: float
    end_us: float
    args: dict[str, Any]
    child_intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def inclusive_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def self_us(self) -> float:
        return max(0.0, self.inclusive_us - interval_union(self.child_intervals))


def interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_events(
    path: Path,
    *,
    start_us: float | None = None,
    end_us: float | None = None,
    thread_regex: re.Pattern[str] | None = None,
) -> tuple[list[CpuEvent], dict[tuple[int, int], str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    thread_names: dict[tuple[int, int], str] = {}
    for raw in raw_events:
        if raw.get("ph") == "M" and raw.get("name") == "thread_name":
            args = raw.get("args", {})
            thread_names[(int(raw.get("pid", 0)), int(raw.get("tid", 0)))] = str(
                args.get("name", "")
            )

    events = []
    for raw in raw_events:
        if raw.get("ph") != "X" or "dur" not in raw or "ts" not in raw:
            continue
        category = str(raw.get("cat", ""))
        if category.lower() in GPU_CATEGORIES:
            continue
        pid = int(raw.get("pid", 0))
        tid = int(raw.get("tid", 0))
        if thread_regex and not thread_regex.search(thread_names.get((pid, tid), "")):
            continue
        left = float(raw["ts"])
        right = left + float(raw["dur"])
        if start_us is not None and right <= start_us:
            continue
        if end_us is not None and left >= end_us:
            continue
        left = max(left, start_us) if start_us is not None else left
        right = min(right, end_us) if end_us is not None else right
        if right <= left:
            continue
        events.append(
            CpuEvent(
                name=str(raw.get("name", "<unnamed>")),
                category=category,
                pid=pid,
                tid=tid,
                start_us=left,
                end_us=right,
                args=raw.get("args", {}) if isinstance(raw.get("args"), dict) else {},
            )
        )
    return events, thread_names


def assign_children(events: list[CpuEvent]) -> None:
    by_thread: dict[tuple[int, int], list[CpuEvent]] = defaultdict(list)
    for event in events:
        by_thread[(event.pid, event.tid)].append(event)
    for rows in by_thread.values():
        rows.sort(key=lambda event: (event.start_us, -event.end_us, event.name))
        stack: list[CpuEvent] = []
        for event in rows:
            while stack and (
                event.start_us >= stack[-1].end_us
                or event.end_us > stack[-1].end_us
            ):
                stack.pop()
            if stack:
                parent = stack[-1]
                parent.child_intervals.append((event.start_us, event.end_us))
            stack.append(event)


def classify(event: CpuEvent) -> str:
    name = event.name
    leaf = name.rsplit("::", 1)[-1].split("(", 1)[0]
    details = f"{name} {event.category} {json.dumps(event.args, ensure_ascii=False)}".lower()
    if SYNC_RE.fullmatch(leaf):
        return "cuda_sync_wait_wall"
    if SUBMIT_RE.fullmatch(leaf):
        return "cuda_submission_runtime"
    if "python_gil" in details or "gil" in event.category.lower():
        return "python_gil"
    if any(token in details for token in ("numpy", "multiarray", "umath", "ndarray")):
        return "numpy"
    if any(token in details for token in (
        "get_num_common_prefix_blocks", "scheduler.schedule", "vllm/v1/core/sched",
        "scheduler.py"
    )):
        return "vllm_scheduler"
    if any(token in details for token in (
        "torch._dynamo", "torchinductor", "triton", "compile", "jit"
    )):
        return "python_jit_or_compilation"
    if any(token in details for token in (
        "tolist", "to_cpu", "serialization", "pickle", "json", "tokenizer"
    )):
        return "python_output_or_serialization"
    if event.category == "python_function" or "python" in event.category.lower():
        return "python_other"
    if event.category == "cpu_op" or name.startswith("aten::"):
        return "aten_cpu_op"
    if event.category in {"osrt", "os_runtime"}:
        return "os_runtime_wall"
    if event.category == "cuda_runtime":
        return "cuda_runtime_other"
    if "user_annotation" in event.category:
        return "user_annotation_self"
    return "other_cpu_wall"


def summarize(
    events: list[CpuEvent], thread_names: dict[tuple[int, int], str]
) -> dict[str, Any]:
    assign_children(events)
    total_self = sum(event.self_us for event in events)
    grouped: dict[str, list[CpuEvent]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[CpuEvent]] = defaultdict(list)
    by_thread: dict[tuple[int, int], float] = defaultdict(float)
    for event in events:
        grouped[classify(event)].append(event)
        by_name[(event.name, event.category)].append(event)
        by_thread[(event.pid, event.tid)] += event.self_us

    group_rows = []
    for category, rows in grouped.items():
        values = [event.self_us for event in rows]
        category_total = sum(values)
        group_rows.append(
            {
                "category": category,
                "calls": len(rows),
                "self_wall_ms": category_total / 1_000,
                "share_of_summed_self_wall_pct": (
                    category_total / total_self * 100 if total_self else None
                ),
                "p50_self_us": percentile(values, 0.50),
                "p95_self_us": percentile(values, 0.95),
                "p99_self_us": percentile(values, 0.99),
                "max_self_us": max(values) if values else None,
            }
        )
    group_rows.sort(key=lambda row: float(row["self_wall_ms"]), reverse=True)

    function_rows = []
    for (name, category), rows in by_name.items():
        self_total = sum(event.self_us for event in rows)
        inclusive_total = sum(event.inclusive_us for event in rows)
        function_rows.append(
            {
                "name": name,
                "trace_category": category,
                "analysis_category": classify(rows[0]),
                "calls": len(rows),
                "self_wall_ms": self_total / 1_000,
                "inclusive_wall_ms": inclusive_total / 1_000,
                "share_of_summed_self_wall_pct": (
                    self_total / total_self * 100 if total_self else None
                ),
                "p50_self_us": percentile((event.self_us for event in rows), 0.50),
                "p99_self_us": percentile((event.self_us for event in rows), 0.99),
            }
        )
    function_rows.sort(key=lambda row: float(row["self_wall_ms"]), reverse=True)

    thread_rows = [
        {
            "pid": pid,
            "tid": tid,
            "thread_name": thread_names.get((pid, tid), ""),
            "summed_self_wall_ms": value / 1_000,
        }
        for (pid, tid), value in sorted(
            by_thread.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return {
        "events": len(events),
        "summed_cpu_self_wall_ms": total_self / 1_000,
        "by_analysis_category": group_rows,
        "top_functions": function_rows[:100],
        "by_thread": thread_rows,
        "notes": [
            "Self wall excludes nested recorded trace events on the same thread.",
            "Summing threads can exceed process wall time because threads overlap.",
            "cuda_sync_wait_wall includes sleep/descheduling and is not active CPU time.",
            "Use Nsys/perf leaf CPU samples for true on-CPU self-time shares.",
        ],
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def print_summary(payload: dict[str, Any]) -> None:
    print(
        f"CPU trace events={payload['events']} "
        f"summed self wall={payload['summed_cpu_self_wall_ms']:.3f} ms"
    )
    print("CATEGORY".ljust(34), "CALLS".rjust(8), "SELF_MS".rjust(12), "SHARE%".rjust(10))
    for row in payload["by_analysis_category"]:
        print(
            str(row["category"])[:34].ljust(34),
            str(row["calls"]).rjust(8),
            _fmt(row["self_wall_ms"]).rjust(12),
            _fmt(row["share_of_summed_self_wall_pct"]).rjust(10),
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument("--thread-regex")
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if not args.trace.is_file():
        raise SystemExit(f"Trace does not exist: {args.trace}")
    if args.start_us is not None and args.end_us is not None and args.start_us >= args.end_us:
        raise SystemExit("--start-us must be smaller than --end-us")
    thread_re = re.compile(args.thread_regex) if args.thread_regex else None
    events, names = load_events(
        args.trace,
        start_us=args.start_us,
        end_us=args.end_us,
        thread_regex=thread_re,
    )
    payload = summarize(events, names)
    payload["source"] = str(args.trace)
    print_summary(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
