#!/usr/bin/env python3
"""Analyze CUDA wait/launch boundaries and sampled CPU self time in Nsys SQLite.

The script intentionally keeps three metrics separate:

* CUDA API wall duration: caller entry to return, including sleep/descheduling.
* Device-event wait and post-event host tail: available only with event trace.
* Sampled on-CPU self time: leaf CPU samples, excluding off-CPU waiting.

It uses only the Python standard library and tolerates missing optional Nsys tables.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYNC_RE = re.compile(
    r"^(?:cuda(?:Device|Event|Stream|Thread)Synchronize|"
    r"cu(?:Ctx|Event|Stream)Synchronize)(?:_v\d+)?$"
)
SUBMIT_RE = re.compile(
    r"^(?:cudaGraphLaunch|cudaLaunchKernel(?:ExC)?|cudaMemcpyAsync|"
    r"cudaMemsetAsync|cuGraphLaunch|cuLaunchKernel|cuMemcpy\w*Async)(?:_v\d+)?$"
)


@dataclass(frozen=True)
class RuntimeCall:
    start: int
    end: int
    global_tid: int | None
    correlation_id: int | None
    name: str


@dataclass(frozen=True)
class GpuActivity:
    start: int
    end: int
    stream_key: tuple[int | None, int | None, int | None, int | None]
    correlation_id: int | None
    table: str


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def metric_summary(values_ns: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values_ns)
    total = sum(values)
    return {
        "count": len(values),
        "total_ms": total / 1_000_000,
        "mean_us": total / len(values) / 1_000 if values else None,
        "p50_us": _scale(percentile(values, 0.50), 1_000),
        "p90_us": _scale(percentile(values, 0.90), 1_000),
        "p95_us": _scale(percentile(values, 0.95), 1_000),
        "p99_us": _scale(percentile(values, 0.99), 1_000),
        "max_us": max(values) / 1_000 if values else None,
    }


def _scale(value: float | None, divisor: float) -> float | None:
    return value / divisor if value is not None else None


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    escaped = table.replace('"', '""')
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{escaped}")')}


def _select_expr(table_columns: set[str], name: str) -> str:
    return name if name in table_columns else f"NULL AS {name}"


def load_strings(connection: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    return {int(row[0]): str(row[1]) for row in connection.execute(
        "SELECT id, value FROM StringIds"
    )}


def load_runtime_calls(
    connection: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
) -> list[RuntimeCall]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in tables:
        return []
    cols = columns(connection, table)
    required = {"start", "end", "nameId"}
    if not required.issubset(cols):
        return []
    query = (
        "SELECT start, end, "
        f"{_select_expr(cols, 'globalTid')}, "
        f"{_select_expr(cols, 'correlationId')}, nameId FROM {table}"
    )
    calls = []
    for start, end, global_tid, correlation_id, name_id in connection.execute(query):
        calls.append(
            RuntimeCall(
                start=int(start),
                end=int(end),
                global_tid=int(global_tid) if global_tid is not None else None,
                correlation_id=(
                    int(correlation_id) if correlation_id is not None else None
                ),
                name=strings.get(int(name_id), f"nameId={name_id}"),
            )
        )
    return calls


def load_sync_links(
    connection: sqlite3.Connection, tables: set[str]
) -> dict[int, list[dict[str, int | None]]]:
    table = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    if table not in tables:
        return {}
    cols = columns(connection, table)
    if "correlationId" not in cols:
        return {}
    wanted = (
        "start",
        "end",
        "eventSyncId",
        "eventId",
        "contextId",
        "globalPid",
    )
    query = "SELECT correlationId, " + ", ".join(
        _select_expr(cols, name) for name in wanted
    ) + f" FROM {table}"
    output: dict[int, list[dict[str, int | None]]] = defaultdict(list)
    for row in connection.execute(query):
        if row[0] is None:
            continue
        output[int(row[0])].append(
            {
                name: int(value) if value is not None else None
                for name, value in zip(wanted, row[1:])
            }
        )
    return output


def load_cuda_events(
    connection: sqlite3.Connection, tables: set[str]
) -> dict[tuple[int | None, int | None, int | None], list[int]]:
    table = "CUPTI_ACTIVITY_KIND_CUDA_EVENT"
    if table not in tables:
        return {}
    cols = columns(connection, table)
    if "timestamp" not in cols:
        return {}
    wanted = ("eventSyncId", "eventId", "contextId")
    query = "SELECT timestamp, " + ", ".join(
        _select_expr(cols, name) for name in wanted
    ) + f" FROM {table}"
    output: dict[tuple[int | None, int | None, int | None], list[int]] = defaultdict(list)
    for timestamp, *key_values in connection.execute(query):
        if timestamp is None or int(timestamp) <= 0:
            continue
        key = tuple(
            int(value) if value is not None else None for value in key_values
        )
        output[key].append(int(timestamp))
    return output


def _best_sync_link(
    call: RuntimeCall, candidates: list[dict[str, int | None]]
) -> dict[str, int | None] | None:
    if not candidates:
        return None
    call_pid = nsys_pid(call.global_tid)
    compatible = [
        item
        for item in candidates
        if call_pid is None
        or item.get("globalPid") is None
        or item.get("globalPid") == call_pid
    ]
    if compatible:
        candidates = compatible
    return min(
        candidates,
        key=lambda item: abs((item.get("start") or call.start) - call.start),
    )


def nsys_pid(global_tid: int | None) -> int | None:
    """Decode the PID stored in Nsys' 24-bit TID composite when available."""
    if global_tid is None:
        return None
    pid = global_tid >> 24
    return pid or None


def _event_timestamp(
    call: RuntimeCall,
    link: dict[str, int | None] | None,
    cuda_events: dict[tuple[int | None, int | None, int | None], list[int]],
) -> int | None:
    if link is None:
        return None
    key = (link.get("eventSyncId"), link.get("eventId"), link.get("contextId"))
    candidates = cuda_events.get(key, [])
    if not candidates:
        return None
    # Prefer the completion associated with this API window. If the event was
    # already complete on entry, the latest prior completion is the right one.
    not_after_return = [timestamp for timestamp in candidates if timestamp <= call.end]
    return max(not_after_return) if not_after_return else min(candidates)


def load_sched_intervals(
    connection: sqlite3.Connection,
    tables: set[str],
    trace_end: int,
) -> dict[int, list[tuple[int, int, bool, int | None, int | None]]]:
    if "SCHED_EVENTS" not in tables:
        return {}
    cols = columns(connection, "SCHED_EVENTS")
    if not {"start", "isSchedIn", "globalTid"}.issubset(cols):
        return {}
    query = (
        "SELECT start, isSchedIn, globalTid, "
        f"{_select_expr(cols, 'threadState')}, "
        f"{_select_expr(cols, 'threadBlock')} "
        "FROM SCHED_EVENTS WHERE globalTid IS NOT NULL "
        "ORDER BY globalTid, start"
    )
    events: dict[int, list[tuple[int, bool, int | None, int | None]]] = defaultdict(list)
    for start, is_in, tid, state, block in connection.execute(query):
        events[int(tid)].append(
            (
                int(start),
                bool(is_in),
                int(state) if state is not None else None,
                int(block) if block is not None else None,
            )
        )
    intervals: dict[int, list[tuple[int, int, bool, int | None, int | None]]] = {}
    for tid, rows in events.items():
        built = []
        for index, (start, running, state, block) in enumerate(rows):
            end = rows[index + 1][0] if index + 1 < len(rows) else trace_end
            if end > start:
                built.append((start, end, running, state, block))
        intervals[tid] = built
    return intervals


def interval_state_overlap(
    start: int,
    end: int,
    intervals: list[tuple[int, int, bool, int | None, int | None]],
) -> tuple[int, int]:
    on_cpu = 0
    covered = 0
    for left, right, running, _state, _block in intervals:
        if right <= start:
            continue
        if left >= end:
            break
        overlap = max(0, min(end, right) - max(start, left))
        covered += overlap
        if running:
            on_cpu += overlap
    return on_cpu, covered


def summarize_sync_calls(
    calls: list[RuntimeCall],
    sync_links: dict[int, list[dict[str, int | None]]],
    cuda_events: dict[tuple[int | None, int | None, int | None], list[int]],
    sched_intervals: dict[int, list[tuple[int, int, bool, int | None, int | None]]],
) -> dict[str, Any]:
    rows = []
    for call in calls:
        if not SYNC_RE.fullmatch(call.name.rsplit("::", 1)[-1].split("(", 1)[0]):
            continue
        correlation_key = (
            call.correlation_id if call.correlation_id is not None else -1
        )
        link = _best_sync_link(call, sync_links.get(correlation_key, []))
        event_time = _event_timestamp(call, link, cuda_events)
        on_cpu, covered = interval_state_overlap(
            call.start,
            call.end,
            sched_intervals.get(
                call.global_tid if call.global_tid is not None else -1, []
            ),
        )
        duration = call.end - call.start
        rows.append(
            {
                "name": call.name,
                "start_ns": call.start,
                "end_ns": call.end,
                "duration_ns": duration,
                "global_tid": call.global_tid,
                "correlation_id": call.correlation_id,
                "event_completion_ns": event_time,
                "device_not_ready_ns": (
                    max(0, event_time - call.start) if event_time is not None else None
                ),
                "post_event_host_tail_ns": (
                    max(0, call.end - max(call.start, event_time))
                    if event_time is not None
                    else None
                ),
                "on_cpu_ns": on_cpu if covered else None,
                "off_cpu_covered_ns": covered - on_cpu if covered else None,
                "sched_coverage_pct": covered / duration * 100 if duration else None,
            }
        )

    grouped = []
    for name in sorted({str(row["name"]) for row in rows}):
        members = [row for row in rows if row["name"] == name]
        exact = [row for row in members if row["event_completion_ns"] is not None]
        sched = [row for row in members if row["on_cpu_ns"] is not None]
        grouped.append(
            {
                "name": name,
                "api_wall": metric_summary(row["duration_ns"] for row in members),
                "exact_event_matches": len(exact),
                "device_not_ready": metric_summary(
                    int(row["device_not_ready_ns"]) for row in exact
                ),
                "post_event_host_tail": metric_summary(
                    int(row["post_event_host_tail_ns"]) for row in exact
                ),
                "sched_matches": len(sched),
                "sampled_on_cpu_inside_api": metric_summary(
                    int(row["on_cpu_ns"]) for row in sched
                ),
                "sampled_off_cpu_inside_api": metric_summary(
                    int(row["off_cpu_covered_ns"]) for row in sched
                ),
            }
        )
    return {"calls": rows, "by_api": grouped}


def load_gpu_activities(
    connection: sqlite3.Connection, tables: set[str]
) -> list[GpuActivity]:
    prefixes = (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    )
    activities = []
    for table in sorted(tables):
        if not table.startswith(prefixes):
            continue
        cols = columns(connection, table)
        if not {"start", "end", "streamId"}.issubset(cols):
            continue
        wanted = ("globalPid", "deviceId", "contextId", "streamId", "correlationId")
        query = "SELECT start, end, " + ", ".join(
            _select_expr(cols, name) for name in wanted
        ) + f' FROM "{table}"'
        for start, end, global_pid, device, context, stream, correlation in connection.execute(query):
            activities.append(
                GpuActivity(
                    start=int(start),
                    end=int(end),
                    stream_key=(
                        int(global_pid) if global_pid is not None else None,
                        int(device) if device is not None else None,
                        int(context) if context is not None else None,
                        int(stream) if stream is not None else None,
                    ),
                    correlation_id=(
                        int(correlation) if correlation is not None else None
                    ),
                    table=table,
                )
            )
    return activities


def summarize_submissions(
    calls: list[RuntimeCall], activities: list[GpuActivity]
) -> dict[str, Any]:
    by_correlation: dict[int, list[GpuActivity]] = defaultdict(list)
    by_stream: dict[tuple[int | None, int | None, int | None, int | None], list[GpuActivity]] = defaultdict(list)
    for activity in activities:
        if activity.correlation_id is not None:
            by_correlation[activity.correlation_id].append(activity)
        by_stream[activity.stream_key].append(activity)
    for stream_rows in by_stream.values():
        stream_rows.sort(key=lambda item: item.end)

    rows = []
    for call in calls:
        leaf = call.name.rsplit("::", 1)[-1].split("(", 1)[0]
        if not SUBMIT_RE.fullmatch(leaf) or call.correlation_id is None:
            continue
        candidates = by_correlation.get(call.correlation_id, [])
        call_pid = nsys_pid(call.global_tid)
        compatible = [
            item
            for item in candidates
            if call_pid is None
            or item.stream_key[0] is None
            or item.stream_key[0] == call_pid
        ]
        if compatible:
            candidates = compatible
        candidates = [item for item in candidates if item.start >= call.start]
        if not candidates:
            continue
        first = min(candidates, key=lambda item: item.start)
        stream_rows = by_stream[first.stream_key]
        prior_ends = [item.end for item in stream_rows]
        position = bisect.bisect_right(prior_ends, first.start)
        prior_end = max(
            (item.end for item in stream_rows[:position] if item.end <= first.start),
            default=None,
        )
        rows.append(
            {
                "name": call.name,
                "api_duration_ns": call.end - call.start,
                "api_enter_to_gpu_start_ns": first.start - call.start,
                "api_exit_to_gpu_start_ns": first.start - call.end,
                "same_stream_idle_gap_ns": (
                    first.start - prior_end if prior_end is not None else None
                ),
                "critical_launch_bubble_ns": max(
                    0,
                    first.start
                    - max(call.end, prior_end if prior_end is not None else call.end),
                ),
                "gpu_activity_table": first.table,
                "correlation_id": call.correlation_id,
            }
        )
    grouped = []
    for name in sorted({str(row["name"]) for row in rows}):
        members = [row for row in rows if row["name"] == name]
        grouped.append(
            {
                "name": name,
                "api_wall": metric_summary(row["api_duration_ns"] for row in members),
                "api_enter_to_gpu_start": metric_summary(
                    row["api_enter_to_gpu_start_ns"] for row in members
                ),
                "api_exit_to_gpu_start": metric_summary(
                    row["api_exit_to_gpu_start_ns"] for row in members
                ),
                "same_stream_idle_gap": metric_summary(
                    row["same_stream_idle_gap_ns"]
                    for row in members
                    if row["same_stream_idle_gap_ns"] is not None
                ),
                "critical_launch_bubble": metric_summary(
                    row["critical_launch_bubble_ns"] for row in members
                ),
            }
        )
    return {"calls": rows, "by_api": grouped}


def cpu_category(symbol: str, module: str) -> str:
    value = f"{symbol} {module}".lower()
    if any(token in value for token in ("numpy", "multiarray", "umath")):
        return "numpy_native"
    if any(token in value for token in (
        "get_num_common_prefix_blocks", "scheduler.py", "vllm/v1/core/sched"
    )):
        return "vllm_scheduler"
    if any(token in value for token in (
        "take_gil", "drop_gil", "gil", "pythread_acquire", "pthread_mutex"
    )):
        return "gil_or_locking"
    if any(token in value for token in (
        "torch._dynamo", "torchinductor", "triton", "llvm", "compile"
    )):
        return "jit_or_compilation"
    if any(token in value for token in (
        "libcuda", "libcudart", "nvidia", "cudaevent", "cugraphlaunch"
    )):
        return "cuda_runtime_or_driver"
    if any(token in value for token in (
        "pymalloc", "malloc", "free", "memcpy", "memmove"
    )):
        return "allocation_or_memory"
    if any(token in value for token in (
        "pyeval", "_py", "python", "ceval"
    )):
        return "python_interpreter"
    if any(token in value for token in (
        "[kernel", "vmlinux", "libpthread", "libc.so", "libc-"
    )):
        return "os_or_libc"
    if "vllm" in value:
        return "vllm_other"
    return "other_native"


def summarize_cpu_samples(
    connection: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
) -> dict[str, Any]:
    required = {"COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS"}
    if not required.issubset(tables):
        return {"available": False, "total_leaf_samples": 0, "by_category": [], "top": []}
    query = """
        SELECT c.globalTid, s.symbol, s.module, c.cpuCycles
        FROM SAMPLING_CALLCHAINS AS s
        JOIN COMPOSITE_EVENTS AS c ON c.id = s.id
        WHERE s.stackDepth = 0 AND c.cpuCycles = 1
    """
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    total = 0
    for _tid, symbol_id, module_id, weight in connection.execute(query):
        symbol = strings.get(int(symbol_id), f"symbolId={symbol_id}")
        module = strings.get(int(module_id), f"moduleId={module_id}")
        count = int(weight)
        grouped[(symbol, module)] += count
        by_category[cpu_category(symbol, module)] += count
        total += count
    categories = [
        {
            "category": name,
            "samples": count,
            "share_pct": count / total * 100 if total else None,
        }
        for name, count in sorted(by_category.items(), key=lambda item: item[1], reverse=True)
    ]
    top = [
        {
            "symbol": symbol,
            "module": module,
            "samples": count,
            "share_pct": count / total * 100 if total else None,
        }
        for (symbol, module), count in sorted(
            grouped.items(), key=lambda item: item[1], reverse=True
        )[:50]
    ]
    return {
        "available": True,
        "total_leaf_samples": total,
        "by_category": categories,
        "top": top,
    }


def analyze(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = table_names(connection)
        strings = load_strings(connection, tables)
        runtime = load_runtime_calls(connection, tables, strings)
        trace_end = max((call.end for call in runtime), default=0)
        sched = load_sched_intervals(connection, tables, trace_end)
        sync = summarize_sync_calls(
            runtime,
            load_sync_links(connection, tables),
            load_cuda_events(connection, tables),
            sched,
        )
        submissions = summarize_submissions(
            runtime, load_gpu_activities(connection, tables)
        )
        cpu_samples = summarize_cpu_samples(connection, tables, strings)
        return {
            "source": str(path),
            "tables_present": sorted(tables),
            "runtime_calls": len(runtime),
            "sync": sync,
            "submissions": submissions,
            "cpu_self_samples": cpu_samples,
            "notes": [
                "CUDA API duration is host wall time, not active CPU time.",
                "Exact event decomposition requires non-zero CUDA_EVENT timestamps.",
                "CPU self-time percentages are statistical leaf-sample shares.",
                "critical_launch_bubble includes visible post-API handoff but cannot isolate the physical PCIe doorbell.",
                "Memcpy stream gaps may include explicit stream/event dependencies.",
            ],
        }
    finally:
        connection.close()


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def print_summary(payload: dict[str, Any]) -> None:
    print("Synchronization APIs")
    for row in payload["sync"]["by_api"]:
        wall = row["api_wall"]
        device = row["device_not_ready"]
        tail = row["post_event_host_tail"]
        print(
            f"  {row['name']}: calls={wall['count']} total={_fmt(wall['total_ms'])} ms "
            f"p50={_fmt(wall['p50_us'])} us; exact={row['exact_event_matches']} "
            f"device_wait_total={_fmt(device['total_ms'])} ms "
            f"post_event_tail_total={_fmt(tail['total_ms'])} ms"
        )
    print("Submission APIs")
    for row in payload["submissions"]["by_api"]:
        api = row["api_wall"]
        bubble = row["critical_launch_bubble"]
        print(
            f"  {row['name']}: calls={api['count']} api_total={_fmt(api['total_ms'])} ms "
            f"api_p50={_fmt(api['p50_us'])} us "
            f"critical_bubble_total={_fmt(bubble['total_ms'])} ms "
            f"bubble_p50={_fmt(bubble['p50_us'])} us"
        )
    samples = payload["cpu_self_samples"]
    print(f"CPU sampled self time: leaf_samples={samples['total_leaf_samples']}")
    for row in samples["by_category"]:
        print(
            f"  {row['category']}: {row['samples']} samples "
            f"({_fmt(row['share_pct'])}%)"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if not args.sqlite.is_file():
        raise SystemExit(f"SQLite file does not exist: {args.sqlite}")
    payload = analyze(args.sqlite)
    print_summary(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
