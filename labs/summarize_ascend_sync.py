#!/usr/bin/env python3
"""Summarize host-side Ascend synchronization events from trace_view.json."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYNC_PATTERN = re.compile(
    r"(?i)(?:"
    r"(?:aclrt|rt)(?:synchronize(?:device|stream|event)|"
    r"(?:device|stream|event)synchronize)"
    r"|(?:torch_?npu|npu).*(?:synchronize|synchronise)"
    r"|(?:device|stream|event)[_.:]*(?:synchronize|synchronise)"
    r")"
)


@dataclass(frozen=True)
class SyncEvent:
    name: str
    family: str
    duration_us: float
    pid: str
    tid: str
    source: str


def _family(name: str) -> str:
    lowered = name.lower()
    if "aclrt" in lowered:
        return "aclrt"
    if re.search(r"(?:^|[^a-z])rt(?:synchronize|device|stream|event)", lowered):
        return "rt"
    return "framework"


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _trace_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    raise ValueError("trace must be a list or contain traceEvents")


def load_sync_events(path: Path, *, duration_unit: str) -> list[SyncEvent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    multiplier = {"us": 1.0, "ns": 0.001, "ms": 1000.0}[duration_unit]
    samples: list[SyncEvent] = []
    for event in _trace_events(payload):
        name = str(event.get("name", "")).strip()
        if not name or not SYNC_PATTERN.search(name):
            continue
        if event.get("ph") not in {None, "X"}:
            continue
        raw_duration = event.get("dur")
        if not isinstance(raw_duration, (int, float)):
            continue
        samples.append(
            SyncEvent(
                name=name,
                family=_family(name),
                duration_us=float(raw_duration) * multiplier,
                pid=str(event.get("pid", "")),
                tid=str(event.get("tid", "")),
                source=str(path),
            )
        )
    return samples


def _row(name: str, samples: list[SyncEvent]) -> dict[str, Any]:
    durations = [sample.duration_us for sample in samples]
    total_us = sum(durations)
    return {
        "name": name,
        "calls": len(samples),
        "total_ms": total_us / 1000,
        "mean_us": total_us / len(samples),
        "p50_us": percentile(durations, 0.50),
        "p95_us": percentile(durations, 0.95),
        "p99_us": percentile(durations, 0.99),
        "max_us": max(durations),
    }


def summarize(
    samples: list[SyncEvent],
    *,
    decode_steps: int,
    generated_tokens: int,
) -> dict[str, Any]:
    families: dict[str, list[SyncEvent]] = defaultdict(list)
    by_api: dict[tuple[str, str], list[SyncEvent]] = defaultdict(list)
    by_thread: dict[tuple[str, str, str, str, str], list[SyncEvent]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[SyncEvent]] = defaultdict(list)
    source_families: dict[str, dict[str, list[SyncEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        families[sample.family].append(sample)
        by_api[(sample.family, sample.name)].append(sample)
        by_thread[
            (sample.family, sample.name, sample.pid, sample.tid, sample.source)
        ].append(sample)
        by_source[(sample.family, sample.source)].append(sample)
        source_families[sample.source][sample.family].append(sample)

    preferred_family = next(
        (name for name in ("aclrt", "rt", "framework") if families.get(name)),
        None,
    )
    preferred_by_source = {
        source: next(
            (family for family in ("aclrt", "rt", "framework") if grouped.get(family)),
            None,
        )
        for source, grouped in source_families.items()
    }
    selected = [
        sample
        for sample in samples
        if sample.family == preferred_by_source.get(sample.source)
    ]
    selected_total_us = sum(sample.duration_us for sample in selected)
    api_rows = []
    for (family, name), grouped in by_api.items():
        row = _row(name, grouped)
        row["family"] = family
        api_rows.append(row)
    api_rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)

    thread_rows = []
    for (family, name, pid, tid, source), grouped in by_thread.items():
        row = _row(name, grouped)
        row.update(
            {
                "family": family,
                "pid": pid,
                "tid": tid,
                "source": source,
            }
        )
        thread_rows.append(row)
    thread_rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)

    selected_by_source = {
        source: (
            sum(
                sample.duration_us
                for sample in grouped[preferred_by_source[source] or ""]
            )
            / 1000
        )
        for source, grouped in source_families.items()
    }

    return {
        "input_sync_events": len(samples),
        "preferred_nonduplicated_family": preferred_family,
        "preferred_nonduplicated_family_by_source": preferred_by_source,
        "selected_sync_host_wall_ms": selected_total_us / 1000,
        "selected_sync_host_wall_ms_per_decode_step": (
            selected_total_us / decode_steps / 1000 if decode_steps else None
        ),
        "selected_sync_host_wall_us_per_generated_token": (
            selected_total_us / generated_tokens if generated_tokens else None
        ),
        "selected_sync_host_wall_ms_by_source": selected_by_source,
        "family_totals_ms": {
            family: sum(sample.duration_us for sample in grouped) / 1000
            for family, grouped in families.items()
        },
        "by_api": api_rows,
        "by_pid_tid_api": thread_rows,
        "notes": [
            "Durations are Host API wall time, not active CPU cycles.",
            "aclrt/rt/framework layers can be nested; do not sum family totals.",
            "TP ranks can wait concurrently; summing ranks is not request critical path.",
            "Use CPU thread state plus NPU/HCCL/copy overlap for attribution.",
            "Chrome trace duration defaults to microseconds; override --duration-unit if needed.",
        ],
    }


def print_summary(payload: dict[str, Any]) -> None:
    print(
        "FAMILY".ljust(10),
        "API".ljust(42),
        "CALLS".rjust(7),
        "TOTAL_MS".rjust(11),
        "P50_US".rjust(11),
        "P95_US".rjust(11),
        "P99_US".rjust(11),
        "MAX_US".rjust(11),
    )
    print("-" * 120)
    for row in payload["by_api"]:
        print(
            str(row["family"])[:10].ljust(10),
            str(row["name"])[:42].ljust(42),
            str(row["calls"]).rjust(7),
            f"{row['total_ms']:.3f}".rjust(11),
            f"{row['p50_us']:.3f}".rjust(11),
            f"{row['p95_us']:.3f}".rjust(11),
            f"{row['p99_us']:.3f}".rjust(11),
            f"{row['max_us']:.3f}".rjust(11),
        )
    print()
    print(
        "Preferred non-duplicated family: "
        f"{payload['preferred_nonduplicated_family'] or '-'}"
    )
    print(
        "Selected synchronization Host wall time: "
        f"{payload['selected_sync_host_wall_ms']:.3f} ms"
    )
    for source, total_ms in payload["selected_sync_host_wall_ms_by_source"].items():
        print(f"  {source}: {total_ms:.3f} ms")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_view_json", type=Path, nargs="+")
    parser.add_argument(
        "--duration-unit",
        choices=("us", "ns", "ms"),
        default="us",
        help="Unit used by each trace event's dur field. Chrome trace defaults to us.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=127,
        help="Additional steady decode iterations; use 0 for a 1024->1 control.",
    )
    parser.add_argument("--generated-tokens", type=int, default=8 * 128)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    if args.decode_steps < 0 or args.generated_tokens < 1:
        raise SystemExit(
            "decode steps must be non-negative and generated tokens positive"
        )
    samples = []
    try:
        for path in args.trace_view_json:
            samples.extend(load_sync_events(path, duration_unit=args.duration_unit))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    payload = summarize(
        samples,
        decode_steps=args.decode_steps,
        generated_tokens=args.generated_tokens,
    )
    print_summary(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
