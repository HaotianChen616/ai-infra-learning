#!/usr/bin/env python3
"""Summarize host-side CUDA synchronization API durations from Nsight CSV.

Input must be the CSV produced by the Nsight Systems ``cuda_api_trace`` report.
The reported durations are host API wall-clock durations, not active CPU cycles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYNC_API_PATTERN = re.compile(
    r"^(?:"
    r"cuda(?:Device|Event|Stream|Thread)Synchronize"
    r"|cu(?:Ctx|Event|Stream)Synchronize"
    r")(?:_v\d+)?$"
)


@dataclass(frozen=True)
class ApiSample:
    name: str
    duration_ns: int
    start_ns: int | None
    pid: str
    tid: str
    thread_name: str


def percentile(values: Iterable[int], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_field(
    fieldnames: Iterable[str],
    aliases: tuple[str, ...],
    *,
    required: bool,
) -> str | None:
    normalized = {_normalize_header(name): name for name in fieldnames}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for normalized_name, original in normalized.items():
        if any(normalized_name.startswith(alias) for alias in aliases):
            return original
    if required:
        raise ValueError(f"missing CSV field matching: {', '.join(aliases)}")
    return None


def _parse_integer(value: str) -> int:
    normalized = value.strip().replace(",", "")
    if not normalized:
        raise ValueError("numeric field is empty")
    return int(float(normalized))


def _csv_payload(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for index, line in enumerate(lines):
        normalized = _normalize_header(line)
        if "," in line and "duration" in normalized and "name" in normalized:
            return "\n".join(lines[index:]) + "\n"
    raise ValueError("could not find the cuda_api_trace CSV header")


def load_api_samples(path: Path) -> list[ApiSample]:
    reader = csv.DictReader(_csv_payload(path).splitlines())
    fieldnames = reader.fieldnames or []
    name_field = _find_field(fieldnames, ("name", "apiname", "function"), required=True)
    duration_field = _find_field(
        fieldnames,
        ("durationns", "duration"),
        required=True,
    )
    start_field = _find_field(fieldnames, ("startns", "start"), required=False)
    pid_field = _find_field(fieldnames, ("pid", "processid"), required=False)
    tid_field = _find_field(fieldnames, ("tid", "threadid"), required=False)
    thread_field = _find_field(
        fieldnames,
        ("threadname",),
        required=False,
    )

    samples: list[ApiSample] = []
    for row in reader:
        raw_name = (row.get(name_field) or "").strip()
        if not raw_name:
            continue
        raw_start = (row.get(start_field) or "").strip() if start_field else ""
        samples.append(
            ApiSample(
                name=raw_name,
                duration_ns=_parse_integer(row.get(duration_field) or ""),
                start_ns=_parse_integer(raw_start) if raw_start else None,
                pid=(row.get(pid_field) or "").strip() if pid_field else "",
                tid=(row.get(tid_field) or "").strip() if tid_field else "",
                thread_name=(row.get(thread_field) or "").strip()
                if thread_field
                else "",
            )
        )
    return samples


def is_sync_api(name: str) -> bool:
    leaf_name = name.rsplit("::", 1)[-1].split("(", 1)[0].strip()
    return SYNC_API_PATTERN.fullmatch(leaf_name) is not None


def _summary_row(
    name: str,
    samples: list[ApiSample],
    all_cuda_api_ns: int,
) -> dict[str, Any]:
    durations = [sample.duration_ns for sample in samples]
    total_ns = sum(durations)
    return {
        "name": name,
        "calls": len(samples),
        "total_ms": total_ns / 1_000_000,
        "share_of_summed_cuda_api_duration_pct": (
            total_ns / all_cuda_api_ns * 100 if all_cuda_api_ns else None
        ),
        "mean_us": total_ns / len(samples) / 1_000 if samples else None,
        "p50_us": (percentile(durations, 0.50) or 0) / 1_000,
        "p95_us": (percentile(durations, 0.95) or 0) / 1_000,
        "p99_us": (percentile(durations, 0.99) or 0) / 1_000,
        "max_us": max(durations) / 1_000 if durations else None,
    }


def summarize(
    samples: list[ApiSample],
    *,
    decode_steps: int,
    generated_tokens: int,
) -> dict[str, Any]:
    sync_samples = [sample for sample in samples if is_sync_api(sample.name)]
    grouped: dict[str, list[ApiSample]] = {}
    thread_grouped: dict[tuple[str, str, str, str], list[ApiSample]] = {}
    for sample in sync_samples:
        grouped.setdefault(sample.name, []).append(sample)
        thread_grouped.setdefault(
            (sample.pid, sample.tid, sample.thread_name, sample.name),
            [],
        ).append(sample)

    all_cuda_api_ns = sum(sample.duration_ns for sample in samples)
    total_sync_ns = sum(sample.duration_ns for sample in sync_samples)
    api_rows = sorted(
        (
            _summary_row(name, grouped_samples, all_cuda_api_ns)
            for name, grouped_samples in grouped.items()
        ),
        key=lambda row: float(row["total_ms"]),
        reverse=True,
    )
    thread_rows = []
    for (pid, tid, thread_name, name), grouped_samples in thread_grouped.items():
        row = _summary_row(name, grouped_samples, all_cuda_api_ns)
        row.update({"pid": pid, "tid": tid, "thread_name": thread_name})
        thread_rows.append(row)
    thread_rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)

    starts = [sample.start_ns for sample in samples if sample.start_ns is not None]
    ends = [
        sample.start_ns + sample.duration_ns
        for sample in samples
        if sample.start_ns is not None
    ]
    observed_span_ns = max(ends) - min(starts) if starts and ends else None
    return {
        "input_samples": len(samples),
        "sync_samples": len(sync_samples),
        "total_sync_host_wall_ms": total_sync_ns / 1_000_000,
        "sync_share_of_summed_cuda_api_duration_pct": (
            total_sync_ns / all_cuda_api_ns * 100 if all_cuda_api_ns else None
        ),
        "observed_cuda_api_span_ms": (
            observed_span_ns / 1_000_000 if observed_span_ns is not None else None
        ),
        "sync_host_wall_ms_per_decode_step": (
            total_sync_ns / decode_steps / 1_000_000 if decode_steps else None
        ),
        "sync_host_wall_us_per_generated_token": (
            total_sync_ns / generated_tokens / 1_000 if generated_tokens else None
        ),
        "by_api": api_rows,
        "by_thread_and_api": thread_rows,
        "notes": [
            "Durations are host API wall-clock time, not active on-CPU time.",
            "Self means time not nested under another recorded API; waits and descheduling may be included.",
            "Summing across tensor-parallel worker processes is not request critical-path time.",
            "Use Nsight thread states and GPU correlation to attribute each long call.",
        ],
    }


def _format_optional(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def print_summary(payload: dict[str, Any]) -> None:
    print(
        "API".ljust(30),
        "CALLS".rjust(7),
        "TOTAL_MS".rjust(11),
        "MEAN_US".rjust(11),
        "P50_US".rjust(11),
        "P95_US".rjust(11),
        "P99_US".rjust(11),
        "MAX_US".rjust(11),
    )
    print("-" * 109)
    for row in payload["by_api"]:
        print(
            str(row["name"])[:30].ljust(30),
            str(row["calls"]).rjust(7),
            _format_optional(row["total_ms"]).rjust(11),
            _format_optional(row["mean_us"]).rjust(11),
            _format_optional(row["p50_us"]).rjust(11),
            _format_optional(row["p95_us"]).rjust(11),
            _format_optional(row["p99_us"]).rjust(11),
            _format_optional(row["max_us"]).rjust(11),
        )
    print()
    print(
        "Total synchronization host wall time: "
        f"{payload['total_sync_host_wall_ms']:.3f} ms"
    )
    print(
        "Share of summed CUDA API durations: "
        f"{_format_optional(payload['sync_share_of_summed_cuda_api_duration_pct'])}%"
    )
    print(
        "Synchronization wall time per assumed decode step: "
        f"{_format_optional(payload['sync_host_wall_ms_per_decode_step'])} ms"
    )
    print(
        "Synchronization wall time per generated token: "
        f"{_format_optional(payload['sync_host_wall_us_per_generated_token'])} us"
    )
    if not payload["by_api"]:
        print("No blocking CUDA synchronization APIs matched.")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda_api_trace_csv", type=Path)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=8 * 128)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    if args.decode_steps < 1:
        raise SystemExit("--decode-steps must be at least 1")
    if args.generated_tokens < 1:
        raise SystemExit("--generated-tokens must be at least 1")
    try:
        samples = load_api_samples(args.cuda_api_trace_csv)
    except (OSError, ValueError) as error:
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
