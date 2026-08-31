#!/usr/bin/env python3
"""Summarize JSON results emitted by the MetaX mxFIO matrix runner."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


PARSED_FIELDS = (
    "run_id",
    "repetition",
    "mode",
    "gpu",
    "io_size",
    "numjobs",
    "region_size",
    "file",
    "json_file",
    "exit_code",
    "parse_status",
    "fio_version",
    "job_count",
    "read_gib_s",
    "read_iops",
    "read_gib",
    "runtime_s",
    "latency_mean_us",
    "latency_p99_us",
    "usr_cpu_percent",
    "sys_cpu_percent",
)

SUMMARY_FIELDS = (
    "mode",
    "gpu",
    "io_size",
    "numjobs",
    "samples",
    "read_gib_s_median",
    "read_gib_s_min",
    "read_gib_s_max",
    "read_iops_median",
    "latency_mean_us_median",
    "latency_p99_us_median",
    "usr_cpu_percent_median",
    "sys_cpu_percent_median",
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latency_block(operation: Mapping[str, object]) -> tuple[Mapping[str, object], float]:
    for name, scale_to_us in (
        ("clat_ns", 0.001),
        ("lat_ns", 0.001),
        ("clat_us", 1.0),
        ("lat_us", 1.0),
        ("clat_ms", 1000.0),
        ("lat_ms", 1000.0),
    ):
        value = operation.get(name)
        if isinstance(value, Mapping):
            return value, scale_to_us
    return {}, 1.0


def _percentile(block: Mapping[str, object], target: float) -> float | None:
    values = block.get("percentile")
    if not isinstance(values, Mapping) or not values:
        return None
    numeric: list[tuple[float, float]] = []
    for key, value in values.items():
        try:
            numeric.append((float(key), float(value)))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return min(numeric, key=lambda item: abs(item[0] - target))[1]


def parse_fio_json(payload: Mapping[str, object]) -> dict[str, object]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("fio JSON does not contain jobs")

    bw_bytes_s = 0.0
    iops = 0.0
    io_bytes = 0.0
    runtime_ms = 0.0
    usr_cpu = 0.0
    sys_cpu = 0.0
    weighted_latency = 0.0
    latency_weight = 0.0
    p99_values: list[float] = []

    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        read = job.get("read")
        if not isinstance(read, Mapping):
            continue
        job_bw = _number(read.get("bw_bytes"))
        if not job_bw:
            job_bw = _number(read.get("bw")) * 1024.0
        bw_bytes_s += job_bw
        iops += _number(read.get("iops"))
        io_bytes += _number(read.get("io_bytes"))
        runtime_ms = max(runtime_ms, _number(read.get("runtime")))
        usr_cpu += _number(job.get("usr_cpu"))
        sys_cpu += _number(job.get("sys_cpu"))

        latency, scale = _latency_block(read)
        mean_us = _number(latency.get("mean")) * scale
        total_ios = _number(read.get("total_ios"), 1.0)
        if mean_us:
            weighted_latency += mean_us * max(total_ios, 1.0)
            latency_weight += max(total_ios, 1.0)
        p99 = _percentile(latency, 99.0)
        if p99 is not None:
            p99_values.append(p99 * scale)

    return {
        "fio_version": str(payload.get("fio version", payload.get("fio_version", ""))),
        "job_count": len(jobs),
        "read_gib_s": bw_bytes_s / (1024.0**3),
        "read_iops": iops,
        "read_gib": io_bytes / (1024.0**3),
        "runtime_s": runtime_ms / 1000.0,
        "latency_mean_us": weighted_latency / latency_weight if latency_weight else "",
        "latency_p99_us": max(p99_values) if p99_values else "",
        "usr_cpu_percent": usr_cpu,
        "sys_cpu_percent": sys_cpu,
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    required = {
        "run_id",
        "repetition",
        "mode",
        "gpu",
        "io_size",
        "numjobs",
        "region_size",
        "file",
        "json_file",
        "exit_code",
    }
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def parse_manifest_runs(manifest: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in read_manifest(manifest):
        row: dict[str, object] = {
            key: source.get(key, "")
            for key in (
                "run_id",
                "repetition",
                "mode",
                "gpu",
                "io_size",
                "numjobs",
                "region_size",
                "file",
                "json_file",
                "exit_code",
            )
        }
        json_path = Path(source["json_file"])
        if not json_path.is_absolute():
            json_path = manifest.parent / json_path
        if not json_path.exists():
            row["parse_status"] = "missing_json"
            rows.append(row)
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("top-level JSON value is not an object")
            row.update(parse_fio_json(payload))
            row["parse_status"] = "ok"
        except (json.JSONDecodeError, ValueError) as error:
            row["parse_status"] = f"invalid_json:{error}"
        rows.append(row)
    return rows


def _median(rows: Iterable[Mapping[str, object]], field: str) -> object:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.median(values) if values else ""


def summarize_runs(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    key_fields = ("mode", "gpu", "io_size", "numjobs")
    for row in rows:
        if row.get("parse_status") != "ok" or str(row.get("exit_code")) != "0":
            continue
        key = tuple(str(row.get(field, "")) for field in key_fields)
        groups[key].append(row)

    output: list[dict[str, object]] = []
    for key, samples in sorted(groups.items()):
        throughputs = [float(sample["read_gib_s"]) for sample in samples]
        result: dict[str, object] = dict(zip(key_fields, key))
        result.update(
            {
                "samples": len(samples),
                "read_gib_s_median": statistics.median(throughputs),
                "read_gib_s_min": min(throughputs),
                "read_gib_s_max": max(throughputs),
                "read_iops_median": _median(samples, "read_iops"),
                "latency_mean_us_median": _median(samples, "latency_mean_us"),
                "latency_p99_us_median": _median(samples, "latency_p99_us"),
                "usr_cpu_percent_median": _median(samples, "usr_cpu_percent"),
                "sys_cpu_percent_median": _median(samples, "sys_cpu_percent"),
            }
        )
        output.append(result)
    return output


def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    columns = (
        "mode",
        "gpu",
        "io_size",
        "numjobs",
        "samples",
        "read_gib_s_median",
        "read_iops_median",
        "latency_mean_us_median",
        "latency_p99_us_median",
        "usr_cpu_percent_median",
        "sys_cpu_percent_median",
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_manifest_runs(args.manifest)
    summary = summarize_runs(parsed)
    write_csv(output_dir / "parsed.csv", PARSED_FIELDS, parsed)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary)
    write_markdown(output_dir / "summary.md", summary)

    failures = [
        row
        for row in parsed
        if row.get("parse_status") != "ok" or str(row.get("exit_code")) != "0"
    ]
    print(f"Parsed {len(parsed)} run(s), summarized {len(summary)} group(s), failures={len(failures)}")
    print(f"Summary: {output_dir / 'summary.md'}")
    return 1 if args.fail_on_errors and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
