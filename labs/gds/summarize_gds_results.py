#!/usr/bin/env python3
"""Parse gdsio logs produced by run_gdsio_matrix.sh.

The parser intentionally depends only on the Python standard library so that it
can run on a freshly provisioned GPU node.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


RESULT_RE = re.compile(
    r"IoType:\s*(?P<io_type>\S+)\s+"
    r"XferType:\s*(?P<xfer_type>\S+)\s+"
    r"Threads:\s*(?P<threads>\d+)\s+"
    r"(?:IoDepth:\s*\d+\s+)?"
    r"DataSetSize:\s*(?P<dataset_size>.*?)\s+"
    r"IOSize:\s*(?P<reported_io_size>.*?)\s+"
    r"Throughput:\s*(?P<throughput>[0-9.eE+-]+)\s+GiB/sec,?\s+"
    r"Avg_Latency:\s*(?P<latency>[0-9.eE+-]+)\s+usecs\s+"
    r"ops:\s*(?P<ops>\d+)\s+"
    r"total_time\s+(?P<total_time>[0-9.eE+-]+)\s+secs",
    flags=re.IGNORECASE | re.DOTALL,
)

CPU_PERCENT_RE = re.compile(r"Percent of CPU this job got:\s*([0-9.]+)%")
USER_TIME_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYSTEM_TIME_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")

TRANSFER_NAMES = {
    "0": "gds",
    "1": "cpu",
    "2": "cpu_gpu",
    "5": "gds_async",
    "6": "gds_batch",
    "7": "gds_batch_stream",
}

PARSED_FIELDS = (
    "run_id",
    "repetition",
    "gpu",
    "operation",
    "transfer_code",
    "transfer_name",
    "io_size",
    "workers",
    "dataset_size",
    "file",
    "log_file",
    "exit_code",
    "parse_status",
    "io_type",
    "xfer_type",
    "reported_threads",
    "reported_dataset_size",
    "reported_io_size",
    "throughput_gib_s",
    "avg_latency_us",
    "ops",
    "total_time_s",
    "cpu_percent",
    "user_time_s",
    "system_time_s",
)

SUMMARY_FIELDS = (
    "gpu",
    "operation",
    "transfer_code",
    "transfer_name",
    "io_size",
    "workers",
    "samples",
    "throughput_gib_s_median",
    "throughput_gib_s_min",
    "throughput_gib_s_max",
    "avg_latency_us_median",
    "total_time_s_median",
    "cpu_percent_median",
    "user_time_s_median",
    "system_time_s_median",
)


def parse_gdsio_output(text: str) -> dict[str, object] | None:
    """Return the final gdsio result found in a log, if present."""

    matches = list(RESULT_RE.finditer(text))
    if not matches:
        return None
    values = matches[-1].groupdict()
    result: dict[str, object] = {
        "io_type": values["io_type"],
        "xfer_type": values["xfer_type"],
        "reported_threads": int(values["threads"]),
        "reported_dataset_size": " ".join(values["dataset_size"].split()),
        "reported_io_size": " ".join(values["reported_io_size"].split()),
        "throughput_gib_s": float(values["throughput"]),
        "avg_latency_us": float(values["latency"]),
        "ops": int(values["ops"]),
        "total_time_s": float(values["total_time"]),
    }
    optional_metrics = (
        ("cpu_percent", CPU_PERCENT_RE),
        ("user_time_s", USER_TIME_RE),
        ("system_time_s", SYSTEM_TIME_RE),
    )
    for name, pattern in optional_metrics:
        optional_match = pattern.search(text)
        if optional_match:
            result[name] = float(optional_match.group(1))
    return result


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    required = {
        "run_id",
        "repetition",
        "gpu",
        "operation",
        "transfer_code",
        "io_size",
        "workers",
        "dataset_size",
        "file",
        "log_file",
        "exit_code",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        missing = sorted(required.difference(reader.fieldnames or []))
        raise ValueError(f"manifest is missing columns: {', '.join(missing)}")
    return rows


def _resolve_log(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def parse_manifest_runs(manifest_path: Path) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for row in read_manifest(manifest_path):
        transfer_code = row["transfer_code"]
        base: dict[str, object] = {
            field: row.get(field, "")
            for field in (
                "run_id",
                "repetition",
                "gpu",
                "operation",
                "transfer_code",
                "io_size",
                "workers",
                "dataset_size",
                "file",
                "log_file",
                "exit_code",
            )
        }
        base["transfer_name"] = row.get("transfer_name") or TRANSFER_NAMES.get(
            transfer_code, f"xfer_{transfer_code}"
        )
        log_path = _resolve_log(manifest_path, row["log_file"])
        if not log_path.exists():
            base["parse_status"] = "missing_log"
            parsed.append(base)
            continue
        metrics = parse_gdsio_output(log_path.read_text(encoding="utf-8", errors="replace"))
        if metrics is None:
            base["parse_status"] = "no_result"
            parsed.append(base)
            continue
        base["parse_status"] = "ok"
        base.update(metrics)
        parsed.append(base)
    return parsed


def summarize_runs(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    key_fields = (
        "gpu",
        "operation",
        "transfer_code",
        "transfer_name",
        "io_size",
        "workers",
    )
    for row in rows:
        if row.get("parse_status") != "ok" or str(row.get("exit_code")) != "0":
            continue
        key = tuple(str(row.get(field, "")) for field in key_fields)
        groups[key].append(row)

    summary: list[dict[str, object]] = []
    for key, samples in sorted(groups.items()):
        throughputs = [float(item["throughput_gib_s"]) for item in samples]
        latencies = [float(item["avg_latency_us"]) for item in samples]
        total_times = [float(item["total_time_s"]) for item in samples]
        cpu_percentages = [float(item["cpu_percent"]) for item in samples if "cpu_percent" in item]
        user_times = [float(item["user_time_s"]) for item in samples if "user_time_s" in item]
        system_times = [float(item["system_time_s"]) for item in samples if "system_time_s" in item]
        result: dict[str, object] = dict(zip(key_fields, key))
        result.update(
            {
                "samples": len(samples),
                "throughput_gib_s_median": statistics.median(throughputs),
                "throughput_gib_s_min": min(throughputs),
                "throughput_gib_s_max": max(throughputs),
                "avg_latency_us_median": statistics.median(latencies),
                "total_time_s_median": statistics.median(total_times),
                "cpu_percent_median": statistics.median(cpu_percentages) if cpu_percentages else "",
                "user_time_s_median": statistics.median(user_times) if user_times else "",
                "system_time_s_median": statistics.median(system_times) if system_times else "",
            }
        )
        summary.append(result)
    return summary


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(
    path: Path,
    parsed: list[Mapping[str, object]],
    summary: list[Mapping[str, object]],
) -> None:
    failures = [
        row
        for row in parsed
        if row.get("parse_status") != "ok" or str(row.get("exit_code")) != "0"
    ]
    lines = [
        "# GDSIO 结果摘要",
        "",
        f"- 清单记录：{len(parsed)}",
        f"- 成功解析：{len(parsed) - len(failures)}",
        f"- 失败或缺失：{len(failures)}",
        "",
        "| GPU | 操作 | 路径 | I/O size | workers | 样本 | 吞吐中位数 GiB/s | 平均延迟中位数 us | CPU% 中位数 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {gpu} | {operation} | {transfer_name} | {io_size} | {workers} | "
            "{samples} | {throughput:.6f} | {latency:.3f} | {cpu_percent} |".format(
                gpu=row["gpu"],
                operation=row["operation"],
                transfer_name=row["transfer_name"],
                io_size=row["io_size"],
                workers=row["workers"],
                samples=row["samples"],
                throughput=float(row["throughput_gib_s_median"]),
                latency=float(row["avg_latency_us_median"]),
                cpu_percent=(
                    f"{float(row['cpu_percent_median']):.1f}"
                    if row.get("cpu_percent_median") != ""
                    else "N/A"
                ),
            )
        )
    if failures:
        lines.extend(
            [
                "",
                "## 失败记录",
                "",
                "| run_id | exit_code | parse_status | log |",
                "|---|---:|---|---|",
            ]
        )
        for row in failures:
            lines.append(
                f"| {row.get('run_id', '')} | {row.get('exit_code', '')} | "
                f"{row.get('parse_status', '')} | `{row.get('log_file', '')}` |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="runs.tsv path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; defaults to the manifest directory",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="return non-zero when a log is missing, unparseable, or has a non-zero exit",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    manifest = args.manifest.resolve()
    output_dir = (args.output_dir or manifest.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_manifest_runs(manifest)
    summary = summarize_runs(parsed)
    write_csv(output_dir / "parsed_runs.csv", parsed, PARSED_FIELDS)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_markdown(output_dir / "summary.md", parsed, summary)
    print(f"Parsed {len(parsed)} runs; summary: {output_dir / 'summary.md'}")
    if args.fail_on_errors and any(
        row.get("parse_status") != "ok" or str(row.get("exit_code")) != "0"
        for row in parsed
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
