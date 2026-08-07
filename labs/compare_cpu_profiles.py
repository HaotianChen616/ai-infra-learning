#!/usr/bin/env python3
"""Compare JSON summaries emitted by the vLLM CPU profiling analyzers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def flatten(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "sync" in payload:
        for row in payload["sync"].get("by_api", []):
            name = row["name"]
            for section, prefix in (
                ("api_wall", "sync_wall"),
                ("device_not_ready", "sync_device_wait"),
                ("post_event_host_tail", "sync_host_tail"),
                ("sampled_on_cpu_inside_api", "sync_on_cpu"),
                ("sampled_off_cpu_inside_api", "sync_off_cpu"),
            ):
                values = row.get(section, {})
                for metric in ("count", "total_ms", "p50_us", "p99_us", "max_us"):
                    value = values.get(metric)
                    if value is not None:
                        metrics[f"{prefix}.{name}.{metric}"] = float(value)
        for row in payload.get("submissions", {}).get("by_api", []):
            name = row["name"]
            for section, prefix in (
                ("api_wall", "submit_api"),
                ("critical_launch_bubble", "submit_critical_bubble"),
            ):
                values = row.get(section, {})
                for metric in ("count", "total_ms", "p50_us", "p99_us", "max_us"):
                    value = values.get(metric)
                    if value is not None:
                        metrics[f"{prefix}.{name}.{metric}"] = float(value)
        samples = payload.get("cpu_self_samples", {})
        metrics["cpu_sample.total_leaf_samples"] = float(
            samples.get("total_leaf_samples", 0)
        )
        for row in samples.get("by_category", []):
            if row.get("share_pct") is not None:
                metrics[f"cpu_sample_share.{row['category']}.pct"] = float(
                    row["share_pct"]
                )
    if "by_analysis_category" in payload:
        metrics["torch_trace.summed_cpu_self_wall_ms"] = float(
            payload.get("summed_cpu_self_wall_ms", 0)
        )
        for row in payload.get("by_analysis_category", []):
            category = row["category"]
            metrics[f"torch_self.{category}.ms"] = float(row["self_wall_ms"])
            share = row.get("share_of_summed_self_wall_pct")
            if share is not None:
                metrics[f"torch_self_share.{category}.pct"] = float(share)
    if "perf_stat" in payload or "pyspy_all" in payload:
        perf = payload.get("perf_stat") or {}
        for name, value in (perf.get("counters") or {}).items():
            if value is not None:
                metrics[f"perf_counter.{name}"] = float(value)
        for name, value in (perf.get("derived") or {}).items():
            if value is not None:
                metrics[f"perf_derived.{name}"] = float(value)
        all_samples = payload.get("pyspy_all") or {}
        metrics["pyspy.total_samples"] = float(all_samples.get("total_samples", 0))
        for row in all_samples.get("by_category", []):
            share = row.get("share_pct")
            if share is not None:
                metrics[f"pyspy_share.{row['category']}.pct"] = float(share)
        gil_samples = payload.get("pyspy_gil") or {}
        metrics["pyspy_gil.total_samples"] = float(
            gil_samples.get("total_samples", 0)
        )
        proxy = payload.get("gil_sample_ratio_proxy_pct")
        if proxy is not None:
            metrics["pyspy_gil.sample_ratio_proxy_pct"] = float(proxy)
    return metrics


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("comparison label cannot be empty")
    return label, Path(raw_path)


def compare(inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    runs = []
    for label, path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append({"label": label, "path": str(path), "metrics": flatten(payload)})
    baseline = runs[0]["metrics"]
    all_names = sorted({name for run in runs for name in run["metrics"]})
    rows = []
    for name in all_names:
        values = []
        base_value = baseline.get(name)
        for run in runs:
            value = run["metrics"].get(name)
            delta = value - base_value if value is not None and base_value is not None else None
            delta_pct = (
                delta / base_value * 100
                if delta is not None and base_value not in (None, 0)
                else None
            )
            values.append(
                {
                    "label": run["label"],
                    "value": value,
                    "delta_vs_baseline": delta,
                    "delta_pct_vs_baseline": delta_pct,
                }
            )
        rows.append({"metric": name, "values": values})
    return {
        "baseline": runs[0]["label"],
        "runs": [{"label": run["label"], "path": run["path"]} for run in runs],
        "metrics": rows,
        "notes": [
            "Compare only identical workload windows and profiler configurations.",
            "CPU sample shares are statistical; compare sample count and confidence.",
            "A smaller blocking API wall time is not necessarily smaller on-CPU time.",
        ],
    }


def markdown(payload: dict[str, Any]) -> str:
    labels = [run["label"] for run in payload["runs"]]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for row in payload["metrics"]:
        rendered = []
        for index, value in enumerate(row["values"]):
            number = value["value"]
            if number is None:
                rendered.append("-")
            elif index == 0 or value["delta_pct_vs_baseline"] is None:
                rendered.append(f"{number:.4f}")
            else:
                rendered.append(
                    f"{number:.4f} ({value['delta_pct_vs_baseline']:+.2f}%)"
                )
        lines.append(f"| `{row['metric']}` | " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=parse_input,
        metavar="LABEL=SUMMARY.json",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("provide at least two summaries")
    for _label, path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"summary does not exist: {path}")
    payload = compare(args.inputs)
    rendered = markdown(payload)
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
