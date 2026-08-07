#!/usr/bin/env python3
"""Diff two /proc/interrupts or /proc/softirqs snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_snapshot(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty snapshot: {path}")
    cpus = [token for token in lines[0].split() if token.startswith("CPU")]
    if not cpus:
        raise ValueError(f"CPU header not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        raw_label, payload = line.split(":", 1)
        label = raw_label.strip()
        tokens = payload.split()
        if len(tokens) < len(cpus):
            continue
        try:
            counts = [int(token.replace(",", "")) for token in tokens[: len(cpus)]]
        except ValueError:
            continue
        description = " ".join(tokens[len(cpus) :])
        rows[label] = {
            "label": label,
            "description": description,
            "counts": counts,
        }
    return cpus, rows


def diff(before: Path, after: Path) -> dict[str, Any]:
    before_cpus, before_rows = parse_snapshot(before)
    after_cpus, after_rows = parse_snapshot(after)
    if before_cpus != after_cpus:
        raise ValueError("CPU headers differ between snapshots")
    rows = []
    for label in sorted(set(before_rows) | set(after_rows)):
        before_row = before_rows.get(label, {"counts": [0] * len(before_cpus), "description": ""})
        after_row = after_rows.get(label, {"counts": [0] * len(after_cpus), "description": ""})
        deltas = [right - left for left, right in zip(before_row["counts"], after_row["counts"])]
        rows.append(
            {
                "label": label,
                "description": after_row["description"] or before_row["description"],
                "total_delta": sum(deltas),
                "per_cpu_delta": dict(zip(before_cpus, deltas)),
            }
        )
    rows.sort(key=lambda row: int(row["total_delta"]), reverse=True)
    return {
        "before": str(before),
        "after": str(after),
        "cpus": before_cpus,
        "rows": rows,
        "notes": [
            "Counter deltas include every workload on the host during the window.",
            "Correlate NVIDIA IRQ labels and busy CPUs with affinity and scheduler evidence.",
        ],
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.top < 1:
        raise SystemExit("--top must be positive")
    try:
        payload = diff(args.before, args.after)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("TOTAL".rjust(12), "IRQ".ljust(12), "DESCRIPTION")
    for row in payload["rows"][: args.top]:
        print(str(row["total_delta"]).rjust(12), row["label"].ljust(12), row["description"])
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
