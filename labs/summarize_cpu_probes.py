#!/usr/bin/env python3
"""Summarize perf-stat CSV and py-spy folded/raw profiles."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_number(value: str) -> float | None:
    normalized = value.strip().replace(" ", "")
    if not normalized or normalized.startswith("<"):
        return None
    normalized = normalized.replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_perf_stat(path: Path) -> dict[str, Any]:
    counters: dict[str, float] = {}
    units: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        value = parse_number(fields[0])
        event = fields[2]
        if value is None or not event:
            continue
        counters[event] = value
        units[event] = fields[1]

    def find(*names: str) -> float | None:
        for name in names:
            if name in counters:
                return counters[name]
        return None

    cycles = find("cycles", "cpu-cycles")
    instructions = find("instructions")
    branches = find("branches", "branch-instructions")
    branch_misses = find("branch-misses")
    cache_refs = find("cache-references")
    cache_misses = find("cache-misses")
    task_clock_ms = find("task-clock")
    derived = {
        "ipc": instructions / cycles if instructions is not None and cycles else None,
        "branch_miss_pct": (
            branch_misses / branches * 100
            if branch_misses is not None and branches
            else None
        ),
        "cache_miss_pct": (
            cache_misses / cache_refs * 100
            if cache_misses is not None and cache_refs
            else None
        ),
        "context_switches_per_on_cpu_second": _per_second(
            find("context-switches", "cs"), task_clock_ms
        ),
        "cpu_migrations_per_on_cpu_second": _per_second(
            find("cpu-migrations", "migrations"), task_clock_ms
        ),
        "page_faults_per_on_cpu_second": _per_second(
            find("page-faults", "faults"), task_clock_ms
        ),
    }
    return {"counters": counters, "units": units, "derived": derived}


def _per_second(value: float | None, task_clock_ms: float | None) -> float | None:
    if value is None or not task_clock_ms:
        return None
    return value / (task_clock_ms / 1_000)


def frame_category(frame: str) -> str:
    value = frame.lower()
    if any(token in value for token in ("numpy", "multiarray", "umath", "ndarray")):
        return "numpy"
    if any(token in value for token in (
        "get_num_common_prefix_blocks", "scheduler.py", "vllm/v1/core/sched"
    )):
        return "vllm_scheduler"
    if any(token in value for token in ("torch._dynamo", "triton", "compile", "jit")):
        return "python_jit_or_compilation"
    if any(token in value for token in ("tolist", "json", "tokenizer", "serialize")):
        return "python_output_or_serialization"
    if any(token in value for token in ("cuda", "synchronize", "torch/cuda")):
        return "cuda_python_boundary"
    if "vllm" in value:
        return "vllm_other"
    return "python_other"


def parse_pyspy_raw(path: Path) -> dict[str, Any]:
    by_leaf: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(.*)\s+(\d+)$", line)
        if not match:
            continue
        stack = match.group(1)
        count = int(match.group(2))
        leaf = stack.rsplit(";", 1)[-1]
        by_leaf[leaf] += count
        by_category[frame_category(leaf)] += count
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
            "leaf": leaf,
            "samples": count,
            "share_pct": count / total * 100 if total else None,
        }
        for leaf, count in sorted(by_leaf.items(), key=lambda item: item[1], reverse=True)[:50]
    ]
    return {"total_samples": total, "by_category": categories, "top_leaf": top}


def summarize(
    perf_stat: Path | None,
    pyspy_all: Path | None,
    pyspy_gil: Path | None,
    *,
    requests: int | None = None,
    decode_steps: int | None = None,
) -> dict[str, Any]:
    if requests is not None and requests < 1:
        raise ValueError("requests must be positive")
    if decode_steps is not None and decode_steps < 1:
        raise ValueError("decode_steps must be positive")
    perf_payload = parse_perf_stat(perf_stat) if perf_stat else None
    if perf_payload:
        counters = perf_payload["counters"]
        perf_payload["normalization"] = {
            "requests": requests,
            "decode_steps": decode_steps,
            "counters_per_request": (
                {name: value / requests for name, value in counters.items()}
                if requests
                else {}
            ),
            "counters_per_decode_step": (
                {name: value / decode_steps for name, value in counters.items()}
                if decode_steps
                else {}
            ),
        }
    payload: dict[str, Any] = {
        "perf_stat": perf_payload,
        "pyspy_all": parse_pyspy_raw(pyspy_all) if pyspy_all else None,
        "pyspy_gil": parse_pyspy_raw(pyspy_gil) if pyspy_gil else None,
        "notes": [
            "Treat task-clock, cycles, and instructions as actual on-CPU optimization metrics.",
            "Compare perf counters over identical windows or normalize them by completed requests/steps.",
            "py-spy --gil reports sampled Python stacks holding the GIL, not GIL wait duration.",
            "py-spy results are statistical and separate runs must use identical load.",
        ],
    }
    all_samples = payload["pyspy_all"]["total_samples"] if payload["pyspy_all"] else 0
    gil_samples = payload["pyspy_gil"]["total_samples"] if payload["pyspy_gil"] else 0
    payload["gil_sample_ratio_proxy_pct"] = (
        gil_samples / all_samples * 100 if all_samples else None
    )
    return payload


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-stat", type=Path)
    parser.add_argument("--pyspy-all", type=Path)
    parser.add_argument("--pyspy-gil", type=Path)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    paths = (args.perf_stat, args.pyspy_all, args.pyspy_gil)
    if not any(paths):
        raise SystemExit("provide at least one input")
    for path in paths:
        if path is not None and not path.is_file():
            raise SystemExit(f"input does not exist: {path}")
    payload = summarize(
        args.perf_stat,
        args.pyspy_all,
        args.pyspy_gil,
        requests=args.requests,
        decode_steps=args.decode_steps,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
