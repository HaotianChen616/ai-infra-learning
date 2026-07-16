#!/usr/bin/env python3
"""Measure a single streaming request to an OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkResult:
    ttft_seconds: float | None
    total_seconds: float
    content_chunks: int
    inter_chunk_seconds: tuple[float, ...]


def parse_sse_data(line: bytes | str) -> dict | None:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def extract_content(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_benchmark(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    timeout: float,
) -> BenchmarkResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    first_content_at: float | None = None
    previous_content_at: float | None = None
    intervals: list[float] = []
    chunks = 0

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            event = parse_sse_data(line)
            if event is None:
                continue
            content = extract_content(event)
            if not content:
                continue

            now = time.perf_counter()
            if first_content_at is None:
                first_content_at = now
            if previous_content_at is not None:
                intervals.append(now - previous_content_at)
            previous_content_at = now
            chunks += 1
            print(content, end="", flush=True)

    end = time.perf_counter()
    print()
    return BenchmarkResult(
        ttft_seconds=(first_content_at - start) if first_content_at else None,
        total_seconds=end - start,
        content_chunks=chunks,
        inter_chunk_seconds=tuple(intervals),
    )


def format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain KV cache in three sentences.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    try:
        result = run_benchmark(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach endpoint: {error.reason}") from error

    print("\nTiming summary")
    print("==============")
    print(f"TTFT (first content chunk): {format_seconds(result.ttft_seconds)}")
    print(f"End-to-end:                {format_seconds(result.total_seconds)}")
    print(f"Content chunks:            {result.content_chunks}")
    print(f"Inter-chunk p50:           {format_seconds(percentile(result.inter_chunk_seconds, 0.50))}")
    print(f"Inter-chunk p95:           {format_seconds(percentile(result.inter_chunk_seconds, 0.95))}")
    print("\nA streaming chunk is not guaranteed to equal one model token.")


if __name__ == "__main__":
    main()
