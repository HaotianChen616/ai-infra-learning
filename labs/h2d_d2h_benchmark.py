#!/usr/bin/env python3
"""Profile host-to-device and device-to-host copies from the host and device sides.

PyTorch is imported lazily so the repository's standard-library-only tests still
run without CUDA or Ascend. Run this program on a Linux NVIDIA or Ascend host.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import multiprocessing
import os
import platform
import resource
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


BYTE_UNITS = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
}


@dataclass(frozen=True)
class Scenario:
    direction: str
    size_bytes: int
    pinned: bool
    non_blocking: bool
    sync_policy: str

    @property
    def name(self) -> str:
        memory = "pinned" if self.pinned else "pageable"
        mode = "nonblocking" if self.non_blocking else "blocking"
        return (
            f"{self.direction}/{format_bytes(self.size_bytes)}/"
            f"{memory}/{mode}/{self.sync_policy}"
        )


@dataclass(frozen=True)
class TransferSample:
    scenario: str
    iteration: int
    direction: str
    size_bytes: int
    pinned: bool
    non_blocking: bool
    sync_policy: str
    cpu_prepare_ms: float
    host_api_ms: float
    device_copy_ms: float
    completion_ms: float
    pipeline_ms: float


@dataclass(frozen=True)
class AcceleratorRuntime:
    backend: str
    torch: Any
    api: Any
    extension: Any | None = None


def parse_size(value: str) -> int:
    """Parse values such as 4096, 4KiB, 1MiB, or 0.5GB."""
    normalized = value.strip().lower().replace(" ", "")
    if not normalized:
        raise ValueError("size cannot be empty")

    suffix = ""
    number = normalized
    for unit in sorted(BYTE_UNITS, key=len, reverse=True):
        if normalized.endswith(unit):
            suffix = unit
            number = normalized[: -len(unit)]
            break

    try:
        parsed = float(number)
    except ValueError as error:
        raise ValueError(f"invalid size: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("size must be finite and greater than zero")

    multiplier = BYTE_UNITS.get(suffix, 1)
    size_bytes = int(parsed * multiplier)
    if size_bytes <= 0:
        raise ValueError("size rounds down to zero bytes")
    return size_bytes


def parse_csv_choices(value: str, allowed: set[str], label: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{label} cannot be empty")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"invalid {label}: {', '.join(invalid)}")
    return values


def parse_cpu_affinity(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {part!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError("CPU IDs must be non-negative")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU affinity cannot be empty")
    return tuple(sorted(cpus))


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


def bandwidth_gbps(size_bytes: int, milliseconds: float | None) -> float | None:
    if milliseconds is None or milliseconds <= 0:
        return None
    return size_bytes / (milliseconds * 1_000_000)


def format_bytes(size_bytes: int) -> str:
    for unit, factor in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size_bytes >= factor and size_bytes % factor == 0:
            return f"{size_bytes // factor}{unit}"
    return f"{size_bytes}B"


def _require_runtime(backend: str) -> AcceleratorRuntime:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required for this lab. Install the build matching the "
            "selected accelerator backend."
        ) from error

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available to PyTorch. Run with --backend cuda on a "
                "Linux NVIDIA GPU host."
            )
        return AcceleratorRuntime("cuda", torch, torch.cuda)

    try:
        import torch_npu
    except ImportError as error:
        raise SystemExit(
            "torch_npu is required for --backend npu. Install a torch_npu build "
            "matching the server's PyTorch and CANN versions."
        ) from error
    npu_api = torch_npu.npu
    if not npu_api.is_available():
        raise SystemExit(
            "Ascend NPU is not available to torch_npu. Check the driver, firmware, "
            "CANN environment, device permissions, and ASCEND_VISIBLE_DEVICES."
        )
    return AcceleratorRuntime("npu", torch, npu_api, torch_npu)


def _run_python_work(iterations: int) -> int:
    accumulator = 0
    for index in range(iterations):
        accumulator = ((accumulator << 5) - accumulator + index) & 0xFFFFFFFF
    return accumulator


@contextlib.contextmanager
def annotated_range(
    runtime: AcceleratorRuntime,
    label: str,
    enabled: bool,
    stream: Any | None = None,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    range_id: int | None = None
    if runtime.backend == "cuda":
        runtime.api.nvtx.range_push(label)
    else:
        try:
            range_id = runtime.api.mstx.range_start(label, stream)
        except AttributeError as error:
            raise RuntimeError(
                "This torch_npu build does not expose mstx markers. Disable "
                "--annotate or use a torch_npu/CANN version with mstx support."
            ) from error
    try:
        with runtime.torch.profiler.record_function(label):
            yield
    finally:
        if runtime.backend == "cuda":
            runtime.api.nvtx.range_pop()
        elif range_id is not None:
            runtime.api.mstx.range_end(range_id)


def _prepare_buffers(
    runtime: AcceleratorRuntime,
    scenario: Scenario,
    device: Any,
) -> tuple[Any, Any]:
    torch = runtime.torch
    if scenario.direction == "h2d":
        source = torch.empty(
            scenario.size_bytes,
            dtype=torch.uint8,
            pin_memory=scenario.pinned,
        )
        source.fill_(17)
        destination = torch.empty(
            scenario.size_bytes,
            dtype=torch.uint8,
            device=device,
        )
    else:
        source = torch.empty(
            scenario.size_bytes,
            dtype=torch.uint8,
            device=device,
        )
        source.fill_(23)
        destination = torch.empty(
            scenario.size_bytes,
            dtype=torch.uint8,
            pin_memory=scenario.pinned,
        )
    runtime.api.synchronize(device)
    return source, destination


def _copy(destination: Any, source: Any, non_blocking: bool) -> None:
    destination.copy_(source, non_blocking=non_blocking)


def run_scenario(
    runtime: AcceleratorRuntime,
    scenario: Scenario,
    *,
    device: Any,
    iterations: int,
    warmup: int,
    python_work: int,
    annotate: bool,
    verify: bool,
) -> list[TransferSample]:
    source, destination = _prepare_buffers(runtime, scenario, device)
    stream = runtime.api.Stream(device=device)

    with runtime.api.stream(stream):
        for _ in range(warmup):
            _copy(destination, source, scenario.non_blocking)
    stream.synchronize()

    events = [
        (
            runtime.api.Event(enable_timing=True),
            runtime.api.Event(enable_timing=True),
        )
        for _ in range(iterations)
    ]
    raw_samples: list[dict[str, float | int]] = []

    batch_started_ns = time.perf_counter_ns()
    for iteration, (start_event, end_event) in enumerate(events):
        prepare_started_ns = time.perf_counter_ns()
        with annotated_range(runtime, f"{scenario.name}/cpu_prepare", annotate):
            _run_python_work(python_work)
        prepare_finished_ns = time.perf_counter_ns()

        with runtime.api.stream(stream):
            start_event.record(stream)
            completion_started_ns = time.perf_counter_ns()
            with annotated_range(
                runtime,
                f"{scenario.name}/device_submit",
                annotate,
                stream,
            ):
                host_started_ns = time.perf_counter_ns()
                _copy(destination, source, scenario.non_blocking)
                host_finished_ns = time.perf_counter_ns()
            end_event.record(stream)

        if scenario.sync_policy == "each":
            with annotated_range(runtime, f"{scenario.name}/completion_wait", annotate):
                end_event.synchronize()
            completion_finished_ns = time.perf_counter_ns()
            pipeline_ms = (completion_finished_ns - prepare_started_ns) / 1_000_000
            completion_ms = (completion_finished_ns - completion_started_ns) / 1_000_000
        else:
            pipeline_ms = math.nan
            completion_ms = math.nan

        raw_samples.append(
            {
                "iteration": iteration,
                "cpu_prepare_ms": (prepare_finished_ns - prepare_started_ns)
                / 1_000_000,
                "host_api_ms": (host_finished_ns - host_started_ns) / 1_000_000,
                "completion_ms": completion_ms,
                "pipeline_ms": pipeline_ms,
            }
        )

    if scenario.sync_policy == "batch":
        with annotated_range(runtime, f"{scenario.name}/batch_wait", annotate):
            events[-1][1].synchronize()
        batch_finished_ns = time.perf_counter_ns()
        amortized_ms = (batch_finished_ns - batch_started_ns) / 1_000_000 / iterations
        for raw_sample in raw_samples:
            raw_sample["completion_ms"] = amortized_ms
            raw_sample["pipeline_ms"] = amortized_ms

    samples: list[TransferSample] = []
    for raw_sample, (start_event, end_event) in zip(raw_samples, events):
        samples.append(
            TransferSample(
                scenario=scenario.name,
                iteration=int(raw_sample["iteration"]),
                direction=scenario.direction,
                size_bytes=scenario.size_bytes,
                pinned=scenario.pinned,
                non_blocking=scenario.non_blocking,
                sync_policy=scenario.sync_policy,
                cpu_prepare_ms=float(raw_sample["cpu_prepare_ms"]),
                host_api_ms=float(raw_sample["host_api_ms"]),
                device_copy_ms=float(start_event.elapsed_time(end_event)),
                completion_ms=float(raw_sample["completion_ms"]),
                pipeline_ms=float(raw_sample["pipeline_ms"]),
            )
        )

    if verify:
        expected = 17 if scenario.direction == "h2d" else 23
        observed = int(destination[0].item())
        if observed != expected:
            raise RuntimeError(
                f"{scenario.name}: copy verification failed: {observed} != {expected}"
            )

    del source, destination, events, stream
    return samples


def summarize_samples(
    scenarios: Iterable[Scenario],
    samples: Iterable[TransferSample],
) -> list[dict[str, Any]]:
    samples_by_name: dict[str, list[TransferSample]] = {}
    for sample in samples:
        samples_by_name.setdefault(sample.scenario, []).append(sample)

    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        grouped = samples_by_name.get(scenario.name, [])
        if not grouped:
            continue
        host_p50 = percentile((sample.host_api_ms for sample in grouped), 0.50)
        host_p95 = percentile((sample.host_api_ms for sample in grouped), 0.95)
        host_p99 = percentile((sample.host_api_ms for sample in grouped), 0.99)
        device_p50 = percentile((sample.device_copy_ms for sample in grouped), 0.50)
        device_p95 = percentile((sample.device_copy_ms for sample in grouped), 0.95)
        device_p99 = percentile((sample.device_copy_ms for sample in grouped), 0.99)
        completion_p50 = percentile((sample.completion_ms for sample in grouped), 0.50)
        completion_p95 = percentile((sample.completion_ms for sample in grouped), 0.95)
        completion_p99 = percentile((sample.completion_ms for sample in grouped), 0.99)
        pipeline_p50 = percentile((sample.pipeline_ms for sample in grouped), 0.50)
        pipeline_p95 = percentile((sample.pipeline_ms for sample in grouped), 0.95)
        pipeline_p99 = percentile((sample.pipeline_ms for sample in grouped), 0.99)
        summaries.append(
            {
                "scenario": scenario.name,
                "direction": scenario.direction,
                "size_bytes": scenario.size_bytes,
                "size": format_bytes(scenario.size_bytes),
                "host_memory": "pinned" if scenario.pinned else "pageable",
                "mode": "nonblocking" if scenario.non_blocking else "blocking",
                "sync_policy": scenario.sync_policy,
                "samples": len(grouped),
                "cpu_prepare_ms_p50": percentile(
                    (sample.cpu_prepare_ms for sample in grouped), 0.50
                ),
                "cpu_prepare_ms_p95": percentile(
                    (sample.cpu_prepare_ms for sample in grouped), 0.95
                ),
                "cpu_prepare_ms_p99": percentile(
                    (sample.cpu_prepare_ms for sample in grouped), 0.99
                ),
                "host_api_ms_p50": host_p50,
                "host_api_ms_p95": host_p95,
                "host_api_ms_p99": host_p99,
                "device_copy_ms_p50": device_p50,
                "device_copy_ms_p95": device_p95,
                "device_copy_ms_p99": device_p99,
                "completion_ms_p50": completion_p50,
                "completion_ms_p95": completion_p95,
                "completion_ms_p99": completion_p99,
                "pipeline_ms_p50": pipeline_p50,
                "pipeline_ms_p95": pipeline_p95,
                "pipeline_ms_p99": pipeline_p99,
                "device_copy_gbps_p50": bandwidth_gbps(scenario.size_bytes, device_p50),
                "effective_gbps_p50": bandwidth_gbps(scenario.size_bytes, pipeline_p50),
            }
        )
    return summaries


def _rusage_snapshot() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _proc_counter_snapshot(path: str) -> dict[str, int]:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    counters: dict[str, int] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        label, remainder = line.split(":", 1)
        count = 0
        description: list[str] = []
        parsing_counts = True
        for field in remainder.split():
            if parsing_counts:
                try:
                    count += int(field)
                    continue
                except ValueError:
                    parsing_counts = False
            description.append(field)
        key = label.strip()
        if description:
            key = f"{key}: {' '.join(description)}"
        counters[key] = count
    return counters


def system_snapshot() -> dict[str, Any]:
    return {
        "rusage": _rusage_snapshot(),
        "interrupts": _proc_counter_snapshot("/proc/interrupts"),
        "softirqs": _proc_counter_snapshot("/proc/softirqs"),
    }


def _numeric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], (int, float))
    }


def _top_counter_delta(
    before: dict[str, int],
    after: dict[str, int],
    limit: int = 20,
) -> list[dict[str, int | str]]:
    deltas = [
        {"name": key, "delta": value - before.get(key, 0)}
        for key, value in after.items()
        if value - before.get(key, 0) > 0
    ]
    return sorted(deltas, key=lambda item: int(item["delta"]), reverse=True)[:limit]


def system_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "process": _numeric_delta(before["rusage"], after["rusage"]),
        "top_interrupt_deltas_system_wide": _top_counter_delta(
            before["interrupts"], after["interrupts"]
        ),
        "top_softirq_deltas_system_wide": _top_counter_delta(
            before["softirqs"], after["softirqs"]
        ),
        "warning": (
            "/proc interrupt counters are system-wide, not attributable to this "
            "process; correlate them with the timeline before drawing conclusions."
        ),
    }


def _gil_burn(stop_event: threading.Event) -> None:
    value = 1
    while not stop_event.is_set():
        value = (value * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF


def _cpu_burn_process(stop_event: Any, affinity: tuple[int, ...] | None) -> None:
    if affinity is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(affinity))
    value = 1
    while not stop_event.is_set():
        value = (value * 1_103_515_245 + 12_345) & 0x7FFFFFFF


class Interference:
    def __init__(
        self,
        mode: str,
        cpu_workers: int,
        affinity: tuple[int, ...] | None,
    ) -> None:
        self.mode = mode
        self.cpu_workers = cpu_workers
        self.affinity = affinity
        self.thread_stop: threading.Event | None = None
        self.thread: threading.Thread | None = None
        self.process_stop: Any = None
        self.processes: list[Any] = []

    def __enter__(self) -> "Interference":
        if self.mode == "gil":
            self.thread_stop = threading.Event()
            self.thread = threading.Thread(
                target=_gil_burn,
                args=(self.thread_stop,),
                name="python-gil-contender",
                daemon=True,
            )
            self.thread.start()
        elif self.mode == "cpu":
            context = multiprocessing.get_context("spawn")
            self.process_stop = context.Event()
            for worker_id in range(self.cpu_workers):
                process = context.Process(
                    target=_cpu_burn_process,
                    args=(self.process_stop, self.affinity),
                    name=f"cpu-contender-{worker_id}",
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.thread_stop is not None:
            self.thread_stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.process_stop is not None:
            self.process_stop.set()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


def collect_metadata(
    runtime: AcceleratorRuntime,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch = runtime.torch
    properties = runtime.api.get_device_properties(device)
    if runtime.backend == "cuda":
        property_names = (
            "name",
            "total_memory",
            "major",
            "minor",
            "multi_processor_count",
            "is_multi_gpu_board",
            "is_integrated",
        )
    else:
        property_names = (
            "name",
            "total_memory",
            "L2_cache_size",
            "cube_core_num",
            "vector_core_num",
        )
    accelerator = {}
    for name in property_names:
        try:
            value = getattr(properties, name)
        except (AttributeError, RuntimeError):
            continue
        if value is not None:
            accelerator[name] = value

    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    extension_version = None
    cann_version = None
    if runtime.extension is not None:
        extension_version = getattr(runtime.extension, "__version__", None)
        get_cann_version = getattr(
            getattr(runtime.extension, "utils", None),
            "get_cann_version",
            None,
        )
        if get_cann_version is not None:
            try:
                cann_version = get_cann_version()
            except (RuntimeError, TypeError):
                pass
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "backend": runtime.backend,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_npu": extension_version,
        "cann": cann_version,
        "device": str(device),
        "accelerator": accelerator,
        "cpu_affinity": affinity,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_LAUNCH_BLOCKING",
                "TASK_QUEUE_ENABLE",
                "CPU_AFFINITY_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "arguments": vars(args),
    }


def measure_host_allocation(
    torch: Any,
    sizes: Iterable[int],
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size_bytes in sizes:
        for pinned in (False, True):
            elapsed: list[float] = []
            for _ in range(iterations):
                started_ns = time.perf_counter_ns()
                buffer = torch.empty(
                    size_bytes,
                    dtype=torch.uint8,
                    pin_memory=pinned,
                )
                buffer.fill_(1)
                elapsed.append((time.perf_counter_ns() - started_ns) / 1_000_000)
                del buffer
            rows.append(
                {
                    "size_bytes": size_bytes,
                    "size": format_bytes(size_bytes),
                    "host_memory": "pinned" if pinned else "pageable",
                    "iterations": iterations,
                    "allocate_and_touch_ms_first": elapsed[0],
                    "allocate_and_touch_ms_p50": percentile(elapsed, 0.50),
                    "allocate_and_touch_ms_p95": percentile(elapsed, 0.95),
                }
            )
    return rows


def write_outputs(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    samples: list[TransferSample],
    summaries: list[dict[str, Any]],
    allocation: list[dict[str, Any]],
    os_delta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "summary": summaries,
        "host_allocation": allocation,
        "os_observation": os_delta,
        "metric_notes": {
            "host_api_ms": "CPU time spent inside Tensor.copy_; enqueue is not completion.",
            "device_copy_ms": (
                "Accelerator event time around the copy on its stream; CUDA Event "
                "for NVIDIA and NPU Event for Ascend."
            ),
            "completion_ms": (
                "For sync=each, submit-to-event-completion wall time. For sync=batch, "
                "whole batch wall time amortized per copy."
            ),
            "pipeline_ms": (
                "CPU prepare through completion. Batch mode is amortized and includes "
                "all inter-copy CPU gaps."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    if samples:
        with (output_dir / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(samples[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)


def print_summary(summaries: list[dict[str, Any]]) -> None:
    columns = (
        ("direction", 4),
        ("size", 8),
        ("host_memory", 8),
        ("mode", 11),
        ("sync_policy", 5),
    )
    header = " ".join(name.upper().ljust(width) for name, width in columns)
    header += " HOST_P50  DEV_P50  PIPE_P95  DEV_GB/s  EFF_GB/s"
    print(header)
    print("-" * len(header))
    for row in summaries:
        prefix = " ".join(
            str(row[name]).ljust(width)[:width] for name, width in columns
        )
        print(
            f"{prefix} "
            f"{row['host_api_ms_p50']:8.3f} "
            f"{row['device_copy_ms_p50']:8.3f} "
            f"{row['pipeline_ms_p95']:8.3f} "
            f"{row['device_copy_gbps_p50']:9.3f} "
            f"{row['effective_gbps_p50']:9.3f}"
        )


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    sizes = tuple(parse_size(value) for value in args.sizes.split(","))
    directions = parse_csv_choices(args.directions, {"h2d", "d2h"}, "directions")
    memories = parse_csv_choices(
        args.host_memory, {"pageable", "pinned"}, "host memory"
    )
    modes = parse_csv_choices(args.modes, {"blocking", "nonblocking"}, "modes")
    sync_policies = parse_csv_choices(
        args.sync_policies, {"each", "batch"}, "sync policies"
    )
    return [
        Scenario(
            direction=direction,
            size_bytes=size_bytes,
            pinned=memory == "pinned",
            non_blocking=mode == "nonblocking",
            sync_policy=sync_policy,
        )
        for direction in directions
        for size_bytes in sizes
        for memory in memories
        for mode in modes
        for sync_policy in sync_policies
    ]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument(
        "--device",
        help="Defaults to cuda:0 for CUDA and npu:0 for Ascend.",
    )
    parser.add_argument("--sizes", default="4KiB,1MiB,16MiB,64MiB")
    parser.add_argument("--directions", default="h2d,d2h")
    parser.add_argument("--host-memory", default="pageable,pinned")
    parser.add_argument("--modes", default="blocking,nonblocking")
    parser.add_argument("--sync-policies", default="each,batch")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--python-work", type=int, default=0)
    parser.add_argument(
        "--interference",
        choices=("none", "gil", "cpu"),
        default="none",
        help="Optional Python GIL or OS scheduler contention.",
    )
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument(
        "--cpu-affinity",
        help="Linux CPU set such as 8 or 8-11; also inherited by CPU contenders.",
    )
    parser.add_argument("--allocation-iterations", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/h2d_d2h"))
    parser.add_argument(
        "--annotate",
        action="store_true",
        help=(
            "Emit NVTX (CUDA) or mstx (Ascend) plus record_function ranges; "
            "enable the matching external profiler marker switch."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Also export a focused CUDA or Ascend PyTorch Profiler trace.",
    )
    parser.add_argument("--trace-size", default="64MiB")
    parser.add_argument("--trace-direction", choices=("h2d", "d2h"), default="h2d")
    parser.add_argument(
        "--trace-host-memory", choices=("pageable", "pinned"), default="pinned"
    )
    parser.add_argument(
        "--trace-mode", choices=("blocking", "nonblocking"), default="nonblocking"
    )
    parser.add_argument(
        "--trace-sync-policy", choices=("each", "batch"), default="each"
    )
    parser.add_argument("--trace-iterations", type=int, default=10)
    parser.add_argument("--trace-stacks", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for name in ("warmup", "iterations", "trace_iterations", "cpu_workers"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.python_work < 0:
        raise ValueError("--python-work must be non-negative")
    if args.allocation_iterations < 0:
        raise ValueError("--allocation-iterations must be non-negative")


def _npu_experimental_config(torch_npu: Any) -> Any:
    profiler = torch_npu.profiler
    common = {
        "profiler_level": profiler.ProfilerLevel.Level0,
        "aic_metrics": profiler.AiCMetrics.AiCoreNone,
        "l2_cache": False,
        "data_simplification": False,
    }
    try:
        return profiler._ExperimentalConfig(mstx=True, **common)
    except TypeError:
        return profiler._ExperimentalConfig(msprof_tx=True, **common)


def export_accelerator_trace(
    runtime: AcceleratorRuntime,
    args: argparse.Namespace,
    device: Any,
    output_dir: Path,
) -> None:
    scenario = Scenario(
        direction=args.trace_direction,
        size_bytes=parse_size(args.trace_size),
        pinned=args.trace_host_memory == "pinned",
        non_blocking=args.trace_mode == "nonblocking",
        sync_policy=args.trace_sync_policy,
    )
    if runtime.backend == "cuda":
        profiler_api = runtime.torch.profiler
        activities = [
            profiler_api.ProfilerActivity.CPU,
            profiler_api.ProfilerActivity.CUDA,
        ]
        profile_options: dict[str, Any] = {}
    else:
        profiler_api = runtime.extension.profiler
        activities = [
            profiler_api.ProfilerActivity.CPU,
            profiler_api.ProfilerActivity.NPU,
        ]
        profile_options = {
            "experimental_config": _npu_experimental_config(runtime.extension)
        }

    with profiler_api.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.trace_stacks,
        **profile_options,
    ) as profiler_result:
        run_scenario(
            runtime,
            scenario,
            device=device,
            iterations=args.trace_iterations,
            warmup=2,
            python_work=args.python_work,
            annotate=True,
            verify=not args.no_verify,
        )

    trace_path = output_dir / f"{runtime.backend}_trace.json"
    profiler_result.export_chrome_trace(str(trace_path))
    key_averages = getattr(profiler_result, "key_averages", None)
    if key_averages is not None:
        (output_dir / "torch_profiler_table.txt").write_text(
            key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=100,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"{runtime.backend.upper()} trace: {trace_path}")


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
        scenarios = build_scenarios(args)
        affinity = parse_cpu_affinity(args.cpu_affinity) if args.cpu_affinity else None
    except ValueError as error:
        parser.error(str(error))

    if affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu-affinity is supported only on Linux")
        os.sched_setaffinity(0, set(affinity))

    runtime = _require_runtime(args.backend)
    device_name = args.device or f"{args.backend}:0"
    device = runtime.torch.device(device_name)
    runtime.api.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[TransferSample] = []
    before = system_snapshot()
    with Interference(args.interference, args.cpu_workers, affinity):
        for index, scenario in enumerate(scenarios, start=1):
            print(f"[{index:02d}/{len(scenarios):02d}] {scenario.name}", flush=True)
            all_samples.extend(
                run_scenario(
                    runtime,
                    scenario,
                    device=device,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    python_work=args.python_work,
                    annotate=args.annotate,
                    verify=not args.no_verify,
                )
            )
    after = system_snapshot()

    allocation = (
        measure_host_allocation(
            runtime.torch,
            sorted({scenario.size_bytes for scenario in scenarios}),
            args.allocation_iterations,
        )
        if args.allocation_iterations
        else []
    )
    summaries = summarize_samples(scenarios, all_samples)
    metadata = collect_metadata(runtime, device, args)
    write_outputs(
        args.output_dir,
        metadata=metadata,
        samples=all_samples,
        summaries=summaries,
        allocation=allocation,
        os_delta=system_delta(before, after),
    )
    print()
    print_summary(summaries)
    print(f"\nRaw samples: {args.output_dir / 'samples.csv'}")
    print(f"Summary:     {args.output_dir / 'summary.json'}")

    if args.trace:
        export_accelerator_trace(runtime, args, device, args.output_dir)


if __name__ == "__main__":
    main()
