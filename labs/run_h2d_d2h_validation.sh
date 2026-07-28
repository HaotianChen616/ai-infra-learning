#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${1:-artifacts/h2d_d2h_validation}"
BENCHMARK="labs/h2d_d2h_benchmark.py"

mkdir -p "${OUTPUT_ROOT}/system"

uname -a >"${OUTPUT_ROOT}/system/uname.txt"
lscpu >"${OUTPUT_ROOT}/system/lscpu.txt"
nvidia-smi -q >"${OUTPUT_ROOT}/system/nvidia-smi-q.txt"
nvidia-smi topo -m >"${OUTPUT_ROOT}/system/nvidia-smi-topo.txt"

if command -v numactl >/dev/null 2>&1; then
  numactl --hardware >"${OUTPUT_ROOT}/system/numa.txt"
fi

"${PYTHON_BIN}" "${BENCHMARK}" \
  --backend cuda \
  --sizes 4KiB,1MiB,16MiB,64MiB,256MiB \
  --iterations 100 \
  --warmup 20 \
  --allocation-iterations 5 \
  --output-dir "${OUTPUT_ROOT}/baseline"

"${PYTHON_BIN}" "${BENCHMARK}" \
  --backend cuda \
  --sizes 64MiB \
  --host-memory pinned \
  --modes nonblocking \
  --sync-policies each \
  --iterations 20 \
  --warmup 5 \
  --trace \
  --trace-size 64MiB \
  --trace-direction h2d \
  --trace-iterations 20 \
  --output-dir "${OUTPUT_ROOT}/torch-profiler-h2d"

"${PYTHON_BIN}" "${BENCHMARK}" \
  --backend cuda \
  --sizes 64MiB \
  --host-memory pinned \
  --modes nonblocking \
  --sync-policies each \
  --iterations 20 \
  --warmup 5 \
  --trace \
  --trace-size 64MiB \
  --trace-direction d2h \
  --trace-iterations 20 \
  --output-dir "${OUTPUT_ROOT}/torch-profiler-d2h"

if command -v nsys >/dev/null 2>&1; then
  mkdir -p "${OUTPUT_ROOT}/nsys"
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --output="${OUTPUT_ROOT}/nsys/h2d_d2h" \
    "${PYTHON_BIN}" "${BENCHMARK}" \
      --backend cuda \
      --sizes 4KiB,64MiB \
      --iterations 20 \
      --warmup 5 \
      --annotate \
      --output-dir "${OUTPUT_ROOT}/nsys/run"
fi

if command -v perf >/dev/null 2>&1; then
  mkdir -p "${OUTPUT_ROOT}/perf"
  perf stat \
    --event task-clock,context-switches,cpu-migrations,page-faults,minor-faults,major-faults \
    --output "${OUTPUT_ROOT}/perf/stat.txt" \
    "${PYTHON_BIN}" "${BENCHMARK}" \
      --backend cuda \
      --sizes 4KiB,64MiB \
      --iterations 100 \
      --warmup 20 \
      --output-dir "${OUTPUT_ROOT}/perf/run"
fi

echo "Validation artifacts: ${OUTPUT_ROOT}"
