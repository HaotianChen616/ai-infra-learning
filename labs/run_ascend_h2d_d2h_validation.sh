#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${1:-artifacts/ascend_h2d_d2h_validation}"
BENCHMARK="labs/h2d_d2h_benchmark.py"

mkdir -p "${OUTPUT_ROOT}/system"

uname -a >"${OUTPUT_ROOT}/system/uname.txt"
lscpu >"${OUTPUT_ROOT}/system/lscpu.txt"
npu-smi info >"${OUTPUT_ROOT}/system/npu-smi-info.txt"
npu-smi info -m >"${OUTPUT_ROOT}/system/npu-smi-mapping.txt"
npu-smi info -t topo >"${OUTPUT_ROOT}/system/npu-smi-topo.txt" 2>&1 || true

if command -v numactl >/dev/null 2>&1; then
  numactl --hardware >"${OUTPUT_ROOT}/system/numa.txt"
fi

"${PYTHON_BIN}" -c \
  'import torch, torch_npu; print("torch:", torch.__version__); print("torch_npu:", torch_npu.__version__); print("device:", torch_npu.npu.get_device_name(0)); print(torch_npu.npu.get_device_properties(0)); get_cann_version = getattr(torch_npu.utils, "get_cann_version", None); print("CANN:", get_cann_version() if get_cann_version else "unavailable")' \
  >"${OUTPUT_ROOT}/system/torch-npu.txt"

"${PYTHON_BIN}" "${BENCHMARK}" \
  --backend npu \
  --sizes 4KiB,1MiB,16MiB,64MiB,256MiB \
  --iterations 100 \
  --warmup 20 \
  --allocation-iterations 5 \
  --output-dir "${OUTPUT_ROOT}/baseline"

for direction in h2d d2h; do
  "${PYTHON_BIN}" "${BENCHMARK}" \
    --backend npu \
    --sizes 64MiB \
    --host-memory pinned \
    --modes nonblocking \
    --sync-policies each \
    --iterations 20 \
    --warmup 5 \
    --trace \
    --trace-size 64MiB \
    --trace-direction "${direction}" \
    --trace-iterations 20 \
    --output-dir "${OUTPUT_ROOT}/torch-npu-profiler-${direction}"
done

if command -v msprof >/dev/null 2>&1; then
  mkdir -p "${OUTPUT_ROOT}/msprof"
  msprof \
    --output="${OUTPUT_ROOT}/msprof" \
    --msproftx=on \
    --sys-profiling=on \
    --sys-pid-profiling=on \
    --sys-interconnection-profiling=on \
    "${PYTHON_BIN}" "${BENCHMARK}" \
      --backend npu \
      --sizes 4KiB,64MiB \
      --iterations 20 \
      --warmup 5 \
      --annotate \
      --output-dir "${OUTPUT_ROOT}/msprof-run"
fi

if command -v perf >/dev/null 2>&1; then
  mkdir -p "${OUTPUT_ROOT}/perf"
  perf stat \
    --event task-clock,context-switches,cpu-migrations,page-faults,minor-faults,major-faults \
    --output "${OUTPUT_ROOT}/perf/stat.txt" \
    "${PYTHON_BIN}" "${BENCHMARK}" \
      --backend npu \
      --sizes 4KiB,64MiB \
      --iterations 100 \
      --warmup 20 \
      --output-dir "${OUTPUT_ROOT}/perf/run"
fi

echo "Ascend validation artifacts: ${OUTPUT_ROOT}"
