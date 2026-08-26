#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
OUTPUT="${1:-${SCRIPT_DIR}/build/cufile_verify}"
NVCC="${NVCC:-${CUDA_ROOT}/bin/nvcc}"

if [[ ! -x "${NVCC}" ]]; then
  echo "nvcc not found at ${NVCC}; set CUDA_HOME or NVCC" >&2
  exit 127
fi
if [[ ! -f "${CUDA_ROOT}/include/cufile.h" ]]; then
  echo "cufile.h not found under ${CUDA_ROOT}; install libcufile-dev/nvidia-gds" >&2
  exit 127
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${NVCC}" \
  -O2 -std=c++17 \
  -I"${CUDA_ROOT}/include" \
  -L"${CUDA_ROOT}/lib64" \
  -L"${CUDA_ROOT}/targets/x86_64-linux/lib" \
  -Xlinker -rpath -Xlinker "${CUDA_ROOT}/lib64" \
  -Xlinker -rpath -Xlinker "${CUDA_ROOT}/targets/x86_64-linux/lib" \
  "${SCRIPT_DIR}/cufile_verify.cu" \
  -lcufile -lcuda -lcudart \
  -o "${OUTPUT}"

echo "Built ${OUTPUT}"
