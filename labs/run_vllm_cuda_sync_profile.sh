#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

MODEL="${MODEL:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-int8-w8a8}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
INPUT_TOKENS="${INPUT_TOKENS:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((INPUT_TOKENS * CONCURRENCY))}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/vllm_cuda_sync}"
PROFILE_LABEL="${PROFILE_LABEL:-bs${CONCURRENCY}_${INPUT_TOKENS}_${OUTPUT_TOKENS}}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-baseline_${PROFILE_LABEL}}"
ANALYSIS_LABEL="${ANALYSIS_LABEL:-${PROFILE_LABEL}}"
PYTHON_BIN="${PYTHON:-python3}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command is not available: $1" >&2
    exit 1
  fi
}

print_usage() {
  cat <<'EOF'
Profile host-side CUDA synchronization in a running vLLM serving workload.

Verify a downloaded checkpoint before launch:
  MODEL=/models/Qwen3-32B-INT8-W8A8 \
    bash labs/run_vllm_cuda_sync_profile.sh verify

Terminal A:
  MODEL=/models/Qwen3-32B-INT8-W8A8 TP_SIZE=1 \
    bash labs/run_vllm_cuda_sync_profile.sh server [extra vllm serve args]

Terminal B:
  bash labs/run_vllm_cuda_sync_profile.sh warmup
  bash labs/run_vllm_cuda_sync_profile.sh profile

Against a separately launched server without any profiler:
  bash labs/run_vllm_cuda_sync_profile.sh benchmark

After profile finishes, stop Terminal A with Ctrl-C. Then summarize a report:
  bash labs/run_vllm_cuda_sync_profile.sh stats path/to/report.nsys-rep

Environment defaults:
  SERVED_MODEL_NAME=qwen3-32b-int8-w8a8
  QUANTIZATION=compressed-tensors DTYPE=bfloat16
  HOST=127.0.0.1 PORT=8000 TP_SIZE=1
  INPUT_TOKENS=1024 OUTPUT_TOKENS=128 CONCURRENCY=8
  MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=8192 WARMUP_PROMPTS=16
  OUTPUT_ROOT=artifacts/vllm_cuda_sync
  PROFILE_LABEL=bs8_1024_128 ANALYSIS_LABEL=bs8_1024_128
  BENCHMARK_LABEL=baseline_bs8_1024_128

For the 1024 -> 1 control group, keep the server running and use:
  INPUT_TOKENS=1024 OUTPUT_TOKENS=1 PROFILE_LABEL=bs8_1024_1 \
    bash labs/run_vllm_cuda_sync_profile.sh profile

This workflow intentionally requires a serialized compressed-tensors INT8 W8A8
checkpoint. For a remote model ID, set CONFIG_JSON to a downloaded config.json
and run "verify" before starting the server.
EOF
}

resolve_config_json() {
  if [[ -n "${CONFIG_JSON}" ]]; then
    printf '%s\n' "${CONFIG_JSON}"
  elif [[ -n "${MODEL}" && -f "${MODEL}/config.json" ]]; then
    printf '%s\n' "${MODEL}/config.json"
  fi
}

run_verify() {
  require_command "${PYTHON_BIN}"
  if [[ -z "${MODEL}" && -z "${CONFIG_JSON}" ]]; then
    echo "Set MODEL to a local checkpoint or CONFIG_JSON to config.json." >&2
    exit 1
  fi
  local config_path
  config_path="$(resolve_config_json)"
  if [[ -z "${config_path}" || ! -f "${config_path}" ]]; then
    echo "Cannot find config.json. Set CONFIG_JSON explicitly." >&2
    exit 1
  fi
  "${PYTHON_BIN}" labs/verify_int8_w8a8_config.py "${config_path}" "$@"
}

run_server() {
  require_command nsys
  require_command nvidia-smi
  require_command "${PYTHON_BIN}"
  require_command vllm
  if [[ -z "${MODEL}" ]]; then
    echo "MODEL must point to the Qwen3-32B INT8 W8A8 checkpoint." >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_ROOT}/nsys" "${OUTPUT_ROOT}/logs"
  local config_path
  config_path="$(resolve_config_json)"
  if [[ -n "${config_path}" && -f "${config_path}" ]]; then
    run_verify \
      --output-json "${OUTPUT_ROOT}/logs/int8-w8a8-verification.json"
  else
    echo "WARNING: config.json is not local; INT8 W8A8 metadata was not verified." >&2
    echo "Set CONFIG_JSON and run the verify action before trusting the profile." >&2
  fi
  uname -a >"${OUTPUT_ROOT}/logs/uname.txt"
  nvidia-smi -q >"${OUTPUT_ROOT}/logs/nvidia-smi-q.txt"
  nvidia-smi topo -m >"${OUTPUT_ROOT}/logs/nvidia-smi-topo.txt"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id \
    --format=csv \
    >"${OUTPUT_ROOT}/logs/nvidia-smi-gpus.csv"
  nsys --version >"${OUTPUT_ROOT}/logs/nsys-version.txt"
  nsys status -e >"${OUTPUT_ROOT}/logs/nsys-environment.txt" || true
  vllm --version >"${OUTPUT_ROOT}/logs/vllm-version.txt"

  echo "Starting vLLM under Nsight Systems."
  echo "Run the warmup and profile actions from a second terminal."
  echo "Stop this command with Ctrl-C only after the profile client returns."
  VLLM_WORKER_MULTIPROC_METHOD=spawn exec nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --cudabacktrace=sync:100000 \
    --cuda-event-trace=true \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --wait=all \
    --output="${OUTPUT_ROOT}/nsys/qwen3_int8_sync" \
    vllm serve "${MODEL}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --quantization "${QUANTIZATION}" \
      --dtype "${DTYPE}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${CONCURRENCY}" \
      --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
      --profiler-config.profiler cuda \
      "$@"
}

bench_common=(
  vllm bench serve
  --backend vllm
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL_NAME}"
  --dataset-name random
  --random-input-len "${INPUT_TOKENS}"
  --random-output-len "${OUTPUT_TOKENS}"
  --random-range-ratio 0
  --max-concurrency "${CONCURRENCY}"
  --request-rate inf
  --ignore-eos
  --temperature 0
  --ready-check-timeout-sec 600
)

run_warmup() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${bench_common[@]}" \
    --num-prompts "${WARMUP_PROMPTS}" \
    --save-result \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename warmup.json
}

run_profile() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${bench_common[@]}" \
    --num-prompts "${CONCURRENCY}" \
    --profile \
    --save-result \
    --save-detailed \
    --plot-timeline \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename "profile_${PROFILE_LABEL}.json"
  echo "Profile request completed. Stop the server-side nsys command with Ctrl-C."
}

run_benchmark() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${bench_common[@]}" \
    --num-prompts "${CONCURRENCY}" \
    --save-result \
    --save-detailed \
    --plot-timeline \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename "${BENCHMARK_LABEL}.json"
}

run_stats() {
  require_command nsys
  if [[ $# -ne 1 ]]; then
    echo "stats requires exactly one .nsys-rep path." >&2
    exit 1
  fi
  local report="$1"
  if [[ ! -f "${report}" ]]; then
    echo "Nsight report does not exist: ${report}" >&2
    exit 1
  fi
  local analysis_dir="${OUTPUT_ROOT}/analysis/${ANALYSIS_LABEL}"
  mkdir -p "${analysis_dir}"
  nsys stats \
    --force-overwrite=true \
    --report cuda_api_sum \
    --report cuda_api_trace \
    --format csv,csv \
    --output "${analysis_dir}/summary","${analysis_dir}/trace" \
    "${report}"

  local trace_csv="${analysis_dir}/trace_cuda_api_trace.csv"
  "${PYTHON_BIN}" labs/summarize_cuda_sync.py \
    "${trace_csv}" \
    --decode-steps "${OUTPUT_TOKENS}" \
    --generated-tokens "$((CONCURRENCY * OUTPUT_TOKENS))" \
    --output-json "${analysis_dir}/cuda_sync_summary.json"
  echo "Analysis artifacts: ${analysis_dir}"
}

case "${ACTION}" in
  verify)
    run_verify "$@"
    ;;
  server)
    run_server "$@"
    ;;
  warmup)
    run_warmup
    ;;
  profile)
    run_profile
    ;;
  benchmark)
    run_benchmark
    ;;
  stats)
    run_stats "$@"
    ;;
  help | -h | --help)
    print_usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    print_usage >&2
    exit 1
    ;;
esac
