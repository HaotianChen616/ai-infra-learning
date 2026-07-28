#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON:-python3}"
MODEL="${MODEL:-Eco-Tech/Qwen3.6-27B-w8a8}"
MODEL_DIR="${MODEL_DIR:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-w8a8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
INPUT_TOKENS="${INPUT_TOKENS:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-8}"
DECODE_STEPS="${DECODE_STEPS:-$((OUTPUT_TOKENS > 1 ? OUTPUT_TOKENS - 1 : 0))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((INPUT_TOKENS * CONCURRENCY))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${CONCURRENCY}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/vllm_ascend_e2e}"
PROFILE_LABEL="${PROFILE_LABEL:-bs${CONCURRENCY}_${INPUT_TOKENS}_${OUTPUT_TOKENS}}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-baseline_${PROFILE_LABEL}}"
ANALYSIS_LABEL="${ANALYSIS_LABEL:-${PROFILE_LABEL}}"
TORCH_PROFILER_WITH_STACK="${TORCH_PROFILER_WITH_STACK:-false}"
PROFILE_DIR="${PROFILE_DIR:-${OUTPUT_ROOT}/torch_profile/${PROFILE_LABEL}}"
EXTRA_SERVER_CONFIG="${EXTRA_SERVER_CONFIG:-}"

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command is not available: $1" >&2
    exit 1
  fi
}

write_run_manifest() {
  local path="$1"
  shift
  {
    printf 'timestamp=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'model=%q\n' "${MODEL}"
    printf 'served_model_name=%q\n' "${SERVED_MODEL_NAME}"
    printf 'tp_size=%q\n' "${TP_SIZE}"
    printf 'dp_size=%q\n' "${DP_SIZE}"
    printf 'input_tokens=%q\n' "${INPUT_TOKENS}"
    printf 'output_tokens=%q\n' "${OUTPUT_TOKENS}"
    printf 'concurrency=%q\n' "${CONCURRENCY}"
    printf 'task_queue_enable=%q\n' "${TASK_QUEUE_ENABLE}"
    printf 'ascend_launch_blocking=%q\n' "${ASCEND_LAUNCH_BLOCKING}"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } >"${path}"
}

print_usage() {
  cat <<'EOF'
End-to-end profiling for Qwen3.6-27B Ascend W8A8 in an existing vLLM-Ascend
container. Run all actions from the repository root inside that container.

1. Capture environment and verify a local checkpoint:
  MODEL_DIR=/models/Qwen3.6-27B-w8a8 \
    bash labs/run_vllm_ascend_e2e_profile.sh system
  MODEL_DIR=/models/Qwen3.6-27B-w8a8 \
    bash labs/run_vllm_ascend_e2e_profile.sh verify-model

2. Terminal A, start a dynamically controlled Ascend PyTorch Profiler server:
  MODEL=/models/Qwen3.6-27B-w8a8 TP_SIZE=2 \
    PROFILE_LABEL=full_bs8_1024_128 \
    bash labs/run_vllm_ascend_e2e_profile.sh server-profile [extra serve args]

3. Terminal B:
  bash labs/run_vllm_ascend_e2e_profile.sh warmup
  PROFILE_LABEL=full_bs8_1024_128 \
    bash labs/run_vllm_ascend_e2e_profile.sh profile

4. Analyze and summarize the generated *_ascend_pt data:
  ANALYSIS_LABEL=full_bs8_1024_128 \
    PROFILE_LABEL=full_bs8_1024_128 \
    bash labs/run_vllm_ascend_e2e_profile.sh analyze

5. For profiler-off metrics, restart Terminal A with server-baseline and run:
  bash labs/run_vllm_ascend_e2e_profile.sh server-baseline
  bash labs/run_vllm_ascend_e2e_profile.sh benchmark

Key defaults:
  MODEL=Eco-Tech/Qwen3.6-27B-w8a8
  SERVED_MODEL_NAME=qwen3.6-27b-w8a8
  TP_SIZE=2 DP_SIZE=1
  INPUT_TOKENS=1024 OUTPUT_TOKENS=128 CONCURRENCY=8
  MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=8192 MAX_NUM_SEQS=8
  TORCH_PROFILER_WITH_STACK=false
  OUTPUT_ROOT=artifacts/vllm_ascend_e2e

For the 1024 -> 1 control:
  Restart server-profile with:
    OUTPUT_TOKENS=1 PROFILE_LABEL=prefill_control_bs8_1024_1 \
      bash labs/run_vllm_ascend_e2e_profile.sh server-profile
  Then run warmup, profile and analyze with the same two variables.

EXTRA_SERVER_CONFIG is optional JSON passed through --additional-config.
All additional arguments after server-profile/server-baseline are appended to
vllm serve, so container-specific working arguments can override defaults.
Use one PROFILE_LABEL/PROFILE_DIR per server lifetime. This prevents full and
control captures from being aggregated into one synchronization summary.
EOF
}

capture_system() {
  require_command "${PYTHON_BIN}"
  mkdir -p "${OUTPUT_ROOT}/system"
  uname -a >"${OUTPUT_ROOT}/system/uname.txt"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu >"${OUTPUT_ROOT}/system/lscpu.txt"
  fi

  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"${OUTPUT_ROOT}/system/npu-smi-info.txt"
    npu-smi info -m >"${OUTPUT_ROOT}/system/npu-smi-mapping.txt" 2>&1 || true
    npu-smi info -t topo >"${OUTPUT_ROOT}/system/npu-smi-topo.txt" 2>&1 || true
  fi
  if command -v numactl >/dev/null 2>&1; then
    numactl --hardware >"${OUTPUT_ROOT}/system/numa.txt"
  fi
  if command -v lspci >/dev/null 2>&1; then
    lspci -tv >"${OUTPUT_ROOT}/system/lspci-tree.txt"
  fi

  "${PYTHON_BIN}" -c \
    'import torch, torch_npu; print("torch:", torch.__version__); print("torch_npu:", torch_npu.__version__); utils = getattr(torch_npu, "utils", None); get_cann_version = getattr(utils, "get_cann_version", None); print("CANN:", get_cann_version() if get_cann_version else "unavailable"); print("device_count:", torch_npu.npu.device_count()); print("device_0:", torch_npu.npu.get_device_name(0))' \
    >"${OUTPUT_ROOT}/system/torch-npu.txt"
  if command -v vllm >/dev/null 2>&1; then
    vllm --version >"${OUTPUT_ROOT}/system/vllm-version.txt"
  fi
  "${PYTHON_BIN}" -m pip show vllm vllm-ascend \
    >"${OUTPUT_ROOT}/system/vllm-packages.txt" 2>&1 || true
  env | grep -E \
    '^(ASCEND_LAUNCH_BLOCKING|ASCEND_RT_VISIBLE_DEVICES|NPU_VISIBLE_DEVICES|TASK_QUEUE_ENABLE|CPU_AFFINITY_CONF|PYTORCH_NPU_ALLOC_CONF|HCCL_BUFFSIZE|OMP_PROC_BIND|OMP_NUM_THREADS|VLLM_ASCEND_|VLLM_USE_MODELSCOPE)=' \
    | sort >"${OUTPUT_ROOT}/system/relevant-env.txt" || true
  echo "System artifacts: ${OUTPUT_ROOT}/system"
}

verify_model() {
  require_command "${PYTHON_BIN}"
  if [[ -z "${MODEL_DIR}" ]]; then
    if [[ -d "${MODEL}" ]]; then
      MODEL_DIR="${MODEL}"
    else
      echo "MODEL_DIR must point to a downloaded Qwen3.6-27B W8A8 directory." >&2
      exit 1
    fi
  fi
  mkdir -p "${OUTPUT_ROOT}/system"
  "${PYTHON_BIN}" labs/verify_ascend_w8a8_model.py \
    "${MODEL_DIR}" \
    --output-json "${OUTPUT_ROOT}/system/ascend-w8a8-verification.json"
}

build_serve_args() {
  SERVE_ARGS=(
    vllm serve "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --data-parallel-size "${DP_SIZE}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed 1024
    --quantization ascend
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
    --no-enable-prefix-caching
  )
  if [[ -n "${EXTRA_SERVER_CONFIG}" ]]; then
    SERVE_ARGS+=(--additional-config "${EXTRA_SERVER_CONFIG}")
  fi
}

run_server_profile() {
  require_command vllm
  require_command "${PYTHON_BIN}"
  capture_system
  mkdir -p "${PROFILE_DIR}" "${OUTPUT_ROOT}/logs"
  local profiler_config
  profiler_config="$(
    "${PYTHON_BIN}" -c \
      'import json, sys; print(json.dumps({"profiler": "torch", "torch_profiler_dir": sys.argv[1], "torch_profiler_with_stack": sys.argv[2].lower() == "true", "torch_profiler_record_shapes": False, "torch_profiler_with_memory": False}))' \
      "${PROFILE_DIR}" "${TORCH_PROFILER_WITH_STACK}"
  )"
  build_serve_args
  local final_args=(
    "${SERVE_ARGS[@]}"
    --profiler-config "${profiler_config}"
    "$@"
  )
  write_run_manifest \
    "${OUTPUT_ROOT}/logs/server_${PROFILE_LABEL}_command.txt" \
    "${final_args[@]}"
  echo "Starting profiled vLLM-Ascend server. Warm up before running profile."
  exec "${final_args[@]}"
}

run_server_baseline() {
  require_command vllm
  capture_system
  mkdir -p "${OUTPUT_ROOT}/logs"
  build_serve_args
  local final_args=("${SERVE_ARGS[@]}" "$@")
  write_run_manifest \
    "${OUTPUT_ROOT}/logs/server_${BENCHMARK_LABEL}_command.txt" \
    "${final_args[@]}"
  echo "Starting profiler-off vLLM-Ascend server."
  exec "${final_args[@]}"
}

BENCH_COMMON=(
  vllm bench serve
  --backend vllm
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL_NAME}"
  --dataset-name random
  --random-input-len "${INPUT_TOKENS}"
  --random-output-len "${OUTPUT_TOKENS}"
  --random-range-ratio 0
  --request-rate inf
  --ignore-eos
  --temperature 0
  --ready-check-timeout-sec 1800
)

run_warmup() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${BENCH_COMMON[@]}" \
    --num-prompts "${WARMUP_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --save-result \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename "warmup_${PROFILE_LABEL}.json"
}

run_profile() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${BENCH_COMMON[@]}" \
    --num-prompts "${CONCURRENCY}" \
    --max-concurrency "${CONCURRENCY}" \
    --profile \
    --save-result \
    --save-detailed \
    --plot-timeline \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename "profile_${PROFILE_LABEL}.json"
  echo "Profile request completed. Wait for /stop_profile to flush before analysis."
}

run_benchmark() {
  require_command vllm
  mkdir -p "${OUTPUT_ROOT}/results"
  "${BENCH_COMMON[@]}" \
    --num-prompts "${CONCURRENCY}" \
    --max-concurrency "${CONCURRENCY}" \
    --save-result \
    --save-detailed \
    --plot-timeline \
    --result-dir "${OUTPUT_ROOT}/results" \
    --result-filename "${BENCHMARK_LABEL}.json"
}

run_analyze() {
  require_command "${PYTHON_BIN}"
  local analysis_dir="${OUTPUT_ROOT}/analysis/${ANALYSIS_LABEL}"
  mkdir -p "${analysis_dir}"
  mapfile -d '' profile_dirs < <(
    find "${PROFILE_DIR}" -type d -name '*_ascend_pt' -print0
  )
  if [[ ${#profile_dirs[@]} -eq 0 ]]; then
    echo "No *_ascend_pt directories found under ${PROFILE_DIR}." >&2
    exit 1
  fi
  for profile_path in "${profile_dirs[@]}"; do
    "${PYTHON_BIN}" -c \
      'import sys; from torch_npu.profiler.profiler import analyse; analyse(sys.argv[1])' \
      "${profile_path}"
  done
  mapfile -d '' trace_files < <(
    find "${PROFILE_DIR}" -type f -path '*/ASCEND_PROFILER_OUTPUT/trace_view.json' -print0
  )
  if [[ ${#trace_files[@]} -eq 0 ]]; then
    echo "No analyzed trace_view.json files found." >&2
    exit 1
  fi
  "${PYTHON_BIN}" labs/summarize_ascend_sync.py \
    "${trace_files[@]}" \
    --decode-steps "${DECODE_STEPS}" \
    --generated-tokens "$((CONCURRENCY * OUTPUT_TOKENS))" \
    --output-json "${analysis_dir}/ascend_sync_summary.json"
  printf '%s\n' "${trace_files[@]}" >"${analysis_dir}/trace-files.txt"
  echo "Analysis artifacts: ${analysis_dir}"
}

case "${ACTION}" in
  system)
    capture_system
    ;;
  verify-model)
    verify_model
    ;;
  server-profile)
    run_server_profile "$@"
    ;;
  server-baseline)
    run_server_baseline "$@"
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
  analyze)
    run_analyze
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
