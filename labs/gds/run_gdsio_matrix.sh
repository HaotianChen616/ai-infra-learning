#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_gdsio_matrix.sh --file ABSOLUTE_PATH [options]

Required:
  --file FILE              Existing regular test file. This script never creates it.

Options:
  --output-dir DIR         Default: artifacts/gds/gdsio-<UTC timestamp>
  --gpu INDEX              Default: 0
  --operation CODE         gdsio -I code; default 0 (sequential read)
  --io-sizes CSV           Default: 4K,1M,16M
  --workers CSV            Default: 1,4,16
  --transfers CSV          Default: 0,1,2 (GDS, CPU, CPU->GPU)
  --dataset-size SIZE      gdsio -s value; default 16G
  --duration SECONDS       gdsio -T value; default 10
  --repetitions N          Default: 1
  --random-seed VALUE      gdsio -k value for RANDREAD; default 20260824
  --cufile-config FILE     Sets CUFILE_ENV_PATH_JSON for every run
  --gdsio PATH             Override gdsio discovery
  --python PATH            Default: python3

Safety:
  The default operation is read-only. Write operation codes are rejected; perform
  write and persistence experiments manually after reviewing the experiment guide.
EOF
}

FILE_PATH=""
OUTPUT_DIR=""
GPU=0
OPERATION=0
IO_SIZES="4K,1M,16M"
WORKERS="1,4,16"
TRANSFERS="0,1,2"
DATASET_SIZE="16G"
DURATION=10
REPETITIONS=1
RANDOM_SEED=20260824
CUFILE_CONFIG=""
GDSIO_PATH="${GDSIO:-}"
PYTHON_BIN="${PYTHON:-python3}"

while (($#)); do
  case "$1" in
    --file) FILE_PATH="${2:?--file requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --gpu) GPU="${2:?--gpu requires a value}"; shift 2 ;;
    --operation) OPERATION="${2:?--operation requires a value}"; shift 2 ;;
    --io-sizes) IO_SIZES="${2:?--io-sizes requires a value}"; shift 2 ;;
    --workers) WORKERS="${2:?--workers requires a value}"; shift 2 ;;
    --transfers) TRANSFERS="${2:?--transfers requires a value}"; shift 2 ;;
    --dataset-size) DATASET_SIZE="${2:?--dataset-size requires a value}"; shift 2 ;;
    --duration) DURATION="${2:?--duration requires a value}"; shift 2 ;;
    --repetitions) REPETITIONS="${2:?--repetitions requires a value}"; shift 2 ;;
    --random-seed) RANDOM_SEED="${2:?--random-seed requires a value}"; shift 2 ;;
    --cufile-config) CUFILE_CONFIG="${2:?--cufile-config requires a value}"; shift 2 ;;
    --gdsio) GDSIO_PATH="${2:?--gdsio requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${FILE_PATH}" ]]; then
  echo "--file is required" >&2
  exit 2
fi
if [[ "${FILE_PATH}" != /* ]]; then
  echo "--file must be an absolute path on the dedicated test filesystem" >&2
  exit 2
fi
if [[ ! -f "${FILE_PATH}" ]]; then
  echo "Test file does not exist or is not regular: ${FILE_PATH}" >&2
  exit 2
fi
if [[ "${OPERATION}" != "0" && "${OPERATION}" != "2" ]]; then
  echo "Only read-only gdsio operations 0 (READ) and 2 (RANDREAD) are allowed" >&2
  exit 2
fi
if ! [[ "${DURATION}" =~ ^[1-9][0-9]*$ && "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--duration and --repetitions must be positive integers" >&2
  exit 2
fi
if [[ -n "${CUFILE_CONFIG}" && ! -f "${CUFILE_CONFIG}" ]]; then
  echo "cuFile config not found: ${CUFILE_CONFIG}" >&2
  exit 2
fi

find_gdsio() {
  if [[ -n "${GDSIO_PATH}" && -x "${GDSIO_PATH}" ]]; then
    printf '%s\n' "${GDSIO_PATH}"
    return 0
  fi
  if command -v gdsio >/dev/null 2>&1; then
    command -v gdsio
    return 0
  fi
  local candidate
  for candidate in /usr/local/cuda/gds/tools/gdsio /usr/local/cuda-*/gds/tools/gdsio; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

GDSIO_PATH="$(find_gdsio 2>/dev/null || true)"
if [[ -z "${GDSIO_PATH}" ]]; then
  echo "gdsio not found; install gds-tools or pass --gdsio" >&2
  exit 127
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="artifacts/gds/gdsio-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${OUTPUT_DIR}/logs"
MANIFEST="${OUTPUT_DIR}/runs.tsv"
printf 'run_id\trepetition\tgpu\toperation\ttransfer_code\ttransfer_name\tio_size\tworkers\tdataset_size\tfile\tlog_file\texit_code\tcommand\n' >"${MANIFEST}"

USE_GNU_TIME=0
if [[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU time'; then
  USE_GNU_TIME=1
fi

transfer_name() {
  case "$1" in
    0) echo gds ;;
    1) echo cpu ;;
    2) echo cpu_gpu ;;
    5) echo gds_async ;;
    6) echo gds_batch ;;
    7) echo gds_batch_stream ;;
    *) echo "xfer_$1" ;;
  esac
}

IFS=',' read -r -a IO_SIZE_VALUES <<<"${IO_SIZES}"
IFS=',' read -r -a WORKER_VALUES <<<"${WORKERS}"
IFS=',' read -r -a TRANSFER_VALUES <<<"${TRANSFERS}"

FAILURES=0
for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  for io_size in "${IO_SIZE_VALUES[@]}"; do
    for workers in "${WORKER_VALUES[@]}"; do
      for transfer in "${TRANSFER_VALUES[@]}"; do
        if ! [[ "${workers}" =~ ^[1-9][0-9]*$ && "${transfer}" =~ ^[0-9]+$ ]]; then
          echo "Invalid matrix value: workers=${workers}, transfer=${transfer}" >&2
          exit 2
        fi
        safe_io="${io_size//[^[:alnum:]]/_}"
        run_id="r${repetition}-gpu${GPU}-op${OPERATION}-x${transfer}-io${safe_io}-w${workers}"
        log_rel="logs/${run_id}.log"
        log_path="${OUTPUT_DIR}/${log_rel}"
        name="$(transfer_name "${transfer}")"
        command=(
          "${GDSIO_PATH}"
          -f "${FILE_PATH}"
          -d "${GPU}"
          -w "${workers}"
          -s "${DATASET_SIZE}"
          -i "${io_size}"
          -I "${OPERATION}"
          -x "${transfer}"
          -T "${DURATION}"
        )
        if [[ "${OPERATION}" == "2" ]]; then
          command+=(-k "${RANDOM_SEED}")
        fi
        if ((USE_GNU_TIME)); then
          full_command=(/usr/bin/time -v "${command[@]}")
        else
          full_command=("${command[@]}")
        fi
        printf -v command_text '%q ' "${full_command[@]}"
        echo "[${run_id}] ${name} io=${io_size} workers=${workers}"
        set +e
        if [[ -n "${CUFILE_CONFIG}" ]]; then
          CUFILE_ENV_PATH_JSON="${CUFILE_CONFIG}" "${full_command[@]}" >"${log_path}" 2>&1
        else
          "${full_command[@]}" >"${log_path}" 2>&1
        fi
        exit_code=$?
        set -e
        if ((exit_code != 0)); then
          FAILURES=$((FAILURES + 1))
          echo "  failed with exit ${exit_code}; see ${log_path}" >&2
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${run_id}" "${repetition}" "${GPU}" "${OPERATION}" "${transfer}" \
          "${name}" "${io_size}" "${workers}" "${DATASET_SIZE}" "${FILE_PATH}" \
          "${log_rel}" "${exit_code}" "${command_text//$'\t'/ }" >>"${MANIFEST}"
      done
    done
  done
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_gds_results.py" \
  --manifest "${MANIFEST}" --output-dir "${OUTPUT_DIR}" --fail-on-errors
SUMMARY_EXIT=$?
set -e

echo "Artifacts: ${OUTPUT_DIR}"
if ((FAILURES != 0 || SUMMARY_EXIT != 0)); then
  echo "${FAILURES} gdsio run(s) failed; parser status=${SUMMARY_EXIT}" >&2
  exit 1
fi
