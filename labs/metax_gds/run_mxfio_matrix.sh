#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_mxfio_matrix.sh --file ABSOLUTE_PATH --fio ABSOLUTE_PATH [options]

Required:
  --file FILE          Existing regular file on a dedicated test filesystem
  --fio PATH           mxFIO executable containing the libcufile engine

Options:
  --output-dir DIR     Default: artifacts/metax-gds/mxfio-<UTC timestamp>
  --gpu INDEX          Default: 0
  --modes CSV          Default: cpu,staging,mas
                       cpu=psync, staging=libcufile+posix,
                       mas=libcufile+cufile
  --io-sizes CSV       Default: 4K,1M,16M
  --numjobs CSV        Default: 1,4,16
  --size SIZE          Read region per job; default: 4G
  --runtime SECONDS    Measurement time; default: 10
  --ramp-time SECONDS  Warm-up time; default: 4
  --repetitions N      Default: 3
  --python PATH        Default: python3

Safety:
  Only sequential reads are issued. The file must already exist, its requested
  region cannot exceed the file size, and fio is invoked with --readonly.
EOF
}

FILE_PATH=""
FIO_PATH=""
OUTPUT_DIR=""
GPU=0
MODES="cpu,staging,mas"
IO_SIZES="4K,1M,16M"
NUMJOBS="1,4,16"
REGION_SIZE="4G"
RUNTIME=10
RAMP_TIME=4
REPETITIONS=3
PYTHON_BIN="${PYTHON:-python3}"

while (($#)); do
  case "$1" in
    --file) FILE_PATH="${2:?--file requires a value}"; shift 2 ;;
    --fio) FIO_PATH="${2:?--fio requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --gpu) GPU="${2:?--gpu requires a value}"; shift 2 ;;
    --modes) MODES="${2:?--modes requires a value}"; shift 2 ;;
    --io-sizes) IO_SIZES="${2:?--io-sizes requires a value}"; shift 2 ;;
    --numjobs) NUMJOBS="${2:?--numjobs requires a value}"; shift 2 ;;
    --size) REGION_SIZE="${2:?--size requires a value}"; shift 2 ;;
    --runtime) RUNTIME="${2:?--runtime requires a value}"; shift 2 ;;
    --ramp-time) RAMP_TIME="${2:?--ramp-time requires a value}"; shift 2 ;;
    --repetitions) REPETITIONS="${2:?--repetitions requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${FILE_PATH}" || -z "${FIO_PATH}" ]]; then
  echo "--file and --fio are required" >&2
  exit 2
fi
if [[ "${FILE_PATH}" != /* || "${FIO_PATH}" != /* ]]; then
  echo "--file and --fio must both be absolute paths" >&2
  exit 2
fi
if [[ ! -f "${FILE_PATH}" || ! -r "${FILE_PATH}" ]]; then
  echo "Test file does not exist, is not regular, or is not readable: ${FILE_PATH}" >&2
  exit 2
fi
if [[ ! -x "${FIO_PATH}" ]]; then
  echo "mxFIO is not executable: ${FIO_PATH}" >&2
  exit 2
fi
if ! [[ "${GPU}" =~ ^[0-9]+$ && "${RUNTIME}" =~ ^[1-9][0-9]*$ && "${RAMP_TIME}" =~ ^[0-9]+$ && "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU, runtime, ramp-time, and repetitions must be non-negative integers; runtime/repetitions must be positive" >&2
  exit 2
fi
if ! command -v numfmt >/dev/null 2>&1; then
  echo "GNU numfmt is required to validate --size" >&2
  exit 127
fi

set +e
ENGINE_PROBE="$("${FIO_PATH}" --enghelp=libcufile 2>&1)"
ENGINE_STATUS=$?
set -e
if ((ENGINE_STATUS != 0)) || grep -Eqi 'no such|not found|unknown|failed|not loadable' <<<"${ENGINE_PROBE}"; then
  echo "The selected fio does not appear to provide the libcufile engine:" >&2
  echo "${ENGINE_PROBE}" >&2
  exit 2
fi

REGION_BYTES="$(numfmt --from=iec "${REGION_SIZE}" 2>/dev/null || true)"
FILE_BYTES="$(stat -c %s -- "${FILE_PATH}")"
if ! [[ "${REGION_BYTES}" =~ ^[0-9]+$ ]] || ((REGION_BYTES <= 0)); then
  echo "Invalid --size value: ${REGION_SIZE}" >&2
  exit 2
fi
if ((REGION_BYTES > FILE_BYTES)); then
  echo "Requested region ${REGION_SIZE} (${REGION_BYTES} bytes) exceeds file size ${FILE_BYTES}" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="artifacts/metax-gds/mxfio-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${OUTPUT_DIR}/logs"
MANIFEST="${OUTPUT_DIR}/runs.tsv"
printf 'run_id\trepetition\tmode\tgpu\tio_size\tnumjobs\tregion_size\tfile\tjson_file\tstderr_file\ttime_file\texit_code\tcommand\n' >"${MANIFEST}"

IFS=',' read -r -a MODE_VALUES <<<"${MODES}"
IFS=',' read -r -a IO_SIZE_VALUES <<<"${IO_SIZES}"
IFS=',' read -r -a NUMJOB_VALUES <<<"${NUMJOBS}"

FAILURES=0
for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  for mode in "${MODE_VALUES[@]}"; do
    case "${mode}" in
      cpu) ENGINE=psync; CUDA_IO="" ;;
      staging) ENGINE=libcufile; CUDA_IO=posix ;;
      mas) ENGINE=libcufile; CUDA_IO=cufile ;;
      *) echo "Unknown mode: ${mode}; expected cpu, staging, or mas" >&2; exit 2 ;;
    esac
    for io_size in "${IO_SIZE_VALUES[@]}"; do
      for jobs in "${NUMJOB_VALUES[@]}"; do
        if ! [[ "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
          echo "Invalid numjobs value: ${jobs}" >&2
          exit 2
        fi
        safe_io="${io_size//[^[:alnum:]]/_}"
        run_id="r${repetition}-${mode}-gpu${GPU}-io${safe_io}-j${jobs}"
        json_rel="logs/${run_id}.json"
        stderr_rel="logs/${run_id}.stderr.log"
        time_rel="logs/${run_id}.time.log"
        json_path="${OUTPUT_DIR}/${json_rel}"
        stderr_path="${OUTPUT_DIR}/${stderr_rel}"
        time_path="${OUTPUT_DIR}/${time_rel}"
        command=(
          "${FIO_PATH}"
          --readonly
          --name="${run_id}"
          --filename="${FILE_PATH}"
          --ioengine="${ENGINE}"
          --rw=read
          --direct=1
          --invalidate=1
          --thread=1
          --bs="${io_size}"
          --size="${REGION_SIZE}"
          --time_based=1
          --runtime="${RUNTIME}"
          --ramp_time="${RAMP_TIME}"
          --numjobs="${jobs}"
          --group_reporting=1
          --output-format=json
          --output="${json_path}"
        )
        if [[ -n "${CUDA_IO}" ]]; then
          command+=(--gpu_dev_ids="${GPU}" --cuda_io="${CUDA_IO}")
        fi
        printf -v command_text '%q ' "${command[@]}"
        echo "[${run_id}] engine=${ENGINE} cuda_io=${CUDA_IO:-none}"
        set +e
        if [[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU time'; then
          /usr/bin/time -v -o "${time_path}" "${command[@]}" 2>"${stderr_path}"
        else
          : >"${time_path}"
          "${command[@]}" 2>"${stderr_path}"
        fi
        exit_code=$?
        set -e
        if ((exit_code != 0)); then
          FAILURES=$((FAILURES + 1))
          echo "  failed with exit ${exit_code}; see ${stderr_path}" >&2
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${run_id}" "${repetition}" "${mode}" "${GPU}" "${io_size}" \
          "${jobs}" "${REGION_SIZE}" "${FILE_PATH}" "${json_rel}" \
          "${stderr_rel}" "${time_rel}" "${exit_code}" "${command_text//$'\t'/ }" >>"${MANIFEST}"
      done
    done
  done
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_mxfio_results.py" \
  --manifest "${MANIFEST}" --output-dir "${OUTPUT_DIR}" --fail-on-errors
SUMMARY_EXIT=$?
set -e

echo "Artifacts: ${OUTPUT_DIR}"
if ((FAILURES != 0 || SUMMARY_EXIT != 0)); then
  echo "${FAILURES} mxFIO run(s) failed; parser status=${SUMMARY_EXIT}" >&2
  exit 1
fi
