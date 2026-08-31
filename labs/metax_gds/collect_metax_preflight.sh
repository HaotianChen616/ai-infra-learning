#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: collect_metax_preflight.sh [options]

Options:
  --output-dir DIR   Default: artifacts/metax-gds/preflight-<UTC timestamp>
  --test-file FILE   Optional existing test file to inspect
  --fio PATH         mxFIO executable; defaults to $MXFIO or fio in PATH
  -h, --help         Show this help

This script only collects system state. It never changes drivers, module
parameters, mounts, filesystems, IOMMU/ACS settings, or the test file.
EOF
}

OUTPUT_DIR=""
TEST_FILE=""
FIO_PATH="${MXFIO:-}"

while (($#)); do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --test-file) TEST_FILE="${2:?--test-file requires a value}"; shift 2 ;;
    --fio) FIO_PATH="${2:?--fio requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="artifacts/metax-gds/preflight-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${OUTPUT_DIR}"
STATUS_FILE="${OUTPUT_DIR}/status.tsv"
printf 'name\texit_code\tcommand\n' >"${STATUS_FILE}"

capture() {
  local name="$1"
  shift
  local destination="${OUTPUT_DIR}/${name}.txt"
  local command_text
  printf -v command_text '%q ' "$@"
  "$@" >"${destination}" 2>&1
  local exit_code=$?
  printf '%s\t%s\t%s\n' "${name}" "${exit_code}" "${command_text//$'\t'/ }" >>"${STATUS_FILE}"
  return 0
}

capture_shell() {
  local name="$1"
  local expression="$2"
  capture "${name}" bash -lc "${expression}"
}

find_command() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" ]] && command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${FIO_PATH}" ]]; then
  FIO_PATH="$(find_command fio 2>/dev/null || true)"
fi
METAX_SMI_PATH="${METAX_SMI:-}"
if [[ -z "${METAX_SMI_PATH}" ]]; then
  METAX_SMI_PATH="$(find_command mx-smi mxc-smi 2>/dev/null || true)"
fi

capture date_utc date -u +%Y-%m-%dT%H:%M:%SZ
capture uname uname -a
capture_shell os_release 'cat /etc/os-release'
capture_shell kernel_cmdline 'cat /proc/cmdline'
capture_shell glibc_version 'getconf GNU_LIBC_VERSION; ldd --version | head -n 1'
capture lscpu lscpu
capture numactl_hardware numactl --hardware
capture lsblk lsblk -e 7 -o NAME,KNAME,MODEL,SERIAL,SIZE,ROTA,TRAN,FSTYPE,MOUNTPOINTS
capture nvme_list nvme list
capture findmnt findmnt -D
capture df df -hT
capture lspci_tree lspci -tv
capture lspci_devices lspci -Dnn
capture_shell pcie_verbose 'lspci -Dvv 2>/dev/null'
capture_shell iommu_dmesg 'dmesg 2>&1 | grep -i iommu'
capture_shell acs_devices 'lspci -vv 2>/dev/null | grep -B2 -A4 -i "Access Control Services\|ACSCtl"'
capture_shell loaded_modules 'lsmod | grep -Ei "mx|maca|cufile|nvme|ib_|rdma|mlx"'
capture_shell maca_layout 'find /opt/maca /opt/mxdriver -maxdepth 4 -type f 2>/dev/null | sort'
capture_shell maca_libraries 'ldconfig -p 2>/dev/null | grep -Ei "cufile|maca|mxdriver"'
capture_shell installed_packages 'if command -v dpkg-query >/dev/null; then dpkg-query -W 2>/dev/null | grep -Ei "maca|metax|mxdriver|cufile|fio"; elif command -v rpm >/dev/null; then rpm -qa | grep -Ei "maca|metax|mxdriver|cufile|fio"; fi'

if [[ -n "${METAX_SMI_PATH}" ]]; then
  capture metax_smi_path readlink -f "${METAX_SMI_PATH}"
  capture metax_smi "${METAX_SMI_PATH}"
  capture metax_smi_help "${METAX_SMI_PATH}" --help
else
  printf 'metax_smi\t127\tmx-smi/mxc-smi not found\n' >>"${STATUS_FILE}"
fi

if [[ -n "${FIO_PATH}" && -x "${FIO_PATH}" ]]; then
  capture mxfio_path readlink -f "${FIO_PATH}"
  capture mxfio_version "${FIO_PATH}" --version
  capture mxfio_engines "${FIO_PATH}" --enghelp
  capture mxfio_libcufile_options "${FIO_PATH}" --enghelp=libcufile
else
  printf 'mxfio\t127\tmxFIO executable not found; pass --fio\n' >>"${STATUS_FILE}"
fi

if [[ -n "${TEST_FILE}" ]]; then
  if [[ -f "${TEST_FILE}" ]]; then
    capture test_file_stat stat -- "${TEST_FILE}"
    capture test_file_mount findmnt -T "${TEST_FILE}"
    capture test_file_df df -hT "${TEST_FILE}"
    capture test_file_fragments filefrag -v "${TEST_FILE}"
  else
    printf 'test_file\t2\ttest file does not exist or is not regular: %s\n' "${TEST_FILE}" >>"${STATUS_FILE}"
  fi
fi

{
  echo "MetaX C500 MAS preflight artifact"
  echo "output_dir=${OUTPUT_DIR}"
  echo "metax_smi=${METAX_SMI_PATH:-NOT_FOUND}"
  echo "mxfio=${FIO_PATH:-NOT_FOUND}"
  echo "test_file=${TEST_FILE:-NOT_PROVIDED}"
  if grep -Eqi 'C500|MXMACA-C500' "${OUTPUT_DIR}/metax_smi.txt" "${OUTPUT_DIR}/lspci_devices.txt" 2>/dev/null; then
    echo "c500_detected=yes"
  else
    echo "c500_detected=no_or_unknown"
  fi
  if grep -qi 'libcufile' "${OUTPUT_DIR}/mxfio_engines.txt" "${OUTPUT_DIR}/mxfio_libcufile_options.txt" 2>/dev/null; then
    echo "libcufile_engine=yes"
  else
    echo "libcufile_engine=no_or_unknown"
  fi
  if grep -qi 'libcufile' "${OUTPUT_DIR}/maca_libraries.txt" 2>/dev/null; then
    echo "libcufile_library=yes"
  else
    echo "libcufile_library=no_or_unknown"
  fi
  echo "iommu_and_acs=review_only_do_not_change_without_vendor_guidance"
  echo "Review status.tsv and every non-zero command before benchmarking."
} >"${OUTPUT_DIR}/summary.txt"

cat "${OUTPUT_DIR}/summary.txt"
