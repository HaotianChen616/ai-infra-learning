#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: collect_gds_preflight.sh [--output-dir DIR] [--test-file FILE]

Collects read-only A100/GDS qualification evidence. The script never changes
GRUB, drivers, module parameters, mounts, filesystems, or the test file.
EOF
}

OUTPUT_DIR=""
TEST_FILE=""
while (($#)); do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --test-file)
      TEST_FILE="${2:?--test-file requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="artifacts/gds/preflight-$(date -u +%Y%m%dT%H%M%SZ)"
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

find_gdscheck() {
  if [[ -n "${GDSCHECK:-}" && -x "${GDSCHECK}" ]]; then
    printf '%s\n' "${GDSCHECK}"
    return 0
  fi
  local candidate
  for candidate in gdscheck.py gdscheck; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  for candidate in /usr/local/cuda/gds/tools/gdscheck.py /usr/local/cuda-*/gds/tools/gdscheck.py; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

capture date_utc date -u +%Y-%m-%dT%H:%M:%SZ
capture uname uname -a
capture_shell os_release 'cat /etc/os-release'
capture_shell kernel_cmdline 'cat /proc/cmdline'
capture lscpu lscpu
capture numactl_hardware numactl --hardware
capture lsblk lsblk -e 7 -o NAME,KNAME,MODEL,SERIAL,SIZE,ROTA,TRAN,FSTYPE,MOUNTPOINTS
capture nvme_list nvme list
capture findmnt findmnt -D
capture df df -hT
capture nvidia_smi nvidia-smi -q
capture nvidia_smi_topology nvidia-smi topo -m
capture nvidia_smi_pcie nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv
capture lspci_tree lspci -tv
capture lspci_devices lspci -Dnn
capture_shell iommu_dmesg 'dmesg 2>&1 | grep -i iommu'
capture_shell acs_devices 'lspci -vv 2>/dev/null | grep -B2 -A4 -i "Access Control Services\|ACSCtl"'
capture_shell nvme_multipath 'cat /sys/module/nvme_core/parameters/multipath'
capture_shell nvidia_params 'cat /proc/driver/nvidia/params'
capture_shell loaded_modules 'lsmod | grep -E "^(nvidia|nvme|ib_|mlx|rdma)"'
capture_shell nvidia_fs_module 'modinfo nvidia_fs'
capture_shell nvidia_peermem_module 'modinfo nvidia_peermem'
capture_shell p2pdma_symbol 'grep -i p2pdma_pgmap_ops /proc/kallsyms'
capture_shell cuda_layout 'ls -ld /usr/local/cuda*; find /usr/local/cuda*/gds -maxdepth 2 -type f 2>/dev/null | sort'
capture_shell cufile_libraries 'ldconfig -p 2>/dev/null | grep -i cufile'
capture_shell cufile_config 'if test -f /etc/cufile.json; then cat /etc/cufile.json; else echo /etc/cufile.json_NOT_FOUND; fi'
capture_shell installed_gds_packages 'if command -v dpkg-query >/dev/null; then dpkg-query -W "*cufile*" "*gds*" "*nvidia-fs*" 2>&1; elif command -v rpm >/dev/null; then rpm -qa | grep -Ei "cufile|gds|nvidia-fs"; fi'

GDSCHECK_PATH="$(find_gdscheck 2>/dev/null || true)"
if [[ -n "${GDSCHECK_PATH}" ]]; then
  capture gdscheck_version "${GDSCHECK_PATH}" -v
  capture gdscheck_platform "${GDSCHECK_PATH}" -p
  capture gdscheck_filesystems "${GDSCHECK_PATH}" -V
  capture gdscheck_topology "${GDSCHECK_PATH}" -t
  if [[ -n "${TEST_FILE}" ]]; then
    if [[ -f "${TEST_FILE}" ]]; then
      capture gdscheck_file "${GDSCHECK_PATH}" -f "${TEST_FILE}"
    else
      printf 'gdscheck_file\t2\ttest file does not exist: %s\n' "${TEST_FILE}" >>"${STATUS_FILE}"
    fi
  fi
else
  printf 'gdscheck\t127\tgdscheck not found\n' >>"${STATUS_FILE}"
fi

{
  echo "GDS preflight artifact"
  echo "output_dir=${OUTPUT_DIR}"
  echo "gdscheck=${GDSCHECK_PATH:-NOT_FOUND}"
  echo "test_file=${TEST_FILE:-NOT_PROVIDED}"
  if grep -qi 'A100' "${OUTPUT_DIR}/nvidia_smi.txt" 2>/dev/null; then
    echo "a100_detected=yes"
  else
    echo "a100_detected=no"
  fi
  if grep -Eqi 'NVMe[^:]*:[[:space:]]*(p2pdma|nvfs)' "${OUTPUT_DIR}/gdscheck_platform.txt" 2>/dev/null; then
    echo "direct_mode_advertised=yes"
  else
    echo "direct_mode_advertised=no_or_unknown"
  fi
  if grep -Eqi 'IOMMU:[[:space:]]*(disabled|pass-through)' "${OUTPUT_DIR}/gdscheck_platform.txt" 2>/dev/null; then
    echo "iommu_gate=pass"
  else
    echo "iommu_gate=review"
  fi
  echo "Review status.tsv and every non-zero command before benchmarking."
} >"${OUTPUT_DIR}/summary.txt"

cat "${OUTPUT_DIR}/summary.txt"
