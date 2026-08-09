#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/vllm_cpu_experiments}"
PROFILE_SECONDS="${PROFILE_SECONDS:-30}"
PERF_FREQ="${PERF_FREQ:-199}"
PYSPY_RATE="${PYSPY_RATE:-100}"
TRACE_REQUESTS="${TRACE_REQUESTS:-}"
TRACE_DECODE_STEPS="${TRACE_DECODE_STEPS:-}"
TRACE_THREAD_REGEX="${TRACE_THREAD_REGEX:-}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command is unavailable: $1" >&2
    exit 1
  fi
}

require_pid() {
  local pid="$1"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || [[ ! -d "/proc/${pid}" ]]; then
    echo "PID is not visible in this namespace: ${pid}" >&2
    exit 1
  fi
}

normalize_label() {
  local value="$1"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Label may contain only letters, digits, dot, underscore, and dash." >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}

print_usage() {
  cat <<'EOF'
Collect vLLM CUDA wait, CPU self-time, scheduler, GIL, and IRQ evidence.

Environment:
  OUTPUT_ROOT=artifacts/vllm_cpu_experiments
  PROFILE_SECONDS=30 PERF_FREQ=199 PYSPY_RATE=100
  TRACE_REQUESTS=1 TRACE_DECODE_STEPS=99  # example; count actual trace steps
  TRACE_THREAD_REGEX='EngineCor|Worker'  # optional thread-name filter

System inventory:
  bash labs/run_vllm_cpu_experiments.sh doctor
  bash labs/run_vllm_cpu_experiments.sh snapshot <engine-or-worker-pid> <label>

Run a vLLM server under one Nsight configuration. The server must enable the
CUDA profiler endpoint, and the load client must call /start_profile and
/stop_profile (vllm bench serve --profile does this):
  bash labs/run_vllm_cpu_experiments.sh nsys-low <label> -- vllm serve ...
  bash labs/run_vllm_cpu_experiments.sh nsys-deep <label> -- vllm serve ...
  bash labs/run_vllm_cpu_experiments.sh nsys-cpu <label> -- vllm serve ...

Attach one collector at a time to an already-warmed server while another
terminal generates the identical steady workload:
  bash labs/run_vllm_cpu_experiments.sh perf-stat <pid> <label>
  bash labs/run_vllm_cpu_experiments.sh perf-record <pid> <label>
  bash labs/run_vllm_cpu_experiments.sh perf-sched <pid> <label>
  bash labs/run_vllm_cpu_experiments.sh pyspy <pid> <label>
  bash labs/run_vllm_cpu_experiments.sh irq <label>

Export/analyze:
  bash labs/run_vllm_cpu_experiments.sh export <report.nsys-rep> <label>
  bash labs/run_vllm_cpu_experiments.sh torch-trace <trace.json> <label>
  bash labs/run_vllm_cpu_experiments.sh probe-summary <label>
  bash labs/run_vllm_cpu_experiments.sh compare \
    baseline=summary.json pinned=summary.json [more...]

Wait-policy source experiment for vLLM 0.26:
  python3 labs/patch_vllm_event_wait.py status
  python3 labs/patch_vllm_event_wait.py apply --dry-run
  python3 labs/patch_vllm_event_wait.py apply
  VLLM_CUDA_EVENT_WAIT_MODE=blocking|spin|python_poll|hybrid vllm serve ...
  VLLM_CUDA_EVENT_HYBRID_SPIN_US=50  # hybrid only
  python3 labs/patch_vllm_event_wait.py restore

Do not combine collectors in one run. Each profiler has a different observer
effect; repeat the same workload in separate runs.
EOF
}

run_doctor() {
  mkdir -p "${OUTPUT_ROOT}/system"
  uname -a | tee "${OUTPUT_ROOT}/system/uname.txt"
  lscpu | tee "${OUTPUT_ROOT}/system/lscpu.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi topo -m | tee "${OUTPUT_ROOT}/system/nvidia-smi-topo.txt"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv \
      | tee "${OUTPUT_ROOT}/system/nvidia-smi-gpus.csv"
  fi
  if command -v numactl >/dev/null 2>&1; then
    numactl --hardware | tee "${OUTPUT_ROOT}/system/numactl-hardware.txt"
  fi
  for command in nsys perf py-spy taskset numactl; do
    if command -v "${command}" >/dev/null 2>&1; then
      printf '%s: %s\n' "${command}" "$(command -v "${command}")"
    else
      printf '%s: missing\n' "${command}"
    fi
  done | tee "${OUTPUT_ROOT}/system/tools.txt"
  if command -v nsys >/dev/null 2>&1; then
    nsys --version | tee "${OUTPUT_ROOT}/system/nsys-version.txt"
    nsys status -e >"${OUTPUT_ROOT}/system/nsys-status.txt" || true
  fi
  if command -v perf >/dev/null 2>&1; then
    perf --version | tee "${OUTPUT_ROOT}/system/perf-version.txt"
  fi
  "${PYTHON_BIN}" - <<'PY' | tee "${OUTPUT_ROOT}/system/python-packages.txt"
import importlib.metadata
import platform
print(platform.python_version())
for package in ("vllm", "torch", "numpy", "py-spy"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=missing")
PY
  cat /sys/devices/system/cpu/online >"${OUTPUT_ROOT}/system/cpu-online.txt"
  if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor \
      >"${OUTPUT_ROOT}/system/cpu-governors.txt" || true
  fi
}

run_snapshot() {
  if [[ $# -ne 2 ]]; then
    echo "snapshot requires PID and label" >&2
    exit 1
  fi
  local pid="$1"
  local label
  label="$(normalize_label "$2")"
  require_pid "${pid}"
  local directory="${OUTPUT_ROOT}/${label}/snapshot"
  mkdir -p "${directory}"
  ps -L -p "${pid}" -o pid,tid,psr,pcpu,stat,comm,wchan:40 \
    >"${directory}/threads.txt"
  taskset -pc "${pid}" >"${directory}/process-affinity.txt" 2>&1 || true
  cp "/proc/${pid}/status" "${directory}/proc-status.txt"
  cp "/proc/${pid}/sched" "${directory}/proc-sched.txt"
  cp "/proc/${pid}/numa_maps" "${directory}/proc-numa-maps.txt" 2>/dev/null || true
  cp /proc/interrupts "${directory}/interrupts.txt"
  cp /proc/softirqs "${directory}/softirqs.txt"
  {
    for affinity in /proc/irq/*/smp_affinity_list; do
      [[ -r "${affinity}" ]] || continue
      printf '%s ' "${affinity}"
      tr -d '\n' <"${affinity}"
      printf '\n'
    done
  } >"${directory}/irq-affinity-list.txt"
  echo "Snapshot: ${directory}"
}

run_nsys() {
  local mode="$1"
  shift
  if [[ $# -lt 2 ]]; then
    echo "nsys-${mode} requires label and command" >&2
    exit 1
  fi
  local label
  label="$(normalize_label "$1")"
  shift
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ $# -eq 0 ]]; then
    echo "server command is empty" >&2
    exit 1
  fi
  require_command nsys
  local directory="${OUTPUT_ROOT}/${label}/nsys"
  mkdir -p "${directory}"
  local options=()
  case "${mode}" in
    low)
      options=(
        --trace=cuda,nvtx
        --sample=none
        --cpuctxsw=none
        --cudabacktrace=none
        --cuda-event-trace=false
        --cuda-graph-trace=graph
      )
      ;;
    deep)
      options=(
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=process-tree
        --cudabacktrace=none
        --cuda-event-trace=true
        --cuda-graph-trace=node
      )
      ;;
    cpu)
      options=(
        --trace=cuda,nvtx,osrt
        --sample=process-tree
        --cpuctxsw=process-tree
        --backtrace=fp
        --cudabacktrace=none
        --cuda-event-trace=false
        --cuda-graph-trace=graph
      )
      ;;
    *)
      echo "unknown Nsys mode: ${mode}" >&2
      exit 1
      ;;
  esac
  printf '%q ' "$@" >"${directory}/server-command.txt"
  printf '\n' >>"${directory}/server-command.txt"
  exec nsys profile \
    --force-overwrite=true \
    "${options[@]}" \
    --trace-fork-before-exec=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --wait=all \
    --output="${directory}/${label}_${mode}" \
    "$@"
}

run_perf_stat() {
  if [[ $# -ne 2 ]]; then
    echo "perf-stat requires PID and label" >&2
    exit 1
  fi
  local pid="$1"
  local label
  label="$(normalize_label "$2")"
  require_pid "${pid}"
  require_command perf
  local directory="${OUTPUT_ROOT}/${label}/perf"
  mkdir -p "${directory}"
  local events="task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults,minor-faults,major-faults"
  if ! perf stat -x, -e "${events}" -p "${pid}" \
    -o "${directory}/perf-stat.csv" -- sleep "${PROFILE_SECONDS}"; then
    echo "Hardware counters unavailable; retrying software counters only." >&2
    perf stat -x, \
      -e task-clock,context-switches,cpu-migrations,page-faults,minor-faults,major-faults \
      -p "${pid}" -o "${directory}/perf-stat.csv" -- sleep "${PROFILE_SECONDS}"
  fi
  echo "Perf stat: ${directory}/perf-stat.csv"
}

run_perf_record() {
  if [[ $# -ne 2 ]]; then
    echo "perf-record requires PID and label" >&2
    exit 1
  fi
  local pid="$1"
  local label
  label="$(normalize_label "$2")"
  require_pid "${pid}"
  require_command perf
  local directory="${OUTPUT_ROOT}/${label}/perf"
  mkdir -p "${directory}"
  local data="${directory}/perf.data"
  if ! perf record -F "${PERF_FREQ}" -e cycles:u -g --call-graph dwarf,16384 \
    -p "${pid}" -o "${data}" -- sleep "${PROFILE_SECONDS}"; then
    echo "cycles:u unavailable; retrying cpu-clock." >&2
    perf record -F "${PERF_FREQ}" -e cpu-clock -g --call-graph dwarf,16384 \
      -p "${pid}" -o "${data}" -- sleep "${PROFILE_SECONDS}"
  fi
  perf report --stdio --no-children --sort comm,dso,symbol --percent-limit 0.1 \
    -i "${data}" >"${directory}/perf-report-self.txt"
  perf report --stdio --children --sort comm,dso,symbol --percent-limit 0.1 \
    -i "${data}" >"${directory}/perf-report-inclusive.txt"
  echo "Perf record: ${directory}"
}

run_perf_sched() {
  if [[ $# -ne 2 ]]; then
    echo "perf-sched requires PID and label" >&2
    exit 1
  fi
  local pid="$1"
  local label
  label="$(normalize_label "$2")"
  require_pid "${pid}"
  require_command perf
  local directory="${OUTPUT_ROOT}/${label}/perf"
  mkdir -p "${directory}"
  local data="${directory}/perf-sched.data"
  perf sched record -a -o "${data}" -- sleep "${PROFILE_SECONDS}"
  if ! perf sched timehist -i "${data}" -p "${pid}" \
    >"${directory}/perf-sched-timehist.txt"; then
    perf sched timehist -i "${data}" >"${directory}/perf-sched-timehist.txt"
  fi
  perf sched latency -i "${data}" >"${directory}/perf-sched-latency.txt" || true
  echo "Perf sched: ${directory}"
}

run_pyspy() {
  if [[ $# -ne 2 ]]; then
    echo "pyspy requires PID and label" >&2
    exit 1
  fi
  local pid="$1"
  local label
  label="$(normalize_label "$2")"
  require_pid "${pid}"
  require_command py-spy
  local directory="${OUTPUT_ROOT}/${label}/pyspy"
  mkdir -p "${directory}"
  py-spy record --pid "${pid}" --duration "${PROFILE_SECONDS}" \
    --rate "${PYSPY_RATE}" --subprocesses --format raw \
    --output "${directory}/all-python.raw"
  echo "Repeat the identical workload for the GIL-only pass." >&2
  py-spy record --pid "${pid}" --duration "${PROFILE_SECONDS}" \
    --rate "${PYSPY_RATE}" --subprocesses --gil --format raw \
    --output "${directory}/gil-holder.raw"
  echo "py-spy: ${directory}"
}

run_irq() {
  if [[ $# -ne 1 ]]; then
    echo "irq requires label" >&2
    exit 1
  fi
  local label
  label="$(normalize_label "$1")"
  local directory="${OUTPUT_ROOT}/${label}/irq"
  mkdir -p "${directory}"
  cp /proc/interrupts "${directory}/interrupts-before.txt"
  cp /proc/softirqs "${directory}/softirqs-before.txt"
  sleep "${PROFILE_SECONDS}"
  cp /proc/interrupts "${directory}/interrupts-after.txt"
  cp /proc/softirqs "${directory}/softirqs-after.txt"
  "${PYTHON_BIN}" labs/diff_proc_interrupts.py \
    "${directory}/interrupts-before.txt" \
    "${directory}/interrupts-after.txt" \
    --output-json "${directory}/interrupt-delta.json" \
    >"${directory}/interrupt-delta.txt"
  "${PYTHON_BIN}" labs/diff_proc_interrupts.py \
    "${directory}/softirqs-before.txt" \
    "${directory}/softirqs-after.txt" \
    --output-json "${directory}/softirq-delta.json" \
    >"${directory}/softirq-delta.txt"
  echo "IRQ counters: ${directory}"
}

run_export() {
  if [[ $# -ne 2 ]]; then
    echo "export requires report.nsys-rep and label" >&2
    exit 1
  fi
  local report="$1"
  local label
  label="$(normalize_label "$2")"
  [[ -f "${report}" ]] || { echo "Report does not exist: ${report}" >&2; exit 1; }
  require_command nsys
  local directory="${OUTPUT_ROOT}/${label}/analysis"
  mkdir -p "${directory}"
  local sqlite="${directory}/${label}.sqlite"
  nsys export --type=sqlite --force-overwrite=true --output="${sqlite}" "${report}"
  "${PYTHON_BIN}" labs/analyze_nsys_sqlite.py "${sqlite}" \
    --output-json "${directory}/nsys-summary.json" \
    | tee "${directory}/nsys-summary.txt"
  echo "Nsys analysis: ${directory}"
}

run_torch_trace() {
  if [[ $# -ne 2 ]]; then
    echo "torch-trace requires trace.json and label" >&2
    exit 1
  fi
  local trace="$1"
  local label
  label="$(normalize_label "$2")"
  [[ -f "${trace}" ]] || { echo "Trace does not exist: ${trace}" >&2; exit 1; }
  local directory="${OUTPUT_ROOT}/${label}/analysis"
  local arguments=()
  mkdir -p "${directory}"
  if [[ -n "${TRACE_REQUESTS}" ]]; then
    arguments+=(--requests "${TRACE_REQUESTS}")
  fi
  if [[ -n "${TRACE_DECODE_STEPS}" ]]; then
    arguments+=(--decode-steps "${TRACE_DECODE_STEPS}")
  fi
  if [[ -n "${TRACE_THREAD_REGEX}" ]]; then
    arguments+=(--thread-regex "${TRACE_THREAD_REGEX}")
  fi
  "${PYTHON_BIN}" labs/analyze_torch_trace_cpu.py "${trace}" \
    "${arguments[@]}" \
    --output-json "${directory}/torch-cpu-self-summary.json" \
    | tee "${directory}/torch-cpu-self-summary.txt"
}

run_probe_summary() {
  if [[ $# -ne 1 ]]; then
    echo "probe-summary requires label" >&2
    exit 1
  fi
  local label
  label="$(normalize_label "$1")"
  local root="${OUTPUT_ROOT}/${label}"
  local directory="${root}/analysis"
  local arguments=()
  mkdir -p "${directory}"
  if [[ -f "${root}/perf/perf-stat.csv" ]]; then
    arguments+=(--perf-stat "${root}/perf/perf-stat.csv")
  fi
  if [[ -f "${root}/pyspy/all-python.raw" ]]; then
    arguments+=(--pyspy-all "${root}/pyspy/all-python.raw")
  fi
  if [[ -f "${root}/pyspy/gil-holder.raw" ]]; then
    arguments+=(--pyspy-gil "${root}/pyspy/gil-holder.raw")
  fi
  if [[ ${#arguments[@]} -eq 0 ]]; then
    echo "No perf-stat or py-spy input found below: ${root}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" labs/summarize_cpu_probes.py "${arguments[@]}" \
    --output-json "${directory}/cpu-probes-summary.json" \
    | tee "${directory}/cpu-probes-summary.txt"
}

run_compare() {
  if [[ $# -lt 2 ]]; then
    echo "compare requires at least two LABEL=summary.json inputs" >&2
    exit 1
  fi
  local directory="${OUTPUT_ROOT}/comparison"
  mkdir -p "${directory}"
  "${PYTHON_BIN}" labs/compare_cpu_profiles.py "$@" \
    --output-json "${directory}/comparison.json" \
    --output-markdown "${directory}/comparison.md"
}

case "${ACTION}" in
  doctor)
    run_doctor
    ;;
  snapshot)
    run_snapshot "$@"
    ;;
  nsys-low)
    run_nsys low "$@"
    ;;
  nsys-deep)
    run_nsys deep "$@"
    ;;
  nsys-cpu)
    run_nsys cpu "$@"
    ;;
  perf-stat)
    run_perf_stat "$@"
    ;;
  perf-record)
    run_perf_record "$@"
    ;;
  perf-sched)
    run_perf_sched "$@"
    ;;
  pyspy)
    run_pyspy "$@"
    ;;
  irq)
    run_irq "$@"
    ;;
  export)
    run_export "$@"
    ;;
  torch-trace)
    run_torch_trace "$@"
    ;;
  probe-summary)
    run_probe_summary "$@"
    ;;
  compare)
    run_compare "$@"
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    print_usage >&2
    exit 1
    ;;
esac
