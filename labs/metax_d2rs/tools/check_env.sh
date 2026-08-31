#!/usr/bin/env bash
set -uo pipefail

C500_BDF=""
NIC_BDF=""

usage() {
    echo "Usage: $0 [--c500-bdf DOMAIN:BUS:DEV.F] [--nic-bdf DOMAIN:BUS:DEV.F]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --c500-bdf)
            C500_BDF="${2:-}"
            shift 2
            ;;
        --nic-bdf)
            NIC_BDF="${2:-}"
            shift 2
            ;;
        --help|-h)
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

section() {
    echo
    echo "[$1]"
}

run_if_present() {
    local command_name="$1"
    shift
    echo "+ ${command_name} $*"
    if command -v "${command_name}" >/dev/null 2>&1; then
        "${command_name}" "$@" 2>&1 || true
    else
        echo "NOT_FOUND: ${command_name}"
    fi
}

show_pci_device() {
    local label="$1"
    local bdf="$2"
    if [[ -z "${bdf}" ]]; then
        echo "${label}_BDF_NOT_SET"
        return
    fi
    echo "${label}_BDF=${bdf}"
    run_if_present lspci -Dnn -s "${bdf}"
    run_if_present lspci -Dvv -s "${bdf}"
    local sysfs="/sys/bus/pci/devices/${bdf}"
    if [[ -d "${sysfs}" ]]; then
        echo "numa_node=$(<"${sysfs}/numa_node" 2>/dev/null || echo unknown)"
        echo "iommu_group=$(readlink "${sysfs}/iommu_group" 2>/dev/null || echo none)"
        echo "driver=$(readlink "${sysfs}/driver" 2>/dev/null || echo none)"
    else
        echo "SYSFS_DEVICE_NOT_FOUND=${sysfs}"
    fi
}

section "host"
run_if_present uname -a
if [[ -r /etc/os-release ]]; then
    sed -n '1,40p' /etc/os-release
fi
if [[ -r /proc/cmdline ]]; then
    echo "kernel_cmdline=$(</proc/cmdline)"
fi

section "pci-topology"
run_if_present lspci -Dnn
run_if_present lspci -Dtv
show_pci_device C500 "${C500_BDF}"
show_pci_device RNIC "${NIC_BDF}"

section "kernel-modules"
run_if_present lsmod
for module in ib_core ib_uverbs rdma_cm mlx5_core mlx5_ib ubcore uburma udma; do
    run_if_present modinfo "${module}"
done

section "rdma"
run_if_present rdma link show
run_if_present ibv_devices
run_if_present ibv_devinfo
run_if_present udevadm info /dev/infiniband/uverbs0
if [[ -d /dev/infiniband ]]; then
    ls -la /dev/infiniband
else
    echo "NOT_FOUND: /dev/infiniband"
fi

section "ub-urma"
if [[ -d /sys/class/ubcore ]]; then
    find /sys/class/ubcore -maxdepth 2 -type f -print 2>/dev/null | sort | head -200
else
    echo "NOT_FOUND: /sys/class/ubcore"
fi
find /dev -maxdepth 2 \( -name 'uburma*' -o -name 'udma*' \) -print 2>/dev/null | sort
run_if_present urma_admin show
run_if_present urma_perftest --help

section "metax"
run_if_present mx-smi
run_if_present maca-smi
if [[ -d /opt/maca ]]; then
    find /opt/maca -maxdepth 2 -type f \( -name '*mcfile*' -o -name '*cufile*' -o -name '*gdr*' \) -print 2>/dev/null | sort
else
    echo "NOT_FOUND: /opt/maca"
fi
if [[ -d /opt/mxdriver ]]; then
    find /opt/mxdriver -maxdepth 2 -type f -print 2>/dev/null | sort | head -200
else
    echo "NOT_FOUND: /opt/mxdriver"
fi

section "libraries-and-symbols"
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -E 'ibverbs|rdmacm|urma|ummu|mcruntime|mcfile|ndsfs' || true
else
    echo "NOT_FOUND: ldconfig"
fi
if command -v nm >/dev/null 2>&1; then
    for library in /usr/lib64/libibverbs.so* /usr/lib/x86_64-linux-gnu/libibverbs.so*; do
        [[ -e "${library}" ]] || continue
        echo "+ nm -D ${library} | search ibv_reg_dmabuf_mr"
        nm -D "${library}" 2>/dev/null | grep ibv_reg_dmabuf_mr || true
        break
    done
fi

section "dma-buf"
if [[ -r /sys/kernel/debug/dma_buf/bufinfo ]]; then
    sed -n '1,240p' /sys/kernel/debug/dma_buf/bufinfo
else
    echo "UNAVAILABLE: /sys/kernel/debug/dma_buf/bufinfo (debugfs/root may be required)"
fi
if [[ -d /sys/kernel/dmabuf/buffers ]]; then
    find /sys/kernel/dmabuf/buffers -maxdepth 2 -type f -print 2>/dev/null | sort | head -240
else
    echo "NOT_FOUND: /sys/kernel/dmabuf/buffers"
fi

section "iommu"
if [[ -d /sys/kernel/iommu_groups ]]; then
    find /sys/kernel/iommu_groups -maxdepth 2 -type l -print 2>/dev/null | sort | head -240
else
    echo "NOT_FOUND: /sys/kernel/iommu_groups"
fi
if command -v dmesg >/dev/null 2>&1; then
    dmesg 2>/dev/null | grep -Ei 'iommu|dmar|amd-vi|dma.?buf|p2p|acs|peer' | tail -240 || true
fi

section "result-hints"
echo "PASS candidates: C500 and RNIC share a suitable PCIe hierarchy; RDMA device is ACTIVE;"
echo "libibverbs exports ibv_reg_dmabuf_mr; MetaX can export a C500 allocation; MR registration succeeds."
echo "Do not claim D2RS from topology or symbol presence alone. The end-to-end write and device-side checksum are mandatory."
