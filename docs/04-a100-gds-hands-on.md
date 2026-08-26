# A100 40GB GPUDirect Storage 实验手册

本手册用于在一台 Linux + NVIDIA A100 40GB 服务器上完成可重复的 GDS
资格检查、路径证明、性能对比和最小 cuFile 编程实验。实验入口集中在
[`labs/gds/`](../labs/gds/)。

实验最终要回答四个问题：

1. 目标文件实际支持 `p2pdma`、`nvfs`，还是只能 `compat`？
2. GDS 相比 CPU staging 的吞吐、延迟和 CPU 成本有什么差异？
3. I/O size、并发数和 buffer registration 如何改变结果？
4. 能否用 strict mode、统计、PCIe 流量和 checksum 共同证明真实路径？

不要把“cuFile API 返回成功”作为 GDS 跑通的证据。compatibility mode 也可以成功返回。

## 1. 实验边界和安全规则

本仓库中的脚本遵循以下规则：

- 环境采集脚本只读系统状态，不修改 GRUB、驱动、模块、挂载或文件系统。
- 性能矩阵脚本只接受已经存在的普通文件，并且只允许 `READ` 和 `RANDREAD`。
- 脚本不会执行 `mkfs`、分区、RAID 创建、drop cache 或覆盖已有文件。
- 写入、checkpoint 和持久化实验需要在确认专用测试盘后手工执行。

上机前确认：

- 测试文件不在系统盘、模型盘或含有重要数据的生产挂载点。
- 测试期间没有训练、推理或存储维护任务共用目标 NVMe。
- 修改 IOMMU、ACS、驱动参数或 initramfs 已取得机器管理员许可。
- 所有版本和拓扑信息随结果一并保存，不能只复制一行 GiB/s。

## 2. 目标环境

### 2.1 最小环境

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| GPU | 1× A100 40GB | 2× GPU，便于做拓扑对照 |
| 平台 | x86_64 裸金属 Linux | DGX A100 或拓扑清晰的 PCIe 服务器 |
| 系统 | GDS 支持的 Ubuntu/RHEL | Ubuntu 22.04/24.04 |
| CUDA/GDS | 安装 `libcufile` 和 `gds-tools` | CUDA 12.8+，使用匹配版本的驱动和 GDS 包 |
| 存储 | 1× 专用本地 NVMe | 2× 位于不同 PCIe 路径的 NVMe |
| 文件系统 | EXT4 或 XFS | EXT4 显式 `data=ordered` |
| 空间 | 64 GiB | 128–256 GiB |
| 权限 | 能运行 CUDA/GDS | 能读取 dmesg、PCIe 配置并由管理员调整启动项 |

A100 显存是 40GB 不影响 GDS 资格。微基准只需要 64MiB～数 GiB GPU buffer；
测试文件应显著大于 buffer，避免把 GPU 内存容量误当成数据集上限。

### 2.2 两条受支持路线

现代本地 NVMe 首选：

```text
cuFile → EXT4/XFS → Linux NVMe PCI P2PDMA → A100 VRAM
```

- CUDA 12.8 或更新版本。
- Ubuntu 内核具备 PCI P2PDMA；NVIDIA 当前文档给出的起点为 6.2。
- 单 NVMe，不能在当前 P2PDMA 实验中使用 RAID。
- A100 需要检查静态 BAR1、`ForceP2P=0` 和 write-combine 相关驱动设置。
- 该本地 NVMe 路径不依赖 `nvidia-fs.ko` 或定制 NVMe 补丁。

已有 DGX/旧内核可能采用：

```text
cuFile → nvidia-fs.ko → GDS-aware 文件系统/NVMe 驱动 → A100 VRAM
```

- `gdscheck` 应显示 `NVMe: nvfs, compat`。
- `nvidia-fs`、NVIDIA 驱动、CUDA/GDS 和存储驱动必须匹配。
- 不要为了追求 `p2pdma` 标签，在不了解 DGX OS 支持矩阵时直接升级内核。

两条路线都属于真正的 GDS。`compat` 才是经过 CPU memory bounce buffer 的兜底路径。

官方依据：

- [GDS Getting Started](https://docs.nvidia.com/gpudirect-storage/getting-started/)
- [NVMe P2PDMA 要求](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#nvme-p2pdma-troubleshooting)
- [GDS Installation and Troubleshooting](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)

## 3. 实验产物布局

```text
labs/gds/
├── collect_gds_preflight.sh       # 只读环境证据
├── run_gdsio_matrix.sh            # 只读三路径矩阵
├── summarize_gds_results.py       # 解析日志，生成 CSV/Markdown
├── cufile_verify.cu               # 最小同步读取 + GPU checksum
├── build_cufile_verify.sh
└── config/
    ├── cufile-strict.json         # 禁止 compatibility fallback
    └── cufile-p2pdma-strict.json  # strict + 明确请求 P2PDMA
```

每次实验应创建独立目录：

```text
artifacts/gds/<machine>-<date>/
├── preflight/
├── baseline/
├── matrix/
├── cufile/
└── monitoring/
```

`artifacts/` 默认不提交 Git；需要归档时单独打包。

## 4. Phase 0：上机后先采集，不先改配置

在 A100 节点进入仓库根目录，设置本次实验目录：

```bash
export GDS_RUN_DIR="artifacts/gds/a100-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${GDS_RUN_DIR}"
bash labs/gds/collect_gds_preflight.sh \
  --output-dir "${GDS_RUN_DIR}/preflight"
```

脚本即使遇到缺失命令也会继续采集。首先检查：

```bash
cat "${GDS_RUN_DIR}/preflight/summary.txt"
column -ts $'\t' "${GDS_RUN_DIR}/preflight/status.tsv" | less -S
cat "${GDS_RUN_DIR}/preflight/gdscheck_platform.txt"
cat "${GDS_RUN_DIR}/preflight/nvidia_smi_topology.txt"
cat "${GDS_RUN_DIR}/preflight/lsblk.txt"
cat "${GDS_RUN_DIR}/preflight/findmnt.txt"
```

### 4.1 Phase 0 通过门槛

| 检查 | 通过条件 | 未通过时的含义 |
|---|---|---|
| GPU | 能识别 A100 40GB | 驱动、容器或 GPU 分配有问题 |
| GDS 工具 | `gdscheck -v` 能运行 | GDS 包未安装或路径不正确 |
| NVMe mode | 出现 `p2pdma` 或 `nvfs` | 可能只能 compatibility mode |
| 文件系统 | 测试挂载为 EXT4/XFS | 本地块存储 direct path 不满足 |
| EXT4 options | 显式 `data=ordered` | `cuFileHandleRegister` 可能拒绝文件 |
| IOMMU | x86 为 disabled，或平台验证过的 passthrough | P2P 可能失败或跨 Root 绕行 |
| ACS | 最优路径上的 switch 未强制 redirect | P2P 流量可能上行 Root Complex |
| topology | GPU 与 NVMe 同 Root/Switch 最佳 | 能运行不代表能达到最佳性能 |

如果 `gdscheck` 不存在，但 CUDA 已安装，可以先检查：

```bash
ls -l /usr/local/cuda/gds/tools
ldconfig -p | grep libcufile
```

在已正确配置 NVIDIA CUDA 软件源的 Ubuntu 上，官方元包安装方式是：

```bash
sudo apt-get update
sudo apt-get install nvidia-gds
```

不要让该命令顺带升级生产节点的驱动/CUDA 主版本。先用 `apt-cache policy
nvidia-gds` 和现有 CUDA 版本确认将安装的候选版本；DGX OS 应优先遵循对应 DGX OS 文档。

## 5. Phase 1：确定 P2PDMA 或 NVFS 路线

### 5.1 识别当前能力

关注 `gdscheck -p` 的 DRIVER CONFIGURATION：

```text
NVMe : p2pdma, compat
```

表示平台可用 Linux P2PDMA；当前版本同时保留 compatibility fallback。

```text
NVMe : nvfs, compat
```

表示 NVMe 通过 `nvidia-fs.ko` 路线工作。

```text
NVMe : compat
```

不能进入性能实验，应先修环境。

### 5.2 P2PDMA 特有检查

只检查，不立即修改：

```bash
uname -r
grep -i p2pdma_pgmap_ops /proc/kallsyms
cat /sys/module/nvme_core/parameters/multipath
grep -E 'RegistryDwords|StaticBar|ForceP2P' /proc/driver/nvidia/params
```

对 A100，NVIDIA 当前指南要求重点核对：

```text
NVMe multipath: N
RMForceStaticBar1=1
ForceP2P=0
RmForceDisableIomapWC=1
```

这些是需要重建 initramfs 和重启的管理员配置。只有当 `gdscheck` 明确指出
P2PDMA 不可用，并且机器维护窗口允许时，才按当前 NVIDIA 文档修改。

### 5.3 IOMMU 和 ACS

查看：

```bash
cat /proc/cmdline
dmesg | grep -i iommu
lspci -vv | grep -B2 -A4 -i 'Access Control Services\|ACSCtl'
```

普通 x86 裸金属优先 `iommu=off`；DGX A100 可以采用厂商验证的
`iommu=pt`。关闭 IOMMU 会降低 DMA 隔离能力，不能在多租户节点上擅自修改。
ACS 与 IOMMU 是不同机制：ACS 可能把同一 PCIe switch 内的 P2P 流量强制
送到 Root Complex，表现为功能正常但性能下降。

## 6. Phase 2：准备专用测试文件

先选择已经挂载的专用测试文件系统：

```bash
export GDS_MOUNT=/mnt/gds
export GDS_TEST_FILE="${GDS_MOUNT}/a100-gds-read-64g.bin"
findmnt -T "${GDS_MOUNT}" -o SOURCE,TARGET,FSTYPE,OPTIONS
df -hT "${GDS_MOUNT}"
```

必须确认：

- `SOURCE` 是计划压测的专用 NVMe 或管理员指定的测试卷。
- P2PDMA 首轮实验不是 software RAID、device mapper multipath 或系统盘。
- EXT4 options 包含 `data=ordered`，或者文件系统是受支持的 XFS。

若管理员已经准备好空文件系统，可以创建目录：

```bash
sudo install -d -m 0755 -o "$(id -un)" -g "$(id -gn)" "${GDS_MOUNT}/gds-lab"
export GDS_TEST_FILE="${GDS_MOUNT}/gds-lab/a100-gds-read-64g.bin"
```

下面命令会写入 64GiB。它必须只对一个不存在的新文件执行：

```bash
test ! -e "${GDS_TEST_FILE}" || { echo "Refusing to overwrite ${GDS_TEST_FILE}"; exit 1; }
/usr/local/cuda/gds/tools/gdsio \
  -x 1 -d 0 -w 1 -s 64G -i 1M -f "${GDS_TEST_FILE}" -I 1
sync
ls -lh "${GDS_TEST_FILE}"
```

`-x 1` 使用 CPU memory 准备文件，`-I 1` 是顺序写。不要把写入吞吐混入后续
读取结论。

文件存在后重新采集目标文件资格：

```bash
bash labs/gds/collect_gds_preflight.sh \
  --output-dir "${GDS_RUN_DIR}/preflight-with-file" \
  --test-file "${GDS_TEST_FILE}"
cat "${GDS_RUN_DIR}/preflight-with-file/gdscheck_file.txt"
```

## 7. Phase 3：三路径 smoke test

先跑 4K、1M、16M × 1/4/16 workers × 三种路径，每组 10 秒：

```bash
bash labs/gds/run_gdsio_matrix.sh \
  --file "${GDS_TEST_FILE}" \
  --output-dir "${GDS_RUN_DIR}/baseline" \
  --gpu 0 \
  --operation 0 \
  --io-sizes 4K,1M,16M \
  --workers 1,4,16 \
  --transfers 0,1,2 \
  --dataset-size 16G \
  --duration 10 \
  --repetitions 1
```

路径映射：

| `-x` | 汇总名称 | 数据路径 |
|---:|---|---|
| 0 | `gds` | Storage → GPU |
| 1 | `cpu` | Storage → CPU memory |
| 2 | `cpu_gpu` | Storage → CPU memory → GPU |

结果位于：

```text
baseline/runs.tsv
baseline/logs/*.log
baseline/parsed_runs.csv
baseline/summary.csv
baseline/summary.md
```

Linux 上如果存在 GNU `/usr/bin/time`，矩阵脚本还会把每个进程的 CPU%、user time
和 system time 写入日志与汇总；若没有该工具，对应列显示 `N/A`。

查看：

```bash
cat "${GDS_RUN_DIR}/baseline/summary.md"
```

Smoke test 的目的只是暴露命令、权限、空间和路径错误，不能作为最终性能数据。

## 8. Phase 4：strict mode 路径证明

首先使用通用 strict 配置，它禁止 compatibility fallback，但不强制选择
P2PDMA 或 NVFS：

```bash
export CUFILE_ENV_PATH_JSON="$(pwd)/labs/gds/config/cufile-strict.json"

bash labs/gds/run_gdsio_matrix.sh \
  --file "${GDS_TEST_FILE}" \
  --output-dir "${GDS_RUN_DIR}/strict-gds" \
  --gpu 0 \
  --io-sizes 1M,4M,16M \
  --workers 1,4,8,16 \
  --transfers 0 \
  --dataset-size 16G \
  --duration 30 \
  --repetitions 3 \
  --cufile-config "${CUFILE_ENV_PATH_JSON}"
```

通过条件：

- 所有 `-x 0` 命令退出码为 0。
- 日志中没有 compatibility/fallback。
- `gdscheck` 显示目标 NVMe 具备 `p2pdma` 或 `nvfs`。
- cuFile stats 出现对应 GDS I/O。
- GPU `rxpci` 与 NVMe 读取流量同时增长。

如果当前能力明确为 P2PDMA，可以再使用：

```bash
export CUFILE_ENV_PATH_JSON="$(pwd)/labs/gds/config/cufile-p2pdma-strict.json"
```

不要在只支持 NVFS 的机器上使用 P2PDMA 配置。

## 9. Phase 5：正式参数矩阵

正式读取矩阵：

```bash
bash labs/gds/run_gdsio_matrix.sh \
  --file "${GDS_TEST_FILE}" \
  --output-dir "${GDS_RUN_DIR}/matrix/sequential-read" \
  --gpu 0 \
  --operation 0 \
  --io-sizes 4K,64K,256K,1M,4M,16M \
  --workers 1,2,4,8,16,32 \
  --transfers 0,1,2 \
  --dataset-size 16G \
  --duration 30 \
  --repetitions 3 \
  --cufile-config "$(pwd)/labs/gds/config/cufile-strict.json"
```

随机读取使用 `--operation 2`：

```bash
bash labs/gds/run_gdsio_matrix.sh \
  --file "${GDS_TEST_FILE}" \
  --output-dir "${GDS_RUN_DIR}/matrix/random-read" \
  --gpu 0 \
  --operation 2 \
  --io-sizes 4K,64K,256K,1M \
  --workers 1,4,8,16,32 \
  --transfers 0,2 \
  --dataset-size 16G \
  --duration 30 \
  --repetitions 3 \
  --random-seed 20260824 \
  --cufile-config "$(pwd)/labs/gds/config/cufile-strict.json"
```

正式矩阵耗时较长。每组之间保持相同 GPU clocks/power policy、CPU affinity、
系统负载和测试文件，不要边跑边更改 `cufile.json`。

### 9.1 同步采集可观测性

在另一个终端启动：

```bash
mkdir -p "${GDS_RUN_DIR}/monitoring"
nvidia-smi dmon -i 0 -s putcm -d 1 \
  >"${GDS_RUN_DIR}/monitoring/nvidia-smi-dmon.txt" &
export GDS_DMON_PID=$!

iostat -cxzk 1 >"${GDS_RUN_DIR}/monitoring/iostat.txt" &
export GDS_IOSTAT_PID=$!

pidstat -dur 1 >"${GDS_RUN_DIR}/monitoring/pidstat.txt" &
export GDS_PIDSTAT_PID=$!
```

矩阵完成后停止采集：

```bash
kill "${GDS_DMON_PID}" "${GDS_IOSTAT_PID}" "${GDS_PIDSTAT_PID}"
wait "${GDS_DMON_PID}" "${GDS_IOSTAT_PID}" "${GDS_PIDSTAT_PID}" 2>/dev/null || true
```

读取 GDS 时应观察 GPU `rxpci`；只看 SM utilization 没有意义，因为存储 DMA
并不要求 SM 忙碌。

## 10. Phase 6：最小 cuFile API 与 checksum

构建：

```bash
bash labs/gds/build_cufile_verify.sh
mkdir -p "${GDS_RUN_DIR}/cufile"
```

使用 strict mode 读取 64MiB，重复 10 次，并在最后比较 GPU 与 POSIX checksum：

```bash
CUFILE_ENV_PATH_JSON="$(pwd)/labs/gds/config/cufile-strict.json" \
  labs/gds/build/cufile_verify \
  --file "${GDS_TEST_FILE}" \
  --gpu 0 \
  --bytes 64M \
  --offset 0 \
  --iterations 10 \
  --verify \
  | tee "${GDS_RUN_DIR}/cufile/registered.txt"
```

对比未显式注册 buffer：

```bash
CUFILE_ENV_PATH_JSON="$(pwd)/labs/gds/config/cufile-strict.json" \
  labs/gds/build/cufile_verify \
  --file "${GDS_TEST_FILE}" \
  --gpu 0 \
  --bytes 64M \
  --iterations 10 \
  --no-register \
  --verify \
  | tee "${GDS_RUN_DIR}/cufile/unregistered.txt"
```

程序计时只覆盖同步 `cuFileRead`，checksum 在计时结束后执行。它用于验证 API
生命周期和数据正确性，不替代 `gdsio` 的并发吞吐结果。

## 11. Phase 7：拓扑对照，可选

需要至少两块 GPU 或两个位于不同 PCIe 路径的 NVMe。先查看：

```bash
/usr/local/cuda/gds/tools/gdscheck.py -t
nvidia-smi topo -m
lspci -tv
```

选择：

- near：GPU 和 NVMe 同一 PCIe switch/root。
- far：跨 root、跨 socket，或者需要动态路由。

保持文件大小、I/O size、workers、测试时长和次数不变。至少报告：

```text
拓扑路径 / GiB/s / latency / CPU% / NVMe await / GPU rxpci / fallback mode
```

如果机器只有一块 GPU 和一块 NVMe，跳过本阶段，不能虚构“跨 NUMA GDS”结论。

## 12. 结果解释

### 12.1 大块顺序读

重点看 1M～16M：

- GDS 和 CPU staging 谁先达到 NVMe 上限？
- GDS 是否在相似吞吐下降低 CPU 使用和 DRAM 流量？
- workers 增加后吞吐何时饱和、延迟何时恶化？
- GPU `rxpci` 与 `iostat` 的读带宽能否互相解释？

GDS 吞吐没有高于 CPU staging，不等于失败。如果单 NVMe 已经饱和，GDS 的价值
可能主要表现为 CPU 和 DRAM 成本下降。

### 12.2 小块随机读

重点看 4K～256K：

- 小请求可能被 API、队列和设备延迟主导。
- GDS 不保证每个 4K I/O 都比 POSIX 更低延迟。
- 应同时报告 IOPS 和尾延迟；`gdsio` 的平均延迟不够时，需要后续自定义异步程序。

### 12.3 注册与未注册

预注册 buffer 适合重复使用的权重 shard、dataset batch buffer 和 KV block pool。
未注册路径更易用，但 cuFile 可能使用内部 bounce buffer 或额外管理工作。比较时不能
把一次性 `cuFileBufRegister` 时间混入每个 I/O，除非实验目的就是测注册成本。

## 13. AI 场景的下一步

微基准通过后再做应用实验：

### 13.1 模型权重冷加载

- 把 24–32GiB tensor payload 组织为 4K 对齐的连续 shard。
- 对比 POSIX+pinned H2D 和 cuFile 直接读取。
- 记录打开/注册、实际读取、校验、首个 kernel 的分段耗时。
- 避免把 Python 反序列化和文件格式解析误算成 GDS 本身。

### 13.2 KV Cache 换入

- 使用大于 40GB 的 NVMe 数据集。
- 以 256K、1M、4M block 做随机读。
- 使用独立 CUDA stream，把下一块读取与当前块 kernel 重叠。
- 报告 P50/P95/P99、有效吞吐、GPU stall 和 CPU%。

### 13.3 Checkpoint 写入

- 只使用专用测试文件。
- 分开记录 `cuFileWrite` 完成、`fsync` 和最终持久化边界。
- 注入 short write、磁盘满和进程中止，不能只测顺利路径。

## 14. 常见失败与判断顺序

### `gdscheck` 只有 `compat`

1. 确认 GPU 是 A100 且运行在裸金属/受支持虚拟化环境。
2. 确认目标是 EXT4/XFS 本地 NVMe，而不是 overlayfs、tmpfs 或普通系统盘路径。
3. 检查 CUDA/GDS/驱动版本是否匹配。
4. P2PDMA 路线检查 kernel symbol、multipath 和 A100 驱动参数。
5. NVFS 路线检查 `nvidia_fs` 是否加载、版本是否匹配。

### `cuFileHandleRegister` 失败

1. `findmnt -T <file>` 检查真实 mount 和 options。
2. EXT4 确认 `data=ordered`。
3. 确认 fd 使用 `O_DIRECT`，offset/size 4K 对齐。
4. 用 `gdscheck -f <file>`，不要只跑平台级 `-p`。

### strict mode 失败、普通模式成功

这通常是有价值的结果：普通模式很可能在 fallback。保存两份日志，不要把普通模式
数字标为 GDS。

### GDS 正确但慢

按顺序检查：

1. 单 NVMe 的 `fio --direct=1` 上限。
2. I/O size 和 workers 是否足以填满设备。
3. GPU/NVMe 是否跨 Root/Socket。
4. ACS/IOMMU 是否导致绕行。
5. 是否频繁分配和注册 buffer。
6. `iostat` 显示设备已经 100% utilization，还是软件没有喂满。

## 15. 最终验收清单

- [ ] 保存完整 preflight 目录。
- [ ] `gdscheck -f` 对目标文件通过。
- [ ] 明确记录实际候选模式为 `p2pdma` 或 `nvfs`。
- [ ] strict/no-fallback 的 `-x 0` 成功。
- [ ] `cufile_verify --verify` 输出 `verification=PASS`。
- [ ] 同参数完成 GDS、CPU、CPU→GPU 三路径比较。
- [ ] 至少三次正式重复，报告中位数而非最好的一次。
- [ ] 同步保存 `iostat`、CPU 和 GPU PCIe 观测。
- [ ] 解释吞吐上限来自 NVMe、PCIe、队列还是软件路径。
- [ ] 报告测试文件、文件系统和挂载参数。
- [ ] 没有把 compatibility fallback 写成 GDS 结果。

建议最终报告表头：

```text
machine / GPU / driver / CUDA / GDS / kernel / mode / filesystem / mount options
NVMe model / GPU-NVMe topology / IOMMU / ACS / operation / I/O size / workers
throughput / IOPS / avg latency / P99 if available / CPU% / rxpci / checksum / notes
```

## 16. 官方参考

- [GPUDirect Storage 文档入口](https://docs.nvidia.com/gpudirect-storage/)
- [Getting Started](https://docs.nvidia.com/gpudirect-storage/getting-started/)
- [Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
- [Installation and Troubleshooting](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)
- [Benchmarking and Configuration](https://docs.nvidia.com/gpudirect-storage/configuration-guide/)
- [Best Practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/)
