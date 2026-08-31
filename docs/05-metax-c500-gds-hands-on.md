# 沐曦 C500 MAS / GDS 等价能力实验手册

本手册用于在 Linux + 沐曦 C500 + 本地 NVMe 环境中，对 MACA MAS 的
Storage↔GPU 能力进行可重复的资格检查、三路径性能对比和直达路径取证。
实验入口集中在 [`labs/metax_gds/`](../labs/metax_gds/)。

这里将 `cuda_io=cufile` 称为“MAS 直达候选路径”，而不直接称为已经证明的
GDS。原因是：沐曦公开资料确认了 API 和 mxFIO 测试方式，但没有公开
strict/no-fallback 配置、逐 I/O 路径统计以及完整硬件支持矩阵。实验必须把
“API 能运行”和“数据确实绕过 Host DRAM”分开验收。

实验最终回答五个问题：

1. C500、MXMACA、`libcufile` 和 mxFIO 是否安装且版本匹配？
2. Storage→CPU、Storage→CPU→GPU、Storage→GPU 三条路径能否分别运行？
3. 三条路径的带宽、IOPS、延迟和 CPU 成本有何差异？
4. I/O size、并发 job 数和 PCIe/NUMA 拓扑如何影响结果？
5. 现有证据能把结论推进到“API 可用”“强证据直达”还是“厂商严格确认”？

## 1. 名词和证据边界

沐曦官方 mxFIO README 对三条路径的定义是：

| 本手册名称 | mxFIO 配置 | 数据路径 |
|---|---|---|
| `cpu` | `ioengine=psync` | Storage↔CPU |
| `staging` | `ioengine=libcufile`、`cuda_io=posix` | Storage→CPU→GPU |
| `mas` | `ioengine=libcufile`、`cuda_io=cufile` | Storage↔GPU 的 MAS 候选路径 |

官方资料明确说明 mxFIO 是移植到 MACA 软件栈的 fio，可执行文件仍叫 `fio`；
`cuda_io=cufile` 使用 `libcufile` API 在存储和显存之间读写，
`cuda_io=posix` 则先通过 `pread/pwrite` 到主机内存，再进入显存。

注意下面几组不等价关系：

```text
mxFIO 命令成功       ≠ 已证明绕过 Host DRAM
存在 libcufile.so    ≠ 目标文件系统一定支持直达
支持 dma-buf/GDR     ≠ 支持文件语义的 MAS/GDS
mas 比 staging 更快  ≠ 单独足以证明直达
```

本手册使用四级结论：

| 等级 | 可以下的结论 | 必要证据 |
|---|---|---|
| L0 | 环境不完整 | C500、MACA、mxFIO 或 `libcufile` 缺失 |
| L1 | MAS API 可用 | `cuda_io=cufile` 成功并产生合理结果 |
| L2 | 强证据支持直达 | L1 + staging 对照 + CPU/DRAM/PCIe 观测一致 |
| L3 | 该配置已严格确认 | L2 + 厂商 no-fallback 方法/路径计数器或正式支持矩阵 |

在拿不到厂商 strict mode 或路径统计时，最终报告最多写 L2，不写“已经严格证明”。

## 2. 官方依据与已知限制

截至本手册编写时，可公开核对的依据包括：

- [沐曦 C500 产品页](https://www.metax-tech.com/prod.html?cid=107&id=21)：
  C500 是 PCIe 形态的通用计算 GPU，采用 MXMACA 软件栈。
- [沐曦开发者论坛：GPUDirect Storage 特性是否支持](https://developer.metax-tech.com/forum/t/gpudirect-storagete-xing-shi-fou-zhi-chi/288/)：
  官方人员确认支持，并建议使用 mxFIO 测试。
- [MetaX-MACA/mxFIO](https://github.com/MetaX-MACA/mxFIO)：
  给出 `libcufile` 引擎、`cuda_io=cufile/posix`、构建方法和示例配置。
- [MXMACA-C500 发布说明入口](https://developer.metax-tech.com/doc/222)：
  上机前应以实际安装版本对应的发布说明为准。

公开 mxFIO README 还说明：

- 当前移植基于 fio 3.40。
- 运行环境要求 glibc 2.30 或更高版本。
- 如果启用 Huawei NDS 文件系统，需要额外头文件、`libndsfs.so` 和
  `--enable-nds`；这不属于本地 NVMe 基础实验。

公开资料没有给出以下内容，因此不能自行猜测：

- C500 MAS 支持的内核、文件系统、NVMe 控制器完整白名单。
- IOMMU 必须 `off`、`pt` 还是可以正常翻译。
- ACS、BAR、NUMA 的强制配置。
- 等价于 NVIDIA `gdscheck` 的路径诊断输出。
- 禁止 CPU fallback 的配置和运行时计数器。

遇到这些问题，应携带 Phase 0 产物向沐曦支持确认，而不是直接套用 NVIDIA
的 `nvidia-fs`、P2PDMA 或 `cufile.json` 配置。

## 3. 安全规则

仓库脚本遵循以下边界：

- 预检脚本只读系统状态，不加载模块、不改 IOMMU/ACS、不改挂载。
- 矩阵脚本只接受已存在的普通文件，只运行顺序读，并向 mxFIO 传入
  `--readonly`。
- 矩阵脚本要求显式指定 mxFIO 的绝对路径，防止误用发行版自带 fio。
- 请求的测试区域不能大于现有文件。
- 结果写入 `artifacts/metax-gds/`，默认不提交 Git。

上机前人工确认：

- 测试挂载点是专用盘或明确批准的测试空间。
- 测试文件不是模型、checkpoint、数据库、系统镜像或生产数据。
- 没有训练、推理、巡检或存储维护任务共享 C500 和目标 NVMe。
- 修改驱动、内核、IOMMU、ACS 或固件必须经过管理员和厂商支持确认。

## 4. 目标环境

| 项目 | 最小要求 | 推荐 |
|---|---|---|
| GPU | 1× 沐曦 C500 | 2× C500 或 2 个不同拓扑槽位用于对照 |
| 平台 | x86_64 裸金属 Linux | GPU 与 NVMe 同 NUMA/Root Port 或同 PCIe Switch |
| 软件栈 | 匹配的 MXMACA、mxdriver | 同一发布包验证过的版本组合 |
| 用户态库 | `/opt/maca/lib` 中可发现 `libcufile` | 保留安装包版本与 checksum |
| mxFIO | 带 `libcufile` 引擎的沐曦版本 | 从官方仓库固定 commit 构建 |
| glibc | 2.30+ | 与官方构建环境一致 |
| 存储 | 本地 NVMe 上的普通文件 | 专用企业级 NVMe，空间不少于 64 GiB |
| 文件系统 | 由实际版本支持矩阵决定 | 先测试 EXT4/XFS，但不擅自称其已认证 |
| 权限 | 能运行 C500 和 mxFIO | 能读取 dmesg、PCIe、NUMA 与存储计数器 |

先记录而不是先修改：

```bash
uname -a
cat /etc/os-release
getconf GNU_LIBC_VERSION
cat /proc/cmdline
mx-smi
lspci -tv
lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS
```

## 5. 实验产物

```text
labs/metax_gds/
├── collect_metax_preflight.sh   # 只读环境证据
├── run_mxfio_matrix.sh          # 三路径只读矩阵
└── summarize_mxfio_results.py   # 解析 fio JSON，输出 CSV/Markdown

artifacts/metax-gds/<machine>-<date>/
├── preflight/
├── smoke/
├── matrix/
└── monitoring/
```

每次实验使用新目录，不能把不同驱动或不同拓扑的结果混在一起。

## 6. Phase 0：环境预检

在 C500 节点进入仓库根目录：

```bash
export METAX_RUN_DIR="artifacts/metax-gds/c500-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${METAX_RUN_DIR}"

bash labs/metax_gds/collect_metax_preflight.sh \
  --output-dir "${METAX_RUN_DIR}/preflight"
```

查看：

```bash
cat "${METAX_RUN_DIR}/preflight/summary.txt"
column -ts $'\t' "${METAX_RUN_DIR}/preflight/status.tsv" | less -S
cat "${METAX_RUN_DIR}/preflight/glibc_version.txt"
cat "${METAX_RUN_DIR}/preflight/metax_smi.txt"
cat "${METAX_RUN_DIR}/preflight/maca_libraries.txt"
cat "${METAX_RUN_DIR}/preflight/lspci_tree.txt"
```

预检门槛：

| 检查 | 通过条件 | 未通过的处理 |
|---|---|---|
| GPU | `mx-smi`/PCIe 能识别 C500 | 先修驱动、设备分配或容器权限 |
| glibc | 2.30+ | 更换受支持 OS/容器，不能只替换单个 libc 文件 |
| MACA | `/opt/maca` 与 `/opt/mxdriver` 完整 | 使用同一发布包重装或找厂商 |
| `libcufile` | 动态链接器能发现 | 核对库路径和版本，不从 NVIDIA CUDA 混拷 |
| mxFIO | 能列出 `libcufile` 引擎 | 构建的是错误 fio 或构建时未启用该引擎 |
| 存储 | NVMe、挂载点、空间正确 | 换到专用测试文件系统 |
| 拓扑 | GPU/NVMe 路径清晰 | 记录跨 NUMA/Root 的限制并做槽位对照 |

## 7. Phase 1：构建并固定 mxFIO

优先使用沐曦软件包已经提供且与驱动匹配的 mxFIO。如果需要从官方仓库构建：

```bash
git clone https://github.com/MetaX-MACA/mxFIO.git
cd mxFIO
git rev-parse HEAD

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/mxdriver/lib

mkdir -p build
cd build
../configure --enable-cuda --enable-libcufile
make -j"$(nproc)"
```

不要急着 `sudo make install`，先使用构建目录中的二进制，避免覆盖发行版 fio：

```bash
export MXFIO_BIN="$(realpath ./fio)"
"${MXFIO_BIN}" --version
"${MXFIO_BIN}" --enghelp | grep -i libcufile
"${MXFIO_BIN}" --enghelp=libcufile
```

如果 `../configure` 找不到 CUDA/MACA 兼容头文件或 `libcufile`：

```bash
printf '%s\n' "${MACA_PATH}" "${LD_LIBRARY_PATH}"
find /opt/maca /opt/mxdriver -maxdepth 4 \
  \( -iname '*cufile*' -o -iname '*maca*' \) 2>/dev/null
```

不要通过软链接伪造 NVIDIA CUDA 目录来让 configure 通过；应按该 MXMACA
版本的安装说明补齐开发包。

再次采集预检，并把准确的 mxFIO 路径和目标文件加入证据：

```bash
bash labs/metax_gds/collect_metax_preflight.sh \
  --fio "${MXFIO_BIN}" \
  --test-file /mnt/metax-gds/test.bin \
  --output-dir "${METAX_RUN_DIR}/preflight-with-fio"
```

## 8. Phase 2：准备专用测试文件

优先复用管理员已经准备好的普通文件。若必须创建，下面命令会写满 64 GiB，
只能在已确认的专用测试挂载点执行：

```bash
export TEST_FILE=/mnt/metax-gds/c500-mas-test-64g.bin
test ! -e "${TEST_FILE}"
dd if=/dev/zero of="${TEST_FILE}" bs=1M count=65536 \
  oflag=direct conv=excl status=progress
sync
stat "${TEST_FILE}"
findmnt -T "${TEST_FILE}"
```

`conv=excl` 用于拒绝覆盖已有文件。不要使用稀疏文件或只有未写入 extent 的
`fallocate` 文件做存储带宽测试，否则可能测到读零优化而不是真实介质吞吐。

采集源文件指纹是可选的，64 GiB 校验会完整读取一次磁盘：

```bash
sha256sum "${TEST_FILE}" | tee "${METAX_RUN_DIR}/test-file.sha256"
```

该指纹只能证明测试前后文件未改变，不能证明 mxFIO 已正确校验 GPU 中的数据。

## 9. Phase 3：三路径冒烟实验

先跑最小矩阵，确认三个模式都能执行：

```bash
bash labs/metax_gds/run_mxfio_matrix.sh \
  --file "${TEST_FILE}" \
  --fio "${MXFIO_BIN}" \
  --output-dir "${METAX_RUN_DIR}/smoke" \
  --io-sizes 1M \
  --numjobs 1 \
  --size 4G \
  --runtime 10 \
  --ramp-time 2 \
  --repetitions 1
```

查看结果：

```bash
cat "${METAX_RUN_DIR}/smoke/summary.md"
column -ts, "${METAX_RUN_DIR}/smoke/parsed.csv" | less -S
find "${METAX_RUN_DIR}/smoke/logs" -name '*.stderr.log' \
  -size +0 -print -exec sed -n '1,120p' {} \;
```

冒烟门槛：

- `cpu` 成功：普通存储基线可运行。
- `staging` 成功：`libcufile` 引擎能走主机内存中转。
- `mas` 成功：MAS API 至少达到 L1。
- 三种模式读到的字节量和运行时间合理，没有 0 B/s、短读或异常退出。

若 `mas` 失败而 `staging` 成功，问题通常在 MAS 直达资格、文件系统、驱动或
拓扑，不在普通文件权限。保留 stderr、dmesg 和预检目录后再排障。

## 10. Phase 4：正式性能矩阵

默认矩阵为 3 种路径 × 3 个块大小 × 3 个并发 × 3 次重复；按默认
`runtime=10`、`ramp_time=4` 计算，约需 19 分钟，加上初始化和冷却时间会更长。

```bash
bash labs/metax_gds/run_mxfio_matrix.sh \
  --file "${TEST_FILE}" \
  --fio "${MXFIO_BIN}" \
  --output-dir "${METAX_RUN_DIR}/matrix" \
  --modes cpu,staging,mas \
  --io-sizes 4K,1M,16M \
  --numjobs 1,4,16 \
  --size 4G \
  --runtime 10 \
  --ramp-time 4 \
  --repetitions 3
```

输出包括：

- `runs.tsv`：每次运行的参数、日志和退出码。
- `logs/*.json`：mxFIO 原始 JSON。
- `logs/*.time.log`：GNU time 的进程资源数据。
- `parsed.csv`：逐次结果。
- `summary.csv` 和 `summary.md`：按模式、块大小、并发聚合的中位数。

比较时遵循：

1. 同一块大小、并发和数据区域内比较三种模式。
2. 先看三次重复的离散程度，再比较中位数。
3. 4 KiB 重点看 IOPS/延迟；1–16 MiB 重点看带宽和 CPU 成本。
4. `mas` 的价值可能表现为 CPU/DRAM 降低，不一定只表现为带宽翻倍。
5. fio JSON 中的 CPU% 只覆盖 fio 进程，不能替代全机 DRAM/内核线程观测。

## 11. Phase 5：同时采集 CPU、DRAM、NVMe 和 PCIe 证据

在另一个终端启动基础观测：

```bash
mkdir -p "${METAX_RUN_DIR}/monitoring"
iostat -dxm 1 | tee "${METAX_RUN_DIR}/monitoring/iostat.log"
```

如果已安装 sysstat，可按 mxFIO PID 观察：

```bash
pidstat -dur -p <MXFIO_PID> 1 \
  | tee "${METAX_RUN_DIR}/monitoring/pidstat.log"
```

全机 DRAM 带宽需要使用服务器厂商支持的工具，例如 Intel PCM、AMD uProf
或 BMC/PMU 方案。记录工具版本、socket 和计数器名称。判断方向应是：

```text
cpu 路径：      NVMe 流量高，CPU/Host DRAM 有读流量，无 GPU 数据搬运
staging 路径：  NVMe 流量高，Host DRAM 与 CPU 成本显著，随后搬到 GPU
mas 候选路径： NVMe 流量高，GPU PCIe 流量对应，Host DRAM/CPU 明显下降
```

只看到 NVMe 带宽不够，因为三个模式都会读 NVMe；只看到 GPU 利用率也不够，
因为数据搬运不一定表现为计算利用率。

如果 `mx-smi --help` 中提供 PCIe 吞吐、DMA 或内存流量采样命令，优先使用与
当前驱动版本配套的命令，并把帮助输出一起归档。不要假设它与
`nvidia-smi dmon` 参数兼容。

## 12. Phase 6：拓扑对照

用以下信息把 C500 与 NVMe 定位到 PCIe 树：

```bash
mx-smi
lspci -Dnn
lspci -tv
cat /sys/class/nvme/nvme0/device/numa_node
```

如果有两个 NVMe 或两个 GPU 槽位，固定所有软件参数，只改变拓扑：

```text
A：C500 与 NVMe 同 NUMA/Root/Switch
B：跨 PCIe Switch
C：跨 CPU Socket/Root Complex
```

拓扑实验要回答的是性能和可达性差异，不要为了制造同 Root 路径而在生产节点
擅自关闭 ACS 或 IOMMU。沐曦公开 mxFIO 文档没有要求 `iommu=off`；在厂商给出
正式矩阵前，只记录当前状态。

## 13. Phase 7：数据正确性

本仓库矩阵只读现有文件，并依赖 mxFIO 报告完成字节数。它可以做性能和路径
候选验证，但不是端到端 GPU 数据 checksum 程序。

严格正确性实验需要厂商提供或确认以下任一方式：

- MACA `libcufile` 最小读样例：读入 C500 显存，再由 GPU kernel 或回拷计算
  checksum，与 CPU 基线比较。
- mxFIO 在 `cuda_io=cufile` 下受支持的 verify 模式、校验数据生成流程和限制。
- MAS 驱动提供的每请求错误、短读和 fallback 统计。

提交给厂商的问题可以直接写成：

> 请提供 C500 在当前 MXMACA/驱动版本上，使用 libcufile 将普通文件直接读入
> 显存并计算 checksum 的最小样例；同时说明如何禁止或识别 Host DRAM
> fallback，以及可用于证明路径的驱动计数器。

未获得确认前，不要把 NVIDIA 的 `cuFileBufRegister`、`CU_FILE_DRIVER_OPEN` 或
`cufile.json` 配置原样套用到 MACA 实现。

## 14. dma-buf、GDR 与本实验的关系

如果 C500 能把显存导出为 dma-buf，并且网卡能通过 `ibv_reg_dmabuf_mr`
注册，只能证明 GPU 显存具备被网卡直接访问的 GDR 基础：

```text
远端内存/存储 → RDMA NIC → C500 显存
```

它不能单独证明本地文件的 MAS 路径：

```text
本地文件 → 文件系统 → NVMe → C500 显存
```

后者还要求文件系统、块层、NVMe 驱动和 MACA `libcufile` 共同支持。报告中应
分别记录“dma-buf/GDR 验证”和“mxFIO/MAS 验证”，不能互相替代。

## 15. NDS 与远端存储扩展

mxFIO README 提到可以在构建时增加 `--enable-nds`，并安装 NDS 头文件和
`libndsfs.so`。这属于额外的文件系统/远端存储集成，不应混入本地 NVMe
资格实验。

推荐顺序：

1. 先在本地 NVMe 上完成 L1/L2。
2. 固定 C500、驱动和 mxFIO，仅替换为 NDS 文件系统。
3. 重新执行 preflight、三路径矩阵、CPU/DRAM/NIC 计数器采集。
4. 远端路径必须同时证明 NIC↔GPU 的 GDR 和文件系统没有 Host bounce。

远端文件系统的缓存会显著影响结果。记录客户端缓存、服务端缓存、RDMA 网卡、
MTU、队列数、网络拓扑和后端介质，不能只记录一个挂载路径。

## 16. 常见失败和解释

| 现象 | 更可能的原因 | 下一步 |
|---|---|---|
| 普通 fio 有 `libaio` 但没有 `libcufile` | 用错 fio | 显式传入官方 mxFIO 绝对路径 |
| mxFIO 启动时报 glibc symbol/version | glibc < 2.30 或构建环境不匹配 | 换受支持系统或本机重编译 |
| `staging` 成功、`mas` 失败 | MAS 资格、文件系统或拓扑问题 | 保存 stderr/dmesg，向厂商确认矩阵 |
| 三种模式带宽几乎一样 | NVMe 已饱和、块太大或路径回退 | 看 CPU/DRAM/PCIe，不只看带宽 |
| `mas` CPU 很高 | 回退、轮询、锁竞争或统计口径 | 对照 staging，并检查内核线程/DRAM |
| 4 KiB 性能很差 | 同步 I/O、队列深度或引擎开销 | 增加 numjobs，分别报告延迟与 IOPS |
| 多 job 反而下降 | GPU/存储队列、锁、NUMA 或带宽饱和 | 扫描 1/2/4/8/16，找拐点 |
| 跨 socket 明显下降 | Root Complex/NUMA 路径 | 固定 CPU/GPU/NVMe 亲和性后重测 |
| dma-buf RDMA 成功但 mxFIO MAS 失败 | 只有 GDR，文件/块栈未直通 | 分开排查存储栈，不把 GDR 当 GDS |

出现内核错误时立即停止矩阵并保存：

```bash
dmesg -T | tail -n 300
journalctl -k -b --no-pager | tail -n 500
cp -a "${METAX_RUN_DIR}/preflight" "${METAX_RUN_DIR}/failure-preflight"
```

## 17. 验收报告模板

```text
机器/日期：
C500 型号、数量、PCI BDF：
MXMACA/驱动版本：
mxFIO commit、版本、绝对路径：
glibc、OS、内核：
NVMe 型号/固件：
文件系统/挂载参数：
GPU-NVMe PCIe/NUMA 拓扑：
IOMMU/ACS 当前状态：

cpu 路径：通过/失败，峰值与中位带宽：
staging 路径：通过/失败，峰值与中位带宽：
mas 路径：通过/失败，峰值与中位带宽：
CPU/DRAM/NVMe/GPU PCIe 证据：
端到端 checksum：通过/未执行：
厂商 no-fallback/路径统计：有/无：

结论等级：L0 / L1 / L2 / L3
当前可以声称：
当前不能声称：
遗留问题和下一步：
```

推荐的审慎结论示例：

> 在 C500 + 驱动 X + MXMACA Y + 内核 Z + 文件系统 F + NVMe N 的具体配置
> 上，mxFIO `cuda_io=cufile` 可稳定完成只读测试。相较
> `cuda_io=posix`，CPU 与 Host DRAM 流量下降，NVMe 和 GPU PCIe 流量吻合，
> 因此达到 L2，存在强证据支持 MAS 直达；由于尚无厂商 strict/no-fallback
> 统计，本次不宣称达到 L3。

## 18. 一页执行清单

```bash
# 1. 固定运行目录
export METAX_RUN_DIR="artifacts/metax-gds/c500-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${METAX_RUN_DIR}"

# 2. 固定已验证的 mxFIO 和专用测试文件
export MXFIO_BIN=/absolute/path/to/mxFIO/build/fio
export TEST_FILE=/mnt/metax-gds/c500-mas-test-64g.bin

# 3. 预检
bash labs/metax_gds/collect_metax_preflight.sh \
  --fio "${MXFIO_BIN}" --test-file "${TEST_FILE}" \
  --output-dir "${METAX_RUN_DIR}/preflight"

# 4. 冒烟
bash labs/metax_gds/run_mxfio_matrix.sh \
  --fio "${MXFIO_BIN}" --file "${TEST_FILE}" \
  --io-sizes 1M --numjobs 1 --repetitions 1 \
  --runtime 10 --ramp-time 2 \
  --output-dir "${METAX_RUN_DIR}/smoke"

# 5. 正式矩阵
bash labs/metax_gds/run_mxfio_matrix.sh \
  --fio "${MXFIO_BIN}" --file "${TEST_FILE}" \
  --modes cpu,staging,mas --io-sizes 4K,1M,16M \
  --numjobs 1,4,16 --size 4G --repetitions 3 \
  --output-dir "${METAX_RUN_DIR}/matrix"

# 6. 查看结果
cat "${METAX_RUN_DIR}/matrix/summary.md"
```

完成清单不代表自动达到 L3。最终结论必须同时引用预检、原始 mxFIO JSON、
系统观测和厂商对 fallback 的说明。
