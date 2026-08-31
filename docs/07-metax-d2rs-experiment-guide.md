# 沐曦 C500 D2RS 实验手册

> 目标：按“模拟契约 → 本地 MAS → Host RDMA → C500 GDR → RDMA D2RS → Host URMA → C500 URMA”的顺序逐层验收，任何一层失败都能明确定位到软件栈。
>
> 设计文档：[06-metax-d2rs-design.md](06-metax-d2rs-design.md)；Demo：[labs/metax_d2rs](../labs/metax_d2rs/README.md)。

## 0. 实验纪律

### 0.1 先确定路径等级

每份结果必须填写一个且只能填写一个：

- `SIMULATION`：Host memory + 本地文件模拟；
- `COMPAT`：远端数据落到计算节点 Host memory，再 H2D；
- `CLIENT_DIRECT_RDMA`：Agent Host staging → RNIC → C500；
- `END_TO_END_DIRECT_RDMA`：远端存储设备 → RNIC → C500；
- `CLIENT_DIRECT_URMA`；
- `END_TO_END_DIRECT_URMA`。

不允许仅凭 `ibv_reg_dmabuf_mr` 或 `urma_register_seg` 成功就声称 D2RS。最终证据必须包含实际传输和 C500 kernel 校验。

### 0.2 数据安全

- 所有 fio/mxFIO 实验使用专用文件 `/mnt/nvme/d2rs-test.bin`；
- 本手册默认只读，不对生产文件或裸盘执行写测试；
- `--self-test` 只创建并删除 `/tmp/d2rs-self-test-XXXXXX`，不会生成或保留 `/mnt/nvme/d2rs-test.bin`；
- 首次运行前确认挂载点，并按 3.4 节显式创建持久测试文件；
- 修改 IOMMU、ACS、PFC、驱动或内核参数前保存原配置和回滚步骤；
- `iommu=off` 只用于隔离故障的 A/B 实验，不作为默认或生产配置。

### 0.3 建议保存的元数据

```bash
export D2RS_ROOT=/opt/d2rs-lab
export D2RS_RESULTS=/var/tmp/d2rs-results
export D2RS_REPO=/path/to/ai-infra
mkdir -p "${D2RS_RESULTS}"
date -Iseconds
uname -a
```

不要复用 `HOME` 等系统环境变量保存实验配置。

## 1. 实验拓扑与环境要求

### 1.1 最小两节点拓扑

```text
Compute node                              Storage node
┌──────────────────────┐                  ┌─────────────────────┐
│ MetaX C500           │                  │ NVMe / test file    │
│        │ PCIe P2P    │                  │       │ pread        │
│ RDMA/UB NIC ═════════╪══════════════════╪═ RDMA/UB NIC        │
│ D2rsClient           │ control network  │ Storage Agent       │
└──────────────────────┘                  └─────────────────────┘
```

客户端 C500 与 RNIC/UB NIC 应优先位于同一 PCIe switch/Root Complex，并处于合适的 NUMA node。跨 CPU socket、ACS 强制上行和不兼容的 IOMMU 映射都可能阻断 P2P。

### 1.2 Phase 1 推荐软件

| 组件 | 要求 |
|---|---|
| OS | 厂商支持的 Linux；优先选择双方都认证的 openEuler/Ubuntu |
| Kernel | 同时受 MetaX 驱动和 RNIC dma-buf MR 支持；不要只按版本号判断 |
| MACA/MAS | `/opt/maca`、`/opt/mxdriver`，包含 C500 runtime 与 mcfile/MAS |
| rdma-core | `libibverbs` 中存在 `ibv_reg_dmabuf_mr` |
| RNIC | Host RDMA 已通过，Provider 支持 dma-buf import |
| Storage | 专用测试文件或 NVMe，第一版允许 Agent Host staging |
| Build | C++17 compiler、CMake 3.16+；可选 liburing、SPDK |

### 1.3 Phase 2 推荐软件

当前本地 UMDK master 文档指定 kernel 6.6 构建环境，并依赖 `ubcore`、`uburma`、UDMA 和 libummu。以实验机实际获得的 UMDK/驱动发布包说明为准，不混装不同 release 的用户态和内核态组件。

## 2. 软件栈改造清单

### 2.1 Phase 1 必须新增或修改

| 顺序 | 软件栈 | 动作 | 验收接口 |
|---:|---|---|---|
| 1 | MetaXProvider | 对接厂商显存分配、dma-buf/GDR 导出、fence、释放 | `allocate/export/make_visible/release` |
| 2 | RdmaBackend | 创建 PD/CQ/QP；调用 dma-buf MR；管理 rkey 租约 | `register/submit/poll/revoke` |
| 3 | 控制协议 | 传 object range、region ID/generation、RDMA window | request/completion |
| 4 | Storage Agent | 读取文件、管理 staging MR、发起 RDMA WRITE | Agent metrics |
| 5 | StorageBackend | 先 pread，后 io_uring/SPDK | async read |
| 6 | C500 校验 kernel | 在 device 上计算 CRC/比较 pattern | device result |
| 7 | 应用插件 | vLLM/权重或 KV Cache loader 调用 D2RS | direct policy |

可能需要修改，但应由最小实验触发：

- 沐曦驱动没有显存 dma-buf exporter：修改沐曦内核驱动；
- RNIC Provider 返回 `EOPNOTSUPP`：升级/修改 RNIC Provider 或驱动；
- dma-buf 能导出但 attachment/map 失败：联合修改 exporter/importer；
- 数据到达但 C500 读到旧值：由沐曦提供正确的 cache/fence API。

默认不修改：Linux 主线、rdma-core 公共 API、SPDK 核心、UBS IO。

### 2.2 Phase 2 必须新增或修改

| 顺序 | 软件栈 | 动作 | 判定 |
|---:|---|---|---|
| 1 | UrmaBackend | Host VA 上实现 context/segment/Jetty/JFC/write/poll | 先过 Host↔Host |
| 2 | Agent Transport | verbs 替换为 URMA，StorageBackend 不变 | 同一 workload |
| 3 | MetaXProvider | 提供 URMA 能消费的 C500 memory identity | 厂商接口 |
| 4 | UDMA Provider | 增加 C500 device-memory import/map（若现有 VA 注册失败） | Segment 注册成功 |
| 5 | UMMU | 建立 C500 IOVA/UBVA 翻译和权限 | 远端可访问 |
| 6 | 沐曦驱动 | 提供 dma-buf attachment 或 UDMA 协同接口 | 生命周期闭环 |

原则上不先修改 `liburma` 公共 API、`ubcore` 和 `uburma`。如果 Provider 无法从 VA/IOVA 找到 C500 allocation，才评审 `register_seg_ex` 或 Provider 私有扩展。

## 3. E0：在任意主机跑通契约 Demo

本实验不需要 C500/RNIC，目的是先验证上层代码和测试方法。

### 3.1 构建

```bash
cd "${D2RS_REPO}/labs/metax_d2rs"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

没有 CMake：

```bash
c++ -std=c++17 -O2 -pthread -Iinclude \
  src/d2rs.cpp src/main.cpp -o d2rs_demo
./d2rs_demo --self-test --json
```

### 3.2 自测文件的生命周期

`--self-test` 不依赖预先准备的数据文件。Demo 会通过 `mkstemp` 创建一个约 9 MiB 的
`/tmp/d2rs-self-test-XXXXXX`，写入确定性 pattern，执行非对齐范围读取和校验，随后在正常退出或异常路径中删除该文件。

因此，执行自测后看不到 `d2rs-test.bin` 是预期行为。构建、`ctest` 和 `--self-test` 都不会创建后续 fio/mxFIO 使用的
`/mnt/nvme/d2rs-test.bin`。

### 3.3 通过标准

预期关键字段：

```json
{
  "backend": "sim",
  "path": "simulation",
  "direct_data_path": false,
  "registration_misses": 1,
  "verified": true
}
```

`direct_data_path=false` 是正确结果。模拟后端经过 Host staging，不是硬件直通。

### 3.4 创建持久测试文件并做范围测试

`d2rs-test.bin` 是本实验生成的合成数据，不需要从外部下载。应在专用 NVMe 挂载点显式创建；以下命令在文件已经存在时拒绝覆盖：

```bash
export D2RS_FILE=/mnt/nvme/d2rs-test.bin
if test -e "${D2RS_FILE}"; then
  echo "文件已存在，拒绝覆盖：${D2RS_FILE}" >&2
else
  fio \
    --name=create-d2rs-data \
    --filename="${D2RS_FILE}" \
    --rw=write \
    --ioengine=psync \
    --bs=1M \
    --size=16G \
    --direct=1 \
    --buffer_pattern=0x0123456789abcdef \
    --end_fsync=1

  ls -lh "${D2RS_FILE}"
  sha256sum "${D2RS_FILE}" > "${D2RS_RESULTS}/d2rs-test.sha256"
fi
```

只做 Demo 功能验证时，可先创建 1 GiB 零填充文件：

```bash
export D2RS_FILE=/mnt/nvme/d2rs-test.bin
dd if=/dev/zero \
  of="${D2RS_FILE}" \
  bs=1M count=1024 \
  oflag=direct conv=excl \
  status=progress
```

`conv=excl` 同样会在目标文件已存在时失败。若挂载点不支持 direct I/O，去掉 `oflag=direct` 只影响文件准备过程，不代表 D2RS 数据面已通过验证。

然后运行范围测试：

```bash
./build/d2rs_demo \
  --input "${D2RS_FILE}" \
  --offset 4K \
  --length 1G \
  --chunk 4M \
  --iodepth 8 \
  --json
```

依次扫描：

```text
chunk:    64K, 256K, 1M, 2M, 4M, 8M, 16M
iodepth:  1, 2, 4, 8, 16, 32
offset:   0, 4K, 64K, 123（负向/非对齐）
length:   4K, 1M, 4M+777, 1G
```

## 4. E1：采集 C500/RNIC/UB 环境证据

### 4.1 找到 BDF

```bash
lspci -Dnn
lspci -Dtv
```

记录 C500 和 RNIC 的完整 BDF，例如：

```bash
export D2RS_C500_BDF=0000:41:00.0
export D2RS_RNIC_BDF=0000:42:00.0
```

### 4.2 运行只读探测脚本

```bash
cd "${D2RS_REPO}/labs/metax_d2rs"
./tools/check_env.sh \
  --c500-bdf "${D2RS_C500_BDF}" \
  --nic-bdf "${D2RS_RNIC_BDF}" \
  > "${D2RS_RESULTS}/env.txt" 2>&1
```

必须人工确认：

1. C500/RNIC 驱动都已绑定；
2. RNIC 的 RDMA link 是 `ACTIVE`；
3. `libibverbs` 导出 `ibv_reg_dmabuf_mr`；
4. 两设备的 PCIe 层级和 NUMA 合理；
5. ACSCtl 是否把 P2P 流量强制上送；
6. IOMMU 当前模式；
7. MetaX runtime/MAS 库实际版本；
8. Phase 2 时 `/sys/class/ubcore`、`uburma`、UDMA 是否存在。

符号存在只表示用户态 API 可用，不表示 RNIC 能导入 C500 dma-buf。

## 5. E2：IOMMU 与 ACS 对照实验

### 5.1 默认策略

优先使用厂商认证配置。若支持，先测试 IOMMU 开启并使用 passthrough：

```text
Intel: intel_iommu=on iommu=pt
AMD:   amd_iommu=on iommu=pt
ARM:   使用平台/厂商给出的 SMMU passthrough 配置
```

`iommu=off` 会失去 DMA 隔离，只用于确认“失败是否来自 IOMMU 映射”。如果关闭后成功，结论应该是“需要修复/支持 IOMMU P2P 映射”，不是“生产必须关闭 IOMMU”。

### 5.2 A/B 方法

1. 保存当前 kernel cmdline、IOMMU group、dmesg；
2. 使用当前/认证配置完成一轮 E4/E5；
3. 仅在隔离实验节点修改启动参数并重启；
4. 使用完全相同的 BDF、buffer size 和 workload 重跑；
5. 立即恢复原参数；
6. 比较 MR 注册错误码、dmesg、带宽、CRC，而不只比较吞吐。

不要在正式环境使用 `pcie_acs_override` 掩盖拓扑问题。

## 6. E3：本地 MAS/GDS 基线

这一步回答“C500 是否能本地存储直读显存”，不能替代 GDR 验证。

### 6.1 准备 mxFIO

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/mxdriver/lib
export MXFIO_SRC=/path/to/MetaX-MACA/mxFIO
cd "${MXFIO_SRC}"
./configure --enable-cuda --enable-libcufile
make -j 12
```

配置阶段应找到 `-lruntime_cu -lmcruntime -lmcfile -ldl`。失败时保存 `config.log`，不要用 NVIDIA 的 libcufile 替代沐曦库。

### 6.2 准备只读测试文件

使用 3.4 节显式创建的专用文件；不要再次运行创建命令覆盖它。开始基线测试前检查大小和初始校验：

```bash
test -f /mnt/nvme/d2rs-test.bin
ls -lh /mnt/nvme/d2rs-test.bin
sha256sum -c "${D2RS_RESULTS}/d2rs-test.sha256"
```

### 6.3 三路径对比

配置模板位于 `labs/metax_d2rs/configs`：

```bash
fio "${D2RS_REPO}/labs/metax_d2rs/configs/fio-host-read.fio" \
  > "${D2RS_RESULTS}/fio-host.txt"

fio "${D2RS_REPO}/labs/metax_d2rs/configs/mxfio-posix-read.fio" \
  > "${D2RS_RESULTS}/mxfio-posix.txt"

fio "${D2RS_REPO}/labs/metax_d2rs/configs/mxfio-cufile-read.fio" \
  > "${D2RS_RESULTS}/mxfio-cufile.txt"
```

对应关系：

| job | 路径 |
|---|---|
| fio host | Storage → Host |
| mxFIO posix | Storage → Host → C500 |
| mxFIO cufile | Storage → C500/MAS |

通过标准：cufile job 成功、数据正确、日志显示使用沐曦 mcfile/MAS，且与 posix 路径的 CPU/带宽特征不同。不能只看 job 名称。

## 7. E4：Host↔Host RDMA 基线

先排除网络本身的问题。

### 7.1 设备与网络

两端执行：

```bash
rdma link show
ibv_devices
ibv_devinfo
ip addr
```

RoCE 环境还要按交换机/RNIC 运维规范核对 VLAN、MTU、PFC/ECN、GID index；这些参数不应由 D2RS 程序偷偷修改。

### 7.2 带宽测试

先用 `ib_write_bw --help` 核对当前 perftest 版本。典型流程：

存储端：

```bash
ib_write_bw -d mlx5_0 -F --report_gbits
```

计算端：

```bash
ib_write_bw -d mlx5_0 -F --report_gbits STORAGE_RDMA_IP
```

再扫描 4 KiB、64 KiB、1 MiB 和多 QP。保存两端命令、GID/port、错误计数和 NIC counters。

通过标准：Host MR 的 write/read 正确，链路无持续丢包/重传，带宽达到网络合理区间。Host RDMA 不通过时不要进入 C500 GDR。

## 8. E5：C500 GDR 原语

### 8.1 向沐曦索取的材料

必须获得厂商提供或确认的：

- C500 GDR sample；
- 显存导出/注册头文件与库；
- 支持的 RNIC/OFED/rdma-core/内核矩阵；
- RNIC → C500 和 C500 → RNIC 两个方向的同步要求；
- 是否能导出 dma-buf fd、offset 和 allocation identity；
- 异常退出、显存释放、device reset 的清理语义。

公开论坛的“支持 GDR”不是可执行 API 规范。

### 8.2 最小正确性矩阵

| 方向 | 发送端 pattern | 目的端校验 | 必须测试 |
|---|---|---|---|
| Host/RNIC → C500 | 递增、全 0、全 1、随机 | C500 kernel CRC | 4K～1G、对齐/非对齐 |
| C500 → Host/RNIC | C500 kernel 生成 | CPU/RNIC 接收 CRC | flush/fence 开关 |

每一组至少运行：

```text
size:      4K, 64K, 1M, 4M, 64M, 1G
iodepth:   1, 2, 8, 32
repeats:   correctness 1000 次，稳定性 1 小时
topology:  同 switch；若有条件再测跨 switch/socket
```

通过标准：两个方向 CRC 全对；cache/fence 语义有明确调用点；无 Xid/driver reset/IOMMU fault。

## 9. E6：C500 dma-buf → RDMA MR

### 9.1 检查用户态入口

```bash
nm -D /usr/lib64/libibverbs.so 2>/dev/null | grep ibv_reg_dmabuf_mr
```

不同发行版库路径不同，可使用 `ldconfig -p | grep ibverbs` 定位。

### 9.2 实现 MetaXProvider

复制并填充：

```text
labs/metax_d2rs/adapters/metax_provider_template.cpp
```

只替换四个 `ENOSYS`：

1. `d2rs_metax_initialize`；
2. `d2rs_metax_allocate_and_export`；
3. `d2rs_metax_make_visible_to_device`；
4. `d2rs_metax_release`。

不要自行猜测私有 ioctl 号。若厂商只提供 GDR registration handle 而非 dma-buf，则新增独立 Provider，保持上层 `DeviceAllocation` 不变。

### 9.3 构建 RDMA 注册适配器

```bash
cd "${D2RS_REPO}/labs/metax_d2rs"
cmake -S . -B build-rdma \
  -DD2RS_BUILD_RDMA_ADAPTER=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rdma -j
```

适配器调用的是：

```text
ibv_reg_dmabuf_mr(pd, dmabuf_offset, length, iova, fd,
                  LOCAL_WRITE | REMOTE_WRITE)
```

`iova` 与 `dmabuf_offset` 必须具有相同 page offset。

### 9.4 注册生命周期测试

按以下顺序写最小测试程序，复用 `adapter_c.h`：

1. 初始化 MetaX device；
2. 分配 4 MiB C500 buffer并导出 fd；
3. 打开 RNIC context、创建 PD；
4. 调用 `d2rs_rdma_register_dmabuf`；
5. 记录 MR 的 iova/lkey/rkey，绝不记录或输出 fd 到网络；
6. 注销 MR；
7. 关闭 fd；
8. 释放显存；
9. 循环 10,000 次，检查 fd/MR 泄漏；
10. 增加“有 in-flight WR 时注销”的负向测试，预期被上层拒绝。

通过标准：注册/注销稳定，dmesg 无 DMA/IOMMU fault，`/proc/<pid>/fd` 和 locked memory 不持续增长。

## 10. E7：实现 RDMA D2RS Agent 与 Client

这是从当前契约 Demo 到真机 PoC 需要增加的代码。

### 10.1 Client 改造顺序

1. 用 MetaXProvider 替换 `make_sim_device_provider()`；
2. 用 RdmaBackend 替换 `make_sim_transport()`；
3. Region 注册后生成 `{region_id, generation, iova, rkey, len, lease}`；
4. 控制面连接 Agent，完成身份认证和 QP 信息交换；
5. 下发 `ReadRequest`；
6. 收割完成；
7. 调用 `make_visible_to_device`；
8. 启动 C500 checksum kernel；
9. revoke/deregister 后再释放 buffer。

### 10.2 Agent 改造顺序

1. 根目录白名单下打开对象，禁止任意路径；
2. 创建固定数量、固定大小的 aligned staging buffers；
3. 每个 staging buffer 只注册一次 Host MR；
4. 接收请求后校验 Region、generation、offset、length、lease；
5. `pread`/`io_uring` 读取一个 chunk；
6. 计算 CRC；
7. `IBV_WR_RDMA_WRITE` 到 `iova + destination_offset`；
8. poll send CQ；
9. 发送 Completion；
10. 复用 staging buffer，记录 `server_staging_bytes`。

第一版使用一个 QP、一个连接、无重叠目的区间、关闭 relaxed ordering。正确性通过后再增加多 QP 和乱序。

### 10.3 端到端测试

测试顺序：

1. 4 KiB 单请求、iodepth=1；
2. 4 MiB 单请求；
3. `4 MiB + 777` 验证尾块；
4. 1 GiB 顺序读；
5. 1,000 个随机不重叠 range；
6. iodepth 1/2/4/8/16/32；
7. 注册缓存关闭/开启；
8. Agent kill、网线断开、超时、C500 reset；
9. 注销后发送旧 generation 请求，必须拒绝；
10. 24 小时稳定性。

### 10.4 D2RS 通过证据

必须同时提供：

- C500 device checksum 正确；
- Client 日志 `path=CLIENT_DIRECT_RDMA`；
- Client `host_staging_bytes=0`；
- RNIC TX/RX 和 PCIe counters 有对应增量；
- MetaXProvider 的 dma-buf/MR 路径被调用；
- compat/H2D 拷贝计数为 0；
- Agent staging 字节数明确记录；
- 失败时 `DIRECT_REQUIRED` 不得静默回退。

## 11. E8：性能与瓶颈定位

### 11.1 路径矩阵

| ID | 路径 | 用途 |
|---|---|---|
| A | Storage → Host | 存储上限 |
| B | Storage → Host → C500 | compat 基线 |
| C | 本地 MAS → C500 | 本地直通基线 |
| D | Agent Host → RDMA → Host | 网络上限 |
| E | Agent Host → RDMA → C500 | Phase 1 D2RS |
| F | Agent storage P2P → RDMA → C500 | 端到端直通 |

### 11.2 参数矩阵

```text
block/chunk: 4K, 64K, 256K, 1M, 2M, 4M, 8M, 16M
iodepth:     1, 2, 4, 8, 16, 32, 64
jobs/QP:     1, 2, 4, 8
file size:   至少大于客户端和 Agent 可用内存缓存
mode:        cold/warm 分开
```

### 11.3 观测命令

按系统实际工具选择：

```bash
pidstat -p AGENT_PID 1
pidstat -p CLIENT_PID 1
iostat -x 1
sar -n DEV 1
numastat -p AGENT_PID
perf stat -p AGENT_PID
ethtool -S RDMA_NETDEV
rdma statistic show
```

同时采集 C500 利用率/显存带宽、RNIC 端口计数和 PCIe link 状态。所有监控输出与 workload 使用同一个时间戳目录。

## 12. E9：Host URMA 基线

### 12.1 安装与驱动

严格跟随当前 UMDK release 的 `README_zh.md`。本地 master 的关键检查是：

```bash
uname -r
modprobe ubcore
modprobe uburma
ls -la /sys/class/ubcore
ls -la /dev | grep -E 'uburma|udma'
```

UDMA 模块参数、安装路径和版本必须来自同一个发布包，不能照抄其他机器的 `insmod` 参数。

### 12.2 URMA write 基线

先用工具确认设备名：

```bash
urma_admin show
urma_perftest -h
```

服务端：

```bash
urma_perftest write_bw -d UB_DEVICE -s 1048576 -n 100000
```

客户端：

```bash
urma_perftest write_bw -d UB_DEVICE -s 1048576 -n 100000 -S SERVER_IP
```

再测试 `write_lat`、不同 size 和并发。Host Segment 基线不通过时不要进入 C500 Segment。

### 12.3 构建 Demo URMA 注册边界

```bash
cd "${D2RS_REPO}/labs/metax_d2rs"
cmake -S . -B build-urma \
  -DD2RS_BUILD_URMA_ADAPTER=ON \
  -DUMDK_ROOT=/path/to/umdk-or-install-root
cmake --build build-urma -j
```

该适配器执行 `urma_register_seg` 和 `urma_get_seg_ctx`，用于验证 Segment 注册/序列化契约。

## 13. E10：C500 URMA/UMMU 接入

### 13.1 先直接验证现有 Provider

1. 通过 MetaXProvider 分配 C500 buffer；
2. 获得厂商认可的 VA/IOVA；
3. 调用 `d2rs_urma_register_va`；
4. 导出 Segment context；
5. 远端 `urma_import_seg`；
6. 用 `urma_write` 写入非零 pattern；
7. `urma_poll_jfc` 成功后执行 MetaX device fence；
8. C500 kernel 校验。

若注册返回空或内核报 pin/map 错误，不能把 Host VA 强转继续试。

### 13.2 Provider/UMMU 需要的改造

如果现有 VA 注册不支持 C500：

```text
MetaX allocation
  → dma-buf exporter / vendor peer-memory handle
  → UDMA Provider import
  → dma_buf_attach + map_attachment（或厂商等价接口）
  → 获取 device-visible sg/IOVA
  → UMMU 建翻译表与权限表
  → urma_target_seg
```

必须实现：

- map/unmap 对称；
- dma-buf move-notify/失效；
- token/lease revoke；
- device reset 和进程退出清理；
- 并发 unregister 与 in-flight write 隔离；
- IOMMU/SMMU 开启时的正确映射；
- C500 device visibility fence。

### 13.3 替换 Agent 数据面

将 Agent 的三个动作替换：

```text
ibv_reg_mr/staging MR     → urma_register_seg
ibv_post_send(RDMA_WRITE) → urma_write
ibv_poll_cq               → urma_poll_jfc
```

控制协议只替换 Transport 私有 payload：`iova/rkey` 变为 serialized Segment/Token/Jetty identity。文件语义、chunk、CRC 和 StorageBackend 不变。

## 14. 故障定位表

| 现象 | 最可能层 | 首查 |
|---|---|---|
| mxFIO cufile 失败，posix 成功 | MAS/mcfile/本地 GDS | 库版本、文件系统、对齐、驱动日志 |
| Host RDMA 失败 | 网络/RNIC | link、GID、MTU、PFC、firewall、QP 状态 |
| `ibv_reg_dmabuf_mr` 符号不存在 | rdma-core 太旧/错误库 | `ldconfig`、`nm -D` |
| MR 返回 `EINVAL` | 参数/对齐/access | fd、offset、length、iova page offset |
| MR 返回 `EOPNOTSUPP` | verbs Provider/exporter 不兼容 | RNIC 支持矩阵、Provider 日志 |
| MR 成功但 RDMA WRITE 报 protection error | iova/rkey/range/lease | WC syndrome、QP、Region generation |
| 传输成功但 C500 读到旧数据 | 显存一致性 | MetaX fence/cache API、方向性限制 |
| `iommu=off` 后才成功 | IOMMU P2P 映射缺口 | IOMMU fault、sg/IOVA、厂商支持矩阵 |
| iodepth=1 正确，高并发错 | 排序/生命周期 | 重叠区间、buffer 复用、CQ/完成映射 |
| URMA Host 成功，C500 Segment 失败 | UDMA/UMMU device memory | Provider register_seg、pin/map 日志 |
| URMA write 完成但 C500 数据错误 | UMMU 映射/设备 fence | UBVA/IOVA、token、cacheable flag |
| 断链后旧请求写入新 buffer | Region 生命周期漏洞 | generation、lease revoke、drain 顺序 |

## 15. 验收报告模板

### 15.1 环境

```text
Date/owner:
Path kind:
OS/kernel:
C500 model/driver/runtime/MAS:
RNIC/firmware/driver/rdma-core:
UB NIC/UMDK/UDMA/UMMU:
Storage/NVMe/filesystem:
PCIe tree/NUMA:
ACS/IOMMU:
Switch/MTU/PFC/ECN:
Git commit/build flags:
```

### 15.2 功能

```text
MetaX export API:
dma-buf fd obtained: PASS/FAIL/NA
RDMA MR registration: PASS/FAIL/NA
URMA Segment registration: PASS/FAIL/NA
Inbound device visibility: PASS/FAIL
C500 device checksum: PASS/FAIL
direct-required fallback count: 0/...
fault injection: PASS/FAIL
```

### 15.3 性能

```text
block/chunk/iodepth/jobs:
bandwidth/IOPS:
P50/P95/P99/P999:
client CPU/agent CPU:
registration latency/cache hit ratio:
client Host staging bytes:
agent Host staging bytes:
retry/error/CRC count:
```

### 15.4 最终判定

```text
SIMULATION / COMPAT / CLIENT_DIRECT_RDMA / END_TO_END_DIRECT_RDMA /
CLIENT_DIRECT_URMA / END_TO_END_DIRECT_URMA / BLOCKED

Evidence:
Blocking layer:
Next owner/action:
```

## 16. 推荐实际执行顺序

```text
第 1 天：E0 + E1，收集环境与接口证据
第 2 天：E3 本地 MAS 三路径
第 3 天：E4 Host RDMA
第 4～5 天：E5 C500 GDR 双向原语
第 6 天：E6 dma-buf MR 生命周期
第 7～10 天：E7 RDMA Agent/Client 正确性
第 11～12 天：E8 性能和故障注入
随后：E9 Host URMA → E10 C500 URMA
```

最关键的停机点是 E5/E6：如果拿不到 C500 显存导出或 RNIC 无法注册，就停止优化上层，直接把最小复现、错误码、BDF 拓扑、IOMMU/ACS 证据交给沐曦和 RNIC 厂商。
