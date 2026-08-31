# 沐曦 C500 D2RS 设计文档：RDMA → URMA 两阶段实现

> 文档状态：v0.2，面向 PoC 与厂商联合适配。
>
> 目标：实现“远端存储对象/文件 → 沐曦 C500 显存”的异步读取。第一阶段使用 RDMA，第二阶段保持上层不变，将数据传输替换为灵衢 UB/URMA。
>
> 配套代码：[MetaX D2RS Demo](../labs/metax_d2rs/README.md)；配套实验：[07-metax-d2rs-experiment-guide.md](07-metax-d2rs-experiment-guide.md)。

## 1. 结论先行

方案可行与否取决于一个硬门槛：

> C500 显存分配必须能被 RNIC/UB NIC 注册并执行 Peer DMA，同时沐曦运行时必须给出入站 DMA 完成后的显存可见性语义。

沐曦开发者论坛已经确认 C500 “支持 GDR”，mxFIO 也公开展示了通过 `libcufile` 测试 MACA MAS 的路径；但这些证据还不能回答以下工程问题：

1. C500 是否能把指定显存分配导出为 dma-buf；
2. 哪些 RNIC、驱动和拓扑组合支持该 dma-buf 的导入；
3. 入站 RDMA WRITE 完成后，C500 kernel 前是否需要 cache invalidate、stream event 或其他 fence；
4. GDR 接口是公开 SDK、MCCL 内部接口，还是需要厂商驱动适配。

因此项目分为三个判定结果，而不是简单的“支持/不支持”：

| 结果 | 数据路径 | 可交付名称 |
|---|---|---|
| `COMPAT` | 远端存储 → 客户端 Host pinned memory → C500 | 兼容路径，不称为 D2RS |
| `CLIENT_DIRECT` | 远端存储节点 DRAM → RNIC → C500，客户端无 Host bounce | 第一阶段 D2RS |
| `END_TO_END_DIRECT` | 远端 NVMe/块设备 → RNIC/UB NIC → C500，两端均无 Host staging | 增强型 D2RS |

第一阶段先实现 `CLIENT_DIRECT`。远端 Agent 使用 Host DRAM staging 是允许的，但报告必须如实标识，不能把它写成端到端零拷贝。

## 2. 目标与非目标

### 2.1 第一阶段目标

- 只读：`remote file/object range → C500 buffer`；
- C500 buffer 支持显式注册和注销；
- 4 KiB 对齐的大 I/O 走 RDMA 直达；
- 请求按可协商的 chunk 拆分，支持多请求并发和乱序完成；
- 具备 timeout、取消、CRC、端到端设备侧校验；
- 明确区分 `direct-required` 和允许回退的请求；
- 上报真实路径、注册命中率、Host staging 字节数、P50/P99 和 CPU 利用率。

### 2.2 暂不包含

- POSIX 透明拦截或 FUSE；
- 写路径、原子性和持久化事务；
- 多副本、分区重平衡和分布式缓存；
- 在没有厂商接口时伪造 C500 dma-buf/GDR API；
- 第一阶段直接修改 Linux、rdma-core、SPDK 或 UBS IO 主干。

## 3. 已知证据与未知项

| 项目 | 当前证据 | 结论 |
|---|---|---|
| C500 支持 GDR | [沐曦开发者论坛](https://developer.metax-tech.com/forum/t/mu-xi-c500shi-fou-zhi-chi-gpudirect-rdma/445/)回复“支持GDR” | 产品能力存在，接口形态待确认 |
| C500 存储直通 | [MetaX-MACA/mxFIO](https://github.com/MetaX-MACA/mxFIO) 说明 `cuda_io=cufile` 测试 MACA MAS | 可以先做本地 MAS 基线 |
| Linux dma-buf | [Linux 官方文档](https://docs.kernel.org/driver-api/dma-buf.html)定义跨驱动 DMA buffer 和 dma-fence | 推荐的跨驱动共享机制 |
| RDMA dma-buf MR | [rdma-core 官方 man page](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)提供 `ibv_reg_dmabuf_mr(pd, offset, len, iova, fd, access)` | Phase 1 首选注册入口 |
| UBS IO/NDS | [官方说明](https://gitcode.com/openeuler/ubs-io)覆盖 RDMA/UB 外置存储直通 NPU HBM及建链/内存注册协同 | 借鉴架构，不能假设 NPU 代码可用于 C500 |
| URMA 显存注册 | 公共 `urma_seg_cfg_t` 只有 `va/len/iova/token/flag`，没有 dma-buf fd | 需要验证/扩展 UDMA、UMMU 或 Provider |

未知项必须在实验报告中保持 `UNKNOWN`，直到相应的最小测试通过。

## 4. 总体架构

```text
┌───────────────────────────────────────────────────────────────────┐
│ AI Framework / mxFIO / d2rs_bench                                 │
├───────────────────────────────────────────────────────────────────┤
│ D2rsClient                                                        │
│  open / register / read_async / poll / cancel / unregister        │
├───────────────────────────────────────────────────────────────────┤
│ RequestPlanner        RegionRegistry        Completion/Fence       │
│ Flow/Slice/chunk      注册缓存与租约          路径与错误统计          │
├───────────────────────┬───────────────────────────────────────────┤
│ DeviceMemoryProvider  │ TransportBackend                           │
│ MetaXProvider         │ RdmaBackend        UrmaBackend             │
│ SimProvider           │ ibverbs            UMDK/URMA               │
├───────────────────────┴───────────────────────────────────────────┤
│ 控制面：认证 RPC，传输 Region 能力、对象范围、请求 ID、generation   │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────┐
│ Remote Storage Agent                                               │
│ RequestValidator → StorageBackend → staging/MR → TransportBackend   │
│                     pread/io_uring     RDMA WRITE / URMA WRITE       │
└───────────────────────────────────────────────────────────────────┘
```

### 4.1 为什么使用存储 Agent

自研 Agent 能控制以下关键步骤：

- 对象/文件到 LBA 或文件 offset 的解析；
- Host staging、SPDK buffer 或设备 P2P buffer 的选择；
- 目的 C500 Region 的访问窗口和租约；
- 分片、重试、CRC、限流和完成通知；
- RDMA 与 URMA 的传输替换。

直接从标准 NVMe-oF 开始会同时引入 initiator、target、块层和设备显存映射四个变量，不适合作为第一个可行性实验。

## 5. 核心接口

### 5.1 北向接口

建议先提供显式异步 API，不做 POSIX 透明替换：

```c
int d2rs_init(const struct d2rs_options *options);
int d2rs_open(const char *uri, struct d2rs_file **file);

int d2rs_register_device_buffer(
    int device_id, void *device_ptr, size_t length,
    struct d2rs_buffer **buffer);

int d2rs_read_async(
    struct d2rs_file *file,
    off_t file_offset,
    size_t length,
    struct d2rs_buffer *destination,
    size_t destination_offset,
    uint32_t flags,
    uint64_t *request_id);

int d2rs_poll(struct d2rs_completion *completions,
              uint32_t max_count, int timeout_ms);
int d2rs_cancel(uint64_t request_id);
int d2rs_unregister_device_buffer(struct d2rs_buffer *buffer);
```

重要语义：

- `D2RS_DIRECT_REQUIRED`：直达不可用时直接失败；
- `D2RS_ALLOW_COMPAT`：允许 Host pinned buffer + H2D 回退；
- `poll` 成功意味着 transport 已完成且 Provider 已执行 device-visible fence；
- `unregister` 只有在所有引用该 Region 的请求完成或取消后才能返回成功。

### 5.2 DeviceMemoryProvider

```text
allocate / import_application_buffer
export_dma_buf_or_vendor_handle
get_allocation_identity_and_generation
make_visible_to_device
release
```

`generation` 用于防止显存释放后地址复用造成旧 rkey/token 写入新对象。

### 5.3 TransportBackend

```text
query_caps
register_region
export_remote_descriptor
connect
submit_write_from_storage
poll
revoke_region
disconnect
```

公共层只保存 `region_id + generation + offset + length`。RDMA 地址/rkey 和 URMA Segment/Token 只能存在于 Transport 私有描述符中。

### 5.4 StorageBackend

```text
open(uri)
read_async(handle, offset, iov, callback)
cancel(request_id)
close(handle)
query_alignment_and_max_io
```

PoC 顺序：

1. `pread + O_DIRECT`；
2. `io_uring`；
3. SPDK bdev/blob/NVMe；
4. Ceph、对象存储或自研分布式存储插件。

## 6. 控制协议

### 6.1 请求描述

```text
ReadRequest
  protocol_version
  request_id
  tenant_id
  object_id / canonical path
  source_offset
  length
  destination_region_id
  destination_generation
  destination_offset
  chunk_size
  checksum_type
  direct_policy
  deadline
```

### 6.2 Region 能力描述

```text
CommonRegion
  region_id
  generation
  memory_type
  device_id
  length
  transport_type
  lease_id
  expires_at

RDMA private payload
  iova
  rkey
  qp/session identity

URMA private payload
  serialized urma_seg context
  token policy/value reference
  target Jetty/JFR identity
```

禁止在网络协议中发送 dma-buf fd。fd 只在 C500 所在节点用于 `ibv_reg_dmabuf_mr` 或本地 Provider 导入。

### 6.3 完成协议

MVP 可以在 Agent 的 RDMA WRITE CQE 后通过控制通道发送完成消息。优化阶段改用 WRITE WITH IMM 或等价通知。

```text
Completion
  request_id
  chunk_index
  bytes
  storage_status
  transport_status
  checksum
  path_kind
  server_staging_bytes
```

客户端收到所有 chunk 完成后执行 MetaX Provider 的可见性操作，然后才向应用报告成功。

## 7. Phase 1：基于 RDMA 的 D2RS

### 7.1 客户端初始化

1. 应用通过 MACA runtime 分配 C500 buffer；
2. MetaXProvider 导出 dma-buf fd、offset、allocation generation；
3. RdmaBackend 创建 context、PD、CQ、QP；
4. 调用：

```c
mr = ibv_reg_dmabuf_mr(
    pd,
    dmabuf_offset,
    length,
    iova,
    dmabuf_fd,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
```

5. 本地保存 `mr` 和 dma-buf fd；只把 `iova/rkey/len/lease` 发送给 Agent；
6. 注册缓存以 allocation identity、device、offset、length、generation、PD 为 key；
7. 注销顺序：停止新请求 → drain CQ → revoke lease → `ibv_dereg_mr` → close dma-buf fd → free C500 buffer。

`iova` 与 dma-buf offset 必须有相同 page offset。不能用普通 `ibv_reg_mr(device_ptr, len)` 代替，除非沐曦明确提供可被当前 verbs Provider 识别的 peer-memory VA 路径。

### 7.2 Agent 读取与推送

```text
RPC ReadRequest
    ↓
鉴权、路径白名单、Region lease/generation 校验
    ↓
StorageBackend 把 chunk 读入 Agent local MR
    ↓
ibv_post_send(IBV_WR_RDMA_WRITE, remote_addr=iova+offset, rkey)
    ↓
poll send CQ
    ↓
Completion RPC / WRITE WITH IMM
```

PoC 使用每个队列一组预注册 Host staging buffers，避免每个 I/O 注册 MR。大文件按 chunk pipeline：读盘 N+1 与传输 N 重叠。

### 7.3 第二级：远端存储设备 P2P

在 `CLIENT_DIRECT` 稳定后，才把 Agent Host staging 替换为：

- SPDK 分配的 DMA-safe buffer；或
- NVMe CMB/P2PDMA buffer；或
- 支持 device-to-device DMA 的 DPU/存储卡 buffer。

验收时必须分别统计：

- 客户端 Host staging bytes；
- Agent Host staging bytes；
- 存储设备 DMA bytes；
- RNIC DMA bytes。

只有两端 Host staging 都为 0，才标记 `END_TO_END_DIRECT`。

## 8. Phase 2：替换为 URMA

### 8.1 保持不变

- 北向文件/KV API；
- RequestPlanner、Flow/Slice；
- Region ID、generation、lease；
- StorageBackend；
- CRC、重试、超时、指标；
- 应用和远端对象语义。

### 8.2 verbs 与 URMA 映射

| RDMA | URMA |
|---|---|
| device/context | URMA device/context |
| PD/MR | UMMU protection + Segment |
| QP | Jetty 或 JFS/JFR |
| CQ | JFC/JFCE |
| `ibv_reg_dmabuf_mr` | 目标应为 C500-capable `urma_register_seg`/Provider 扩展 |
| `ibv_post_send(RDMA_WRITE)` | `urma_write` |
| `ibv_poll_cq` | `urma_poll_jfc` |
| addr/rkey | serialized segment + token |

### 8.3 当前不可跳过的缺口

公共 UMDK 的 `urma_seg_cfg_t` 是：

```text
va / len / token_id / token_value / flag / user_ctx / iova
```

没有 dma-buf fd。示例也是 `memalign()` 得到 Host VA 后调用 `urma_register_seg()`。因此“把 RDMA API 换成 URMA API”不足以实现 C500 D2RS。

需要按以下顺序定位改造点：

1. 验证 UDMA Provider 能否识别 C500 VA 或 vendor IOVA；
2. 若不能，在 Provider/驱动增加 device-memory import；
3. 将 dma-buf attachment/sg-table 或沐曦私有映射交给 UDMA；
4. 在 UMMU 建立地址翻译和权限表；
5. 让 `urma_register_seg` 返回可被远端 import 的 Segment；
6. 验证 `urma_write` 后的 C500 device visibility；
7. 验证 revoke/unregister 与显存释放的竞态。

可能需要修改 UDMA Provider/UMMU/沐曦驱动，但不应先修改 URMA 上层 API。只有 VA 模型确实无法承载 dma-buf 时，再设计 `register_seg_ex` 或 Provider 私有扩展。

## 9. 软件栈修改矩阵

| 软件栈 | Phase 1 RDMA | Phase 2 URMA | 所有者 |
|---|---|---|---|
| AI 框架/vLLM/推理服务 | 新增 D2RS loader/plugin；保留 POSIX fallback | 不变 | 应用团队 |
| D2rsClient/Core | 新建 | 不变 | 自研 |
| MetaXProvider | 对接 MACA/MAS/GDR；显存导出、generation、fence | 增加 URMA 可注册句柄 | 沐曦 + 自研 |
| 沐曦用户态 runtime | 有公开接口则不改；否则增加受支持的 export/sync API | 可能增加 UB 注册协同 | 沐曦 |
| 沐曦内核驱动 | 有 dma-buf exporter/GDR 接口则不改；否则必须增加 | 可能增加 UDMA/UMMU attachment | 沐曦 |
| Linux dma-buf/IOMMU | 优先只配置，不改主线 | 优先不改主线 | 平台团队 |
| rdma-core/libibverbs | 直接使用 `ibv_reg_dmabuf_mr`，不 fork | 被 UrmaBackend 替代 | 网络团队 |
| RNIC verbs Provider/驱动 | 支持 dma-buf importer 则不改；否则升级或适配 | 不适用 | NIC 厂商 |
| Remote Storage Agent | 新建 RDMA backend | 新增 URMA backend | 自研 |
| StorageBackend | 新建 pread/io_uring/SPDK adapter | 不变 | 存储团队 |
| UMDK/liburma | 不使用 | 原则上不改公共层 | 灵衢团队 |
| ubcore/uburma | 不使用 | Provider 能力够则不改 | 灵衢团队 |
| UDMA/UMMU Provider | 不使用 | 很可能增加 C500 device-memory 注册路径 | 灵衢 + 沐曦 |
| UBS IO/NDS | 仅作架构参考 | 可选接入，不作为 PoC 依赖 | 后续联合团队 |

## 10. 从 UBS IO 借鉴什么

本地可读的 BoostIO 代码展示了以下成熟模式：

1. SDK、网络、消息、缓存、Flow、磁盘、UFS、集群管理分层；
2. `bio-ctrl-` 与 `bio-data-` 两套连接；
3. `GetRequest/PutRequest` 携带 MR 信息，数据面执行单边 Get/Put；
4. 大 MR 预注册后按 Block 切分；
5. Flow/Slice 支持拆分、预取、CRC 和生命周期；
6. UFS 屏蔽 Local/Ceph/HDFS 差异；
7. partition version、网络重试、断链回调和统计。

对应源码：

- [整体架构](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/docs/zh/user_guide.md)
- [控制/数据通道与 HCOM 类型](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/src/net/net_common.h)
- [MR 池与单边操作](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/src/net/net_engine.h)
- [Get/Put 协议](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/src/message/message.h)
- [Flow/Slice](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/src/flow/flow.h)
- [后端存储接口](https://gitcode.com/openeuler/ubs-io/blob/master/ubsio-boostio/src/underfs/file_system.h)

不能照搬：

- 公共头文件直接 alias HCOM/RDMA 类型；
- wire message 中直接暴露 `uintptr_t + mrKey`；
- Host MR 分配与注册绑在一起；
- 固定 4 MiB 上限而不做 capability negotiation；
- 在基础直通未打通前引入 ZooKeeper、双副本和完整缓存策略。

本地 `ubsio-nds/NDS直通.txt` 为空，无法从公开源码验证 NPU HBM 注册和 UB 建链实现，相关部分只能作为产品方向参考。

## 11. 一致性与完成语义

一次读取成功必须同时满足：

```text
存储读取完成
  AND 传输 CQ/JFC 完成
  AND 所有 chunk 状态成功
  AND checksum 符合
  AND MetaX device-visible fence 完成
```

禁止仅凭 Agent 端 CQE 就让 C500 kernel 消费数据。

Phase 0/1 正确性实验必须在 C500 上运行 checksum kernel；CPU 把显存复制回来比对只能作为辅助。还要分别验证：

- RNIC → C500 写后，C500 kernel 是否立即可见；
- C500 → RNIC 读前是否需要 flush；
- 同一 buffer 的重叠写是否受支持；
- relaxed ordering 是否关闭；
- 多 stream、多 QP 和乱序 chunk 是否保持应用约定。

第一版禁止重叠写，禁止 relaxed ordering，一个 request 的 chunk 使用不重叠目的区间。

## 12. 可靠性与安全

### 12.1 读请求重试

读取是幂等操作，但重试必须复用 request ID，并检查 generation。旧请求晚到时不能写入已经复用的显存。

### 12.2 Region lease

- rkey/Token 只在短租约内有效；
- 每次导出绑定 tenant、connection、region、generation；
- Agent 只能访问 `base + allowed range`；
- 注销先 revoke，再 drain，再 deregister；
- 断链触发 lease 失效和 Region 隔离。

### 12.3 路径安全

Agent 不接收任意绝对路径。控制面传 object ID 或已规范化的相对路径，服务端做根目录限制、权限校验和长度检查。

## 13. 性能设计

初始推荐值是实验起点，不是固定 ABI：

| 参数 | 初值 | 扫描范围 |
|---|---:|---:|
| chunk | 4 MiB | 64 KiB、256 KiB、1/2/4/8/16 MiB |
| iodepth | 8 | 1、2、4、8、16、32、64 |
| Agent staging buffers | 2 × iodepth | 1×、2×、4× |
| QP/Jetty | 1 | 1、2、4、8 |
| Region cache | allocation lifetime | 0%、50%、90%+ 命中率 |

报告至少包含：

- 带宽、IOPS、P50/P95/P99/P999；
- 客户端和 Agent CPU%；
- C500、RNIC、PCIe、存储利用率；
- 注册次数、注册耗时和命中率；
- direct/compat 请求数和字节数；
- 两端 Host staging bytes；
- CRC 错误、timeout、retry、disconnect；
- NUMA、PCIe 拓扑、ACS/IOMMU 配置。

## 14. Demo 与正式实现的关系

当前 Demo 位于 `labs/metax_d2rs`：

```text
include/d2rs/d2rs.hpp
  DeviceMemoryProvider / TransportBackend / Region / Completion

src/d2rs.cpp
  模拟设备内存、注册缓存、Flow/Slice、异步 Agent、CRC

adapters/rdma_dmabuf_registration.cpp
  正确的 ibv_reg_dmabuf_mr 边界

adapters/urma_registration.cpp
  公共 UMDK VA 注册边界与 Segment 序列化

adapters/metax_provider_template.cpp
  厂商需要填充的 allocation/export/fence/release
```

模拟后端用于验证上层契约，输出始终是 `direct_data_path=false`。它不是 RDMA 性能 Demo。

正式实现建议增加：

```text
src/client/
src/agent/
src/control/
src/storage/pread_backend/
src/storage/io_uring_backend/
src/transport/rdma/
src/transport/urma/
src/device/metax/
bench/
tests/fault/
```

## 15. 里程碑与退出条件

| 里程碑 | 交付 | 通过条件 |
|---|---|---|
| M0 契约 Demo | 当前 sim core、文档、env probe | 本机自测 CRC/字节全对 |
| M1 本地 MAS 基线 | mxFIO cufile/posix 对比 | 明确本地直达是否可用及性能 |
| M2 GDR 原语 | NIC ↔ C500 microbenchmark | 两方向正确性、带宽和 cache 语义闭环 |
| M3 dma-buf MR | MetaXProvider + RDMA adapter | `ibv_reg_dmabuf_mr` 成功且可重复注册/注销 |
| M4 RDMA D2RS | Agent pread/io_uring + RDMA WRITE | 客户端 Host staging=0，C500 checksum 全对 |
| M5 工程化 | batch、注册缓存、故障注入、指标 | 24h 稳定性，无越界/旧 generation 写入 |
| M6 URMA Host VA | URMA Agent 与 Host segment | `urma_write/poll_jfc` 基线通过 |
| M7 URMA C500 | C500-capable UDMA/UMMU Provider | UB → C500，客户端 Host staging=0 |
| M8 应用接入 | vLLM/权重/KV Cache loader | 冷启动或 KV 加载收益可复现 |

如果 M2 或 M3 失败，Phase 1 暂停在 `COMPAT`，把错误码、拓扑和厂商接口缺口提交给沐曦/RNIC 厂商；不能通过修改上层 D2RS 代码绕过。

## 16. 当前待厂商确认清单

### 沐曦

1. C500 GDR 的公开头文件、库、sample 和支持矩阵；
2. 显存导出 dma-buf 或 peer-memory handle 的 API；
3. 支持的内核、IOMMU 模式、RNIC 和 PCIe 拓扑；
4. RNIC 写 C500 后的 device visibility/fence；
5. C500 读/写方向是否都支持，是否有 cache 限制；
6. MAS 与 GDR 是否共享同一显存注册机制；
7. dma-buf move-notify、释放和进程退出时的失效语义。

### RNIC/灵衢

1. verbs Provider 是否实现 dma-buf MR；
2. 是否能 import C500 exporter 返回的 attachment/sg-table；
3. URMA/UDMA Provider 如何注册非 Host 内存；
4. UMMU 是否能建立 C500 BAR/IOVA 映射和权限表；
5. Segment revoke、设备 reset、进程异常退出时如何清理；
6. UB NIC 与 C500 的推荐拓扑及 ACS/IOMMU 约束。
