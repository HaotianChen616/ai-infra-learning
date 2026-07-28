# 昇腾 910B4：Qwen3.6-27B-W8A8 vLLM 端到端同步等待分析手册

本文不再把 H2D/D2H 微基准当成主实验。目标是在**另一台已经具备可用
`vllm serve` 环境的现有容器内**，复现实质性的在线推理负载，回答：

> Qwen3.6-27B-W8A8 在昇腾 910B4 上以 vLLM-Ascend 服务时，
> Host 侧同步等待 API 花了多少时间；这些时间是 Python/框架开销、
> CANN Runtime、NPU/HCCL 前序任务、DMA/PCIe，还是 OS 调度造成的；
> 哪些优化有实际收益？

手册不创建镜像、不安装 CANN，也不重新搭建 vLLM。实验命令均从已经能成功
启动该模型的容器内执行。若已有启动命令与本文默认参数不同，以已验证可工作的
容器参数为准，并把差异写入实验记录。

配套工具：

- `labs/run_vllm_ascend_e2e_profile.sh`：启动服务、预热、压测、采集和解析；
- `labs/verify_ascend_w8a8_model.py`：检查本地模型的 W8A8 元数据；
- `labs/summarize_ascend_sync.py`：汇总 `trace_view.json` 中的同步 API Host wall；
- `labs/h2d_d2h_benchmark.py`：仅在 timeline 已指向搬运问题时做补充微基准。

## 1. 固定实验卡

先固定口径，避免每次采集都在改变问题：

| 项目 | 基准值 |
|---|---|
| 设备 | 昇腾 910B4，记录服务器产品形态、卡数、NUMA/HCCS/PCIe 拓扑 |
| 容器 | 已能正常执行 `vllm serve` 的现有 vLLM-Ascend 容器 |
| 模型 | `Eco-Tech/Qwen3.6-27B-w8a8` 或对应本地目录 |
| 量化 | ModelSlim W8A8，服务参数 `--quantization ascend` |
| dtype | `bfloat16`；这是非量化算子的计算类型，不代表权重未量化 |
| 并行 | TP=2、DP=1；显存或部署约束不同时记录实际值 |
| 输入/输出 | 每请求 1024 input tokens、128 output tokens |
| 并发 | 8 个并发请求、一次发出 8 个请求、`request-rate=inf` |
| 输出控制 | `--ignore-eos`，尽量保证每请求生成 128 token |
| Prefix Cache | 关闭，防止第二次相同长度请求改变 Prefill 工作量 |
| Speculative/MTP | 主实验关闭；作为单独 A/B，不混入基线 |
| Task Queue | `TASK_QUEUE_ENABLE=1` |
| Blocking | `ASCEND_LAUNCH_BLOCKING=0` |
| Profiler stack | 主实验 `false`，有明确调用栈问题时再单独开启 |

Qwen3.6-27B 是混合 Gated DeltaNet/全注意力模型，并接受过 MTP 训练。主实验
仍关闭 speculative decoding，因为 MTP 会改变每次调度步接受的 token 数，
使“128 输出 token 对应多少 decode step”不再稳定。

每份结果至少附带：日期、机器型号、CPU/NUMA、910B4 数量、CANN、驱动、
固件、PyTorch、`torch_npu`、vLLM、vLLM-Ascend、模型路径/版本、可见设备、
TP/DP、所有环境变量和完整服务命令。

## 2. 先统一“同步耗时”的定义

关注的典型 API 包括：

- `aclrtSynchronizeDevice`
- `aclrtSynchronizeStream`
- `aclrtSynchronizeEvent`
- 对应的 `*WithTimeout`
- timeline 中可能出现的低层 `rt*Synchronize*`
- `torch_npu.npu.synchronize()` 等框架包装层

这些接口的 Host wall time 可近似写成：

```text
T_sync_host_wall
  = T_framework/runtime_on_cpu
  + T_wait_prior_npu_work
  + T_wait_hccl
  + T_wait_copy_or_dma
  + T_blocked
  + T_runnable_but_not_scheduled
  + T_wakeup_and_return
```

必须区分：

1. **API wall/duration**：进入同步 API 到返回的墙钟时间。CANN API 统计和
   timeline 的 `dur` 主要给出这个值。
2. **on-CPU self time**：线程真正处于 Running 状态、在该函数及其不再展开的
   子路径里执行指令的时间。需要 CPU sampling 或线程状态佐证。
3. **等待时间**：API wall 减去 on-CPU 部分，可能在睡眠，也可能已经 Runnable
   但暂时没有得到 CPU。

因此，看到 `aclrtSynchronizeStream = 4 ms`，不能直接得出“CANN Runtime 消耗
了 4 ms CPU”。它可能只用了几十微秒 CPU，剩余时间在等待 NPU kernel、HCCL、
DMA 或 OS 唤醒。

同一个同步调用还可能同时出现在 framework、`aclrt`、`rt` 三层。三层是嵌套
关系，不能相加。配套汇总器按 `aclrt > rt > framework` 选择一个优先层作为
非重复口径，同时保留各层和各线程明细。

## 3. 容器与目录前提

进入已经可部署模型的容器，确认：

- 仓库目录已挂载或复制进容器；
- 模型目录可读；
- `artifacts/` 所在文件系统有足够空间；
- NPU 设备、驱动和 CANN 环境在容器内可见；
- `vllm serve`、`vllm bench serve`、`torch_npu` 均可用；
- profiling 输出目录可写，且最终能从容器复制到持久存储。

以下示例假设：

```bash
cd /workspace/ai-infra

export MODEL=/models/Qwen3.6-27B-w8a8
export MODEL_DIR=/models/Qwen3.6-27B-w8a8
export OUTPUT_ROOT=/data/profile/qwen36_27b_w8a8
export TP_SIZE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

路径按容器实际挂载修改。不要把 profiler 输出写入容器临时层后直接退出容器。

## 4. 预检：先证明“测的是预期模型和环境”

### 4.1 采集系统信息

在容器内执行：

```bash
bash labs/run_vllm_ascend_e2e_profile.sh system
```

输出位于 `${OUTPUT_ROOT}/system/`，包括 CPU、NUMA、NPU、vLLM、
PyTorch/`torch_npu` 版本与关键环境变量。若容器内看不到拓扑命令，额外在宿主机
采集 `npu-smi info`、`lscpu`、`numactl --hardware`，并和容器结果放在一起。

### 4.2 校验本地 W8A8 checkpoint

```bash
MODEL_DIR="${MODEL_DIR}" \
  bash labs/run_vllm_ascend_e2e_profile.sh verify-model
```

检查项包括：

- `config.json` 的模型类型；
- `quant_model_description.json` 是否存在 W8A8 描述；
- safetensors 权重和索引是否存在。

这只证明 checkpoint 元数据正确。还必须在服务日志中确认：

- 加载的是预期路径和 revision；
- `--quantization ascend` 生效；
- 没有因为不支持算子而整体退回非预期路径；
- 实际 TP/DP、可见 NPU 与预期一致。

某些 BF16 算子仍然存在是正常现象，不能凭一个 BF16 kernel 判断“W8A8
未生效”。

## 5. 先测 profiler-off 服务基线

Profiler 会引入额外事件记录、内存和文件写入，不能拿 profiler-on 的吞吐当
生产基线。

### 5.1 终端 A：使用现有容器启动服务

```bash
mkdir -p "${OUTPUT_ROOT}/logs"

MODEL="${MODEL}" TP_SIZE=2 \
  bash labs/run_vllm_ascend_e2e_profile.sh server-baseline \
  2>&1 | tee "${OUTPUT_ROOT}/logs/server_baseline.log"
```

脚本的默认关键参数为：

```text
--quantization ascend
--dtype bfloat16
--tensor-parallel-size 2
--data-parallel-size 1
--max-model-len 2048
--max-num-seqs 8
--max-num-batched-tokens 8192
--no-enable-prefix-caching
```

若现有可用启动命令还需要设备、模型实现、调度器或其他参数，直接追加在
`server-baseline` 后。脚本会把这些参数追加到 `vllm serve`。先确认健康检查和
一次真实请求成功，再进入压测。

### 5.2 终端 B：预热和正式基线

```bash
export MODEL="${MODEL}"
export OUTPUT_ROOT=/data/profile/qwen36_27b_w8a8

bash labs/run_vllm_ascend_e2e_profile.sh warmup

for run in 1 2 3 4 5; do
  BENCHMARK_LABEL="baseline_run${run}" \
    bash labs/run_vllm_ascend_e2e_profile.sh benchmark
done
```

记录五轮的中位数和 p95，不只保留最好的一轮。重点观察：

- request throughput；
- output token throughput；
- TTFT；
- TPOT/ITL；
- E2E latency；
- 五轮抖动。

完成后停止 baseline 服务，再启动 profiler 服务，避免端口和显存冲突。

## 6. Full：采集 1024→128 端到端 timeline

### 6.1 终端 A：启动可动态控制的 profiler 服务

每个实验使用独立 `PROFILE_LABEL`。Profiler 输出路径在服务启动时已固定，因此
换 full/control/A/B 时应重启服务。

```bash
export PROFILE_LABEL=full_bs8_1024_128

MODEL="${MODEL}" TP_SIZE=2 PROFILE_LABEL="${PROFILE_LABEL}" \
  bash labs/run_vllm_ascend_e2e_profile.sh server-profile \
  2>&1 | tee "${OUTPUT_ROOT}/logs/server_${PROFILE_LABEL}.log"
```

脚本给 vLLM 增加动态 profiler 配置：

```json
{
  "profiler": "torch",
  "torch_profiler_dir": ".../torch_profile/full_bs8_1024_128",
  "torch_profiler_with_stack": false,
  "torch_profiler_record_shapes": false,
  "torch_profiler_with_memory": false
}
```

关闭 stack、shape 和 memory 是为了降低采集扰动。若后续确实要定位 Python
调用栈，单独用 `TORCH_PROFILER_WITH_STACK=true` 重启并只采很短窗口。

### 6.2 终端 B：先预热，再只包围正式请求

```bash
export OUTPUT_ROOT=/data/profile/qwen36_27b_w8a8
export PROFILE_LABEL=full_bs8_1024_128
export INPUT_TOKENS=1024
export OUTPUT_TOKENS=128
export CONCURRENCY=8

bash labs/run_vllm_ascend_e2e_profile.sh warmup
bash labs/run_vllm_ascend_e2e_profile.sh profile
```

`vllm bench serve --profile` 会在正式请求前后调用服务的 `/start_profile` 与
`/stop_profile`。这样模型加载和大部分预热不会污染窗口。

请求配置为：

```text
random dataset
exact input length = 1024
exact requested output length = 128
num prompts = 8
max concurrency = 8
request rate = inf
ignore EOS
temperature = 0
```

等待终端 A 明确完成 stop/flush 后再停止服务。提前杀进程可能留下不完整 JSON。

### 6.3 解析 CANN profiler 输出

```bash
PROFILE_LABEL=full_bs8_1024_128 \
ANALYSIS_LABEL=full_bs8_1024_128 \
OUTPUT_TOKENS=128 \
  bash labs/run_vllm_ascend_e2e_profile.sh analyze
```

该动作对每个 `*_ascend_pt` 目录执行 `torch_npu.profiler.analyse()`，再从
`ASCEND_PROFILER_OUTPUT/trace_view.json` 汇总同步 API，结果写入：

```text
artifacts/vllm_ascend_e2e/
├── system/
├── logs/
├── results/
│   └── profile_full_bs8_1024_128.json
├── torch_profile/
│   └── full_bs8_1024_128/
│       └── ..._ascend_pt/
│           └── ASCEND_PROFILER_OUTPUT/
│               ├── trace_view.json
│               ├── api_statistic.csv
│               ├── kernel_details.csv
│               ├── operator_details.csv
│               ├── op_statistic.csv
│               └── step_trace_time.csv
└── analysis/
    └── full_bs8_1024_128/
        ├── ascend_sync_summary.json
        └── trace-files.txt
```

不同 CANN/`torch_npu` 版本的文件名可能有差异；以实际生成文件为准并保留原始
目录。

## 7. Control：采集 1024→1，分离 Prefill 和 Decode

Full 包含请求调度、tokenize、Prefill、128 个输出 token、Detokenize 和返回。
要估计持续 Decode 的同步成本，需要保持输入和并发不变，只把输出改为 1。

停止 Full 服务，然后重启：

```bash
export PROFILE_LABEL=prefill_control_bs8_1024_1
export OUTPUT_TOKENS=1

MODEL="${MODEL}" TP_SIZE=2 \
  bash labs/run_vllm_ascend_e2e_profile.sh server-profile \
  2>&1 | tee "${OUTPUT_ROOT}/logs/server_${PROFILE_LABEL}.log"
```

另一个终端执行：

```bash
export PROFILE_LABEL=prefill_control_bs8_1024_1
export OUTPUT_TOKENS=1

bash labs/run_vllm_ascend_e2e_profile.sh warmup
bash labs/run_vllm_ascend_e2e_profile.sh profile

ANALYSIS_LABEL=prefill_control_bs8_1024_1 \
  bash labs/run_vllm_ascend_e2e_profile.sh analyze
```

两次采集必须保持模型、TP、并发、输入、缓存设置和 profiler 配置一致。对同一
TP rank/worker、同一 API 层，近似：

```text
持续 Decode 同步 Host wall / batch step
  ≈ (Full_sync_wall - Control_sync_wall) / 127

持续 Decode 同步 Host wall / generated token
  ≈ (Full_sync_wall - Control_sync_wall) / (8 × 127)
```

这里的 127 是 Full 相比 1024→1 多出的生成迭代。该差分仍是近似值：
Continuous Batching、请求完成时刻和 hybrid model 的执行模式可能使各步不完全
同构。至少重复三次，报告中位数和范围。

TP rank 可能并行等待。**各 rank 的同步时长之和不是请求关键路径**。关键路径
优先看同一时段各 rank 的最大值及端到端 E2E/TPOT，再把 rank sum 作为 Host
资源消耗的辅助统计。

## 8. Timeline 用什么看、具体怎么看

首选 MindStudio Insight 打开 `trace_view.json`；快速共享也可使用 Perfetto UI。
`api_statistic.csv` 适合排序统计，不能代替 timeline 的因果关系。

推荐查看顺序：

1. 先用压测结果确定整个 8 请求窗口和 TTFT/TPOT。
2. 在 timeline 找一次 Prefill，再找稳态 Decode 的连续若干步。
3. 展开 API server/EngineCore/scheduler、worker/TP rank、Python/PTA、
   CANN Runtime、NPU kernel、HCCL 和 memcpy 轨道。
4. 搜索 `Synchronize`，定位
   `aclrtSynchronizeDevice/Stream/Event` 及低层 `rt*`。
5. 对每个长同步区间，纵向查看同一时间 NPU/HCCL/memcpy 正在做什么。
6. 若有 CPU thread state，检查线程是 Running、Sleeping 还是
   Runnable/Ready；没有则使用下一节的 CPU 采样补证。

典型判读：

| 长同步区间同时出现什么 | 优先假设 | 下一步验证 |
|---|---|---|
| 连续 NPU kernel | 等待前序 NPU 计算完成 | 看 kernel 类型、排队深度和 graph/task queue |
| HCCL 通信 | 等待 TP collective | 检查 rank 不平衡、HCCS/PCIe 拓扑、TP A/B |
| Memcpy/H2D/D2H | 等待 DMA/搬运 | 检查方向、大小、pinned/NUMA，再做 copy 微基准 |
| NPU 已完成，线程 Sleeping | Runtime 阻塞/唤醒尾部 | 对齐 API return、futex/调度事件 |
| NPU 已完成，线程 Runnable | CPU 争用或绑核问题 | 看 run-queue、CPU affinity、上下文切换 |
| NPU 空闲，Host 线程 Running | Python/PTA/CANN 下发或锁 | CPU flame graph、任务队列、graph mode |
| 多个 rank 同步尾部差异大 | 慢 rank/负载不均成为屏障 | 按 rank 对齐 kernel、HCCL 和 NUMA |

不要仅凭颜色判断。点开事件，确认：

- 完整 API 名称；
- `pid/tid` 与 TP rank；
- start/end/duration；
- args/correlation id；
- 是否存在外层 framework 和内层 `aclrt/rt` 的嵌套；
- 相同区间的 NPU、HCCL、Memcpy 活动。

Chrome trace 通常以微秒表示 `dur`。若某版本输出异常大/小，先手工比对一个事件
的时间轴刻度，再给汇总器传 `--duration-unit ns` 或 `ms`，不要盲目换算。

## 9. 如何得到真正的 CPU on-CPU self time

Ascend profiler 的 API duration 回答“Host 调用多久返回”，不是“函数实际烧了
多少 CPU”。要进一步拆解，在相同 workload 下做一轮短的 CPU sampling。

容器有权限、能看到服务 PID 且安装了 `perf` 时，可在请求窗口采集：

```bash
mkdir -p "${OUTPUT_ROOT}/cpu"

perf record -F 99 -g -p <vllm-worker-pid> \
  -o "${OUTPUT_ROOT}/cpu/perf.data" -- sleep 30

perf report --stdio \
  -i "${OUTPUT_ROOT}/cpu/perf.data" \
  > "${OUTPUT_ROOT}/cpu/perf-report.txt"
```

若要看阻塞与唤醒，优先在宿主机使用能看到容器线程的 PID，并在获准后采集
`sched_switch`/`sched_wakeup` 或等价 eBPF 数据。容器通常需要额外
`perf_event`、BPF、host PID namespace 权限；不要为了采集临时修改生产容器
权限。

分析口径：

```text
同步 API 内 on-CPU self
  = CPU samples 落在 aclrt/rt synchronize 路径的比例
    × 采样窗口内目标线程 on-CPU 时间

同步 API 内 off-CPU wait
  ≈ timeline API wall - 对齐区间内 on-CPU time
```

Sampling 是统计估计。99 Hz 对几十微秒函数分辨率有限：增加重复次数、拉长稳态
窗口，而不是直接把频率升得很高。Python `cProfile` 只看到 Python 调用关系，
无法区分 native blocking 和 OS 调度，不能单独作为结论。

建议同时保留：

- 每线程 CPU 利用率、上下文切换、run-queue；
- CPU flame graph 或 `perf report`；
- CPU affinity/NUMA；
- 与 timeline 相同的开始/结束时间标记。

## 10. 从 `api_statistic.csv` 提取同步 API

先确认表头，因为不同版本列名可能变化：

```bash
find "${PROFILE_DIR}" -path '*/ASCEND_PROFILER_OUTPUT/api_statistic.csv' \
  -exec head -n 5 {} \;
```

在表格中筛选包含 `Synchronize` 的 API，至少输出：

| rank/pid/tid | API | calls | total wall | avg | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|

配套 JSON 还提供按 API 和 `pid/tid` 的统计。CSV 与 trace 结果不一致时依次检查：

1. 是否统计了不同时间窗口；
2. 是否把 framework、`aclrt`、`rt` 嵌套层相加；
3. 是否把所有 TP rank 相加；
4. 是否混入模型加载/预热；
5. 是否使用了错误的时间单位；
6. 是否有某个 rank 的 trace 缺失或未 flush。

## 11. H2D/D2H、PCIe 和 DMA 何时才是主因

在模型已经驻留 NPU 的稳态 Decode 中，不应预设每个 token 都有大规模权重
H2D。只有 timeline 明确显示长同步与 memcpy 重叠时，才按以下链路追踪：

```text
Host tensor/元数据
  → Python/PTA/CANN 下发
  → Host 内存是否可 DMA
  → DMA descriptor / copy engine
  → PCIe/片内互联
  → Device buffer
  → 同步点等待 copy 完成
```

需要逐项确认：

- copy 方向与字节数；
- pageable/pinned 或额外 staging copy；
- Host 内存与目标 910B4 的 NUMA 距离；
- 是否是 KV offload、采样结果回传、日志/指标或请求数据；
- copy 与计算是否重叠；
- 同步是否过早破坏异步流水。

确认后才运行补充微基准：

```bash
python3 labs/h2d_d2h_benchmark.py \
  --backend npu \
  --sizes 4KiB,1MiB,16MiB,64MiB \
  --output-dir "${OUTPUT_ROOT}/copy_microbench"
```

微基准用于回答“链路能力/尺寸曲线是否异常”，不能替代端到端归因。

## 12. 优化 A/B 顺序

每个变量都应重启服务、重新预热，重复至少三次，并同时比较 profiler-off
TTFT/TPOT/吞吐与 profiler-on timeline。

### A. 先消除明显同步和 CPU 调度问题

1. 找到业务代码或框架插件中不必要的显式 device/stream synchronize；
2. 查看线程是否经常 Runnable 但得不到 CPU；
3. 检查 vLLM worker 与 NPU/NUMA 亲和性；
4. 核对当前 vLLM-Ascend 版本的 CPU binding 默认行为，不要盲目覆盖；
5. 减少高频日志、同步指标和请求热路径上的 Python 工作。

### B. Task Queue

基线保持：

```bash
TASK_QUEUE_ENABLE=1
ASCEND_LAUNCH_BLOCKING=0
```

诊断时可用 `TASK_QUEUE_ENABLE=0` 重启做一次 A/B。它会改变下发/同步行为，
可能更易观察，但通常不是生产优化结论。不要使用
`ASCEND_LAUNCH_BLOCKING=1` 的结果评估真实性能。

### C. Graph mode

按当前 vLLM-Ascend 版本支持的 graph 参数比较默认、full-decode graph 与 eager。
关注 NPU 空洞、Host launch gap、同步调用次数和 TPOT，而不只看 kernel 总时长。

### D. Async scheduling

若版本和模型支持，比较 async scheduling 开/关。它可能减少调度阻塞，也可能
改变 timeline 结构；必须保持 workload 与其他参数不变。

### E. TP=2 与 TP=4

只在模型内存或服务目标需要时比较。TP 增大可能减小单卡计算，但增加 HCCL
collective 和同步屏障。若 TP 改变，结果不能用于单独证明“CPU 优化有效”。

### F. MTP/speculative

先完成 MTP 关闭的主实验，再按当前官方 Qwen3.6 配置单独测试 MTP。启用后应
报告 accepted tokens、acceptance rate 和每 accepted token 成本，不能继续用
`输出 token 数 - 1` 当作 decode step 数。

### G. Profiler 选项

- `with_stack=false`：主性能采集；
- `with_stack=true`：短窗口定位调用栈；
- profiler-off：最终性能判断。

若 profiler-on/off 差异明显，缩短采集窗口、减少 stack/shape/memory 信息，
不要用重采集结果代表生产。

## 13. 结果表模板

### 13.1 端到端指标

| Run | Config | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 | output tok/s |
|---|---|---:|---:|---:|---:|
| baseline-1 | profiler off |  |  |  |  |
| baseline-2 | profiler off |  |  |  |  |
| full | profiler on |  |  |  |  |
| control | profiler on, output=1 |  | N/A |  |  |

### 13.2 同步 API

| Rank | API family | API | Calls | Total Host wall | p50 | p95 | p99 | Max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | aclrt |  |  |  |  |  |  |  |
| 1 | aclrt |  |  |  |  |  |  |  |

### 13.3 归因

| 长同步区间 | NPU/HCCL/Memcpy 重叠 | CPU 状态 | 结论 | 证据强度 |
|---|---|---|---|---|
|  |  | Running/Sleeping/Runnable |  | 强/中/弱 |

结论建议写成：

> 在固定的 1024→128、并发 8、TP=2 配置中，rank 0 的
> `aclrtSynchronizeStream` p95 为 X；其中 Y% 区间与 HCCL 重叠，
> Z% 在 NPU 已完成后仍处于 Runnable。关闭非必要 CPU 争用后，
> profiler-off TPOT 中位数从 A 降到 B，重复三轮均成立。

不要只写“同步很慢”或“应该是 PCIe”。

## 14. 应上传/归档哪些数据

每个 run 独立目录，至少保留：

1. `${OUTPUT_ROOT}/system/` 全部文件；
2. 模型 W8A8 验证 JSON；
3. 完整 server log 和完整启动命令；
4. `vllm bench serve` 的 result JSON；
5. 每个 `*_ascend_pt` 原始目录；
6. `ASCEND_PROFILER_OUTPUT/trace_view.json`；
7. `api_statistic.csv`、`kernel_details.csv`、`operator_details.csv`、
   `op_statistic.csv`、`step_trace_time.csv`；
8. HCCL/communication 文件（如果生成）；
9. `ascend_sync_summary.json` 与 `trace-files.txt`；
10. CPU sampling、线程状态和 affinity 数据（如果采集）；
11. profiler-off 五轮基线；
12. 实验 README：机器、版本、时间、变量、异常和已知缺失。

上传前检查日志和路径是否包含访问令牌、内部地址或敏感请求文本。模型权重本身
通常不需要上传。

可将结果目录压缩后从容器复制到持久存储；务必先停止 profiling 并确认 JSON
可被解析。若数据量过大，最小审阅包为：

```text
system/
logs/
results/
analysis/
每个 run 的 trace_view.json
每个 run 的 api_statistic.csv
每个 run 的 kernel_details.csv
实验 README
```

但要做完整二次分析，仍建议保留整个 `*_ascend_pt`。

## 15. 完成标准

实验只有同时满足以下条件才算完成：

- checkpoint 元数据和服务日志均证明 W8A8 路径；
- profiler-off 基线稳定；
- Full 和 Control 使用独立目录、相同核心配置；
- timeline 同时覆盖 Host API、NPU kernel、HCCL 和 memcpy；
- 同步 API 没有跨嵌套层或跨 TP rank错误求和；
- Host wall 与 on-CPU self 明确分开；
- 至少一个优化 A/B 用 profiler-off 指标验证；
- 原始 trace、统计表、日志、版本和实验配置可复查。

## 16. 官方资料

- [vLLM-Ascend：Qwen3.5-27B / Qwen3.6-27B 部署教程](https://docs.vllm.ai/projects/ascend/en/v0.24.0rc/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [vLLM-Ascend：Service Profiling Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [vLLM-Ascend：Supported Models](https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_models.html)
- [vLLM-Ascend：CPU Binding](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/cpu_binding.html)
- [vLLM-Ascend：Graph Mode](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html)
- [vLLM-Ascend：Performance Benchmark](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/performance_benchmark.html)
- [CANN Runtime：Synchronization Management](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0020.html)
- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
