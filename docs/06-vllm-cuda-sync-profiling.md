# A100 INT8 W8A8 vLLM 推理中的 CUDA 同步等待完整手册

本手册面向真实 vLLM 在线推理，不把纯 H2D/D2H 微基准当作最终结论。实验固定为：

```text
模型：Qwen3-32B
量化：INT8 W8A8，LLM Compressor / compressed-tensors
权重：INT8；激活：INT8；非 FP8、非 W8A16
硬件：NVIDIA A100 / SM80
服务：vLLM OpenAI-compatible server
输入：每请求固定 1024 token
输出：每请求固定 128 token
负载：8 个请求同时到达，最大并发 8
采样：ignore_eos，确保每请求生成 128 token
目标：cudaDeviceSynchronize / cudaEventSynchronize /
      cudaStreamSynchronize 等 Host API 的等待时间与优化空间
```

实验入口：

```text
labs/run_vllm_cuda_sync_profile.sh
labs/verify_int8_w8a8_config.py
labs/summarize_cuda_sync.py
labs/attribute_sync.py            # 情况 A/D/E: 长同步窗口与 GPU kernel/memcpy 重叠
labs/attribute_cpu_state.py       # capstone: GPU overlap × on-CPU% → 统一情况 A/B/C/D/E
labs/attribute_cpu_state2.py      # 第13节 T_sync = T_running + T_blocked 的 Running/Blocked 拆分
labs/bench_sync_sdk.py            # OpenAI-SDK 压测, 绕开 vllm bench serve tokenizer 初始化
```

建议首先在 A100 80GB 上使用 TP1，隔离单卡 CUDA Runtime/Driver、kernel 和 Host
调度因素。A100 40GB 若因权重、workspace、CUDA Graph 和 KV cache 无法容纳模型，
可使用 TP2，但 NCCL 会成为同步等待的额外变量；正式结论必须分别报告 TP1/TP2。

## 1. 先纠正指标语义

PyTorch Profiler 或 Nsight Systems 中同步 API 的 CPU duration，首先应称为
**Host API wall time**：

```text
T_sync_api_wall = API return timestamp - API enter timestamp
```

`Self CPU time` 的 `Self` 表示没有计入嵌套 profiler event，不表示这段时间
CPU 一直在执行指令。一次 3 ms 的 `cudaEventSynchronize` 可能包含：

```text
T_sync_api_wall
= T_runtime_and_driver_on_cpu
 + T_wait_for_prior_gpu_kernel
 + T_wait_for_memcpy_or_collective
 + T_thread_sleep
 + T_runnable_but_not_scheduled
 + T_wakeup_and_return
```

因此不能写成：

```text
cudaEventSynchronize Self CPU = 3 ms
⇒ CUDA Runtime 消耗了 3 ms CPU 算力
```

正确解释是：

```text
调用线程有 3 ms 没有离开这个 API
```

再用 GPU timeline 和 CPU thread state 拆解这 3 ms。

## 2. 为什么纯 64 MiB copy 实验不够

[`04-h2d-d2h-profiling.md`](04-h2d-d2h-profiling.md) 中的 copy
微基准仍有价值，但它只是：

- 校准 pinned/pageable 和 PCIe 搬运量级；
- 验证 profiler、Event 和 NVTX 是否工作；
- 建立 `host_api_ms`、`device_copy_ms`、`pipeline_ms` 的区别。

它不能回答真实 vLLM 同步 API 在等待什么。Qwen3-32B-INT8-W8A8 推理中的
`cudaEventSynchronize` 更可能等待：

- Prefill/Decode CUDA Graph 中的 kernel；
- attention、GEMM、W8A8 quant/dequant kernel；
- Tensor Parallel 场景的 NCCL collective；
- sampling 或 output 路径的 D2H；
- allocator、CUDA Graph replay 或跨 stream Event；
- Host 线程的睡眠、唤醒和 OS 调度。

只有 timeline 上同步 API 前方确实存在 `Memcpy HtoD/DtoH`，才能把该次等待的一部分
归因到 PCIe/DMA。同步 API 名称本身不能证明 PCIe 是根因。

## 3. INT8 W8A8 模型验收

本手册仅接受序列化的 `compressed-tensors` INT8 W8A8 checkpoint。模型
`config.json` 至少应满足：

```json
{
  "model_type": "qwen3",
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 8,
          "type": "int"
        },
        "input_activations": {
          "num_bits": 8,
          "type": "int"
        }
      }
    }
  }
}
```

常见方案是 weight per-channel、activation dynamic per-token；`lm_head` 保持
BF16/FP16 是允许的。`torch_dtype=bfloat16` 只描述未量化算子和计算接口，不能据此
否定 W8A8。以下任一情况都不属于本实验：

- `format=float-quantized` 或 weight/activation 的 `type=float`：FP8 W8A8；
- `input_activations=null`：W8A16/weight-only；
- 权重 `num_bits=4`：W4A16/W4A8；
- 只有文件名写着 W8A8，但 `quantization_config` 不符合上述元数据。

本地 checkpoint 执行：

```bash
MODEL=/models/Qwen3-32B-INT8-W8A8 \
bash labs/run_vllm_cuda_sync_profile.sh verify
```

使用 Hugging Face model ID 时，先下载并固定 revision，再将本地 `config.json`
传给校验器：

```bash
MODEL=organization/Qwen3-32B-INT8-W8A8 \
CONFIG_JSON=/models/Qwen3-32B-INT8-W8A8/config.json \
bash labs/run_vllm_cuda_sync_profile.sh verify
```

校验通过只证明 checkpoint 元数据正确。服务启动日志仍需出现
`compressed-tensors`/INT8 对应实现；最终还应在 Nsight kernel 名称和算子范围中
确认没有退回 BF16 GEMM。checkpoint 来自第三方时，还要固定 commit revision、
记录哈希并自行验证精度。

## 4. 固定 BS8 的含义

vLLM 是 continuous batching 系统，“BS8”不能只写一个 batch size 参数。本实验定义：

```text
--num-prompts 8
--max-concurrency 8
--request-rate inf
server --max-num-seqs 8
server --max-num-batched-tokens 8192
```

这表示八个请求尽可能同时到达，服务端最多保留八条 active sequence，并允许一次
调度容纳 `8 × 1024` 个 prefill token。

实际 batch 仍需从 vLLM scheduler/NVTX 和请求 timeline 验证。请求在网络、tokenizer
和 EngineCore 队列中的到达可能有微小偏差，不能只根据命令行断言每个 engine step
始终恰好为 8。

固定输出必须使用：

```text
--random-output-len 128
--random-range-ratio 0
--ignore-eos
```

否则部分请求提前遇到 EOS，后段 decode batch 会从 8 逐步下降，无法比较每步同步开销。

## 5. 采集设计

使用两种 profiler，各自回答不同问题：

| 工具 | 用途 | 不用于 |
|---|---|---|
| Nsight Systems | CUDA Runtime/Driver API、GPU kernel/memcpy/NCCL、CPU thread state、NVTX | PyTorch Python stack 的完整语义 |
| PyTorch Profiler | operator、shape、Python/ATen stack、Self CPU 聚合 | 精确拆 OS off-CPU 与驱动等待 |

主结论以 Nsight Systems 为准。vLLM 官方也建议性能敏感的 profiling 优先使用
Nsight Systems，而 PyTorch Profiler 用于需要 shape/stack 的高开销诊断。

### 动态采集的原因

不能从模型加载开始全程采集。Qwen3-32B 初始化和 CUDA Graph capture 会产生巨大
trace，并污染同步 API 汇总。

本实验使用：

```text
--capture-range=cudaProfilerApi
--capture-range-end=repeat
server --profiler-config.profiler cuda
client --profile
```

客户端 `--profile` 会调用 vLLM 的 `/start_profile` 和 `/stop_profile`，只保存请求窗口。
`repeat` 允许同一个已 warmup 的 server 连续采多组 A/B。

## 6. 环境准备

采集端必须是装有 A100 的 Linux 服务器；macOS 只能安装 Nsight Systems GUI
查看 `.nsys-rep`，不能代替服务器侧采集。建议为 vLLM 使用独立环境，并按照
官方 GPU 安装文档选择与 CUDA/PyTorch 匹配的 wheel，不要混用多个 CUDA wheel。

最低检查：

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi -q -d PCI
nsys --version
nsys status -e
vllm --version
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

验收条件：

- GPU 名称为 A100，compute capability 为 `(8, 0)`；
- 驱动、PyTorch CUDA 和 vLLM wheel 互相兼容；
- `nsys status -e` 没有阻止 CUDA trace/context-switch 的关键错误；
- `nvidia-smi topo -m` 已保存，可识别 GPU 对应 CPU NUMA node；
- profiling 前没有其他进程抢占目标 GPU；
- 不启用 `CUDA_LAUNCH_BLOCKING=1`；
- 不启用 CPU weight offload、KV offload 或会额外引入 PCIe 搬运的功能。

版本不要只写“latest”。正式结果必须保存 `vllm --version`、PyTorch/CUDA、driver、
Nsight Systems 和 checkpoint revision。跨版本比较时一次只变更一个组件。

## 7. 启动服务器

先校验 checkpoint：

```bash
MODEL=/models/Qwen3-32B-INT8-W8A8 \
bash labs/run_vllm_cuda_sync_profile.sh verify
```

终端 A：

```bash
MODEL=/models/Qwen3-32B-INT8-W8A8 \
SERVED_MODEL_NAME=qwen3-32b-int8-w8a8 \
TP_SIZE=1 \
bash labs/run_vllm_cuda_sync_profile.sh server
```

脚本固定传入：

```text
--quantization compressed-tensors
--dtype bfloat16
--max-model-len 2048
--max-num-seqs 8
--max-num-batched-tokens 8192
```

如果 TP1 在 A100 40GB 上 OOM，使用 `TP_SIZE=2` 重新启动，不要在同一条结果中混合
TP1 和 TP2。TP2 必须补充 NCCL 轨道，并按 rank 分析同步 API；跨 rank 的 Host API
duration 不能直接相加当请求关键路径。

脚本对 server 使用：

```text
VLLM_WORKER_MULTIPROC_METHOD=spawn
--trace-fork-before-exec=true
--cuda-graph-trace=node
--sample=process-tree
--cpuctxsw=process-tree
--cudabacktrace=sync:100000
--cuda-event-trace=true
```

它能覆盖 vLLM EngineCore/worker 子进程、CUDA Graph node、长同步调用回栈和 CPU
调度状态。`cuda-event-trace`、node-level CUDA Graph 和 backtrace 都会增加开销，
所以 trace 用于归因，最终 TTFT/TPOT 数值需回到无 profiler 基准。

服务启动后检查日志：

- 明确识别 `compressed-tensors` INT8 W8A8；
- 没有 fallback、unsupported quantization 或重新在线量化警告；
- 没有 weight/KV CPU offload；
- CUDA Graph capture 能完成；
- `/v1/models` 和 health/ready check 正常。

如果无法证明 runtime 使用 INT8 路径，本次 trace 标记为无效，不能按“INT8 W8A8”
发布结论。

## 8. Warmup

终端 B：

```bash
bash labs/run_vllm_cuda_sync_profile.sh warmup
```

默认发送 16 个请求、并发 8，让以下一次性行为离开正式窗口：

- 权重页首次访问；
- Triton/torch.compile 编译；
- CUDA Graph capture；
- allocator 扩容；
- KV cache 首次触碰；
- tokenizer、HTTP connection 和 Python module warmup。

如果 warmup 日志仍然出现编译或 graph capture，应继续 warmup，不能立即正式采集。

## 9. 采集 1024→128、并发 8

终端 B：

```bash
PROFILE_LABEL=full_bs8_1024_128 \
bash labs/run_vllm_cuda_sync_profile.sh profile
```

客户端固定：

```text
8 requests
1024 input tokens/request
128 output tokens/request
1024 generated tokens in total
temperature=0
ignore_eos=true
request_rate=inf
```

命令返回后，确认终端 A 已完成 profile flush，再用 `Ctrl-C` 停止 server/`nsys`。
`.nsys-rep` 只有在 finalize 完成后才可复制。

典型产物：

```text
artifacts/vllm_cuda_sync/
├── nsys/
│   └── qwen3_int8_sync*.nsys-rep
├── results/
│   ├── warmup.json
│   ├── profile_full_bs8_1024_128.json
│   └── *.html
└── logs/
    ├── nsys-version.txt
    ├── vllm-version.txt
    ├── int8-w8a8-verification.json
    ├── nvidia-smi-q.txt
    ├── nvidia-smi-gpus.csv
    └── nvidia-smi-topo.txt
```

## 10. 用 1024→1 分离 Prefill

完整 1024→128 trace 同时包含 Prefill 和约 128 次 Decode step。为了估算 Decode
同步开销，再采一组同服务对照：

```bash
OUTPUT_TOKENS=1 \
PROFILE_LABEL=prefill_control_bs8_1024_1 \
bash labs/run_vllm_cuda_sync_profile.sh profile
```

近似拆分：

```text
T_decode_sync_total
≈ T_sync(1024→128) - T_sync(1024→1)

T_decode_sync_per_step
≈ T_decode_sync_total / 127
```

这不是数学上的严格消元，因为 output=1 与 output=128 可能触发不同 graph bucket、
allocator 和 scheduler 行为。必须同时核对两个 timeline 的 Prefill 区间是否相似。

## 11. 生成 CUDA 同步 API 汇总

对每个 `.nsys-rep`：

```bash
ANALYSIS_LABEL=full_bs8_1024_128 \
bash labs/run_vllm_cuda_sync_profile.sh stats \
  artifacts/vllm_cuda_sync/nsys/qwen3_int8_sync1.nsys-rep
```

脚本调用：

```text
nsys stats / cuda_api_sum
nsys stats / cuda_api_trace
labs/summarize_cuda_sync.py
```

输出：

```text
analysis/full_bs8_1024_128/
├── summary_cuda_api_sum.csv
├── trace_cuda_api_trace.csv
└── cuda_sync_summary.json
```

汇总器只统计可能阻塞 Host 的 API：

```text
cudaDeviceSynchronize
cudaEventSynchronize
cudaStreamSynchronize
cudaThreadSynchronize
cuCtxSynchronize
cuEventSynchronize
cuStreamSynchronize
```

它报告：

- 调用次数；
- Host wall total；
- mean、p50、p95、p99、max；
- 占所有 CUDA API duration 求和的比例；
- 按 PID/TID/thread name 分组；
- 按假设的 128 decode steps 和 1024 generated tokens 归一化。

注意：Tensor Parallel 多 worker 的同步 API 可能同时等待。跨 PID 直接求和可能大于
请求 wall time，所以必须同时保留 per-rank/per-thread 结果。

### PyTorch Profiler 交叉验证

Nsight Systems 是主采集。若需要把长同步调用映射回 PyTorch operator/Python stack，
另启一轮服务，不要与 `nsys` 同时采：

```bash
MODEL=/models/Qwen3-32B-INT8-W8A8

vllm serve "${MODEL}" \
  --served-model-name qwen3-32b-int8-w8a8 \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --profiler-config \
  '{"profiler":"torch","torch_profiler_dir":"./artifacts/vllm_cuda_sync/torch","torch_profiler_with_stack":true,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false}'
```

然后仍使用相同的 `warmup` 和 `profile` client。PyTorch trace 用 Perfetto 打开，搜索
`cudaDeviceSynchronize`、`cudaEventSynchronize`、`cudaStreamSynchronize`，记录
Self CPU total/average/calls 和上层 operator/stack。

PyTorch Profiler 的 Self CPU 用于回答“谁调用”；Nsight 的 CUDA API、GPU correlation
和 thread state 用于回答“在等什么”。两次采集的请求参数必须一致，但数值不能直接
逐微秒对齐，因为 profiler 开销和进程启动已经不同。

## 12. Nsight Systems timeline 怎么看

打开 `.nsys-rep` 后，先定位 `cuda_api_sum` 中 total 或 p99 最大的同步 API，再从
Events View 双击一条长调用跳到 timeline。

需要固定五类轨道：

```text
vLLM frontend / EngineCore / worker CPU thread
CUDA Runtime and Driver API
CPU Thread State / CPU core
CUDA Graph / kernel / NCCL
Memcpy HtoD / DtoH
```

对每条长同步 API 画窗口：

```text
API enter
│
├─ CPU Running：Runtime/Driver 或框架代码
├─ CPU Blocked/Sleeping：等待完成通知
├─ CPU Ready 但未运行：OS 调度延迟
├─ GPU Kernel/NCCL/Memcpy：实际前序 Device 工作
└─ wakeup → CPU Running → API return
```

### 情况 A：同步窗口覆盖 GPU kernel/NCCL

```text
CPU: [cudaEventSynchronize======================]
GPU:      [GEMM][Attention][NCCL][Graph nodes]
```

结论：主要是 GPU backlog/依赖等待。API 本身不是根因，删除同步可能导致错误。

下一步：

- 检查是否能用 CUDA Graph；
- W8A8 kernel 是否命中预期实现；
- TP collective 是否过长；
- batch/graph bucket 是否抖动；
- 是否存在不必要的 Host read 导致过早等待。

### 情况 B：GPU 已完成，线程仍 Blocked

```text
GPU completion ──────── API return
CPU thread:       Blocked
```

结论：检查 driver completion、futex/condition wait、唤醒路径和 profiler correlation。

下一步：

- 看 OSRT 和 blocked-state backtrace；
- 检查 CUDA/driver 版本；
- 对比 Event blocking/spin 行为；
- 检查虚拟化和容器限制。

### 情况 C：GPU 已完成，线程 Ready 但没上 CPU

```text
GPU completion ──────── thread scheduled ─ API return
CPU state:       Ready/off-CPU
```

结论：CPU scheduling 是尾延迟贡献者。

下一步：

- worker 绑 NPU/GPU 本地 NUMA CPU；
- 避免与 NIC/NVMe IRQ、tokenizer、HTTP worker 同核；
- 检查 cgroup quota、CPU migration 和超卖；
- A/B `taskset`/NUMA policy，而不是直接改 GPU 配置。

### 情况 D：同步窗口覆盖 Memcpy

```text
CPU sync: [================================]
GPU:          [Memcpy HtoD / DtoH]
```

只有这种情况才继续查：

- pageable/pinned；
- PCIe link speed/width；
- NUMA；
- D2H 后立即 `.item()`/`.cpu()`；
- KV/weight offload；
- copy stream 与计算 stream 依赖。

### 情况 E：GPU 空闲且 CPU Running

同步 API 内没有 GPU 工作，CPU 轨道持续 Running：

- Runtime/Driver overhead；
- 锁竞争；
- allocator/context 操作；
- profiler 注入；
- 异常错误检查路径。

用 `--cudabacktrace=sync:100000` 的调用栈定位源代码，不要只根据 API 名称猜。

### 自动化归因脚本与实测发现

上面五种情况若逐条在 GUI 里画窗口，样本一多就不可重复。`labs/` 提供三个直接
读 Nsight SQLite 的归因脚本，把情况 A/B/C/D/E 变成可复现的 JSON：

| 脚本 | 输入 | 输出 | 回答 |
|---|---|---|---|
| `attribute_sync.py` | `*.sqlite`（只需 CUDA 轨道） | `analysis/bs8_1024_128/long_sync_attribution.json` | 情况 A/D/E：长同步窗口是否覆盖 kernel/memcpy |
| `attribute_cpu_state2.py` | `*.sqlite`（需 `SCHED_EVENTS`） | `analysis/pinned_cpu_running_blocked.json` | 第13节 `T_sync = T_running + T_blocked` 拆分 |
| `attribute_cpu_state.py` | `*.sqlite`（需 CUDA + `SCHED_EVENTS`） | `analysis/pinned_long_sync_combined_attribution.json` | capstone：GPU overlap × on-CPU% → 统一情况 |

`attribute_cpu_state.py` 是 capstone，交叉两个维度给出统一判定，并把情况 A 拆成
两个性质不同的子类：

```text
A_gpu_blocked   : GPU kernel 在跑, 调用线程基本 off-CPU (教科书式 futex 阻塞等待)
A_gpu_spinning  : GPU kernel 在跑, 调用线程基本 on-CPU  (驱动/运行时 poll + 短睡)
D_memcpy        : 只覆盖 memcpy
E_no_gpu_running: 无 GPU 工作, 线程 on-CPU (Runtime/Driver overhead)
BC_no_gpu_off   : 无 GPU 工作, 线程 off-CPU (OS 调度/唤醒尾延迟)
```

这个 A 的二分是必要的：`cudaEventSynchronize` 名字相同，"线程阻塞等 GPU" 和
"线程在 CPU 上 poll 同时 GPU 在跑" 对 CPU 优化空间结论完全相反。前者 CPU 无可优化
（等 GPU/kernel），后者同步窗口里的 task-clock/cycles 是真实 CPU 消耗，归因对象是
驱动完成路径或 vLLM 等待策略。

方法学注意：`SCHED_EVENTS.threadState` 字段只有同时开 `--sample`（CPU 采样）才会
被填充；只开 `--cpuctxsw` 的 capture 里 `threadState` 全为 `0 (Unknown)`，无法用它
区分情况 A/B/C。`isSchedIn` 切上/切出边沿重建 Running 时段在仅有 context-switch
数据时仍有效，因此 capstone 用 on-CPU% 作为 A/E 与 B/C 的判据。若需要 B 与 C 的
精确区分（Ready vs Blocked），必须补一轮开 `--sample=process-tree` 的采集。

运行（DB 路径在脚本顶部，按实际产物修改）：

```bash
python3 labs/attribute_sync.py
python3 labs/attribute_cpu_state2.py
python3 labs/attribute_cpu_state.py
```

可用 capture 上的实测结果（`qwen3_awq_pinned.sqlite`，含 CUDA + `SCHED_EVENTS`，
339 个 >5ms 长同步）：

```text
A_gpu_spinning   339  (100%)
A_gpu_blocked      0
D_memcpy           0
E_no_gpu_running   0
BC_no_gpu_off      0

长同步窗口总墙钟: 48485 ms
调用线程 on-CPU:  48417 ms  (99.9%)
```

即：每一条长同步都同时覆盖 GPU kernel，且调用线程几乎全程 on-CPU。这不是教科书
"线程在 futex 上长睡等 GPU" 的情况 A，而是 `A_gpu_spinning`——GPU 在跑、线程也在
CPU 上 poll（伴随周期性 ~20–40ms 的极短暂 deschedule，off-CPU 仅 0.14%）。对
CPU 优化而言，这意味着长同步窗口里的 task-clock/cycles 是真实消耗，不能当成
"纯等 GPU、CPU 无可优化" 而跳过；但它消耗在驱动完成路径上，而非 vLLM Python
调度器，所以优化对象是 CUDA Graph 覆盖、kernel/NCCL 时长、驱动版本与等待策略
（见 [`07-vllm-cpu-selftime-experiments.md`](07-vllm-cpu-selftime-experiments.md)
的 wait policy 实验），而不是 NumPy/Python 热点。

> 上述数字来自一次 `qwen3_awq` capture，用于验证脚本可用性；本手册的 INT8 W8A8
> 正式结论必须用同套脚本跑 §9 的 `qwen3_int8_sync*.sqlite`，不能把 AWQ 的归因
> 比例当成 W8A8 结论。脚本本身与量化方式无关，对任何符合 Nsight schema 的
> `.sqlite` 通用。

## 13. 核心指标

正式报告至少包含：

| 指标 | 意义 |
|---|---|
| sync API calls / decode step | 每步发生多少次 Host blocking boundary |
| sync wall total / step | 每个 engine step 被同步 API 包住的 Host wall time |
| p50/p95/p99/max | 是否存在同步尾延迟 |
| Running time inside sync | Runtime/Driver 真正占用 CPU 的近似上限 |
| Blocked time inside sync | GPU/driver completion 等待 |
| Ready off-CPU time | OS 调度和唤醒延迟 |
| GPU busy overlap | 等待 kernel/NCCL/memcpy 的证据 |
| TPOT/ITL correlation | 同步长尾是否进入用户可见 decode 延迟 |

不要直接使用 `cuda_api_sum` 的 `Time %` 当应用 CPU 占比。该百分比是某 API duration
占**所有列出的 CUDA API duration 求和**的比例，不是应用 wall time，也不是 CPU
利用率。

如果要报告“真正消耗的 CPU”，应将同步窗口进一步拆成：

```text
T_sync_host_wall
= T_running_on_cpu
 + T_blocked
 + T_ready_off_cpu
```

`T_running_on_cpu` 才接近 Runtime/Driver 实际占用 CPU 的上限。Nsight thread state
负责拆 Running/Blocked/Ready；Linux `perf sched` 可作为调度延迟的独立交叉验证。
硬中断/softirq 只有在 CPU timeline、`/proc/interrupts` 或 perf 证据显示与长尾相关时
才纳入根因，不能从 `cudaEventSynchronize` 名称直接推断。

## 14. 优化优先级

### P0：先定位同步调用来源

通过 backtrace 和 vLLM/NVTX range 回答：

- 哪个 rank、PID、TID；
- Prefill、Decode、sampling 还是 output processing；
- `.item()`、`.cpu()`、显式 synchronize、Event wait，还是框架内部；
- 每 decode step 几次。

没有调用源，不能谈删除同步。

### P1：缩小同步范围

当依赖允许时：

```text
cudaDeviceSynchronize
→ cudaStreamSynchronize
→ cudaEventSynchronize
→ cudaStreamWaitEvent / stream dependency
```

目标不是机械替换 API，而是从“整个 device 等完”缩小为“只等待真正依赖的数据”。

### P2：延后 Host 可见性

典型反模式：

- decode step 中间 `.item()`；
- 每步把 logits/大 tensor 拉回 CPU；
- Python 分支依赖 GPU 标量；
- debug print/logging 访问 CUDA tensor；
- output processor 在 producer Event 完成前强制同步。

把 Host read 移到真正消费点，或只搬 sampled token/小结果。

### P3：不要把 GPU 等待误当 CPU 优化

如果同步 API 90% 时间与 GPU kernel/NCCL 重叠，CPU 绑核只能改善剩余的唤醒尾部。
主要优化对象是：

- INT8 W8A8 scaled-matmul、activation quantization kernel 与算子融合；
- CUDA Graph 覆盖；
- attention backend；
- TP/NCCL 拓扑；
- graph/batch bucket；
- 调度和 batch 形状。

动态 per-token activation quantization 会在每层引入 scale/reduction/quantization
工作。Decode 的矩阵 `M` 较小时，这些短 kernel 和 launch 固定开销可能更显眼。
应先确认它们是否被 CUDA Graph 捕获、是否命中 vLLM 当前版本针对 SM80 的
compressed-tensors INT8 实现，再讨论 CPU 调度；不能因为 kernel 名包含 BF16
就直接判定回退，因为 attention、norm、embedding、lm_head 等非量化算子仍可用 BF16。

### P4：处理 OS 尾延迟

只有 timeline 显示 Ready/off-CPU 或异常 migration 时，再做：

- CPU affinity；
- NUMA local memory；
- 隔离 IRQ/繁忙服务线程；
- 调整容器 CPU quota/cpuset；
- 将 frontend、tokenizer、EngineCore、worker 合理分核。

### P5：检查 PCIe/DMA

只有同步窗口与 Memcpy 明确重叠，且 copy duration 异常时，才使用
H2D/D2H 微基准、PCIe link、NUMA 和 pinned memory 结果解释。

## 15. 必做 A/B

### 无 profiler 性能基线

先停止 Nsight 服务器，使用相同模型参数直接启动 vLLM：

```bash
vllm serve /models/Qwen3-32B-INT8-W8A8 \
  --served-model-name qwen3-32b-int8-w8a8 \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192
```

另一终端：

```bash
bash labs/run_vllm_cuda_sync_profile.sh warmup

BENCHMARK_LABEL=baseline_bs8_1024_128 \
bash labs/run_vllm_cuda_sync_profile.sh benchmark
```

重复至少 5 轮。Nsight/PyTorch trace 只负责归因；优化前后的 TTFT、TPOT、ITL 和吞吐
只比较这种 profiler-off 基线。

| A/B | 目的 | 预期判读 |
|---|---|---|
| 1024→1 vs 1024→128 | Prefill/Decode 分离 | 差值近似 127 个 decode step |
| CUDA Graph 默认 vs `--enforce-eager` | Graph 对 sync/launch 的影响 | eager 常有更多 launch/API 边界 |
| TP1 vs 实际 TP | 区分单卡 kernel 与 NCCL | TP 后同步增长并伴随 NCCL，优先查通信 |
| `--disable-async-output-proc` on/off | output CPU pipeline | 同步位置和 TPOT 是否改变 |
| CPU local vs remote NUMA | OS/NUMA 尾部 | GPU工作相同而 wakeup 尾部变化 |
| profiler on/off | 测量扰动 | TTFT/TPOT 正式数值只取 profiler off |

`CUDA_LAUNCH_BLOCKING=1` 只能作为调试负对照。它会改变所有异步语义，不能用来做
生产性能基线，也不能用它的 sync API 占比推断正常模式。

## 16. 本场景的优先假设

对 A100 上的 Qwen3-32B-INT8-W8A8、1024→128、并发 8：

1. Prefill 初段的同步长调用更可能等待大 GEMM/attention 或 TP collective。
2. Decode 中重复同步若每 step 一次，可能进入 TPOT 关键路径。
3. W8A8 降低权重带宽/计算成本后，Host launch/sync 固定成本占比可能上升。
4. 默认没有 KV/weight offload 时，PCIe 大块 DMA 通常不是每个 decode step 的主体。
5. D2H sampled token 很小，常见问题是同步/唤醒固定成本，而不是传输带宽。

这些都是待验证假设，不是结论。结论必须来自相同同步调用窗口内的 GPU、Memcpy、
NCCL 和 CPU thread-state 证据。

可用 capture（`qwen3_awq`，AWQ 而非本手册的 INT8 W8A8）上用 §12 的 capstone 脚本
已得到一条需要写进 W8A8 验证计划的经验：长同步全部落在 `A_gpu_spinning`——GPU
在跑、调用线程同时 on-CPU ~100%。这说明假设 1/2 中"等待"二字需要细化：线程并非
在 futex 上长睡，而是在 CPU 上 poll，因此同步窗口内的 task-clock/cycles 是真实
CPU 消耗，假设 3（Host launch/sync 固定成本占比上升）的"成本"是可被 `perf` 采到
的，而不是纯 off-CPU 等待。W8A8 正式 capture 采集后，必须用同套脚本确认这一
`A_gpu_spinning` 是否仍然成立，再决定 CPU 侧优化对象是驱动完成路径还是 vLLM
等待策略。

## 17. 结果报告模板与验收

每轮至少填写：

```text
Run ID:
GPU / memory / PCIe or SXM:
GPU count / TP:
CPU model / NUMA binding:
driver / CUDA / PyTorch / vLLM / Nsight:
checkpoint path or model ID + revision:
quant_method / format:
weight bits+type / activation bits+type:
input / output / prompts / max concurrency:
CUDA Graph enabled:
profiler:
```

同步统计表：

| API | Rank/PID/TID | Calls | Total ms | p50 us | p95 us | p99 us | Max us | Calls/decode step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cudaEventSynchronize` |  |  |  |  |  |  |  |  |
| `cudaStreamSynchronize` |  |  |  |  |  |  |  |  |
| `cudaDeviceSynchronize` |  |  |  |  |  |  |  |  |

长调用归因表（`Running`/`Blocked`/`GPU overlap` 由 `attribute_cpu_state.py`
capstone 自动填写；`Ready` 需开 `--sample` 才能从 threadState 区分，否则留空并
注明）：

| API 区间 | Prefill/Decode | Running | Blocked | Ready | GPU overlap | Memcpy/NCCL | 调用栈 | 结论 |
|---|---|---:|---:|---:|---|---|---|---|
| p99 #1 |  |  |  |  |  |  |  |  |

性能表：

| Run | TTFT p50/p99 | TPOT p50/p99 | ITL p99 | Output tok/s | sync ms/step |
|---|---:|---:|---:|---:|---:|
| profiler off baseline |  |  |  |  | N/A |
| Nsight full 1024→128 |  |  |  |  |  |
| Nsight control 1024→1 |  |  |  |  |  |
| 优化后 profiler off |  |  |  |  | N/A |

一项优化只有同时满足以下条件才接受：

1. INT8 W8A8 runtime 路径和请求形状未改变；
2. 目标同步 API 的 total/p99 或 calls/step 明确下降；
3. TPOT/ITL 或吞吐在 profiler-off 重复实验中改善；
4. 没把等待转移到其他 API、rank 或请求阶段；
5. 输出正确性和模型精度验收没有退化；
6. 至少重复 5 轮，报告中位数和离散程度，不挑最好的一轮。

分享分析数据时至少包含：

- 完整 `*.nsys-rep`；
- full/control 两组 `cuda_sync_summary.json` 和 CUDA API CSV；
- full/control 的 vLLM benchmark JSON；
- `int8-w8a8-verification.json`；
- `nvidia-smi` topology、版本与 server 日志；
- PyTorch trace，仅在需要 operator/stack 交叉验证时上传。

## 18. 官方参考

- [vLLM Profiling：PyTorch Profiler 与 Nsight Systems](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [`vllm bench serve` 参数](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM INT8 W8A8](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int8_w8a8/)
- [vLLM 量化硬件兼容矩阵](https://docs.vllm.ai/en/stable/features/quantization/)
- [LLM Compressor：量化方案选择](https://docs.vllm.ai/projects/llm-compressor/en/stable/steps/choosing-scheme/)
- [Nsight Systems CUDA API summary/trace](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [Nsight Systems CUDA trace、thread state 与 backtrace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

若下一步需要比较 `blocking`、active wait、Python polling、hybrid wait，并把同步
wall time 与真正的 on-CPU self、OS 调度、NUMA、IRQ 分开，继续执行
[`07-vllm-cpu-selftime-experiments.md`](07-vllm-cpu-selftime-experiments.md)。
