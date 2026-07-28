# 实践指南

所有基础实验只依赖 Python 标准库。建议先运行：

```bash
make test
```

## 实验 1：KV Cache 容量计算

```bash
make lab-kv
```

代码：[`labs/kv_cache_calculator.py`](../labs/kv_cache_calculator.py)

需要观察：

- KV 每 token、每序列和总容量如何变化。
- Context Length 或 Concurrency 翻倍时，KV 容量也近似翻倍。
- GQA 减少 KV Head 后，容量如何变化。
- TP 如何把逻辑 KV 分摊到多张设备。
- 给定每卡 KV 预算时，理论并发上限是多少。

可以尝试：

```bash
python3 labs/kv_cache_calculator.py \
  --layers 80 --kv-heads 8 --head-dim 128 --dtype fp8 \
  --context-length 32768 --concurrency 64 --tp-size 8 \
  --kv-capacity-gib-per-device 30
```

说明：计算器给出的是教学用逻辑估算。真实占用还受模型结构、块大小、对齐、并行分片和框架预留空间影响。

## 实验 2：Continuous Batching 调度

```bash
make lab-scheduler
```

代码：[`labs/scheduler_simulator.py`](../labs/scheduler_simulator.py)

模拟器把一次调度迭代抽象为一个 step，并比较：

- Prefill-first：新来的长输入可以优先消耗 token budget。
- Decode-first：先保证已经在生成的请求获得 Decode token。
- Prefill chunk：限制单请求每步最多占用多少 Prefill token。

修改代码中的 `default_requests()`，尝试增加一条晚到达、超长 Prompt 的请求，观察其他请求的 TTFT 和完成时间。

这个实验不模拟真实 GPU 时间、Kernel 或通信，只用于理解调度策略。

## 实验 3：Prefix Cache

```bash
make lab-prefix
```

代码：[`labs/prefix_cache_simulator.py`](../labs/prefix_cache_simulator.py)

需要观察：

- 只有完整 Block 对齐的相同 Token 前缀才能命中。
- 相同长文档加不同问题具有较高复用价值。
- 看起来相同的文本，如果 Chat Template、空格或特殊 Token 不同，可能完全不命中。
- Prefix Cache 主要减少 Prefill 计算，不直接减少输出 Decode 步数。

## 实验 4：连接 vLLM/SGLang 服务

代码：[`labs/openai_stream_benchmark.py`](../labs/openai_stream_benchmark.py)

当本地或远端已经有 OpenAI 兼容接口时运行：

```bash
python3 labs/openai_stream_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model your-model-name \
  --prompt "用三句话解释 KV Cache" \
  --max-tokens 128
```

它会记录：

- HTTP 请求到达第一个有内容的数据块的时间，近似 TTFT。
- 相邻流式内容块的间隔，近似 ITL。
- 整体请求时间。

注意：一次 SSE 内容块不一定等于一个模型 Token，因此这是轻量观测工具，不是严格的 token-level benchmark。

## 实验 5：真实推理中的 Host/Device 同步等待

详细手册：[`docs/04-h2d-d2h-profiling.md`](04-h2d-d2h-profiling.md)

昇腾 910B4 端到端手册：
[`docs/05-ascend-910b4-h2d-d2h-profiling.md`](05-ascend-910b4-h2d-d2h-profiling.md)

这个实验需要 Linux、NVIDIA GPU 与 CUDA 版 PyTorch。建议在目标 A100
服务器上运行：

```bash
make lab-h2d-d2h
```

在已经能正常部署模型的 vLLM-Ascend 容器内，查看 Qwen3.6-27B-W8A8
端到端实验入口：

```bash
make lab-vllm-ascend-e2e
```

NVIDIA 微基准会把一次 CPU↔加速卡 copy 拆成：

- Python/CPU 数据准备时间；
- PyTorch/CUDA 或 `torch_npu` 主机提交时间；
- 设备 event 记录的实际 copy 时间；
- 提交到完成的可见延迟；
- 从准备到完成的端到端时间。

昇腾主实验不是纯 H2D/D2H 微基准，而是固定输入 1024、输出 128、并发 8，
采集 vLLM 服务的 Prefill/Decode、CANN 同步 API、NPU kernel、HCCL、Memcpy
和 CPU 侧等待，并使用 1024→1 control 做差分。只有 timeline 指向 copy
问题时，才用 `make lab-h2d-d2h-ascend` 做链路校准。

## 实验 6：vLLM CUDA 同步等待分析

完整手册：
[`docs/06-vllm-cuda-sync-profiling.md`](06-vllm-cuda-sync-profiling.md)

这个实验在真实 vLLM 服务中分析 `cudaDeviceSynchronize`、
`cudaEventSynchronize` 和 `cudaStreamSynchronize` 等调用的 Host API wall
time，并用 Nsight Systems 拆分：

- GPU kernel、CUDA Graph、NCCL 或 Memcpy 前序等待；
- CUDA Runtime/Driver；
- CPU sleeping、wakeup 和 runnable/off-CPU 调度延迟；
- Prefill、Decode、sampling 与 output processing 的调用来源。

Qwen3-32B INT8 W8A8（`compressed-tensors`）、A100、输入 1024、输出 128、
并发 8 的默认入口：

```bash
bash labs/run_vllm_cuda_sync_profile.sh --help
```

纯 H2D/D2H 微基准只作为 PCIe/pinned-memory 对照，不能代替真实推理同步分析。

## 下一阶段建议

完成基础实验后，再开始真实框架实验：

1. 用同一模型分别启动 vLLM 和 SGLang。
2. 固定输入/输出长度和请求到达率。
3. 记录 TTFT p50/p99、TPOT p50/p99、Goodput、CPU、GPU 和 KV 水位。
4. 一次只改变一个变量，例如 token budget、Prefix Cache 或 Chunked Prefill。
