# AI Infra Learning

一个面向大模型推理基础设施的中文学习仓库，重点关注：

- GPU 服务器中的 CPU 机头与 NUMA/PCIe 拓扑
- Prefill、Decode、Continuous Batching 与调度
- KV Cache、Paged KV / Block Manager、Prefix Cache 与 Offload
- vLLM、SGLang、PyTorch 的分层关系
- TTFT、TPOT、Goodput 等服务指标

## 从哪里开始

1. 阅读 [`docs/01-ai-infra-quickstart.md`](docs/01-ai-infra-quickstart.md)，先建立完整系统地图。
2. 阅读 [`docs/02-hands-on-guide.md`](docs/02-hands-on-guide.md)，按顺序完成实验。
3. 运行 `make test` 验证本地环境。
4. 用 `make lab-kv`、`make lab-scheduler`、`make lab-prefix` 建立性能直觉。
5. 在 Linux NVIDIA GPU 主机阅读并执行
   [`docs/04-h2d-d2h-profiling.md`](docs/04-h2d-d2h-profiling.md)，拆解推理数据搬运。
6. 在已有 vLLM-Ascend 容器的昇腾 910B4 环境，使用
   [`docs/05-ascend-910b4-h2d-d2h-profiling.md`](docs/05-ascend-910b4-h2d-d2h-profiling.md)
   分析 Qwen3.6-27B-W8A8 端到端同步等待。
7. 分析 A100 INT8 W8A8 真实 vLLM 推理中的 CUDA 同步等待，使用
   [`docs/06-vllm-cuda-sync-profiling.md`](docs/06-vllm-cuda-sync-profiling.md)。
8. 对 vLLM 0.26 做阻塞/轮询、CPU self time、绑核、GIL、OS 调度和 IRQ A/B，使用
   [`docs/07-vllm-cpu-selftime-experiments.md`](docs/07-vllm-cpu-selftime-experiments.md)。

## 学习网站

[AI Infra Field Guide](https://ai-infra-field-guide.htchen199905.chatgpt.site)
提供可浏览版本，包含知识地图、vLLM、SGLang、Agent / Gateway
专题与术语表。网站源码位于 `site/`。

```bash
cd site
npm install
npm run dev
```

基础模拟实验只依赖 Python 3.10+ 标准库，不要求加速卡，也不会自动下载模型。
真实硬件实验是可选项：NVIDIA 环境需要 CUDA 版 PyTorch；昇腾端到端实验
假设已有能部署 Qwen3.6-27B-W8A8 的 vLLM-Ascend 容器。

## 仓库结构

```text
.
├── docs/
│   ├── 00-learning-roadmap.md
│   ├── 01-ai-infra-quickstart.md
│   ├── 02-hands-on-guide.md
│   ├── 03-official-reading-list.md
│   ├── 04-h2d-d2h-profiling.md
│   ├── 05-ascend-910b4-h2d-d2h-profiling.md
│   ├── 06-vllm-cuda-sync-profiling.md
│   └── 07-vllm-cpu-selftime-experiments.md
├── labs/
│   ├── h2d_d2h_benchmark.py
│   ├── summarize_ascend_sync.py
│   ├── summarize_cuda_sync.py
│   ├── verify_ascend_w8a8_model.py
│   ├── verify_int8_w8a8_config.py
│   ├── run_h2d_d2h_validation.sh
│   ├── run_vllm_ascend_e2e_profile.sh
│   ├── run_vllm_cuda_sync_profile.sh
│   ├── run_vllm_cpu_experiments.sh
│   ├── analyze_nsys_sqlite.py
│   ├── analyze_torch_trace_cpu.py
│   ├── summarize_cpu_probes.py
│   ├── kv_cache_calculator.py
│   ├── scheduler_simulator.py
│   ├── prefix_cache_simulator.py
│   └── openai_stream_benchmark.py
├── tests/
└── Makefile
```

## 常用命令

```bash
make test
make lab-kv
make lab-scheduler
make lab-prefix
python3 labs/openai_stream_benchmark.py --help
# 仅在 Linux NVIDIA GPU 主机：
make lab-h2d-d2h
# 在已有 vLLM-Ascend 环境的昇腾 910B4 容器内：
make lab-vllm-ascend-e2e
# 真实 vLLM CUDA 同步 API profiling：
bash labs/run_vllm_cuda_sync_profile.sh --help
# vLLM CPU self time、等待策略、绑核、调度与 IRQ 实验：
make lab-vllm-cpu
```

## 学习原则

- 所有性能结论必须注明模型、硬件、输入/输出长度、并发与 SLO。
- 先分清瓶颈发生在排队、CPU、GPU、KV 容量还是通信，再谈优化。
- 实验输出用于理解概念，不代替真实框架和真实生产流量的基准测试。
