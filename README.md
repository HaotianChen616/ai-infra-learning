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
6. 昇腾 910B4 环境使用
   [`docs/05-ascend-910b4-h2d-d2h-profiling.md`](docs/05-ascend-910b4-h2d-d2h-profiling.md)。

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
H2D/D2H 实验是可选的真实硬件实验：NVIDIA 环境需要 CUDA 版 PyTorch，
昇腾环境需要匹配 CANN 的 PyTorch 与 `torch_npu`。

## 仓库结构

```text
.
├── docs/
│   ├── 00-learning-roadmap.md
│   ├── 01-ai-infra-quickstart.md
│   ├── 02-hands-on-guide.md
│   ├── 03-official-reading-list.md
│   ├── 04-h2d-d2h-profiling.md
│   └── 05-ascend-910b4-h2d-d2h-profiling.md
├── labs/
│   ├── h2d_d2h_benchmark.py
│   ├── run_h2d_d2h_validation.sh
│   ├── run_ascend_h2d_d2h_validation.sh
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
# 仅在昇腾 NPU 主机：
make lab-h2d-d2h-ascend
```

## 学习原则

- 所有性能结论必须注明模型、硬件、输入/输出长度、并发与 SLO。
- 先分清瓶颈发生在排队、CPU、GPU、KV 容量还是通信，再谈优化。
- 实验输出用于理解概念，不代替真实框架和真实生产流量的基准测试。
