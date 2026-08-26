# AI Infra Learning

一个面向大模型推理基础设施的中文学习仓库，重点关注：

- GPU 服务器中的 CPU 机头与 NUMA/PCIe 拓扑
- Prefill、Decode、Continuous Batching 与调度
- KV Cache、PagedAttention、Prefix Cache 与 Offload
- vLLM、SGLang、PyTorch 的分层关系
- TTFT、TPOT、Goodput 等服务指标

## 从哪里开始

1. 阅读 [`docs/01-ai-infra-quickstart.md`](docs/01-ai-infra-quickstart.md)，先建立完整系统地图。
2. 阅读 [`docs/02-hands-on-guide.md`](docs/02-hands-on-guide.md)，按顺序完成实验。
3. 运行 `make test` 验证本地环境。
4. 用 `make lab-kv`、`make lab-scheduler`、`make lab-prefix` 建立性能直觉。
5. 有 Linux + A100 + 专用 NVMe 时，按
   [`docs/04-a100-gds-hands-on.md`](docs/04-a100-gds-hands-on.md) 完成 GDS 实验。

全部基础实验只依赖 Python 3.10+ 标准库，不要求 GPU，也不会自动下载模型。

## 仓库结构

```text
.
├── docs/
│   ├── 00-learning-roadmap.md
│   ├── 01-ai-infra-quickstart.md
│   ├── 02-hands-on-guide.md
│   ├── 03-official-reading-list.md
│   └── 04-a100-gds-hands-on.md
├── labs/
│   ├── kv_cache_calculator.py
│   ├── scheduler_simulator.py
│   ├── prefix_cache_simulator.py
│   ├── openai_stream_benchmark.py
│   └── gds/
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
bash labs/gds/collect_gds_preflight.sh --help
bash labs/gds/run_gdsio_matrix.sh --help
```

## 学习原则

- 所有性能结论必须注明模型、硬件、输入/输出长度、并发与 SLO。
- 先分清瓶颈发生在排队、CPU、GPU、KV 容量还是通信，再谈优化。
- 实验输出用于理解概念，不代替真实框架和真实生产流量的基准测试。
