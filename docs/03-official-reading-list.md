# 官方资料阅读顺序

框架更新快，具体参数和默认值应以当前版本官方文档为准。

## 第一层：建立框架地图

- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [SGLang 官方文档](https://docs.sglang.io/)
- [PyTorch Compiler](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html)

## 第二层：理解调度与 KV

- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM KV Offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- [SGLang PD Disaggregation](https://docs.sglang.ai/backend/pd_disaggregation.html)

## 第三层：理解 CPU 开销

- 在 vLLM Optimization 文档中阅读 `CPU Resources for GPU Deployments`。
- 在 PyTorch CUDA Semantics 中阅读 CUDA Graphs。
- 在 SGLang 参数和环境变量文档中搜索 CPU affinity、overlap scheduler、tokenizer 和 offload。

## 阅读方法

不要一开始逐行读源码。先围绕一个请求追踪：

```text
API Server
  → Input Processor / Tokenizer
  → Scheduler
  → KV Manager
  → Model Runner
  → Output Processor
```

每看到一个类或参数，先回答它主要影响：正确性、TTFT、TPOT、吞吐、容量、通信还是可观测性。

## GPUDirect Storage

- [GDS Getting Started](https://docs.nvidia.com/gpudirect-storage/getting-started/)
- [GDS Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
- [GDS Installation and Troubleshooting](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)
- [GDS Benchmarking and Configuration](https://docs.nvidia.com/gpudirect-storage/configuration-guide/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/)

## 沐曦 C500 MAS

- [沐曦 C500 产品页](https://www.metax-tech.com/prod.html?cid=107&id=21)
- [沐曦开发者论坛：GPUDirect Storage 特性是否支持](https://developer.metax-tech.com/forum/t/gpudirect-storagete-xing-shi-fou-zhi-chi/288/)
- [MetaX-MACA/mxFIO](https://github.com/MetaX-MACA/mxFIO)
- [MXMACA-C500 发布说明入口](https://developer.metax-tech.com/doc/222)
- [沐曦开发者论坛：C500 支持 GDR](https://developer.metax-tech.com/forum/t/mu-xi-c500shi-fou-zhi-chi-gpudirect-rdma/445/)

## D2RS、dma-buf 与灵衢 URMA

- [Linux dma-buf 官方文档](https://docs.kernel.org/driver-api/dma-buf.html)
- [rdma-core：`ibv_reg_dmabuf_mr` API](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
- [openEuler UMDK / URMA](https://gitcode.com/openeuler/umdk)
- [openEuler UBS IO / NDS](https://gitcode.com/openeuler/ubs-io)
