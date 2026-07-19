# 7 天学习路线

目标不是在一周内成为内核工程师，而是能顺着一次请求解释 CPU、GPU、KV Cache 和推理框架各自负责什么，并能读懂一份性能报告。

## 第 1 天：请求生命周期

掌握 Token、Prompt、Context、Prefill、Decode、TTFT 和 TPOT。最终能够不看资料画出：

```text
请求 → Tokenize → Scheduler → Prefill → KV Cache → Decode → 流式返回
```

## 第 2 天：Transformer 与 PyTorch

理解 Embedding、Q/K/V、Attention、MLP、RMSNorm，以及 PyTorch Eager、`torch.compile`、TorchInductor 和 CUDA Graph。暂时不用学习反向传播与优化器。

## 第 3 天：KV Cache

运行 `make lab-kv`，能够手算一个模型每 token 的 KV 大小，理解 MHA、MQA、GQA、MLA、Block、Page、Prefix Cache、Offload 和 Quantization。

## 第 4 天：调度

运行 `make lab-scheduler`，修改请求到达时间、输入长度、输出长度和 token budget。观察长 Prefill 如何影响 Decode，以及 Decode-first 和 Chunked Prefill 如何改变尾延迟。

## 第 5 天：vLLM 与 SGLang

理解两者都属于 LLM Serving Engine，不是 PyTorch 的替代品。认识 API Server、Scheduler、Model Worker、Paged KV / Block Manager、RadixAttention 和 PD 分离。

## 第 6 天：CPU 机头与拓扑

认识 Socket、物理核、vCPU、NUMA、DDR、PCIe、Pinned Memory、NIC 和 RDMA。练习回答：某张 GPU 与哪个 CPU Socket、哪块内存、哪张网卡距离最近？

## 第 7 天：性能评测

理解 TTFT、TPOT/ITL、E2E、RPS、Input/Output tokens/s、p99 和 Goodput。对任何 benchmark 都先问：

1. 模型、精度和硬件是什么？
2. 输入和输出长度是什么分布？
3. 并发或到达率是多少？
4. 报的是平均值还是 p99？
5. 是否满足业务 SLO？
6. Prefix Cache 是否命中？

## 完成标准

- 能画出请求链路并标明 CPU/GPU 分工。
- 能解释 Prefill 和 Decode 的主要瓶颈差异。
- 能估算 KV Cache 容量。
- 能区分 PyTorch、vLLM、SGLang、Triton 和 NCCL。
- 能判断一个优化是在改善 TTFT、TPOT、吞吐、容量还是成本。
