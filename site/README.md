# AI Infra Field Guide 网站

本目录是 `ai-infra-learning` 的可浏览学习站，包含三个页面：

- `/`：AI Infra 请求链路、CPU 机头、KV Cache、框架分层与性能指标。
- `/vllm`：vLLM 的请求生命周期、关键机制、源码阅读顺序与实践入口。
- `/sglang`：SGLang 的 RadixAttention、Overlap Scheduler、PD 分离与对照学习。

## 本地运行

需要 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

## 验证

```bash
npm run build
node --test tests/rendered-html.test.mjs
```

基础实验代码位于仓库根目录的 `labs/`。
