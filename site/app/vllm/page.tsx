import Link from "next/link";
import { createPageMetadata } from "../page-metadata";
import { PageProgress, SiteFooter, SiteHeader } from "../site-chrome";

export const metadata = createPageMetadata(
  "vLLM 学习路线",
  "从 Paged KV、Continuous Batching 到 V1 进程架构，循序理解 vLLM。",
  "/vllm",
);

const mechanisms = [
  {
    index: "01",
    title: "Paged KV / Block Manager",
    body: "把每条请求的 KV 划分成固定 Block，通过 Block Table 映射到非连续物理内存，减少碎片并支持按需分配。这是仍然成立的分页思想。",
    effect: "容量 / 并发",
  },
  {
    index: "02",
    title: "Continuous Batching",
    body: "每轮调度都允许完成请求退出、新请求加入，不再等待一个静态 Batch 中所有序列同时结束。",
    effect: "吞吐 / 利用率",
  },
  {
    index: "03",
    title: "Chunked Prefill",
    body: "把长 Prompt 切成 Token Chunk，优先保护 Decode，再使用剩余 Token Budget 安排 Prefill。",
    effect: "TTFT / ITL",
  },
  {
    index: "04",
    title: "Prefix Caching",
    body: "对已经计算的完整 Token 前缀复用 KV Block，跳过共享部分的 Prefill；它不会直接减少 Decode 步数。",
    effect: "Prefill 成本",
  },
  {
    index: "05",
    title: "CUDA Graph",
    body: "录制并重放稳定的 GPU 执行路径，减少 CPU 逐个发射 Kernel 的开销，但要处理动态 Batch 与内存地址约束。",
    effect: "CPU 开销",
  },
  {
    index: "06",
    title: "Distributed Execution",
    body: "通过 TP、PP、DP、EP 等方式跨卡执行；卡越多不一定越快，通信开销必须与计算收益一起衡量。",
    effect: "扩展 / 通信",
  },
];

export default function VllmPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        跳到正文
      </a>
      <SiteHeader active="vllm" />
      <main id="main" className="framework-page vllm-page">
        <section className="framework-hero">
          <div className="shell">
            <div className="breadcrumb">
              <Link href="/">AI Infra</Link>
              <span>/</span>
              <b>vLLM</b>
            </div>
            <div className="framework-hero-grid">
              <div>
                <div className="framework-badge">SERVING ENGINE · 01 · v0.25.1</div>
                <h1>vLLM</h1>
                <p className="framework-thesis">
                  先把它理解成：
                  <strong>请求调度器 + KV 内存管理器 + Model Runner</strong>
                </p>
                <p className="framework-lead">
                  学习重点不是背启动参数，而是追踪一条请求如何进入 Scheduler、获得 KV
                  Block、执行 Prefill，再进入 Decode Loop。
                </p>
                <div className="hero-actions">
                  <a className="button button-primary" href="#learning-path">
                    开始路线 <span aria-hidden="true">↓</span>
                  </a>
                  <a
                    className="button button-ghost-on-dark"
                    href="https://docs.vllm.ai/en/latest/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    官方文档 ↗
                  </a>
                </div>
                <PageProgress current={2} />
              </div>
              <div className="framework-diagram">
                <div className="diagram-title">
                  <span>V1 PROCESS MAP</span>
                  <i>ONLINE</i>
                </div>
                <div className="process-row">
                  <span>API SERVER</span>
                  <small>HTTP · Tokenize · Stream</small>
                </div>
                <div className="diagram-arrow">↓ ZMQ</div>
                <div className="process-row process-core">
                  <span>ENGINE CORE</span>
                  <small>Scheduler · KV Manager</small>
                </div>
                <div className="diagram-arrow">↓ RPC</div>
                <div className="worker-grid">
                  <div>
                    <span>WORKER 0</span>
                    <small>GPU 0</small>
                  </div>
                  <div>
                    <span>WORKER 1</span>
                    <small>GPU 1</small>
                  </div>
                  <div>
                    <span>WORKER N</span>
                    <small>GPU N</small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="framework-summary">
          <div className="shell summary-grid">
            <div>
              <small>MENTAL MODEL</small>
              <strong>Block-based KV</strong>
              <span>像操作系统分页管理内存</span>
            </div>
            <div>
              <small>SCHEDULING UNIT</small>
              <strong>Token Budget</strong>
              <span>不只是请求数量</span>
            </div>
            <div>
              <small>CORE TRADE-OFF</small>
              <strong>Throughput ↔ Latency</strong>
              <span>Batch 越大不一定越好</span>
            </div>
          </div>
        </section>

        <section className="section shell" id="learning-path">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p className="section-kicker">LEARNING PATH</p>
              <h2>按请求生命周期学习</h2>
            </div>
            <p>不要从几百个启动参数开始。先回答每个模块接收什么、保存什么、输出什么。</p>
          </div>
          <div className="route-map">
            {[
              ["1", "API Server", "接收 OpenAI 兼容请求，做输入处理与流式输出。"],
              ["2", "Input Processor", "Chat Template、Tokenize、模型相关输入校验。"],
              ["3", "Scheduler", "从 waiting/running 请求中决定本轮 Token Budget。"],
              ["4", "KV Manager", "分配、复用、驱逐 KV Block，维护 Block Table。"],
              ["5", "Model Runner", "准备 Attention Metadata，发起 GPU Forward。"],
              ["6", "Output Processor", "采样结果、Detokenize、结束条件与返回。"],
            ].map(([index, title, body]) => (
              <article key={index}>
                <span>{index}</span>
                <div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="section framework-mechanisms">
          <div className="shell">
            <div className="section-heading">
              <div>
                <span className="section-number">02</span>
                <p className="section-kicker">CORE MECHANISMS</p>
                <h2>六个机制，覆盖大多数讨论</h2>
              </div>
              <p>听到一个优化时，先判断它改善的是容量、TTFT、TPOT、吞吐还是 CPU 开销。</p>
            </div>
            <div className="mechanism-grid">
              {mechanisms.map((item) => (
                <article key={item.index}>
                  <div className="mechanism-top">
                    <span>{item.index}</span>
                    <i>{item.effect}</i>
                  </div>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </article>
              ))}
            </div>
            <aside className="version-note">
              <b>名称提醒</b>
              <p>
                vLLM v0.25.0 删除的是旧的 <em>PagedAttention</em> attention
                实现；V1 / Model Runner V2 仍使用按 Block 管理 KV、Block Table
                和按需分配。讨论系统设计时，更准确的说法是 Paged KV Cache 或
                Block Manager。
              </p>
              <a
                href="https://github.com/vllm-project/vllm/releases/tag/v0.25.0"
                target="_blank"
                rel="noreferrer"
              >
                查看 v0.25.0 Release Notes ↗
              </a>
            </aside>
            <aside className="hardware-note">
              <b>精度与硬件视角</b>
              <p>
                KV 从 BF16 / FP16 降到 FP8 / INT8，理论字节数约减半；v0.25.0
                还加入了 INT4 per-token-head 与 NVFP4 KV 路径。GPU、Ascend
                NPU、XPU 等后端的 Kernel、内存布局、图捕获与集合通信不同，
                所以“抽象相同”不等于“参数和瓶颈相同”。
              </p>
            </aside>
          </div>
        </section>

        <section className="section shell">
          <div className="section-heading">
            <div>
              <span className="section-number">03</span>
              <p className="section-kicker">SOURCE TOUR</p>
              <h2>读源码时，顺着数据走</h2>
            </div>
            <p>职责边界要记住；链接锁定 v0.25.1，避免 main 分支变化导致对不上。</p>
          </div>
          <div className="source-tour">
            <div className="source-code">
              <div className="code-header">
                <span>request_lifecycle.py</span>
                <i>mental model</i>
              </div>
              <pre>
                <code>{`request = api.parse(payload)
tokens  = processor.encode(request)

while not request.finished:
    batch  = scheduler.schedule()
    blocks = kv_manager.allocate(batch)
    logits = model_runner.execute(batch, blocks)
    output = sampler.sample(logits)
    api.stream(output)`}</code>
              </pre>
            </div>
            <ol className="source-steps source-links">
              <li>
                <span>01</span>
                <div>
                  <a
                    href="https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/engine/core.py"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Engine Core ↗
                  </a>
                  <p>入口、主循环、请求状态与 Worker 协调。</p>
                </div>
              </li>
              <li>
                <span>02</span>
                <div>
                  <a
                    href="https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/core/sched/scheduler.py"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Scheduler ↗
                  </a>
                  <p>Token Budget、waiting/running 队列与抢占。</p>
                </div>
              </li>
              <li>
                <span>03</span>
                <div>
                  <a
                    href="https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/core/kv_cache_manager.py"
                    target="_blank"
                    rel="noreferrer"
                  >
                    KV Cache Manager ↗
                  </a>
                  <p>Block 分配、Prefix Cache 命中与释放。</p>
                </div>
              </li>
              <li>
                <span>04</span>
                <div>
                  <a
                    href="https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py"
                    target="_blank"
                    rel="noreferrer"
                  >
                    GPU Model Runner ↗
                  </a>
                  <p>准备输入与 Attention Metadata，执行模型。</p>
                </div>
              </li>
            </ol>
          </div>
        </section>

        <section className="section framework-lab-section">
          <div className="shell lab-grid">
            <div>
              <span className="section-number">04</span>
              <p className="section-kicker">HANDS-ON</p>
              <h2>用四个实验建立直觉</h2>
              <p>
                学习仓库里的基础实验不需要 GPU。先理解容量、调度和缓存，再连接真实 vLLM 服务测 TTFT。
              </p>
              <a
                className="button button-primary"
                href="https://github.com/HaotianChen616/ai-infra-learning/tree/main/labs"
                target="_blank"
                rel="noreferrer"
              >
                打开实验仓库 ↗
              </a>
            </div>
            <div className="lab-list">
              {[
                ["01", "KV Cache Calculator", "改变 Context、并发、GQA 与 TP。"],
                ["02", "Scheduler Simulator", "比较 Prefill-first 与 Decode-first。"],
                ["03", "Prefix Cache Simulator", "观察精确前缀和 Block 对齐。"],
                ["04", "Streaming Benchmark", "连接接口，记录 TTFT 与内容块间隔。"],
              ].map(([index, title, body]) => (
                <article key={index}>
                  <span>{index}</span>
                  <div>
                    <h3>{title}</h3>
                    <p>{body}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="next-framework">
          <div className="shell next-framework-inner">
            <div>
              <span>NEXT · 03 / 04</span>
              <h2>用 SGLang 对照另一种设计</h2>
              <p>重点观察 RadixAttention、Overlap Scheduler 和 PD 分离。</p>
            </div>
            <Link className="button button-light" href="/sglang">
              前往 SGLang <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
