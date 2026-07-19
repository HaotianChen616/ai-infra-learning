import Link from "next/link";
import { createPageMetadata } from "./page-metadata";
import { PageProgress, SiteFooter, SiteHeader } from "./site-chrome";

export const metadata = createPageMetadata(
  "知识地图",
  "从一次请求出发，理解 CPU、GPU、KV Cache、调度、推理指标和 Serving Framework。",
  "/",
);

export default function Home() {
  return (
    <>
      <a className="skip-link" href="#main">
        跳到正文
      </a>
      <SiteHeader active="home" />
      <main id="main">
        <section className="hero">
          <div className="shell hero-grid">
            <div className="hero-copy">
              <div className="eyebrow">
                <span className="live-dot" />
                LLM INFERENCE · LEARNING MAP
              </div>
              <h1>
                从一次请求，
                <br />
                看懂整个
                <span>推理系统</span>
              </h1>
              <p className="hero-lead">
                把 CPU 机头、GPU 执行、KV Cache 与 Serving Framework
                串成一条完整链路。先能听懂，再能定位，最后动手验证。
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href="#system-map">
                  开始学习 <span aria-hidden="true">↓</span>
                </a>
                <Link className="button button-ghost" href="/vllm">
                  进入 vLLM 专题 <span aria-hidden="true">→</span>
                </Link>
              </div>
              <PageProgress current={1} />
            </div>

            <div className="hero-console" aria-label="推理请求链路摘要">
              <div className="console-topline">
                <span>REQUEST TRACE</span>
                <span className="console-status">RUNNING</span>
              </div>
              <div className="trace-list">
                <div className="trace-row">
                  <span className="trace-index">01</span>
                  <div>
                    <b>Tokenize</b>
                    <small>CPU · text → token ids</small>
                  </div>
                  <span className="trace-time">8 ms</span>
                </div>
                <div className="trace-row">
                  <span className="trace-index">02</span>
                  <div>
                    <b>Schedule</b>
                    <small>CPU · batch + KV blocks</small>
                  </div>
                  <span className="trace-time">2 ms</span>
                </div>
                <div className="trace-row trace-active">
                  <span className="trace-index">03</span>
                  <div>
                    <b>Prefill</b>
                    <small>GPU · process prompt</small>
                  </div>
                  <span className="trace-time">TTFT</span>
                </div>
                <div className="trace-row">
                  <span className="trace-index">04</span>
                  <div>
                    <b>Decode Loop</b>
                    <small>GPU + KV · one token / step</small>
                  </div>
                  <span className="trace-time">TPOT</span>
                </div>
              </div>
              <div className="console-metrics">
                <div>
                  <small>KV WATERMARK</small>
                  <b>67%</b>
                  <span className="meter">
                    <i style={{ width: "67%" }} />
                  </span>
                </div>
                <div>
                  <small>GOODPUT</small>
                  <b>94%</b>
                  <span className="meter meter-lime">
                    <i style={{ width: "94%" }} />
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="signal-strip" aria-label="核心原则">
          <div className="shell signal-inner">
            <span>01 / 先分阶段</span>
            <strong>Prefill 影响 TTFT</strong>
            <span className="signal-arrow">→</span>
            <strong>Decode 影响 TPOT</strong>
            <span className="signal-arrow">→</span>
            <strong>KV 决定并发容量</strong>
          </div>
        </section>

        <section className="section shell" id="system-map">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p className="section-kicker">SYSTEM MAP</p>
              <h2>一条请求，六个关键站点</h2>
            </div>
            <p>
              看问题时不要先猜框架参数。先确认请求卡在排队、CPU、GPU、KV
              还是通信。
            </p>
          </div>

          <div className="pipeline" role="list" aria-label="LLM 推理请求链路">
            {[
              ["01", "API", "HTTP · 鉴权 · 限流", "cpu"],
              ["02", "TOKENIZE", "文本 → Token ID", "cpu"],
              ["03", "SCHEDULE", "排队 · 组 Batch · 分 KV", "cpu"],
              ["04", "PREFILL", "处理完整 Prompt", "gpu"],
              ["05", "KV CACHE", "保存每层 K / V 状态", "memory"],
              ["06", "DECODE", "循环生成新 Token", "gpu"],
            ].map(([index, title, detail, kind]) => (
              <article className={`pipeline-step ${kind}`} key={index} role="listitem">
                <div className="step-top">
                  <span>{index}</span>
                  <i>{kind.toUpperCase()}</i>
                </div>
                <h3>{title}</h3>
                <p>{detail}</p>
              </article>
            ))}
          </div>

          <div className="phase-compare">
            <article className="phase-card prefill-card">
              <div className="phase-label">P</div>
              <div>
                <span className="card-eyebrow">PREFILL PHASE</span>
                <h3>一次读完整个输入</h3>
                <p>
                  大矩阵、高并行，通常更偏计算受限。长 Prompt
                  会拉高首字等待，并可能打断正在输出的请求。
                </p>
              </div>
              <ul>
                <li>看 TTFT / Queue Time</li>
                <li>关注 Prompt 长度</li>
                <li>策略：Chunked Prefill</li>
              </ul>
            </article>
            <article className="phase-card decode-card">
              <div className="phase-label">D</div>
              <div>
                <span className="card-eyebrow">DECODE PHASE</span>
                <h3>每一步生成一个 Token</h3>
                <p>
                  每轮计算小，却要反复读取权重和 KV，通常更偏带宽受限。它决定用户感知到的输出流畅度。
                </p>
              </div>
              <ul>
                <li>看 TPOT / ITL</li>
                <li>关注活跃 Batch</li>
                <li>策略：Continuous Batching</li>
              </ul>
            </article>
          </div>
        </section>

        <section className="section section-dark">
          <div className="shell">
            <div className="section-heading light">
              <div>
                <span className="section-number">02</span>
                <p className="section-kicker">CPU HOST</p>
                <h2>CPU 不是配角，它负责让 GPU 不停工</h2>
              </div>
              <p>
                “CPU 侧顶不住”通常意味着请求处理、分词、调度或输出路径已经无法持续给 GPU 喂任务。
              </p>
            </div>

            <div className="cpu-grid">
              <article>
                <span className="role-id">A</span>
                <h3>控制面</h3>
                <p>API、鉴权、路由、Admission、Scheduler 与进程协调。</p>
                <div className="tag-row">
                  <span>单核性能</span>
                  <span>物理核</span>
                </div>
              </article>
              <article>
                <span className="role-id">B</span>
                <h3>数据面</h3>
                <p>Tokenizer、Chat Template、Detokenize、流式输出与网络栈。</p>
                <div className="tag-row">
                  <span>线程</span>
                  <span>NUMA</span>
                </div>
              </article>
              <article>
                <span className="role-id">C</span>
                <h3>内存层</h3>
                <p>模型加载、权重或 KV Offload，以及 GPU↔Host 数据搬运。</p>
                <div className="tag-row">
                  <span>DDR 带宽</span>
                  <span>PCIe</span>
                </div>
              </article>
            </div>

            <div className="diagnostic-quote">
              <span>DIAGNOSTIC RULE</span>
              <p>
                CPU 总利用率只有 20%，
                <strong>不代表 CPU 不是瓶颈。</strong>
                关键 Scheduler 线程可能已占满一个核。
              </p>
            </div>
          </div>
        </section>

        <section className="section shell">
          <div className="section-heading">
            <div>
              <span className="section-number">03</span>
              <p className="section-kicker">KV CACHE</p>
              <h2>权重基本固定，KV 随并发增长</h2>
            </div>
            <p>
              KV Cache 是 Attention 的中间状态，不是答案缓存。长上下文和高并发经常先打满 KV 容量。
            </p>
          </div>

          <div className="kv-layout">
            <div className="formula-card">
              <span className="card-eyebrow">CAPACITY FORMULA</span>
              <code>
                <span>2</span> × Layers × KV Heads × Head Dim × Bytes
              </code>
              <div className="formula-demo">
                <div>
                  <small>32 层 · 8 KV Heads · BF16</small>
                  <strong>128 KiB</strong>
                  <span>每 Token</span>
                </div>
                <div className="formula-multiply">×</div>
                <div>
                  <small>8K Context</small>
                  <strong>≈ 1 GiB</strong>
                  <span>每序列</span>
                </div>
              </div>
              <a
                href="https://github.com/HaotianChen616/ai-infra-learning/blob/main/labs/kv_cache_calculator.py"
                target="_blank"
                rel="noreferrer"
              >
                打开 KV 计算实验 <span aria-hidden="true">↗</span>
              </a>
            </div>

            <div className="concept-stack">
              {[
                [
                  "Paged KV / Block Manager",
                  "KV 切成固定 Block，按需分配，减少碎片。它是系统设计；不要与 v0.25.0 删除的旧 PagedAttention 实现混为一谈。",
                ],
                ["Prefix Cache", "相同 Token 前缀直接复用 KV，主要节省 Prefill。"],
                [
                  "Quantized KV",
                  "BF16 每元素 2 Bytes；FP8 / INT8 通常为 1 Byte，可近似把逻辑容量减半，但支持范围、精度与性能要实测。",
                ],
                ["KV Offload", "用 CPU/SSD 容量换 GPU 容量，但要支付传输成本。"],
                ["Preemption", "KV 紧张时暂停请求，通过 Evict 或 Recompute 腾空间。"],
              ].map(([title, body], index) => (
                <details key={title} open={index === 0}>
                  <summary>
                    <span>0{index + 1}</span>
                    {title}
                    <i aria-hidden="true">＋</i>
                  </summary>
                  <p>{body}</p>
                </details>
              ))}
            </div>
          </div>
        </section>

        <section className="section framework-section">
          <div className="shell">
            <div className="section-heading">
              <div>
                <span className="section-number">04</span>
                <p className="section-kicker">FRAMEWORKS</p>
                <h2>PyTorch 打地基，Serving Engine 管请求</h2>
              </div>
              <p>
                vLLM 和 SGLang 位于 PyTorch 之上。它们管理在线请求、调度、KV
                和分布式执行，而不是替代张量与算子运行时。
              </p>
            </div>

            <div className="stack-map">
              <div className="stack-layer app-layer">
                <span>APPLICATION</span>
                <strong>Agent · RAG · Chat · API Gateway</strong>
              </div>
              <div className="stack-layer serving-layer">
                <span>SERVING ENGINE</span>
                <strong>vLLM · SGLang</strong>
              </div>
              <div className="stack-layer runtime-layer">
                <span>MODEL RUNTIME</span>
                <strong>PyTorch · torch.compile · Triton</strong>
              </div>
              <div className="stack-layer hardware-layer">
                <span>HARDWARE</span>
                <strong>CPU · GPU · HBM · DDR · PCIe · NIC</strong>
              </div>
            </div>

            <div className="framework-cards">
              <Link className="framework-card vllm-card" href="/vllm">
                <div className="framework-card-top">
                  <span>专题 01</span>
                  <i aria-hidden="true">↗</i>
                </div>
                <h3>vLLM</h3>
                <p>从 Paged KV 出发，理解 Scheduler、KV Block Manager 与 V1 进程架构。</p>
                <ul>
                  <li>Continuous Batching</li>
                  <li>Chunked Prefill</li>
                  <li>Prefix Caching</li>
                </ul>
                <strong>进入学习路线 →</strong>
              </Link>
              <Link className="framework-card sglang-card" href="/sglang">
                <div className="framework-card-top">
                  <span>专题 02</span>
                  <i aria-hidden="true">↗</i>
                </div>
                <h3>SGLang</h3>
                <p>从 RadixAttention 出发，理解前缀复用、Overlap Scheduler 与 PD 分离。</p>
                <ul>
                  <li>Radix Cache</li>
                  <li>Overlap Scheduling</li>
                  <li>PD Disaggregation</li>
                </ul>
                <strong>进入学习路线 →</strong>
              </Link>
              <Link className="framework-card gateway-card" href="/gateway">
                <div className="framework-card-top">
                  <span>专题 03</span>
                  <i aria-hidden="true">↗</i>
                </div>
                <h3>Agent / Gateway</h3>
                <p>把推理引擎放回真实业务链路，理解路由、流控、工具调用、可观测性与提交边界。</p>
                <ul>
                  <li>Admission Control</li>
                  <li>Streaming</li>
                  <li>Tool Runtime</li>
                </ul>
                <strong>进入应用层路线 →</strong>
              </Link>
            </div>
          </div>
        </section>

        <section className="section shell metric-section">
          <div className="metric-intro">
            <span className="section-number">05</span>
            <p className="section-kicker">METRICS</p>
            <h2>开会先盯住这六个数</h2>
          </div>
          <div className="metric-grid">
            {[
              ["TTFT", "首 Token 延迟", "Prefill + 排队"],
              ["TPOT", "每 Token 耗时", "Decode 速度"],
              ["P99", "尾延迟", "最差一批用户"],
              ["GOODPUT", "有效吞吐", "满足 SLO 的请求"],
              ["KV %", "缓存水位", "并发容量压力"],
              ["TOK/S", "系统吞吐", "必须注明负载"],
            ].map(([metric, name, hint]) => (
              <article key={metric}>
                <strong>{metric}</strong>
                <p>{name}</p>
                <small>{hint}</small>
              </article>
            ))}
          </div>
          <div className="benchmark-rule">
            <span>BENCHMARK CHECK</span>
            <p>
              没有模型、硬件、输入输出长度、并发与 SLO 的“吞吐提升 50%”，
              <strong>没有可比性。</strong>
            </p>
          </div>
          <Link className="glossary-link" href="/glossary">
            不熟悉这些缩写？打开按类别索引的术语表 <span aria-hidden="true">→</span>
          </Link>
        </section>

        <section className="closing-cta">
          <div className="shell closing-inner">
            <div>
              <span>下一站 · SERVING ENGINE</span>
              <h2>开始下钻框架实现</h2>
              <p>先走 vLLM 主线，再用 SGLang 对照调度与 Prefix Cache 设计。</p>
            </div>
            <div className="closing-actions">
              <Link className="button button-light" href="/vllm">
                学习 vLLM <span aria-hidden="true">→</span>
              </Link>
              <Link className="button button-outline-light" href="/sglang">
                学习 SGLang
              </Link>
              <Link className="button button-outline-light" href="/gateway">
                学习 Agent / Gateway
              </Link>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
