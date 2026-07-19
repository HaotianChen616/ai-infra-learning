import Link from "next/link";
import { createPageMetadata } from "../page-metadata";
import { PageProgress, SiteFooter, SiteHeader } from "../site-chrome";

export const metadata = createPageMetadata(
  "SGLang 学习路线",
  "从 RadixAttention、Overlap Scheduler 到 PD 分离，循序理解 SGLang Runtime。",
  "/sglang",
);

export default function SglangPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        跳到正文
      </a>
      <SiteHeader active="sglang" />
      <main id="main" className="framework-page sglang-page">
        <section className="framework-hero">
          <div className="shell">
            <div className="breadcrumb">
              <Link href="/">AI Infra</Link>
              <span>/</span>
              <b>SGLang</b>
            </div>
            <div className="framework-hero-grid">
              <div>
                <div className="framework-badge">
                  SERVING ENGINE · 02 · v0.5.15.post1
                </div>
                <h1>SGLang</h1>
                <p className="framework-thesis">
                  先把它理解成：
                  <strong>前缀感知的 KV Runtime + 高性能 CPU Scheduler</strong>
                </p>
                <p className="framework-lead">
                  学习重点是 RadixAttention 如何组织共享前缀，CPU Scheduler
                  如何与 GPU 执行重叠，以及 Prefill/Decode 为什么要分离。
                </p>
                <div className="hero-actions">
                  <a className="button button-primary" href="#radix">
                    开始路线 <span aria-hidden="true">↓</span>
                  </a>
                  <a
                    className="button button-ghost-on-dark"
                    href="https://docs.sglang.io/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    官方文档 ↗
                  </a>
                </div>
                <PageProgress current={3} />
              </div>
              <div className="radix-visual" aria-label="Radix Tree 前缀复用示意">
                <div className="diagram-title">
                  <span>RADIX CACHE</span>
                  <i>3 REQUESTS</i>
                </div>
                <div className="radix-root">SYSTEM PROMPT</div>
                <div className="radix-line" />
                <div className="radix-shared">SHARED DOCUMENT PREFIX</div>
                <div className="radix-branches">
                  <div>
                    <span>Q1</span>
                    <small>What is KV?</small>
                  </div>
                  <div>
                    <span>Q2</span>
                    <small>Why cache it?</small>
                  </div>
                  <div>
                    <span>Q3</span>
                    <small>How to offload?</small>
                  </div>
                </div>
                <p>
                  <span className="reuse-dot" /> 共享前缀只计算一次，后续请求从分叉点继续。
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="framework-summary">
          <div className="shell summary-grid">
            <div>
              <small>MENTAL MODEL</small>
              <strong>Prefix-aware Runtime</strong>
              <span>Radix Tree 组织 KV</span>
            </div>
            <div>
              <small>CPU OPTIMIZATION</small>
              <strong>Overlap Scheduler</strong>
              <span>准备下一批时 GPU 正在执行</span>
            </div>
            <div>
              <small>ARCHITECTURE</small>
              <strong>Prefill / Decode</strong>
              <span>可拆成不同资源池</span>
            </div>
          </div>
        </section>

        <section className="section shell" id="radix">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p className="section-kicker">RADIX ATTENTION</p>
              <h2>先理解“前缀”为什么值得复用</h2>
            </div>
            <p>
              Agent、长文档问答和多轮对话经常重复 System Prompt、工具描述或历史上下文。
            </p>
          </div>
          <div className="radix-explainer">
            <div className="token-rows">
              <div>
                <span>REQ A</span>
                <i className="shared-token">SYSTEM</i>
                <i className="shared-token">DOC</i>
                <i className="unique-token">QUESTION A</i>
              </div>
              <div>
                <span>REQ B</span>
                <i className="shared-token">SYSTEM</i>
                <i className="shared-token">DOC</i>
                <i className="unique-token">QUESTION B</i>
              </div>
              <div>
                <span>REQ C</span>
                <i className="shared-token">SYSTEM</i>
                <i className="unique-token">OTHER DOC</i>
                <i className="unique-token">QUESTION C</i>
              </div>
            </div>
            <div className="radix-notes">
              <article>
                <span>1</span>
                <div>
                  <h3>精确 Token 前缀</h3>
                  <p>不是语义相似。Chat Template 或特殊 Token 不同也可能无法命中。</p>
                </div>
              </article>
              <article>
                <span>2</span>
                <div>
                  <h3>最长公共前缀</h3>
                  <p>Radix Tree 找到可复用的最长路径，从分叉处继续 Prefill。</p>
                </div>
              </article>
              <article>
                <span>3</span>
                <div>
                  <h3>缓存感知调度</h3>
                  <p>命中多少 KV 会影响请求排序、Eviction 和有效吞吐。</p>
                </div>
              </article>
            </div>
          </div>
        </section>

        <section className="section section-dark">
          <div className="shell">
            <div className="section-heading light">
              <div>
                <span className="section-number">02</span>
                <p className="section-kicker">OVERLAP SCHEDULER</p>
                <h2>让 CPU 准备与 GPU 执行重叠</h2>
              </div>
              <p>
                Decode Step 很短时，CPU Scheduler、元数据准备和 Kernel
                Dispatch 可能进入关键路径。
              </p>
            </div>
            <div className="overlap-board">
              <div className="timeline-labels">
                <span>TIME →</span>
                <span>T0</span>
                <span>T1</span>
                <span>T2</span>
              </div>
              <div className="timeline-row">
                <b>GPU</b>
                <span className="timeline-block gpu-block">EXECUTE BATCH N</span>
                <span className="timeline-block gpu-block">EXECUTE N+1</span>
                <span className="timeline-block gpu-block">EXECUTE N+2</span>
              </div>
              <div className="timeline-row">
                <b>CPU</b>
                <span className="timeline-block idle-block">—</span>
                <span className="timeline-block cpu-block">PREPARE N+1</span>
                <span className="timeline-block cpu-block">PREPARE N+2</span>
              </div>
              <div className="overlap-callout">
                <span>OVERLAP WINDOW</span>
                <p>GPU 执行当前 Batch 的同时，CPU 处理下一轮请求状态和 Attention Metadata。</p>
              </div>
            </div>
          </div>
        </section>

        <section className="section shell">
          <div className="section-heading">
            <div>
              <span className="section-number">03</span>
              <p className="section-kicker">RUNTIME MAP</p>
              <h2>沿四个模块追踪请求</h2>
            </div>
            <p>先理解进程与职责，再看 Radix Cache、调度策略和硬件 Backend 的细节。</p>
          </div>
          <div className="sglang-runtime">
            {[
              ["01", "Tokenizer Manager", "接收请求、输入处理、管理流式响应状态。"],
              ["02", "Scheduler", "维护 running/waiting batch、Radix Cache 与 KV 内存池。"],
              ["03", "Model Worker", "模型 Forward、Attention Backend 与分布式通信。"],
              ["04", "Detokenizer", "Token ID 转文本、处理增量输出与结束状态。"],
            ].map(([index, title, body]) => (
              <article key={index}>
                <div>
                  <span>{index}</span>
                  <i aria-hidden="true">→</i>
                </div>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
          </div>
          <div className="pinned-source-row">
            <span>SOURCE · v0.5.15.post1</span>
            <a
              href="https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/managers/tokenizer_manager.py"
              target="_blank"
              rel="noreferrer"
            >
              Tokenizer Manager ↗
            </a>
            <a
              href="https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/managers/scheduler.py"
              target="_blank"
              rel="noreferrer"
            >
              Scheduler ↗
            </a>
            <a
              href="https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/mem_cache/radix_cache.py"
              target="_blank"
              rel="noreferrer"
            >
              Radix Cache ↗
            </a>
          </div>
        </section>

        <section className="section pd-section">
          <div className="shell">
            <div className="section-heading">
              <div>
                <span className="section-number">04</span>
                <p className="section-kicker">PD DISAGGREGATION</p>
                <h2>Prefill 与 Decode 可以用不同资源池</h2>
              </div>
              <p>
                目标是分别优化 TTFT 和 TPOT，并隔离长 Prefill 对 Decode
                尾延迟的干扰，不是自动提升所有场景吞吐。
              </p>
            </div>
            <div className="pd-diagram">
              <div className="pd-router">
                <span>ROUTER</span>
                <small>request-aware routing</small>
              </div>
              <div className="pd-arrow">↓</div>
              <div className="pd-pools">
                <article className="prefill-pool">
                  <span>PREFILL POOL</span>
                  <h3>Compute-oriented</h3>
                  <p>处理长 Prompt，生成完整 KV。</p>
                  <div>TTFT · TP/PP 配置</div>
                </article>
                <div className="kv-transfer">
                  <span>KV TRANSFER</span>
                  <i>→ → →</i>
                  <small>RDMA / NIXL / Mooncake</small>
                </div>
                <article className="decode-pool">
                  <span>DECODE POOL</span>
                  <h3>Bandwidth-oriented</h3>
                  <p>接收 KV，持续生成输出 Token。</p>
                  <div>TPOT · Tail ITL</div>
                </article>
              </div>
            </div>
          </div>
        </section>

        <section className="section shell">
          <div className="section-heading">
            <div>
              <span className="section-number">05</span>
              <p className="section-kicker">COMPARE</p>
              <h2>与 vLLM 对照着学</h2>
            </div>
            <p>两者能力大量重叠；学习时关注核心数据结构和调度思路，而不是先争论谁永远更快。</p>
          </div>
          <div className="compare-table" role="table" aria-label="vLLM 与 SGLang 学习对照">
            <div className="compare-row compare-head" role="row">
              <span role="columnheader">观察角度</span>
              <span role="columnheader">vLLM</span>
              <span role="columnheader">SGLang</span>
            </div>
            {[
              ["KV 组织", "Paged KV Blocks", "Radix Tree + KV Pool"],
              ["前缀复用", "Automatic Prefix Cache", "RadixAttention"],
              ["CPU 调度", "Engine Core Scheduler", "Overlap Scheduler"],
              ["长 Prefill", "Chunked Prefill", "Chunked Prefill"],
              ["PD 分离", "KV Connector / Transfer", "Disaggregation Runtime"],
              ["学习主线", "Block Table 与调度循环", "Radix Cache 与 Scheduler 状态"],
            ].map((row) => (
              <div className="compare-row" role="row" key={row[0]}>
                {row.map((cell) => (
                  <span role="cell" key={cell}>
                    {cell}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </section>

        <section className="section framework-lab-section">
          <div className="shell lab-grid">
            <div>
              <span className="section-number">06</span>
              <p className="section-kicker">PRACTICE CHECKLIST</p>
              <h2>跑同一负载，做一张对照表</h2>
              <p>
                固定模型、精度、输入/输出长度和到达率，只改变框架或一个参数，记录 TTFT、TPOT、Goodput 与 KV 水位。
              </p>
              <a
                className="button button-primary"
                href="https://github.com/HaotianChen616/ai-infra-learning/tree/main/labs"
                target="_blank"
                rel="noreferrer"
              >
                打开实验代码 ↗
              </a>
            </div>
            <div className="checklist-card">
              {[
                "同一模型与量化精度",
                "固定输入 / 输出长度分布",
                "固定请求到达率与并发",
                "分别记录 p50 / p99",
                "检查 Prefix Cache 命中率",
                "注明框架版本与硬件拓扑",
              ].map((item, index) => (
                <div key={item}>
                  <span>0{index + 1}</span>
                  <p>{item}</p>
                  <i>□</i>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="next-framework">
          <div className="shell next-framework-inner">
            <div>
              <span>NEXT · 04 / 04</span>
              <h2>把推理引擎接回业务入口</h2>
              <p>继续学习 Agent / Gateway 的路由、流控、工具调用与可观测性。</p>
            </div>
            <Link className="button button-light" href="/gateway">
              前往 Agent / Gateway <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
