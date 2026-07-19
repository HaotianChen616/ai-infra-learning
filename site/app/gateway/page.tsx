import Link from "next/link";
import { createPageMetadata } from "../page-metadata";
import { PageProgress, SiteFooter, SiteHeader } from "../site-chrome";

export const metadata = createPageMetadata(
  "Agent / Gateway 学习路线",
  "从 Admission、路由、流式输出到工具调用与提交边界，理解推理引擎之上的应用层。",
  "/gateway",
);

const responsibilities = [
  {
    index: "01",
    title: "Admission & Routing",
    body: "鉴权、配额、限流、优先级、模型选择与容量感知路由；先在入口控制排队，不把所有压力直接推给引擎。",
  },
  {
    index: "02",
    title: "Prompt & Context",
    body: "统一 Chat Template、System Prompt、工具 Schema 与上下文裁剪；Token 变化会直接影响 Prefix Cache 命中。",
  },
  {
    index: "03",
    title: "Streaming & Cancel",
    body: "把首包、内容块、结束信号和错误语义定义清楚；客户端断开后要尽快取消下游生成并释放 KV。",
  },
  {
    index: "04",
    title: "Tool Runtime",
    body: "验证工具参数、处理超时与重试、保存幂等键；模型文本不是授权，真正副作用要经过策略与校验。",
  },
  {
    index: "05",
    title: "Reliability",
    body: "区分连接失败、模型失败、解析失败与工具失败；重试必须知道是否已经产生 Token 或外部副作用。",
  },
  {
    index: "06",
    title: "Observability",
    body: "用同一 Trace 串起 Gateway、Engine、Worker 与 Tool；否则只能看到总延迟，无法定位排队、Prefill 或工具耗时。",
  },
];

export default function GatewayPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        跳到正文
      </a>
      <SiteHeader active="gateway" />
      <main id="main" className="framework-page gateway-page">
        <section className="framework-hero">
          <div className="shell">
            <div className="breadcrumb">
              <Link href="/">AI Infra</Link>
              <span>/</span>
              <b>Agent / Gateway</b>
            </div>
            <div className="framework-hero-grid">
              <div>
                <div className="framework-badge">APPLICATION LAYER · 03</div>
                <h1>Agent / Gateway</h1>
                <p className="framework-thesis">
                  先把它理解成：
                  <strong>推理入口 + 状态机 + 工具安全边界</strong>
                </p>
                <p className="framework-lead">
                  Serving Engine 负责高效生成；Gateway
                  决定谁能进、去哪个模型、如何流式返回，以及什么时候允许模型输出变成真实副作用。
                </p>
                <div className="hero-actions">
                  <a className="button button-primary" href="#request-path">
                    开始路线 <span aria-hidden="true">↓</span>
                  </a>
                  <Link className="button button-ghost-on-dark" href="/glossary">
                    打开术语表 →
                  </Link>
                </div>
                <PageProgress current={4} />
              </div>
              <div className="framework-diagram gateway-diagram">
                <div className="diagram-title">
                  <span>END-TO-END TRACE</span>
                  <i>ONE REQUEST</i>
                </div>
                {[
                  ["GATEWAY", "auth · quota · route"],
                  ["AGENT RUNTIME", "plan · tool · memory"],
                  ["SERVING ENGINE", "queue · prefill · decode"],
                  ["TOOL / CLIENT", "side effect · stream"],
                ].map(([title, detail], index) => (
                  <div key={title}>
                    {index > 0 && <div className="diagram-arrow">↓ trace context</div>}
                    <div className={`process-row ${index === 2 ? "process-core" : ""}`}>
                      <span>{title}</span>
                      <small>{detail}</small>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="framework-summary">
          <div className="shell summary-grid">
            <div>
              <small>ENTRY CONTROL</small>
              <strong>Admission before queue</strong>
              <span>容量不足时明确拒绝或降级</span>
            </div>
            <div>
              <small>STATE MODEL</small>
              <strong>Request ≠ one HTTP call</strong>
              <span>一次任务可能跨多轮模型与工具</span>
            </div>
            <div>
              <small>SAFETY BOUNDARY</small>
              <strong>Validate before commit</strong>
              <span>模型输出不能直接变成副作用</span>
            </div>
          </div>
        </section>

        <section className="section shell" id="request-path">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p className="section-kicker">RESPONSIBILITY MAP</p>
              <h2>应用层要管的六件事</h2>
            </div>
            <p>出现慢、贵、不稳定或越权时，先确认责任属于 Gateway、Agent、引擎还是工具。</p>
          </div>
          <div className="mechanism-grid">
            {responsibilities.map((item) => (
              <article key={item.index}>
                <div className="mechanism-top">
                  <span>{item.index}</span>
                  <i>APPLICATION</i>
                </div>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section section-dark">
          <div className="shell">
            <div className="section-heading light">
              <div>
                <span className="section-number">02</span>
                <p className="section-kicker">LATENCY BUDGET</p>
                <h2>总延迟要拆到每一跳</h2>
              </div>
              <p>只看请求总耗时，会把排队、模型生成、工具执行和客户端背压混在一起。</p>
            </div>
            <div className="gateway-budget">
              {[
                ["QUEUE", "Admission → Engine", "queue_time"],
                ["FIRST TOKEN", "Tokenize + Prefill", "ttft"],
                ["STREAM", "Decode + Network", "itl / tpot"],
                ["TOOL", "Parse + Execute", "tool_latency"],
                ["END-TO-END", "全链路完成", "task_latency"],
              ].map(([name, detail, metric]) => (
                <article key={name}>
                  <span>{name}</span>
                  <p>{detail}</p>
                  <code>{metric}</code>
                </article>
              ))}
            </div>
            <div className="diagnostic-quote gateway-rule">
              <span>TRACE RULE</span>
              <p>
                Gateway 必须把 request / session / tool call 与下游 engine request
                关联起来，才能从 P99 追到具体排队阶段、Worker 和工具调用。
              </p>
            </div>
          </div>
        </section>

        <section className="section shell">
          <div className="section-heading">
            <div>
              <span className="section-number">03</span>
              <p className="section-kicker">COMMIT BOUNDARY</p>
              <h2>流式结果何时算“可以相信”</h2>
            </div>
            <p>对纯文本聊天，Token 可以边生成边展示；对工具调用和有副作用动作，边界必须更谨慎。</p>
          </div>
          <div className="boundary-grid">
            <article>
              <span>AUTOREGRESSIVE</span>
              <h3>已输出 Token 通常不会回改</h3>
              <p>可以增量展示文本，但工具参数仍应等到结构闭合、Schema 校验和策略检查后再提交。</p>
            </article>
            <article>
              <span>DIFFUSION / REVISION</span>
              <h3>中间结果可能不是最终承诺</h3>
              <p>
                当生成范式允许迭代修订时，应用层需要把 commit boundary
                后移：展示可标记为 provisional，外部副作用只接受稳定且已验证的结果。
              </p>
            </article>
            <article>
              <span>TOOL SIDE EFFECT</span>
              <h3>不可逆动作最后执行</h3>
              <p>发送消息、下单、删数据等动作要具备授权、幂等键、审计记录和明确的超时 / 重试语义。</p>
            </article>
          </div>
        </section>

        <section className="section framework-lab-section">
          <div className="shell lab-grid">
            <div>
              <span className="section-number">04</span>
              <p className="section-kicker">HARDWARE LENS</p>
              <h2>异构后端，先保持同一套问题框架</h2>
              <p>
                在 CUDA GPU、Ascend NPU、XPU
                等平台上，请求生命周期和指标仍相通；真正不同的是 Attention
                Backend、算子覆盖、内存布局、图编译与通信实现。
              </p>
            </div>
            <div className="checklist-card">
              {[
                "固定模型、精度与请求分布",
                "确认后端实际选中的 Kernel",
                "分别采集 CPU、设备与链路指标",
                "记录 KV 精度、Block Size 与容量",
                "检查图编译 / 重编译与动态形状",
                "注明框架、插件、驱动与固件版本",
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
              <span>REFERENCE</span>
              <h2>遇到缩写，随时回来查</h2>
              <p>术语表按指标、请求阶段、KV、并行与硬件分类。</p>
            </div>
            <Link className="button button-light" href="/glossary">
              打开术语表 <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
