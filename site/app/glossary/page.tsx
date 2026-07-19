import Link from "next/link";
import { createPageMetadata } from "../page-metadata";
import { SiteFooter, SiteHeader } from "../site-chrome";

export const metadata = createPageMetadata(
  "AI Infra 术语表",
  "按性能指标、请求阶段、KV 内存、调度、并行与硬件分类速查 AI Infra 常用术语。",
  "/glossary",
);

const groups = [
  {
    id: "metrics",
    title: "性能与体验指标",
    terms: [
      ["TTFT", "Time To First Token；从请求到首个输出 Token 的延迟，含排队、CPU 前处理与 Prefill。"],
      ["TPOT", "Time Per Output Token；通常用总生成时间除以输出 Token 数，描述平均 Decode 速度。"],
      ["ITL", "Inter-Token Latency；相邻输出 Token 或流式 Chunk 的间隔，更适合观察抖动与尾延迟。"],
      ["Throughput", "单位时间处理的请求或 Token；必须同时说明模型、长度分布、并发与硬件。"],
      ["Goodput", "满足既定 SLO 的有效吞吐；比单纯追求总吞吐更贴近线上价值。"],
      ["P50 / P99", "延迟分布的中位数与高分位数；平均值看不见少数慢请求。"],
    ],
  },
  {
    id: "lifecycle",
    title: "请求生命周期",
    terms: [
      ["Admission Control", "在请求进入引擎前，根据容量、优先级与 SLO 决定接收、排队、限流或拒绝。"],
      ["Prefill", "一次处理输入 Prompt，生成各层 KV；通常并行度高，长 Prompt 会直接推高 TTFT。"],
      ["Decode", "利用已有 KV 逐步生成新 Token；循环次数多，常更受显存带宽与调度开销影响。"],
      ["Continuous Batching", "每个调度步都允许请求进入或退出，让不同长度的序列共享动态 Batch。"],
      ["Chunked Prefill", "把长 Prompt 切块调度，避免一次 Prefill 长时间阻塞正在 Decode 的请求。"],
      ["Backpressure", "下游消费变慢时向上游传递压力，避免流式缓冲、连接和内存无限增长。"],
    ],
  },
  {
    id: "kv",
    title: "KV 与内存",
    terms: [
      ["KV Cache", "Attention 保存的 Key / Value 中间状态；避免每个 Decode Step 重算全部历史 Token。"],
      ["Paged KV", "把 KV 切成固定 Block 并通过 Block Table 间接寻址，以支持按需分配与复用。"],
      ["Block Table", "逻辑 Token Block 到物理 KV Block 的映射表，是分页 KV 的关键元数据。"],
      ["Prefix Cache", "精确 Token 前缀命中后复用已有 KV，主要减少重复 Prefill。"],
      ["KV Offload", "把部分 KV 放到 CPU、SSD 或远端内存，用传输成本换取 GPU 容量。"],
      ["Preemption", "KV 或调度资源不足时暂挂请求；恢复时可能重新加载 KV 或重新计算。"],
      ["KV Quantization", "以 FP8、INT8、INT4 等更低精度保存 KV；节省容量，但需核对后端支持与精度影响。"],
    ],
  },
  {
    id: "parallel",
    title: "并行与分布式",
    terms: [
      ["TP", "Tensor Parallel；把单层张量计算切到多卡，需要频繁集合通信。"],
      ["PP", "Pipeline Parallel；把不同层放到不同设备，吞吐取决于流水线气泡和 micro-batch。"],
      ["DP", "Data Parallel；复制模型处理不同请求，扩吞吐相对直接，但每份副本都占模型内存。"],
      ["EP", "Expert Parallel；MoE 模型把 Expert 分散到不同设备，重点看 All-to-All 与负载均衡。"],
      ["PD Disaggregation", "把 Prefill 与 Decode 放到不同资源池，并传输 KV，以分别优化 TTFT 与 TPOT。"],
      ["KV Connector", "负责跨进程、跨节点或跨存储层交换 KV 的接口或实现。"],
    ],
  },
  {
    id: "hardware",
    title: "硬件与运行时",
    terms: [
      ["HBM", "加速卡上的高带宽内存；模型权重、KV 与中间张量通常共同争用它。"],
      ["NUMA", "多路 CPU 的非一致内存访问；线程、网卡与 GPU 跨 NUMA 会增加延迟和带宽压力。"],
      ["PCIe / CXL", "CPU、加速器与扩展内存之间的数据通路；Offload 与模型加载常受其限制。"],
      ["CUDA Graph", "录制并重放稳定 GPU 执行图，减少 CPU Launch 开销，但对动态形状和地址有约束。"],
      ["Attention Backend", "具体执行 Attention 的 Kernel 路径，如 FlashAttention、FlashInfer 或硬件厂商后端。"],
      ["torch.compile", "PyTorch 的图捕获与编译入口；可能融合算子并减少 Python 开销，也可能触发重编译。"],
    ],
  },
];

export default function GlossaryPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        跳到正文
      </a>
      <SiteHeader active="glossary" />
      <main id="main" className="glossary-page">
        <section className="glossary-hero">
          <div className="shell">
            <div className="breadcrumb">
              <Link href="/">AI Infra</Link>
              <span>/</span>
              <b>Glossary</b>
            </div>
            <p className="framework-badge">QUICK REFERENCE · 31 TERMS</p>
            <h1>术语表</h1>
            <p>
              开会听到缩写时，先用一句话确认大家说的是同一个东西；需要建立上下文，再回到知识地图。
            </p>
            <nav className="glossary-index" aria-label="术语分类">
              {groups.map((group) => (
                <a href={`#${group.id}`} key={group.id}>
                  {group.title}
                </a>
              ))}
            </nav>
          </div>
        </section>

        <div className="shell glossary-groups">
          {groups.map((group, groupIndex) => (
            <section className="glossary-group" id={group.id} key={group.id}>
              <div className="glossary-group-heading">
                <span>0{groupIndex + 1}</span>
                <h2>{group.title}</h2>
              </div>
              <dl>
                {group.terms.map(([term, definition]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{definition}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>

        <section className="closing-cta">
          <div className="shell closing-inner">
            <div>
              <span>BACK TO SYSTEM</span>
              <h2>术语要放回链路里理解</h2>
              <p>从一次请求重新串起 CPU、GPU、KV、调度和性能指标。</p>
            </div>
            <Link className="button button-light" href="/#system-map">
              返回知识地图 <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
