import Link from "next/link";

type ActivePage = "home" | "vllm" | "sglang";

export function SiteHeader({ active }: { active: ActivePage }) {
  return (
    <header className="site-header">
      <div className="shell header-inner">
        <Link className="wordmark" href="/" aria-label="AI Infra Field Guide 首页">
          <span className="wordmark-mark">AI</span>
          <span className="wordmark-copy">
            <b>INFRA</b>
            <small>FIELD GUIDE</small>
          </span>
        </Link>

        <nav className="main-nav" aria-label="主导航">
          <Link className={active === "home" ? "active" : ""} href="/">
            知识地图
          </Link>
          <Link className={active === "vllm" ? "active" : ""} href="/vllm">
            vLLM
          </Link>
          <Link className={active === "sglang" ? "active" : ""} href="/sglang">
            SGLang
          </Link>
        </nav>

        <a
          className="header-github"
          href="https://github.com/HaotianChen616/ai-infra-learning"
          target="_blank"
          rel="noreferrer"
        >
          学习仓库 <span aria-hidden="true">↗</span>
        </a>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="shell footer-inner">
        <div>
          <span className="footer-kicker">AI INFRA FIELD GUIDE</span>
          <p>先建立系统地图，再下钻框架实现。</p>
        </div>
        <div className="footer-links">
          <Link href="/">知识地图</Link>
          <Link href="/vllm">vLLM</Link>
          <Link href="/sglang">SGLang</Link>
          <a
            href="https://github.com/HaotianChen616/ai-infra-learning"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}

export function PageProgress({ current }: { current: 1 | 2 | 3 }) {
  return (
    <div className="page-progress" aria-label={`学习路径，第 ${current} 站，共 3 站`}>
      {[1, 2, 3].map((step) => (
        <span className={step <= current ? "filled" : ""} key={step} />
      ))}
      <small>0{current} / 03</small>
    </div>
  );
}
