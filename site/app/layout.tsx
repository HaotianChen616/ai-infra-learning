import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const siteDescription =
  "从 CPU 机头、KV Cache 到 vLLM 与 SGLang，一站式建立大模型推理基础设施知识地图。";

export async function generateMetadata(): Promise<Metadata> {
  const incomingHeaders = await headers();
  const host =
    incomingHeaders.get("x-forwarded-host") ?? incomingHeaders.get("host");
  const protocol =
    incomingHeaders.get("x-forwarded-proto") ??
    (host?.startsWith("localhost") ? "http" : "https");
  const origin = host ? `${protocol}://${host}` : "http://localhost:3000";

  return {
    metadataBase: new URL(origin),
    title: {
      default: "AI Infra Field Guide",
      template: "%s · AI Infra Field Guide",
    },
    description: siteDescription,
    keywords: [
      "AI Infra",
      "LLM Inference",
      "KV Cache",
      "vLLM",
      "SGLang",
      "PyTorch",
    ],
    openGraph: {
      type: "website",
      locale: "zh_CN",
      url: "/",
      siteName: "AI Infra Field Guide",
      title: "AI Infra Field Guide",
      description: siteDescription,
      images: [
        {
          url: "/og.png",
          width: 1734,
          height: 907,
          alt: "AI Infra Field Guide — 从一次请求，看懂整个推理系统",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "AI Infra Field Guide",
      description: siteDescription,
      images: ["/og.png"],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
