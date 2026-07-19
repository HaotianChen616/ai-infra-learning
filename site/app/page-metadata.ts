import type { Metadata } from "next";

const socialImage = {
  url: "/og.png",
  width: 1734,
  height: 907,
  alt: "AI Infra Field Guide — 从一次请求，看懂整个推理系统",
};

export function createPageMetadata(
  title: string,
  description: string,
  path: string,
): Metadata {
  return {
    title,
    description,
    alternates: {
      canonical: path,
    },
    openGraph: {
      type: "website",
      locale: "zh_CN",
      url: path,
      siteName: "AI Infra Field Guide",
      title: `${title} · AI Infra Field Guide`,
      description,
      images: [socialImage],
    },
    twitter: {
      card: "summary_large_image",
      title: `${title} · AI Infra Field Guide`,
      description,
      images: [socialImage.url],
    },
  };
}
