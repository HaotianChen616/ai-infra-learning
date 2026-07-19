import assert from "node:assert/strict";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set(
    "test",
    `${pathname}-${process.pid}-${Date.now()}-${Math.random()}`,
  );
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the AI Infra knowledge map", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>知识地图 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /从一次请求/);
  assert.match(html, /一条请求，六个关键站点/);
  assert.match(html, /KV CACHE/);
  assert.match(html, /href="\/vllm"/);
  assert.match(html, /href="\/sglang"/);
  assert.match(html, /href="\/gateway"/);
  assert.match(html, /href="\/glossary"/);
  assert.match(html, /property="og:title" content="知识地图 · AI Infra Field Guide"/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("server-renders the vLLM learning page", async () => {
  const response = await render("/vllm");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>vLLM 学习路线 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /Paged KV \/ Block Manager/);
  assert.match(html, /v0\.25\.1/);
  assert.match(html, /v0\.25\.0 删除的是旧的/);
  assert.match(html, /Continuous Batching/);
  assert.match(html, /V1 PROCESS MAP/);
  assert.match(html, /property="og:title" content="vLLM 学习路线 · AI Infra Field Guide"/i);
  assert.match(html, /href="\/sglang"/);
});

test("server-renders the SGLang learning page", async () => {
  const response = await render("/sglang");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>SGLang 学习路线 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /RadixAttention/);
  assert.match(html, /v0\.5\.15\.post1/);
  assert.match(html, /OVERLAP SCHEDULER/);
  assert.match(html, /PD DISAGGREGATION/);
  assert.match(html, /property="og:title" content="SGLang 学习路线 · AI Infra Field Guide"/i);
  assert.match(html, /href="\/vllm"/);
});

test("server-renders the Agent and Gateway learning page", async () => {
  const response = await render("/gateway");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>Agent \/ Gateway 学习路线 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /Admission &amp; Routing/);
  assert.match(html, /COMMIT BOUNDARY/);
  assert.match(html, /异构后端/);
  assert.match(
    html,
    /property="og:title" content="Agent \/ Gateway 学习路线 · AI Infra Field Guide"/i,
  );
  assert.match(html, /href="\/glossary"/);
});

test("server-renders the categorized glossary", async () => {
  const response = await render("/glossary");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>AI Infra 术语表 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /性能与体验指标/);
  assert.match(html, /KV Quantization/);
  assert.match(html, /PD Disaggregation/);
  assert.match(html, /property="og:title" content="AI Infra 术语表 · AI Infra Field Guide"/i);
  assert.match(html, /href="\/#system-map"/);
});
