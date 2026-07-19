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
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("server-renders the vLLM learning page", async () => {
  const response = await render("/vllm");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>vLLM 学习路线 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /PagedAttention/);
  assert.match(html, /Continuous Batching/);
  assert.match(html, /V1 PROCESS MAP/);
  assert.match(html, /href="\/sglang"/);
});

test("server-renders the SGLang learning page", async () => {
  const response = await render("/sglang");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>SGLang 学习路线 · AI Infra Field Guide<\/title>/i);
  assert.match(html, /RadixAttention/);
  assert.match(html, /OVERLAP SCHEDULER/);
  assert.match(html, /PD DISAGGREGATION/);
  assert.match(html, /href="\/vllm"/);
});
