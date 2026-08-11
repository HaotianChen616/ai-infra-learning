#!/usr/bin/env python3
"""OpenAI-SDK 压测脚本:绕开 vllm bench serve 的 tokenizer 初始化坑。

按手册06 (docs/06-vllm-cuda-sync-profiling.md) 的负载定义:
  - 每请求固定 INPUT_TOKENS 输入 / OUTPUT_TOKENS 输出
  - CONCURRENCY 个请求同时到达 (request_rate=inf)
  - ignore_eos 确保每请求生成满 OUTPUT_TOKENS
  - temperature=0
记录 TTFT / TPOT / 吞吐,存 JSON。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from statistics import median


async def one_request(client: "AsyncOpenAI", model: str, prompt: str,
                      input_tokens: int, output_tokens: int) -> dict:
    t0 = time.perf_counter()
    first_tok = None
    n_tok = 0
    try:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=output_tokens,
            temperature=0,
            stream=True,
            extra_body={"ignore_eos": True, "min_tokens": output_tokens},
        )
        async for chunk in stream:
            if first_tok is None:
                first_tok = time.perf_counter()
            # completions stream chunk.choices[0].text
            try:
                n_tok += 1 if chunk.choices[0].text else 0
            except Exception:
                pass
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "tpot_ms": None}
    t_end = time.perf_counter()
    ttft_ms = (first_tok - t0) * 1000 if first_tok else None
    total_ms = (t_end - t0) * 1000
    # decode 时间 ≈ 总时间 - TTFT;decode token 数 ≈ output_tokens
    decode_ms = total_ms - (ttft_ms or 0)
    tpot_ms = decode_ms / output_tokens if output_tokens else None
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tpot_ms": tpot_ms,
        "tokens": n_tok,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-requests", type=int, default=None,
                    help="默认等于 concurrency")
    ap.add_argument("--label", default="run")
    ap.add_argument("--result-dir", default="artifacts/vllm_cuda_sync/results")
    args = ap.parse_args()

    num = args.num_requests or args.concurrency
    # 构造固定长度输入:用重复 token,让 prefill 量≈input_tokens
    # 用一个常见词重复,vllm 会按 BPE 分词,大致接近
    unit = "hello world " * 2  # 约 4 token
    repeat = max(1, args.input_tokens // 4)
    prompt = unit * repeat

    from openai import AsyncOpenAI  # 延迟导入: 让 --help / py_compile 不依赖 openai
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    print(f"发送 {num} 请求, 并发 {args.concurrency}, "
          f"输入~{args.input_tokens}tok, 输出 {args.output_tokens}tok")

    t_start = time.perf_counter()
    tasks = [one_request(client, args.model, prompt,
                         args.input_tokens, args.output_tokens)
             for _ in range(num)]
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start

    ok = [r for r in results if r.get("ttft_ms") is not None]
    errs = [r for r in results if "error" in r]
    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok]
    total_tok = sum(r["tokens"] for r in ok)

    summary = {
        "label": args.label,
        "num_requests": num,
        "concurrency": args.concurrency,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "wall_seconds": wall,
        "successful_requests": len(ok),
        "failed_requests": len(errs),
        "total_output_tokens": total_tok,
        "output_tok_per_sec": total_tok / wall if wall else 0,
        "ttft_ms_p50": median(ttft) if ttft else None,
        "ttft_ms_max": max(ttft) if ttft else None,
        "tpot_ms_p50": median(tpot) if tpot else None,
        "tpot_ms_max": max(tpot) if tpot else None,
    }
    if errs:
        summary["errors"] = errs[:3]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n保存到 {out / (args.label + '.json')}")


if __name__ == "__main__":
    asyncio.run(main())
