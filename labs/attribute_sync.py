#!/usr/bin/env python3
"""手册06 第12节: CUDA 同步 API 长调用归因。

对每个超过阈值的 Host 同步 API (cudaDeviceSynchronize / cudaStreamSynchronize /
cudaEventSynchronize 等), 查它在 [start, end] 时间窗口内:
  - GPU 是否有 kernel 在跑   -> 情况 A (GPU backlog 等待)
  - GPU 是否有 memcpy 在跑   -> 情况 D (PCIe/DMA 等待)
  - 都没有                  -> 情况 E (Runtime/Driver overhead, 无 GPU 工作)

输出每个长调用的归因 + 汇总统计。
注意: 本次采集未开 --sample/--cpuctxsw, 无法拆 CPU Running/Blocked/Ready,
所以情况 B/C (OS 调度尾延迟) 只能标注为 "需 thread state 才能区分"。
"""
from __future__ import annotations
import sqlite3
import statistics
import json
from pathlib import Path

DB = "artifacts/vllm_cuda_sync/nsys/qwen3_awq_full.sqlite"
THRESHOLD_US = 1000  # 只归因 > 1ms 的长同步调用
OUT = Path("artifacts/vllm_cuda_sync/analysis/bs8_1024_128/long_sync_attribution.json")


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    # 1. 取所有 Host 同步 API 调用
    syncs = con.execute("""
        SELECT r.start, r.end, (r.end-r.start) AS dur_ns, s.value AS name,
               r.globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE 'cuda%Synchronize%' OR s.value LIKE 'cu%Synchronize%'
    """).fetchall()

    # 2. 取所有 GPU kernel 和 memcpy (用 start/end 索引加速重叠查询)
    kernels = con.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    """).fetchall()
    memcpys = con.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """).fetchall()

    # 排序 + 转列表便于区间查询
    k_starts = sorted((k["start"], k["end"]) for k in kernels)
    m_starts = sorted((m["start"], m["end"]) for m in memcpys)

    import bisect

    def overlap_exists(intervals_sorted, lo, hi):
        """区间 [lo,hi] 是否与任一已排序 interval 重叠。"""
        keys = [a for a, _ in intervals_sorted]
        # 二分找第一个 start <= hi 的右侧
        idx = bisect.bisect_right(keys, hi)
        # 往左看是否有 end >= lo
        for i in range(idx - 1, -1, -1):
            a, b = intervals_sorted[i]
            if a > hi:
                break
            if b >= lo:  # 重叠
                return True, (b - a)  # 返回是否有重叠 + 一个重叠区间的长度
        return False, 0

    long_calls = []
    cat_counts = {"A_gpu_kernel": 0, "D_memcpy": 0, "E_no_gpu": 0}
    for s in syncs:
        dur_ns = s["dur_ns"]
        if dur_ns < THRESHOLD_US * 1000:
            continue
        lo, hi = s["start"], s["end"]
        k_overlap, _ = overlap_exists(k_starts, lo, hi)
        m_overlap, _ = overlap_exists(m_starts, lo, hi)
        if k_overlap:
            cat = "A_gpu_kernel"  # 情况A: 同步窗口覆盖 GPU kernel
        elif m_overlap:
            cat = "D_memcpy"      # 情况D: 覆盖 memcpy
        else:
            cat = "E_no_gpu"      # 情况E: 无 GPU 工作 -> runtime/driver overhead
        cat_counts[cat] += 1
        long_calls.append({
            "api": s["name"],
            "dur_ms": round(dur_ns / 1e6, 3),
            "category": cat,
        })

    # 按耗时排序, 取 top 20
    long_calls.sort(key=lambda x: x["dur_ms"], reverse=True)

    # 按 API 分组的 p50/p99/max
    by_api = {}
    for s in syncs:
        by_api.setdefault(s["name"], []).append(s["dur_ns"] / 1e3)
    api_stats = {}
    for name, durs in by_api.items():
        durs_sorted = sorted(durs)
        n = len(durs_sorted)
        api_stats[name] = {
            "calls": n,
            "total_ms": round(sum(durs) / 1e3, 1),
            "p50_us": round(durs_sorted[n // 2], 1),
            "p99_us": round(durs_sorted[int(n * 0.99)] if n > 100 else durs_sorted[-1], 1),
            "max_us": round(durs_sorted[-1], 1),
        }

    print("=" * 70)
    print(f"同步 API 长调用归因 (阈值 > {THRESHOLD_US}us)")
    print("=" * 70)
    print(f"\n长调用总数: {len(long_calls)} / 总同步调用 {len(syncs)}")
    print(f"\n[情况分布 - 手册第12节]")
    print(f"  情况A (同步窗口覆盖 GPU kernel, GPU backlog等待): {cat_counts['A_gpu_kernel']}")
    print(f"  情况D (同步窗口覆盖 Memcpy, PCIe/DMA等待):        {cat_counts['D_memcpy']}")
    print(f"  情况E (无GPU工作, Runtime/Driver overhead):       {cat_counts['E_no_gpu']}")
    print(f"  (情况B/C 需 CPU thread state, 本次未采集)")

    print(f"\n[Top 10 最长同步调用]")
    print(f"{'API':<32}{'耗时ms':>10}{'归因':<22}")
    print("-" * 64)
    for c in long_calls[:10]:
        print(f"{c['api']:<32}{c['dur_ms']:>10}{c['category']:<22}")

    print(f"\n[按 API 汇总]")
    print(f"{'API':<32}{'calls':>7}{'total_ms':>11}{'p50_us':>9}{'p99_us':>10}{'max_us':>10}")
    print("-" * 79)
    for name, st in sorted(api_stats.items(), key=lambda x: -x[1]["total_ms"]):
        print(f"{name:<32}{st['calls']:>7}{st['total_ms']:>11}{st['p50_us']:>9}{st['p99_us']:>10}{st['max_us']:>10}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "threshold_us": THRESHOLD_US,
        "category_counts": cat_counts,
        "top_long_calls": long_calls[:20],
        "by_api": api_stats,
        "note": "情况B/C(OS调度尾延迟)需 --sample/--cpuctxsw 采集CPU thread state, 本次未采集",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n归因结果保存到 {OUT}")


if __name__ == "__main__":
    main()
