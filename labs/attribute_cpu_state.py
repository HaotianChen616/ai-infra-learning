#!/usr/bin/env python3
"""手册06 第12/13节: 长同步调用的 GPU-overlap × on-CPU 联合归因 (capstone)。

对每个超过阈值的 Host 同步 API, 同时考察 [start, end] 窗口内的:
  - GPU 状态: 是否覆盖 kernel / memcpy / 都没有
  - CPU 状态: 调用线程的 on-CPU 占比 (用 isSchedIn 切上/切出边沿重建 Running 时段)
两者交叉得到手册第12节情况 A/B/C/D/E 的统一判定, 并区分情况 A 的两种子类:
  - A_gpu_blocked   : GPU kernel 在跑, 但线程基本 off-CPU (教科书式 GPU 等待)
  - A_gpu_spinning  : GPU kernel 在跑, 且线程基本 on-CPU (驱动/运行时 poll+短睡)
  - D_memcpy        : 只覆盖 memcpy (PCIe/DMA 等待)
  - E_no_gpu_running: 无 GPU 工作, 线程 on-CPU (Runtime/Driver overhead)
  - BC_no_gpu_off   : 无 GPU 工作, 线程 off-CPU (OS 调度/唤醒尾延迟)

为什么不用 SCHED_EVENTS.threadState: 该字段只有开启 --sample (CPU 采样) 才会被
填充; 只采 --cpuctxsw 的 capture 里 threadState 全为 0 (Unknown), 情况 A/B/C 无法
据此区分。isSchedIn 边沿重建在仅有 context-switch 数据时仍能给出 on-CPU 占比, 因此
本脚本以 on-CPU% 作为 B/C 与 A/E 的判据, 用 GPU overlap 区分 A 与 E/D。
"""
from __future__ import annotations
import sqlite3
import bisect
import json
from collections import Counter
from pathlib import Path

DB = "artifacts/vllm_cuda_sync/nsys/qwen3_awq_pinned.sqlite"
THRESHOLD_NS = 5_000_000  # 只归因 > 5ms 的长同步调用
ON_CPU_BLOCKED_THRESHOLD = 0.5  # on-CPU% >= 0.5 视为 "spinning/running", 否则 "blocked/off"
OUT = Path("artifacts/vllm_cuda_sync/analysis/pinned_long_sync_combined_attribution.json")


def overlap_exists(intervals_sorted, keys, lo, hi):
    """区间 [lo,hi] 是否与任一已排序 (start,end) 区间重叠。"""
    idx = bisect.bisect_right(keys, hi)
    for i in range(idx - 1, -1, -1):
        a, b = intervals_sorted[i]
        if a > hi:
            break
        if b >= lo:
            return True
    return False


def build_running_reconstructor(con):
    """按 globalTid 分组 SCHED_EVENTS, 返回 (by_tid) 供 running_in_window 查询。"""
    rows = con.execute(
        "SELECT start, isSchedIn, globalTid FROM SCHED_EVENTS "
        "ORDER BY globalTid, start").fetchall()
    by_tid = {}
    for r in rows:
        by_tid.setdefault(r["globalTid"], []).append((r["start"], r["isSchedIn"]))
    return by_tid


def running_fraction(by_tid, tid, lo, hi):
    """该 tid 在 [lo,hi] 内的 Running 占比; 无调度事件返回 None。"""
    if tid not in by_tid:
        return None
    ev = by_tid[tid]
    times = [a for a, _ in ev]
    flags = [b for _, b in ev]
    idx = bisect.bisect_left(times, lo)
    # 进入窗口前最后一次 isSchedIn=1 表示已在 CPU 上
    on_cpu = idx > 0 and flags[idx - 1] == 1
    seg_start = lo if on_cpu else None
    total = 0
    i = idx
    while i < len(times) and times[i] <= hi:
        f = flags[i]
        if f == 1:  # 切上 CPU
            on_cpu = True
            seg_start = times[i]
        else:  # 切出 CPU
            if on_cpu and seg_start is not None:
                total += times[i] - seg_start
            on_cpu = False
            seg_start = None
        i += 1
    if on_cpu and seg_start is not None:
        total += hi - seg_start
    return total / (hi - lo)


def categorize(kernel_overlap, memcpy_overlap, run_frac):
    """GPU overlap × on-CPU% → 统一情况标签。"""
    if kernel_overlap:
        base = "A_gpu"
    elif memcpy_overlap:
        return "D_memcpy"
    else:
        base = "no_gpu"
    if run_frac is None:
        return f"{base}_unknown_cpu"
    spinning = run_frac >= ON_CPU_BLOCKED_THRESHOLD
    if base == "A_gpu":
        return "A_gpu_spinning" if spinning else "A_gpu_blocked"
    # no_gpu
    return "E_no_gpu_running" if spinning else "BC_no_gpu_off"


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    syncs = con.execute("""
      SELECT r.start, r.end, (r.end-r.start) AS dur, s.value AS name, r.globalTid
      FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
      WHERE (s.value LIKE 'cuda%Synchronize%' OR s.value LIKE 'cu%Synchronize%')
        AND (r.end-r.start) > ?
    """, (THRESHOLD_NS,)).fetchall()
    print(f"> {THRESHOLD_NS/1e6:.0f}ms 长同步调用: {len(syncs)} 个  (DB={DB})")

    kernels = sorted((r["start"], r["end"])
                     for r in con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    memcpys = sorted((r["start"], r["end"])
                     for r in con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    k_keys = [a for a, _ in kernels]
    m_keys = [a for a, _ in memcpys]
    by_tid = build_running_reconstructor(con)

    cat_counts = Counter()
    per_call = []
    run_total = win_total = 0
    for s in syncs:
        lo, hi = s["start"], s["end"]
        ko = overlap_exists(kernels, k_keys, lo, hi)
        mo = overlap_exists(memcpys, m_keys, lo, hi)
        rf = running_fraction(by_tid, s["globalTid"], lo, hi)
        cat = categorize(ko, mo, rf)
        cat_counts[cat] += 1
        win_total += (hi - lo)
        if rf is not None:
            run_total += rf * (hi - lo)
        per_call.append({
            "api": s["name"],
            "dur_ms": round((hi - lo) / 1e6, 1),
            "kernel_overlap": ko,
            "memcpy_overlap": mo,
            "on_cpu_pct": round(rf * 100) if rf is not None else None,
            "category": cat,
        })

    overall_run_pct = 100 * run_total / win_total if win_total else 0
    print(f"\n=== 联合归因 (GPU overlap × on-CPU%) ===")
    for k, v in cat_counts.most_common():
        print(f"  {k:<22} {v:>4}  ({100*v/len(syncs):.0f}%)")
    print(f"\n  长同步窗口总墙钟: {win_total/1e6:.0f} ms")
    print(f"  调用线程 on-CPU:  {run_total/1e6:.0f} ms  ({overall_run_pct:.1f}%)")

    print(f"\n=== Top 12 最长同步 ===")
    print(f"{'API':<30}{'ms':>7}{'Run%':>6}  GPU")
    for c in sorted(per_call, key=lambda x: -x["dur_ms"])[:12]:
        gpu = "kernel" if c["kernel_overlap"] else ("memcpy" if c["memcpy_overlap"] else "no-gpu")
        r = f"{c['on_cpu_pct']}%" if c["on_cpu_pct"] is not None else "?"
        print(f"{c['api'][:30]:<30}{c['dur_ms']:>7}{r:>6}  {gpu}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "db": DB,
        "threshold_ns": THRESHOLD_NS,
        "on_cpu_blocked_threshold": ON_CPU_BLOCKED_THRESHOLD,
        "long_sync_count": len(syncs),
        "category_counts": dict(cat_counts),
        "window_ms_total": win_total / 1e6,
        "on_cpu_ms_total": run_total / 1e6,
        "on_cpu_pct_overall": overall_run_pct,
        "note": ("情况 A_gpu_spinning = GPU kernel 在跑且调用线程 on-CPU>=50% "
                 "(驱动/运行时 poll+短睡, 而非教科书式 futex 阻塞); "
                 "A_gpu_blocked = GPU 在跑且线程基本 off-CPU。"),
        "top_calls": sorted(per_call, key=lambda x: -x["dur_ms"])[:30],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n保存到 {OUT}")


if __name__ == "__main__":
    main()
