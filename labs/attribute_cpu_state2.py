#!/usr/bin/env python3
"""手册06 第12/13节: 用 sched_switch 边沿重建 CPU Running/Blocked 时段。

对每个长 Host 同步 API 调用线程, 用 isSchedIn=1(切上CPU) / isSchedIn=0(切出)
边沿, 重建该线程在同步窗口内的 Running 时段, 得到:
  - T_running_on_cpu  (CPU 真在执行)
  - T_blocked         (窗口 - running, 等待驱动/调度)
这是手册第13节 T_sync = T_running + T_blocked 的拆分。
"""
import sqlite3, bisect, json
from collections import Counter
from pathlib import Path

DB = "artifacts/vllm_cuda_sync/nsys/qwen3_awq_pinned.sqlite"
OUT = Path("artifacts/vllm_cuda_sync/analysis/pinned_cpu_running_blocked.json")

def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    # 1. 长同步调用 (>5ms), 含 globalTid
    syncs = con.execute("""
      SELECT r.start, r.end, (r.end-r.start) AS dur, s.value AS name, r.globalTid
      FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
      WHERE (s.value LIKE 'cuda%Synchronize%' OR s.value LIKE 'cu%Synchronize%')
        AND (r.end-r.start) > 5000000
    """).fetchall()

    # 2. 按 tid 分组的调度事件
    all_sched = con.execute("""
      SELECT start, isSchedIn, globalTid FROM SCHED_EVENTS ORDER BY globalTid, start""").fetchall()
    by_tid = {}
    for e in all_sched:
        by_tid.setdefault(e["globalTid"], []).append((e["start"], e["isSchedIn"]))
    for tid in by_tid:
        ts = by_tid[tid]
        by_tid[tid] = ([a for a,_ in ts], [b for _,b in ts])

    def running_in_window(tid, lo, hi):
        """该 tid 在 [lo,hi] 内的 Running 总时长。"""
        if tid not in by_tid:
            return 0
        times, flags = by_tid[tid]
        idx = bisect.bisect_left(times, lo)
        # 判断进入窗口时是否已在 CPU 上 (前一个 isSchedIn=1 且未切出)
        running_on_entry = False
        if idx > 0 and flags[idx-1] == 1:
            running_on_entry = True
        total = 0
        on_cpu = running_on_entry
        seg_start = lo if on_cpu else None
        i = idx
        while i < len(times) and times[i] <= hi:
            f = flags[i]
            if f == 1:  # 切上
                on_cpu = True; seg_start = times[i]
            else:  # 切出
                if on_cpu and seg_start is not None:
                    total += times[i] - seg_start
                on_cpu = False; seg_start = None
            i += 1
        # 窗口末尾若仍在CPU
        if on_cpu and seg_start is not None:
            total += hi - seg_start
        return total

    # 3. 对每个长同步拆分
    results = []
    run_total = block_total = win_total = 0
    for s in syncs:
        win = s["dur"]
        run = running_in_window(s["globalTid"], s["start"], s["end"])
        blk = win - run
        run_total += run; block_total += blk; win_total += win
        results.append({
            "api": s["name"][:30], "dur_ms": round(win/1e6,1),
            "running_ms": round(run/1e6,1), "blocked_ms": round(blk/1e6,1),
            "run_pct": round(100*run/win,0) if win else 0,
        })

    print(f"=== {len(syncs)} 个 >5ms 长同步的 CPU 时间拆分 ===\n")
    print(f"{'API':<24}{'窗口ms':>8}{'Running':>9}{'Blocked':>9}{'Run%':>6}")
    print("-"*60)
    for r in sorted(results, key=lambda x:-x["dur_ms"])[:12]:
        print(f"{r['api']:<24}{r['dur_ms']:>8}{r['running_ms']:>9}{r['blocked_ms']:>9}{r['run_pct']:>5}%")

    print(f"\n=== 汇总 ===")
    print(f"  窗口总时间:     {win_total/1e6:>8.0f} ms")
    print(f"  CPU Running:    {run_total/1e6:>8.0f} ms  ({100*run_total/win_total:.0f}%)")
    print(f"  CPU Blocked:    {block_total/1e6:>8.0f} ms  ({100*block_total/win_total:.0f}%)")
    print(f"\n  解读:")
    rp = 100*run_total/win_total
    if rp < 15:
        print(f"  Running 仅 {rp:.0f}% → 同步等待主要是 Blocked (等GPU/驱动), CPU 没在干实事")
        print(f"  → 手册情况 A/B: 不是 CPU 瓶颈, 优化对象是 GPU kernel 或驱动完成路径")
    else:
        print(f"  Running {rp:.0f}% → 调用线程在同步API内基本 on-CPU (poll/短睡, 而非 futex 长阻塞)")
        print(f"  → 仅凭 on-CPU% 无法区分情况 A (GPU 在跑) 与情况 E (无 GPU 工作):")
        print(f"     需结合 labs/attribute_cpu_state.py 的 GPU overlap 判定。")
        print(f"     GPU overlap=是 → A_gpu_spinning; GPU overlap=否 → E_no_gpu_running。")

    # 按 API 汇总
    print(f"\n=== 按 API 汇总 Running/Blocked ===")
    print(f"{'API':<28}{'次数':>6}{'Running%':>10}{'Blocked%':>10}")
    by_api = {}
    for r in results:
        by_api.setdefault(r["api"], []).append(r)
    for api, lst in sorted(by_api.items(), key=lambda x:-sum(r["dur_ms"] for r in x[1])):
        run = sum(r["running_ms"] for r in lst)
        blk = sum(r["blocked_ms"] for r in lst)
        tot = run+blk
        print(f"{api:<28}{len(lst):>6}{100*run/tot:>9.0f}%{100*blk/tot:>9.0f}%")

    OUT.write_text(json.dumps({
        "summary": {"window_ms": win_total/1e6, "running_ms": run_total/1e6,
                    "blocked_ms": block_total/1e6, "running_pct": 100*run_total/win_total},
        "top_calls": sorted(results, key=lambda x:-x["dur_ms"])[:20],
    }, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\n保存到 {OUT}")

if __name__ == "__main__":
    main()
