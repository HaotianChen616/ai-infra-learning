# vLLM 0.26 CPU self time、CUDA 等待与 OS 调度实验手册

本文给出一套可以直接搬到 Linux NVIDIA 服务器执行的实验。目标不是再做一遍
GPU kernel 性能分析，而是回答以下问题：

1. `cudaEventSynchronize` 的 Host API 时间中，多少是 Device event 尚未完成，多少是
   event 已完成之后 Host 才返回的尾巴；
2. 阻塞、CUDA active wait、Python 轮询和短轮询后阻塞，分别让 PyTorch Profiler
   `Self CPU time` 发生什么变化；
3. 绑核之后同步 API 变短，究竟是 event 更早完成，还是 OS 唤醒/调度尾巴变短；
4. CPU 总 self time 中，NumPy、Python 调度器、JIT/编译、GIL、CUDA Runtime/Driver、
   内存分配和 OS 分别占多少；
5. 从 CPU 提交 CUDA 工作到 GPU activity 可见之间，框架能够观察到多大的提交空洞。

脚本默认不绑定具体模型。下面用当前问题中的形状作为示例：输入 7000 token、输出
100 token、batch/concurrency 1、vLLM 0.26。若服务器参数不同，只修改统一的变量，
不要在 A/B 两组间偷偷改变请求。

> 安全边界：wait policy 源码补丁只用于诊断，默认行为仍是 `blocking`。不要将
> `python_poll` 直接用于生产；它会占满一个 CPU、持有 GIL，并可能让结果更差。

> 本手册唯一的主验收指标是 PyTorch Profiler 的 `Self CPU time total`。端到端时间、
> GPU 时间、`task-clock`、cycles 和 on-CPU samples 都不替代它；后几项只用于解释
> Self CPU 为什么变化以及记录副作用。

## 1. 先统一“CPU self time”的定义

同一个工具界面中的 `Self CPU` 很容易被误读。本文固定保留三套互不替代的指标，
并明确第一项是主指标：

| 指标 | 含义 | 主要工具 | 能否当作真正 CPU 计算量 |
|---|---|---|---|
| PyTorch Self CPU | 一个 CPU 事件扣掉同线程已记录子事件后的 Host 时间 | PyTorch trace | **主验收指标**；同步 sleep 也在里面 |
| on-CPU sampled self | 线程实际被调度到 CPU 上时，采样栈最叶层的占比 | `perf record`、Nsys CPU sampling、`py-spy` | 可以做统计估计 |
| off-CPU time | 线程 blocked/sleeping/ready、但没有执行指令的时间 | Nsys context switch、`perf sched` | 不能算 CPU 计算量 |

固定工作量下的主 KPI 是：

```text
Total Self CPU time
Total Self CPU time / 固定请求数
Total Self CPU time / 固定 decode step 数
各类别 Self CPU time 及其占 Total Self CPU 的比例
```

PyTorch Profiler 的 `Self CPU` 来自记录事件的 Host 起止时间。对同步 API，它可以包含
线程 sleep、deschedule 和 ready 后等待运行的时间；这些时间即使不执行 CPU 指令，仍然
属于本手册选择的指标。`perf`、Nsys 和 OS 数据负责把变化解释为 device wait、Host
return tail、on-CPU Runtime/Driver 或调度延迟，但不改变最终验收口径。

分析器会跨线程求和，以对应 PyTorch `key_averages()` 中各事件 Self CPU 的加总概念。
因为线程能够重叠，该总和可以大于 trace 的真实墙钟窗口；这不是错误，但 A/B 必须使用
相同线程范围。若只关心 EngineCore/worker，应对两组使用完全相同的线程名过滤规则。

## 2. 工具和脚本地图

仓库新增的工具如下：

| 文件 | 用途 |
|---|---|
| `labs/run_vllm_cpu_experiments.sh` | 统一采集入口，避免手工参数漂移 |
| `labs/analyze_nsys_sqlite.py` | 拆同步 event、OS 调度和 CUDA 提交边界 |
| `labs/analyze_torch_trace_cpu.py` | 重建 Chrome trace 的 PyTorch Self CPU，并按请求/step 归一化 |
| `labs/summarize_cpu_probes.py` | 汇总 `perf stat` 和 `py-spy` |
| `labs/diff_proc_interrupts.py` | 计算 `/proc/interrupts`、`softirqs` 的采样窗口增量 |
| `labs/patch_vllm_event_wait.py` | 可恢复地给 vLLM 0.26 增加四种 wait policy |
| `labs/compare_cpu_profiles.py` | 将多组 JSON 展开为统一 A/B 表 |

每种 profiler 都有 observer effect。不要在同一轮同时打开 Nsys CPU sampling、
`perf record` 和 `py-spy`。它们要对同一份负载分别重放。

## 3. 环境准备

在服务器 clone 本仓库并进入仓库根目录，然后运行：

```bash
bash labs/run_vllm_cpu_experiments.sh doctor
```

检查：

```bash
nvidia-smi topo -m
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE,MAXMHZ,MINMHZ
numactl --hardware
nsys --version
nsys status -e
perf --version
py-spy --version
vllm --version
python3 -c 'import torch,numpy,vllm; print(torch.__version__, numpy.__version__, vllm.__version__)'
```

需要的权限可能包括：

- `perf_event_paranoid` 允许 attach 和硬件计数器；
- `py-spy` attach 允许 `ptrace`；
- `perf sched record -a` 能读系统调度 tracepoint；
- `/proc/interrupts` 在容器内可见；
- Nsys 的 context-switch 和 CUDA event trace 可用。

权限不足时不要用 `sudo` 启动一半组件、普通用户启动另一半后直接比较。让管理员只
开放必要能力，或者在相同权限条件下做两组实验，并在结果中记录缺失项。

## 4. 固定工作负载

### 4.1 启动模板

以下只是模板；模型、量化和 TP 参数按实际服务补全：

```bash
export MODEL=/models/your-model
export SERVED_MODEL=cpu-selftime-test
export PORT=8000

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --profiler-config.profiler cuda
```

如果线上服务已有完整命令，优先原样复制，不要为了 profiling 随意关掉 CUDA Graph、
prefix cache 或 speculative decoding。要测试某个选项，必须建立独立 A/B。

### 4.2 固定 7000→100、batch 1

先 warmup，确保模型加载、CUDA Graph capture 和一次性编译已结束：

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:${PORT} \
  --model "${SERVED_MODEL}" \
  --dataset-name random \
  --random-input-len 7000 \
  --random-output-len 100 \
  --random-range-ratio 0 \
  --num-prompts 3 \
  --max-concurrency 1 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0
```

正式窗口建议不只发 1 个请求。固定 concurrency 1，但串行发送 20～50 个请求，让
`perf`/`py-spy` 获得足够样本：

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:${PORT} \
  --model "${SERVED_MODEL}" \
  --dataset-name random \
  --random-input-len 7000 \
  --random-output-len 100 \
  --random-range-ratio 0 \
  --num-prompts 30 \
  --max-concurrency 1 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0
```

这表示服务并发/batch 上限是 1，不代表只测一次。每组保留相同 seed、请求数和参数。
如果要定位单步 timeline，再另采一份短 trace，不要拿短窗口作 CPU sample 百分比结论。

### 4.3 找到正确进程和线程

不要默认 HTTP frontend PID 就是执行同步和 scheduler 的 PID：

```bash
ps -eLo pid,ppid,tid,psr,pcpu,stat,comm,args | grep -E 'vllm|EngineCore|Worker'
```

对候选 PID 分别观察数秒 `top -H -p PID`，选择在 decode 期间执行
`EngineCore`/worker Python 与 CUDA Runtime 调用的 PID。记录快照：

```bash
export ENGINE_PID=12345
bash labs/run_vllm_cpu_experiments.sh snapshot "${ENGINE_PID}" baseline
```

## 5. 总体实验矩阵

每组至少重复 5 次，顺序用 ABBA 或随机化，避免温度/后台任务随时间漂移。

| 实验 | 只改变什么 | 采集 | 回答的问题 |
|---|---|---|---|
| E0 | profiler 配置 | PyTorch 相同配置、Nsys low/deep/cpu | profiler 配置是否可比、诊断开销多大 |
| E1 | 无 | PyTorch trace 为主，perf/py-spy 辅助 | Total Self CPU 在哪里 |
| E2 | 无 | Nsys deep event/context switch | wait 和 Host return tail 各多少 |
| E3 | wait policy | blocking/spin/python_poll/hybrid | 阻塞、轮询和混合等待取舍 |
| E4 | CPU/NUMA placement | perf、sched、IRQ | 绑核收益来自哪里 |
| E5 | scheduler/NumPy 实现 | perf、py-spy、trace | 算法、分配、访存还是 Python 指令 |
| E6 | Python 并发结构 | py-spy `--gil`、perf | GIL/锁争用有多大 |
| E7 | IRQ/CPU 干扰 | IRQ delta、perf sched | ready 延迟、中断和迁核影响 |
| E8 | 无 | Nsys low/deep | CUDA API return 到 activity start 的可见空洞 |

## 6. E0：先量 profiler observer effect

“低开销复测”不是“不采样”，而是相对于 deep trace 关闭最重的功能：

```text
nsys-low:  CUDA + NVTX，graph-level，关闭 CPU sampling/context switch/event trace
nsys-deep: CUDA + NVTX + OS runtime，node-level，打开 context switch/event trace
nsys-cpu:  增加 CPU sampling，用于 leaf self 归因
```

从 Nsys 启动服务：

```bash
bash labs/run_vllm_cpu_experiments.sh nsys-low e0_low -- \
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --profiler-config.profiler cuda
```

第二终端先 warmup，再给正式命令加 `--profile`。该选项调用服务的 profile start/stop，
Nsys 只保存 capture range 中的请求。

分别重启做 `nsys-low`、`nsys-deep`、`nsys-cpu`；另外保留一轮完全不 attach profiler
的 `perf stat` 对照，用来理解诊断工具的 observer effect。最终 Self CPU A/B 必须来自
相同 PyTorch Profiler 配置，不能拿 Nsys API duration 与 PyTorch Self CPU 直接相减。

## 7. E1：PyTorch Self CPU 主实验

### 7.1 PyTorch trace：唯一主验收数据

另开一轮服务启用 torch profiler，配置方式按当前 vLLM 0.26 的
`--profiler-config` 语法设置输出目录、`with_stack=true`、`record_shapes=false`、
`profile_memory=false`。A/B 两组的 profiler 配置必须完全相同，只采 warmup 后的稳定
窗口。

如果一份 trace 包含 1 个请求，先用请求数做不会歧义的归一化：

```bash
TRACE_REQUESTS=1 \
bash labs/run_vllm_cpu_experiments.sh torch-trace \
  /path/to/batch1.pt.trace.json e1_torch
```

`TRACE_DECODE_STEPS` 必须使用 trace 窗口内真实的 decode step 数。若 trace 包含 prefill
和 decode，`Total Self CPU/request` 可以作为全窗口主指标；`/decode step` 只用于明确
选出的 decode 稳态窗口，不能把 prefill Self CPU 除以 decode step 后冒充 decode 成本。
普通非 speculative 路径理论上常见“prefill 产生首 token，再执行 N-1 次 decode”，但
async output pipeline、最终 drain 和额外同步会使 API 调用次数不同；不能用输出 token
数或 `cudaEventSynchronize` calls 直接冒充 step 数。核验后再加入例如
`TRACE_DECODE_STEPS=99`。

如果只统计 EngineCore/worker 线程，两组使用相同正则：

```bash
TRACE_REQUESTS=1 \
TRACE_THREAD_REGEX='EngineCor|Worker' \
bash labs/run_vllm_cpu_experiments.sh torch-trace \
  /path/to/batch1.pt.trace.json e1_engine_threads
```

也可以直接截取时间窗口：

```bash
python3 labs/analyze_torch_trace_cpu.py \
  /path/to/batch1.pt.trace.json \
  --start-us 100000 --end-us 500000 \
  --requests 1 \
  --thread-regex 'EngineCor|Worker' \
  --output-json decode-cpu-self.json
```

主报告固定先看：

```text
total_self_cpu_ms
normalization.self_cpu_ms_per_request
normalization.self_cpu_ms_per_decode_step
by_analysis_category[*].self_wall_ms
by_analysis_category[*].share_of_summed_self_wall_pct
```

`total_self_cpu_ms` 是从 Chrome trace 的同线程嵌套关系重建的 exclusive Self CPU 加总。
保留旧字段 `summed_cpu_self_wall_ms` 只是兼容历史 JSON，两者数值相同。跨线程加总可以
超过 trace 墙钟窗口，因此不能把它叫作进程端到端时间。

真实 PyTorch trace 的线程名可能被截成 `VLLM::EngineCor`，所以示例用 `EngineCor`。
GPU annotation、kernel 等非 CPU 展示轨道会被排除；`threading.wait`、`_thread.lock`、
ZMQ poll 和后台 usage thread 会单列，避免全部落入 `python_other`。

### 7.2 如何防止“同步时间只是搬家”

不能只比较 `cudaEventSynchronize`。例如：

```text
baseline: sync 40 ms + other 60 ms = Total Self CPU 100 ms
candidate: sync 20 ms + other 90 ms = Total Self CPU 110 ms
```

同步下降 20 ms，但其他类别增加 30 ms，候选方案对本目标是退化。统一对比器会在
“CUDA sync Self CPU 下降但 Total Self CPU 未下降”时输出 warning。

### 7.3 `perf stat`：解释副作用，不负责验收

服务器 warmup 后，终端 A：

```bash
PROFILE_SECONDS=60 \
bash labs/run_vllm_cpu_experiments.sh perf-stat "${ENGINE_PID}" e1_baseline_r1
```

终端 B 在同一时间重放固定 workload。下面这些量只解释 Self CPU 变化和资源副作用：

```text
task-clock / 请求数          实际占用 CPU 的预算
cycles / instructions       指令和周期是否变化
IPC                         前端/后端停顿的综合线索
branch-miss %               Python 分支、哈希/调度决策线索
cache-miss %                指针追逐、大数组/NUMA 线索
context switches / CPU s    调度切换压力
CPU migrations / CPU s      迁核和 cache 冷却
```

允许某个候选方案降低 PyTorch Total Self CPU、但提高 `task-clock`，例如 active spin 用
更多实际 CPU 换更短的 Host wait。该方案仍然满足本文主指标，但必须报告资源副作用。

### 7.4 `perf record` 和 `py-spy`：定位代码原因

```bash
PROFILE_SECONDS=60 PERF_FREQ=199 \
bash labs/run_vllm_cpu_experiments.sh perf-record "${ENGINE_PID}" e1_baseline_r1

PROFILE_SECONDS=60 PYSPY_RATE=100 \
bash labs/run_vllm_cpu_experiments.sh pyspy "${ENGINE_PID}" e1_baseline_r1

bash labs/run_vllm_cpu_experiments.sh probe-summary e1_baseline_r1
```

`perf --no-children` 和 `py-spy` 的 leaf 用于说明 Python、NumPy、Runtime/Driver 的
热点来源，不替代 PyTorch Self CPU。`py-spy --gil` 两轮必须重放相同 workload；
`gil_sample_ratio_proxy_pct` 只是诊断 proxy，不是 GIL 等待 Self CPU。

## 8. E2：把 `cudaEventSynchronize` 打开

用 `nsys-deep` 启动服务并采稳定窗口，结束后导出：

```bash
bash labs/run_vllm_cpu_experiments.sh export \
  artifacts/vllm_cpu_experiments/e2_deep/nsys/e2_deep_deep.nsys-rep \
  e2_deep
```

分析器对每个匹配成功的 event sync 计算：

```text
API wall             = API return - API enter
device_not_ready     = max(0, event completion - API enter)
post_event_host_tail = API return - max(API enter, event completion)
on_cpu_inside_api    = SCHED_EVENTS 中 Running 的交集
off_cpu_inside_api   = 已覆盖区间 - Running
```

这里的 event completion 来自 `--cuda-event-trace=true`。如果 Nsys/driver 不支持，
`exact_event_matches` 会是 0，此时不能用附近最后一个 kernel 的结束时间冒充 event
完成时间，因为 event 可能记录在另一条 stream 或带依赖的位置。

### 如何用它证明绑核收益不是“CPU 更晚进入同步”

对 unbound 和 pinned 比较同一类、同一阶段的调用：

1. `device_not_ready` 近似不变，而 `post_event_host_tail` 和 off-CPU/ready 尾巴下降：
   证据支持“更快被唤醒/调度并返回”；
2. `API enter - event record` 或 `device_not_ready` 也下降：提交顺序、CPU 前置调度或
   Device workload 状态改变，不能只归因于唤醒；
3. `on_cpu_inside_api` 下降：Runtime/Driver 路径本身可能更快、cache/NUMA 更好；
4. 只有 API wall 下降、没有 event/context-switch 匹配：证据不足，保留为相关性。

这比“端到端也变快，所以肯定不是晚调用”更严格。端到端变快能排除明显退化，但不能
单独确定 API 内部缩短的是 device wait、调度尾巴还是 Runtime 指令。

## 9. E3：blocking、轮询和 hybrid

vLLM 0.26 的目标源码是官方 tag 中的
[`vllm/v1/worker/gpu/async_utils.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu/async_utils.py)：

```text
vllm/v1/worker/gpu/async_utils.py
```

补丁脚本要求两个 `torch.cuda.Event(blocking=True)` 和两个对应 `synchronize()` 精确
匹配；不匹配会拒绝写文件，不会猜测修改。先检查和 dry run：

```bash
python3 labs/patch_vllm_event_wait.py status
python3 labs/patch_vllm_event_wait.py apply --dry-run
python3 labs/patch_vllm_event_wait.py apply
```

每组都完整重启 vLLM：

```bash
VLLM_CUDA_EVENT_WAIT_MODE=blocking vllm serve ...
VLLM_CUDA_EVENT_WAIT_MODE=spin vllm serve ...
VLLM_CUDA_EVENT_WAIT_MODE=python_poll vllm serve ...
VLLM_CUDA_EVENT_WAIT_MODE=hybrid \
VLLM_CUDA_EVENT_HYBRID_SPIN_US=25 vllm serve ...
```

hybrid 建议扫 `10,25,50,100` 微秒。每组分别采 `perf-stat`、`perf-record`、`pyspy`、
`nsys-deep`，不要同时采。最后恢复：

```bash
python3 labs/patch_vllm_event_wait.py restore
```

预期取舍：

| 模式 | CPU task-clock | 唤醒尾巴 | GIL/其他线程 | 适用判断 |
|---|---:|---:|---|---|
| blocking | 最低 | 可能有 scheduler wakeup 延迟 | 通常友好 | 默认、共享 CPU |
| spin | 高，可能一整核 | 可能最低 | native wait 未必持有 Python GIL，但占核 | 仅专用核实验 |
| python_poll | 最高 | 可能低 | 持有 GIL，伤害最大 | 只做反例/诊断 |
| hybrid | 介于两者 | 短 wait 可能受益 | 取决于阈值 | wait 分布集中且专核时 |

接受 hybrid 的主条件是固定请求数/step 的 PyTorch `Total Self CPU` 下降，而不只是
`cudaEventSynchronize` 单项下降。`task-clock`、CPU 样本和 GIL 压力可以上升，但必须
作为资源副作用报告；P99 `post_event_host_tail` 用于解释同步 Self CPU 为什么变化。

## 10. E4：绑核和 NUMA

先找 GPU 的 CPU affinity/NUMA：

```bash
nvidia-smi topo -m
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
cat /sys/bus/pci/devices/0000:BUS:DEVICE.F/numa_node
```

选择同一 NUMA node 上、不与繁忙 sibling 共用物理核的 CPU。例如候选为 CPU 24，
先确认它的 `CORE` 和 SMT sibling。按以下顺序建立 A/B：

```text
A: 完全不绑核
B: numactl --cpunodebind=<gpu-node> --membind=<gpu-node>
C: taskset -c 24 + 本地 memory policy
D: 专用物理核，避免其 SMT sibling 上有繁忙线程
```

启动整个进程树时可用：

```bash
numactl --cpunodebind=0 --membind=0 \
  taskset -c 24-31 \
  vllm serve ...
```

启动后如果只想改变某个 TID，先确认 vLLM 是否会重新创建线程；单独 `taskset -pc`
只改一个 PID/TID 容易遗漏子进程。优先在启动时约束进程树，或使用 cpuset cgroup。

每组保存：

```bash
bash labs/run_vllm_cpu_experiments.sh snapshot "${ENGINE_PID}" e4_pinned
bash labs/run_vllm_cpu_experiments.sh perf-stat "${ENGINE_PID}" e4_pinned
bash labs/run_vllm_cpu_experiments.sh perf-sched "${ENGINE_PID}" e4_pinned
bash labs/run_vllm_cpu_experiments.sh irq e4_pinned
```

解释顺序：

1. 固定请求数/step 的 PyTorch `Total Self CPU` 是否下降；
2. CUDA sync、Python/NumPy 和其他 Self CPU 是否只是互相转移；
3. wait 内 ready/off-CPU 和 post-event tail 是否下降；
4. migrations、cache miss、cycles 是否解释了该变化；
5. 目标 CPU 上是否有 NVIDIA/NIC/NVMe IRQ 突增；
6. 是否只是把干扰挪到了同物理核的 SMT sibling。

不要一开始就同时改 IRQ affinity、governor、C-state、nice 和 cpuset，否则无法知道哪个
变量生效。

## 11. E5：NumPy 和 Python scheduler

先用证据分四类，不要看到 `.py` 文件就假设是 Python 指令瓶颈：

| 证据 | 更可能的瓶颈 | 下一步 |
|---|---|---|
| 叶样本在 NumPy C symbol，cache miss 高 | native 数组扫描/拷贝 | dtype、连续性、分配和扫描次数 |
| 叶样本在 Python eval/dict/list，branch miss 高 | 解释器/对象图 | 算法、容器、循环和属性访问 |
| `malloc/free/memcpy` 高 | 临时对象/数组分配 | 复用 buffer、预分配、去 concatenate |
| Python self 低但 scheduler inclusive 高 | 子调用或 native 库 | 看 children/call stack，不要 JIT 错对象 |

对 `get_num_common_prefix_blocks` 这类逻辑，先用输入规模做复杂度实验：固定 block 数分别
取 `1x, 2x, 4x`，主看它和 Total Self CPU 的变化，再用 instructions、cache miss 和
task-clock 解释：

- 近线性增长且 Python eval/dict 样本高：算法/解释器遍历；
- instructions 变化不大、cache miss 和 cycles 增长：更偏访存/指针追逐；
- 只在首次调用高：初始化/JIT/缓存构建，不能算稳态瓶颈。

优先优化顺序：

1. 增量维护 common prefix、避免每 decode step 全量重扫；
2. 合并多次元数据遍历，批量计算；
3. 避免 `tolist()`、fancy indexing、`concatenate` 和 dtype 来回转换；
4. 使用连续、紧凑 dtype，预分配并复用输出；
5. 对很小数组，比较纯 Python/紧凑容器和 NumPy，ufunc 调度成本可能高于计算；
6. 只有确认稳定热点后，才用 Cython/mypyc/Numba/C++ extension 搬走循环。

`torch.compile` 主要面向 tensor graph，不会自动消除任意 scheduler 的 dict/list 控制流。
JIT 实验必须把 compile warmup 排除，以稳态 Total Self CPU 为主，同时报告 instructions、
task-clock 和 fallback；否则“首轮更慢、后面未知”不能算优化。

## 12. E6：GIL 和 Python 并发

GIL 优化的前提是确有两个及以上 Python 线程争用。单线程 CPU 热点即使 100% 持有 GIL，
也不等于 GIL 是瓶颈。

先看：

```text
py-spy all-python.raw：哪些 Python 叶函数热
py-spy gil-holder.raw：谁在采样时持有 GIL
perf sched：相关线程是否同时 runnable
线程 CPU 利用率：是否有其他线程因 GIL 得不到运行
```

可测的单变量方案：

- tokenizer/frontend 与 EngineCore 分进程，并分别绑到物理核；
- 将 numeric loop 放入释放 GIL 的 NumPy/Cython/C++ 路径；
- 减少 Python thread 数和高频小 queue 操作，改为批量消息；
- 避免 Python event `query()` busy loop；
- 对纯 Python CPU 并行使用多进程，但计入序列化、共享内存和 NUMA 代价。

free-threaded Python 或实验性 Python JIT 不是 vLLM/PyTorch 的即插即用开关。只有依赖栈
明确支持并通过正确性、稳定性和 CPU profile 后才纳入候选。

## 13. E7：OS 调度、中断和系统噪声

### 13.1 调度延迟

```bash
PROFILE_SECONDS=60 \
bash labs/run_vllm_cpu_experiments.sh perf-sched "${ENGINE_PID}" e7_baseline
```

关注：

- blocked→wakeup 后 ready 了多久才 sched-in；
- 是否频繁迁核；
- 目标核是否被其他 cgroup/daemon/SMT sibling 抢占；
- cgroup 是否有 CPU quota throttling；
- 固定频率策略下 task-clock 不变但 cycles 是否变化。

优化候选：进程树 affinity、cpuset、同 NUMA memory binding、避免繁忙 SMT sibling、
消除 CPU quota、把非关键后台线程挪走。`SCHED_FIFO` 会饿死系统线程，有锁反转和主机
失联风险，不作为常规建议；先用普通调度策略和 cpuset。

### 13.2 IRQ/softirq

终端 A：

```bash
PROFILE_SECONDS=60 bash labs/run_vllm_cpu_experiments.sh irq e7_irq
```

终端 B 重放固定负载。检查 JSON 的 `per_cpu_delta`，结合
`snapshot/.../irq-affinity-list.txt` 判断目标核上的 NVIDIA、NIC、NVMe 和 `NET_RX/TIMER`
增量。结论只能说明相关性，因为 `/proc/interrupts` 是整机计数。

IRQ affinity 的原则不是“全部赶走”或“全部放 GPU 邻近核”：

- EngineCore 专用核上有高频 NIC/存储 IRQ 时，可把 IRQ 移到同 NUMA 的 housekeeping 核；
- 如果驱动完成路径依赖某个 IRQ/内核线程，跨 NUMA 移太远也可能增加唤醒尾巴；
- 每次只移动一组明确的 IRQ，记录原 mask，并准备恢复；
- 不要让实验脚本自动写 `/proc/irq/*/smp_affinity_list`，避免误改整机。

## 14. E8：算子提交和 doorbell 能看到什么

Nsys 能关联 CUDA Runtime API 与后续 kernel/graph/memcpy activity。分析器报告：

```text
api_wall                   CUDA API 自身 Host 墙钟
api_enter_to_gpu_start     API 进入到 activity start
api_exit_to_gpu_start      API 返回到 activity start，可为负或正
same_stream_idle_gap       同 stream 上一个 activity end 到本 activity start
critical_launch_bubble     start - max(API return, previous same-stream end)，下限为 0
```

`critical_launch_bubble` 是“timeline 上可见、且没有被前序同 stream 工作掩盖”的提交空洞。
它可能包含 Runtime/Driver batching、launch queue、GPU front-end 调度和 profiler 误差。

Nsys 不能把一个普通 CUDA launch 继续精确切成：

```text
Python → ATen/CUDA Runtime → UMD → KMD → MMIO/PCIe doorbell → GPU fetch
```

尤其不能仅凭 timeline 给出“doorbell 物理传输用了 X 微秒”。要进一步研究，需 NVIDIA
驱动/硬件公开 tracepoint、CUPTI/厂商工具或内核级 eBPF/ftrace 事件支持；即使有，也要
确认事件语义和时间域。本文把可观察边界诚实地停在 Runtime API 与 GPU activity。

## 15. 多轮汇总和比较

主 Self CPU 对比先分析两份相同窗口、相同线程范围的 PyTorch trace：

```bash
TRACE_REQUESTS=1 \
bash labs/run_vllm_cpu_experiments.sh torch-trace \
  /traces/unbound.pt.trace.json e1_unbound

TRACE_REQUESTS=1 \
bash labs/run_vllm_cpu_experiments.sh torch-trace \
  /traces/pinned.pt.trace.json e4_pinned

bash labs/run_vllm_cpu_experiments.sh compare \
  unbound=artifacts/vllm_cpu_experiments/e1_unbound/analysis/torch-cpu-self-summary.json \
  pinned=artifacts/vllm_cpu_experiments/e4_pinned/analysis/torch-cpu-self-summary.json
```

对比表固定把 `torch_trace.total_self_cpu_ms`、每请求、每 step 指标放在最前面。如果
CUDA sync 类别下降但 Total Self CPU 没下降，表格顶部会显示耗时转移 warning。

下面的 perf/py-spy 和 Nsys 对比只用于解释主结果。

每个 label 的 `perf-stat`/`py-spy` 采完后：

```bash
bash labs/run_vllm_cpu_experiments.sh probe-summary e1_baseline_r1
bash labs/run_vllm_cpu_experiments.sh probe-summary e4_pinned_r1
```

Nsys report 导出后也会生成 JSON。相同类型的 summary 直接比较：

```bash
bash labs/run_vllm_cpu_experiments.sh compare \
  unbound=artifacts/vllm_cpu_experiments/e1_baseline_r1/analysis/cpu-probes-summary.json \
  pinned=artifacts/vllm_cpu_experiments/e4_pinned_r1/analysis/cpu-probes-summary.json
```

或比较 Nsys summary：

```bash
bash labs/run_vllm_cpu_experiments.sh compare \
  unbound=artifacts/vllm_cpu_experiments/e2_unbound/analysis/nsys-summary.json \
  pinned=artifacts/vllm_cpu_experiments/e2_pinned/analysis/nsys-summary.json
```

不同 profiler 类型不要放在同一张百分比变化表里；它们的分母不同。

## 16. 如何估算“有多少优化空间”

先从相同 PyTorch trace 窗口得到 `Total Self CPU`。若某类别占该总量的比例为 `f`，
把它的 Self CPU 加速 `s` 倍，且不把时间转移到其他类别，则总 Self CPU 的理论下降是：

```text
gain = 1 - ((1 - f) + f / s)
```

例：PyTorch trace 中 NumPy 类别占 Total Self CPU 的 20%，即使完全消除，该指标的理论
上限也只是下降 20%；降低一半时总 Self CPU 下降 10%。同步 API 占 40% 时，它对本文
选定指标的理论上限就是 40%，即使其中大部分是 off-CPU sleep；但纯 CPU/OS 手段通常
只能影响 Host return tail、调用次数和部分等待时序，不能承诺消除全部 device wait。

优化空间分三层报告：

1. `measured share`：该类别当前占 PyTorch Total Self CPU 的比例；
2. `addressable share`：在不改变请求工作量的前提下，预计可降低的 Self CPU 比例；
3. `expected gain`：用原型 A/B 实测，而不是拿理论上限当承诺。

统计要求：

- 每组至少 5 次，报告中位数、P25/P75 或 bootstrap CI；
- PyTorch event 调用次数太少时延长窗口，不用极少调用的百分比做结论；
- 按固定请求数或固定 decode steps 归一化；
- A/B 使用相同 PyTorch Profiler 配置、时间阶段和线程过滤；
- profiler 配置、CPU 频率策略、NUMA、后台流量和模型版本完全一致；
- A/B 改动后确认工作量没变，例如 scheduler 没少处理请求、输出长度仍为 100。

## 17. CPU 优化决策表

| 观测 | 首选动作 | 不要先做 |
|---|---|---|
| sync Self CPU 降、Total Self CPU 不降 | 查其他类别增加和时间转移 | 宣称同步优化已经成功 |
| NumPy Self CPU 高、临时分配高 | 预分配、连续化、少拷贝、合并扫描 | 盲目把所有逻辑改成 JIT |
| Python Self CPU 高，eval/dict/list 样本也高 | 降低复杂度、增量缓存、紧凑数据结构 | 微调变量名或只换 Python 小版本 |
| scheduler Self CPU 随 step/block 线性增长 | 增量维护、批量元数据更新 | 只提高 CPU 优先级 |
| GIL holder 高且其他线程 runnable | native 释放 GIL、分进程、批处理 queue | Python busy polling |
| malloc/free 高 | 复用对象/buffer、减少 tolist/concat | 绑核掩盖分配热点 |
| 绑核后 Total Self CPU 降且 ready tail 降 | cpuset、专用物理核、NUMA local | 直接上 SCHED_FIFO |
| IRQ 与 EngineCore 同核高 | 同 NUMA housekeeping 核做单 IRQ A/B | 一次移动所有 IRQ |
| sync Self CPU 高、on-CPU 低 | 拆 device wait 和 wake tail，测试 blocking/hybrid | 当作 Python 算法热点 |
| sync Self CPU 与 on-CPU 都高 | 比较 wait policy、看 Runtime/Driver 栈 | 只看 API 名猜 doorbell |
| JIT/compile Self CPU 只在 warmup 高 | 排除 warmup、缓存产物 | 优化一次性成本冒充稳态收益 |

## 18. 回传数据清单

建议打包整个 `artifacts/vllm_cpu_experiments`，至少包含：

```text
system/*
<label>/snapshot/*
<label>/perf/perf-stat.csv
<label>/perf/perf-report-self.txt
<label>/perf/perf-report-inclusive.txt
<label>/perf/perf-sched-*.txt
<label>/pyspy/*.raw
<label>/irq/*.json
<label>/analysis/*summary.json
<label>/nsys/*.nsys-rep
服务完整命令、环境变量、日志、请求参数和每轮开始/结束时间
```

模型权重、prompt 原文、token 文本和业务数据不需要上传。若 trace 中含敏感 NVTX、文件
路径或请求内容，先在服务器侧脱敏。

## 19. 官方参考

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Systems Analysis Guide 与 SQLite schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [vLLM Profiling](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [vLLM 0.26 `async_utils.py` API/source](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/gpu/async_utils/)
- [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [Linux perf sched](https://perf.wiki.kernel.org/index.php/Tutorial)
- [py-spy 使用说明](https://github.com/benfred/py-spy)
