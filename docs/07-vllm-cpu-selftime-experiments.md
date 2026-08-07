# vLLM 0.26 CPU self time、CUDA 等待与 OS 调度实验手册

本文给出一套可以直接搬到 Linux NVIDIA 服务器执行的实验。目标不是再做一遍
GPU kernel 性能分析，而是回答以下问题：

1. `cudaEventSynchronize` 的 Host API 时间中，多少是 Device event 尚未完成，多少是
   event 已完成之后 Host 才返回的尾巴；
2. 阻塞、CUDA active wait、Python 轮询和短轮询后阻塞，分别消耗多少真正的 CPU；
3. 绑核之后同步 API 变短，究竟是 event 更早完成，还是 OS 唤醒/调度尾巴变短；
4. CPU 总 self time 中，NumPy、Python 调度器、JIT/编译、GIL、CUDA Runtime/Driver、
   内存分配和 OS 分别占多少；
5. 从 CPU 提交 CUDA 工作到 GPU activity 可见之间，框架能够观察到多大的提交空洞。

脚本默认不绑定具体模型。下面用当前问题中的形状作为示例：输入 7000 token、输出
100 token、batch/concurrency 1、vLLM 0.26。若服务器参数不同，只修改统一的变量，
不要在 A/B 两组间偷偷改变请求。

> 安全边界：wait policy 源码补丁只用于诊断，默认行为仍是 `blocking`。不要将
> `python_poll` 直接用于生产；它会占满一个 CPU、持有 GIL，并可能让结果更差。

## 1. 先统一“CPU self time”的定义

同一个工具界面中的 `Self CPU` 很容易被误读。本文固定保留三套互不替代的指标：

| 指标 | 含义 | 主要工具 | 能否当作真正 CPU 计算量 |
|---|---|---|---|
| CPU self wall | 一个 CPU 事件扣掉同线程已记录子事件后的墙钟时间 | PyTorch trace | 不能；同步 sleep 也在里面 |
| on-CPU sampled self | 线程实际被调度到 CPU 上时，采样栈最叶层的占比 | `perf record`、Nsys CPU sampling、`py-spy` | 可以做统计估计 |
| off-CPU time | 线程 blocked/sleeping/ready、但没有执行指令的时间 | Nsys context switch、`perf sched` | 不能算 CPU 计算量 |

因此如果目标是“只优化 CPU 侧总 self time”，主 KPI 应是：

```text
进程 task-clock / 固定请求数
perf 或 Nsys 的 leaf self samples / 固定请求数
cycles、instructions、IPC、cache/branch miss
```

PyTorch trace 中 `cudaEventSynchronize` 的 self wall 要单列为“等待边界”，不能和
NumPy/Python 的 on-CPU 样本相加。同步 API 的墙钟 self time 变短，可能只表示少睡了，
不一定少执行了 CPU 指令。

## 2. 工具和脚本地图

仓库新增的工具如下：

| 文件 | 用途 |
|---|---|
| `labs/run_vllm_cpu_experiments.sh` | 统一采集入口，避免手工参数漂移 |
| `labs/analyze_nsys_sqlite.py` | 拆同步 event、OS 调度和 CUDA 提交边界 |
| `labs/analyze_torch_trace_cpu.py` | 计算 Chrome trace 的 CPU exclusive/self wall |
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
| E0 | profiler 配置 | unattached、Nsys low/deep/cpu | profiler 本身开销多大 |
| E1 | 无 | PyTorch trace、perf、py-spy | CPU self time 在哪里 |
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
的 `perf stat` 对照。后面任何精确微秒结论都要注明来自哪种配置。

## 7. E1：只看 CPU 侧总 self time

### 7.1 `perf stat`：总 CPU 预算

服务器 warmup 后，终端 A：

```bash
PROFILE_SECONDS=60 \
bash labs/run_vllm_cpu_experiments.sh perf-stat "${ENGINE_PID}" e1_baseline_r1
```

终端 B 在同一时间重放固定 workload。输出重点：

```text
task-clock / 请求数          真正占用 CPU 的总预算
cycles / instructions       指令是否真的减少
IPC                         前端/后端停顿的综合线索
branch-miss %               Python 分支、哈希/调度决策线索
cache-miss %                指针追逐、大数组/NUMA 线索
context switches / CPU s    调度切换压力
CPU migrations / CPU s      迁核和 cache 冷却
```

### 7.2 `perf record`：native + Python 叶函数

```bash
PROFILE_SECONDS=60 PERF_FREQ=199 \
bash labs/run_vllm_cpu_experiments.sh perf-record "${ENGINE_PID}" e1_baseline_r1
```

查看：

```text
artifacts/vllm_cpu_experiments/e1_baseline_r1/perf/perf-report-self.txt
artifacts/vllm_cpu_experiments/e1_baseline_r1/perf/perf-report-inclusive.txt
```

`--no-children` 的 self 更接近“CPU 在哪条叶指令上”，`--children` 用于回答它由哪条
Python/框架路径调用。Python 符号不完整时，使用带 debug/unwind 支持的 Python，或用
下一节的 `py-spy` 补充，不要把 `[unknown]` 强行归类。

### 7.3 `py-spy`：Python 栈与 GIL holder

```bash
PROFILE_SECONDS=60 PYSPY_RATE=100 \
bash labs/run_vllm_cpu_experiments.sh pyspy "${ENGINE_PID}" e1_baseline_r1
```

脚本会顺序采两轮，所以终端 B 也必须重放两轮完全相同的 workload。第一轮采所有
Python 栈，第二轮只保留持有 GIL 的栈。汇总：

```bash
bash labs/run_vllm_cpu_experiments.sh probe-summary e1_baseline_r1
```

`gil_sample_ratio_proxy_pct` 只是相同采样率、相同长度、相同负载下的诊断 proxy，
不是线程等待 GIL 的精确时长。

### 7.4 PyTorch trace：墙钟 self 和调用关系

另开一轮服务启用 torch profiler，配置方式按当前 vLLM 0.26 的
`--profiler-config` 语法设置输出目录、`with_stack=true`、`record_shapes=false`、
`profile_memory=false`。只采稳定窗口。导出后运行：

```bash
bash labs/run_vllm_cpu_experiments.sh torch-trace \
  /path/to/batch1.pt.trace.json e1_torch
```

可以用 `--start-us/--end-us` 直接调用 Python 分析器截取 decode 稳态：

```bash
python3 labs/analyze_torch_trace_cpu.py \
  /path/to/batch1.pt.trace.json \
  --start-us 100000 --end-us 500000 \
  --output-json decode-cpu-self.json
```

不要比较一整份 trace 的 prefill 和另一份 trace 的 decode，也不要把多线程
`summed_cpu_self_wall_ms` 当作进程墙钟；多线程有重叠时总和可以超过窗口长度。

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

接受 hybrid 的必要条件不是 API wall 单项下降，而是固定请求数的 `task-clock` 没超预算，
P99 `post_event_host_tail` 确实下降，且没有把 CPU 样本和 GIL 压力转移给其他线程。

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

1. migrations 是否下降；
2. cache miss、cycles 和 on-CPU self 是否下降；
3. wait 内 ready/off-CPU 和 post-event tail 是否下降；
4. 目标 CPU 上是否有 NVIDIA/NIC/NVMe IRQ 突增；
5. 是否只是把干扰挪到了同物理核的 SMT sibling。

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
取 `1x, 2x, 4x`，观察 instructions 和 task-clock：

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
JIT 实验必须把 compile warmup 排除，并同时报告稳态 instructions、task-clock 和 fallback；
否则“首轮更慢、后面未知”不能算优化。

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

先用真正的 on-CPU 样本或 task-clock 归一化。若某类别占 CPU self 的比例为 `f`，把它
加速 `s` 倍，则总 CPU self 的理论下降比例是：

```text
gain = 1 - ((1 - f) + f / s)
```

例：NumPy/native leaf 占 20%，即使完全消除，CPU self 上限也只是下降 20%；加速 2 倍
时总 CPU self 只下降 10%。同步 API 占 PyTorch self wall 40%，不能据此声称 CPU 计算
最多可优化 40%，因为其中可能绝大部分是 off-CPU sleep。

优化空间分三层报告：

1. `measured share`：该类别当前 on-CPU self 占比；
2. `addressable share`：去掉必要工作、不可改库和测量噪声后的可改比例；
3. `expected gain`：用原型 A/B 实测，而不是拿理论上限当承诺。

统计要求：

- 每组至少 5 次，报告中位数、P25/P75 或 bootstrap CI；
- CPU sample 总数太少时延长窗口，不用小样本的 0.1% 排名做结论；
- 按固定请求数或固定 decode steps 归一化；
- profiler 配置、CPU 频率策略、NUMA、后台流量和模型版本完全一致；
- A/B 改动后确认工作量没变，例如 scheduler 没少处理请求、输出长度仍为 100。

## 17. CPU 优化决策表

| 观测 | 首选动作 | 不要先做 |
|---|---|---|
| NumPy leaf 高、临时分配高 | 预分配、连续化、少拷贝、合并扫描 | 盲目把所有逻辑改成 JIT |
| Python eval/dict/list 高 | 降低复杂度、增量缓存、紧凑数据结构 | 微调变量名或只换 Python 小版本 |
| scheduler 每步全量扫描 | 增量维护、批量元数据更新 | 只提高 CPU 优先级 |
| GIL holder 高且其他线程 runnable | native 释放 GIL、分进程、批处理 queue | Python busy polling |
| malloc/free 高 | 复用对象/buffer、减少 tolist/concat | 绑核掩盖分配热点 |
| migrations/ready delay 高 | cpuset、专用物理核、NUMA local | 直接上 SCHED_FIFO |
| IRQ 与 EngineCore 同核高 | 同 NUMA housekeeping 核做单 IRQ A/B | 一次移动所有 IRQ |
| sync wall 高、on-CPU 低 | 先拆 device wait 和 wake tail | 当作 CPU 算法热点优化 |
| sync on-CPU 高 | 比较 blocking/spin、看 Runtime/Driver 栈 | 只看 API 名猜 doorbell |
| JIT/compile 只在 warmup 高 | 排除 warmup、缓存产物 | 优化一次性成本冒充稳态收益 |

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
