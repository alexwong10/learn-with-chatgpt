# 2026-09-24｜第 11 次：端到端性能诊断：TTFT 突然升高，到底该怪谁？

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #11｜端到端性能诊断：TTFT 突然升高，到底该怪谁？

上一轮做到 GPUDirect RDMA data path。这次切到 performance debugging，并把前 10 次练习第一次真正串起来：从用户看到的 TTFT 异常出发，一路向下定位到 Scheduler、KV Cache、CUDA kernel、NCCL 或 RDMA。

今天不学新 API。训练的是 AI Infra 更重要的一项能力：

从观测现象建立 hypothesis tree，用证据逐层缩小 failure domain，而不是看到某个指标异常就直接归因。

---

## 0–7 min｜Incident：vLLM 的 TTFT 从 120 ms 升到 420 ms

假设线上 TP=4 推理服务出现：

~~~
Before                 Now
TTFT P50   120 ms      420 ms
TPOT P50    32 ms       34 ms
Throughput  920 tok/s   880 tok/s
GPU util     72%         69%
~~~

同时：

~~~
running requests ≈ unchanged
waiting requests:
12 → 87
KV cache usage:
62% → 94%
~~~

先不要说“KV Cache 满了导致 TTFT 上升”。

把 TTFT 拆开：

~~~
request arrival
      │
      ▼
queue wait
      │
      ▼
scheduler admission
      │
      ▼
prefill
      │
      ├── GEMM
      ├── attention
      └── TP collective
      │
      ▼
first token
TTFT =
T_queue
+ T_schedule
+ T_prefill
+ T_comm
~~~

TPOT 几乎不变已经是一条非常强的证据。

如果 NCCL 或 GPU compute 整体退化，你通常还应该预期 decode latency 受到影响。

所以第一版 hypothesis ranking 应该是：

~~~
HIGH
scheduler / admission / KV pressure
MEDIUM
prefill-specific compute regression
LOW
global NCCL / RDMA regression
~~~

这不是结论，只是根据症状调整 prior probability。

---

## 7–15 min｜第一轮证据：Scheduler timeline

现在得到一个 request 的 trace：

~~~
arrival             0 ms
enter waiting       1 ms
first scheduled   287 ms
GPU prefill start 294 ms
GPU prefill end   401 ms
first token       418 ms
~~~

自己算：

~~~
queue/scheduler wait ≈ 287 ms
prefill GPU ≈ 107 ms
~~~

历史正常值：

~~~
queue wait ≈ 20 ms
prefill    ≈ 100 ms
~~~

这时 failure domain 已经明显缩小：

~~~
                    TTFT regression
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       before GPU                    on GPU
         ~267 ms                     ~7 ms delta
             │
             ▼
       Scheduler / KV
~~~

因此现在继续开 Nsight Compute 深挖 GEMM 是低价值动作。

这是今天第一条 debugging discipline：

先定位时间消失在哪一层，再选择 profiler。

这和 Linux troubleshooting 一样：

~~~
application latency
       ↓
先判断 CPU / disk / network
       ↓
再进入对应 subsystem
~~~

而不是一开始就 perf record。

---

## 15–27 min｜Hands-on：写一个 Critical-Path Analyzer

今天实际写约 12 分钟 Python。

给定：

~~~python
trace = [
    ("arrival",       0),
    ("queued",        1),
    ("scheduled",   287),
    ("gpu_start",   294),
    ("gpu_end",     401),
    ("first_token", 418),
]
~~~

实现：

~~~python
def breakdown(trace):
    ...
~~~

输出：

~~~
queue_wait       286 ms
dispatch           7 ms
gpu_execution    107 ms
postprocess        17 ms
--------------------------------
TTFT             418 ms
~~~

然后增加 baseline：

~~~python
baseline = {
    "queue_wait": 20,
    "dispatch": 5,
    "gpu_execution": 100,
    "postprocess": 15,
}
~~~

计算：

~~~
component        baseline   current   delta
queue_wait          20        286     +266
dispatch             5          7       +2
gpu_execution      100        107       +7
postprocess         15         17       +2
~~~

再实现：

~~~python
def regression_contribution(current, baseline):
    ...
~~~

输出每层对总 regression 的贡献率：

~~~
queue_wait       ~96%
GPU execution     ~3%
others            ~1%
~~~

这看起来简单，但它是非常实用的性能分析习惯：

~~~
Latency regression
不要问：
"哪个组件看起来慢？"
而问：
"新增 latency 到底出现在哪里？"
~~~

---

## 27–33 min｜第二层：为什么 Scheduler 等这么久？

现在继续向下。

得到：

~~~
token budget utilization: 97%
KV blocks:
total       10000
used         9400
free          600
~~~

waiting queue：

~~~
decode requests:   11
prefill requests:  76
~~~

最近请求平均 prompt：

~~~
Before:
~800 tokens
Now:
~3500 tokens
~~~

现在构造 causal chain：

~~~
longer prompts
      ↓
prefill requires more tokens
      ↓
token budget pressure
      +
KV block demand ↑
      ↓
fewer waiting requests admitted / iteration
      ↓
queue grows
      ↓
TTFT ↑
~~~

注意这里有两个资源：

~~~
compute budget:
max_num_batched_tokens
memory budget:
KV blocks
~~~

不要把二者混为：

~~~
GPU 满了
~~~

一个 request 可能因为 token budget 无法进入本轮；也可能因为 KV capacity 无法安全执行。

这就是 #6 continuous batching 和 #1/#9 KV allocator/PagedAttention 在真实性能问题中的交汇点。

---

## 33–38 min｜加入一个误导性指标：NCCL latency +20%

现在 profiler 又告诉你：

~~~
NCCL AllReduce:
baseline  250 μs
current   300 μs
+20%
~~~

很多 debugging 会立即转向：“NCCL regression!”

但先计算。

假设一次 prefill critical path 有：

~~~
40 AllReduce
~~~

增加：

~~~
50 μs × 40
=
2 ms
~~~

TTFT regression：

~~~
~300 ms
~~~

所以即使 NCCL 确实退化 20%，它也解释不了主要现象。

这是性能分析中极重要的：

~~~
correlation
≠
critical-path contribution
~~~

不要问有没有异常指标，而问这个异常最多能解释多少 latency。

这就是 back-of-the-envelope bounding。

你应该养成习惯：

~~~
Observed regression
       ↓
300 ms
Hypothesis X maximum contribution
       ↓
2 ms
Therefore:
X cannot be primary root cause
~~~

甚至不用继续 profiling X。

---

## 38–42 min｜如果真的是 NCCL，再怎么向下？

现在换一个 incident。

假设：

~~~
TTFT ↑
TPOT ↑
GPU compute kernel duration:
unchanged
AllReduce:
300 μs → 900 μs
~~~

这时通信才真正进入 critical path。

继续拆：

~~~
NCCL
 │
 ├── waiting for rank?
 │
 ├── CUDA stream dependency?
 │
 ├── algorithm/channel?
 │
 └── transport?
        │
        ├── NVLink
        └── RDMA
             │
             ├── GPU↔NIC PCIe
             ├── NIC
             ├── network
             └── remote PCIe
~~~

然后使用我们之前建立的判断：

~~~
rank0 collective enter  10.0 ms
rank1                   10.1
rank2                   10.0
rank3                   10.2
~~~

说明 rank arrival skew 很小，更值得调查 transport。

但如果：

~~~
rank0 enter 10 ms
rank1 enter 10 ms
rank2 enter 11 ms
rank3 enter 35 ms
~~~

那么首先调查 rank3 before collective，而不是 RDMA。

这正是 #5 的 straggler reasoning。

---

## 42–45 min｜最终 Incident Challenge

现在不给答案。

线上数据：

~~~
Application
TTFT:
150 → 390 ms
TPOT:
31 → 33 ms
~~~

Scheduler：

~~~
waiting:
20 → 105
KV usage:
71% → 96%
token-budget utilization:
99%
~~~

GPU：

~~~
GEMM duration:
+3%
Attention:
+2%
GPU utilization:
74% → 71%
~~~

Communication：

~~~
AllReduce:
280 → 350 μs
~~~

RDMA：

~~~
NIC throughput:
38 GB/s → 37 GB/s
retransmission:
unchanged
PCIe throughput:
unchanged
~~~

Traffic：

~~~
average prompt:
900 → 4100 tokens
~~~

用下面格式写出你的 incident report：

~~~
Primary hypothesis:
...
Causal chain:
...
→
...
→
...
Evidence supporting:
1.
2.
3.
Evidence against alternatives:
NCCL:
...
RDMA:
...
GPU kernel:
...
Next measurement:
...
~~~

关键要求：不要写“可能是 KV / Scheduler”。

必须形成可证伪的因果链，例如：

~~~
prompt length distribution changed
        ↓
prefill token demand ↑
        ↓
scheduler token budget saturated
        ↓
KV occupancy ↑
        ↓
admission rate < arrival rate
        ↓
waiting queue ↑
        ↓
queueing dominates TTFT
~~~

然后给出最小验证实验：

~~~
temporarily reduce incoming long-prefill traffic
              ↓
如果 waiting queue 和 TTFT 快速下降
              ↓
hypothesis strengthened
~~~

这才是完整的 systems debugging loop：

~~~
Observe
   ↓
Decompose
   ↓
Hypothesize
   ↓
Bound
   ↓
Measure
   ↓
Falsify / confirm
~~~

## 今日 Takeaway

今天只保留一个原则：

性能调试的核心不是寻找“异常指标”，而是找到新增 latency 位于 critical path 的哪一层，并证明某个机制能够解释它的数量级。

你现在可以把之前的知识组织成一个统一 troubleshooting stack：

~~~
User metric
TTFT / TPOT
      ↓
Scheduler
queue / token budget
      ↓
Memory
KV blocks / PagedAttention
      ↓
GPU execution
GEMM / Attention
      ↓
CUDA
streams / dependencies
      ↓
Collective
NCCL / rank synchronization
      ↓
Transport
RDMA / GPUDirect
      ↓
Hardware
PCIe / NIC / network
~~~

对于你向通信库、runtime 和 AI Infra 转型，这种跨层定位能力比单独背熟某一套 API 更重要，因为它直接建立在你已有的 Linux、网络协议栈和性能排障经验之上。

### Next-step challenge

把今天的 Critical-Path Analyzer 扩展为读取一个简单 CSV trace：

~~~
request_id,event,timestamp
~~~

自动计算每个 request 的：

~~~
queue
scheduler
GPU
communication
postprocess
TTFT
~~~

然后输出 P50/P95/P99 以及每层的 regression contribution。

下一次切换主题，进入 CUDA/runtime implementation：自己实现一个简化的 asynchronous execution engine，模拟 CUDA stream + event + dependency DAG，再用它构造 GEMM → chunked AllReduce → next GEMM，实际求出什么时候 communication-computation overlap 能缩短 critical path、什么时候只是 timeline 上“看起来重叠”。
