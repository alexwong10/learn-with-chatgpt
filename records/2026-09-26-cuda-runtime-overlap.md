# 2026-09-26｜第 12 次：CUDA Runtime：用 Stream + Event + DAG 判断“真 overlap”还是假 overlap

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #12｜CUDA Runtime：用 Stream + Event + DAG 判断“真 overlap”还是假 overlap

上次是端到端性能诊断；这次切到 **hands-on runtime implementation**。今天不依赖 GPU：用 Python 写一个极简 CUDA execution simulator，亲手构造 GEMM → chunked AllReduce → GEMM，计算 critical path。

核心问题：

> **两个 CUDA stream 上的 kernel 时间线发生重叠，为什么不一定带来任何性能收益？**

---

## 0–7 min｜先建立 CUDA 的异步执行模型

考虑：

~~~cpp
kernel_A<<<..., stream0>>>();
kernel_B<<<..., stream1>>>();
~~~

CPU 侧大致是：

~~~
CPU:
launch A
launch B
continue...
~~~

GPU 侧则可能是：

~~~
stream0: |------ A ------|

stream1:     |------ B ------|
~~~

关键区别：

~~~
CPU launch order
≠
GPU execution order
~~~

同一个 stream：

~~~
A → B → C
~~~

天然有顺序约束。

不同 stream：

~~~
stream0: A
stream1: B
~~~

只是允许并发，不意味着一定并发。

还必须同时满足：

~~~
dependency permits
+
hardware resources permit
~~~

---

## 7–13 min｜Event 本质上是在 DAG 中加边

假设：

~~~
stream0:
GEMM_A
   │
record event E

stream1:
wait E
   │
AllReduce
~~~

逻辑关系：

~~~
GEMM_A
   │
   ▼
event E
   │
   ▼
AllReduce
~~~

本质不是“event 很神秘”，而是：

~~~
dependency edge
~~~

因此整个 GPU workload 可以看成 DAG：

~~~
        GEMM A
        /    \
       ▼      ▼
    kernel   kernel
       \      /
        ▼    ▼
       AllReduce
           │
           ▼
        GEMM B
~~~

Runtime 的一个核心职责就是：

~~~
DAG
 ↓
streams
 ↓
kernel launches
 ↓
GPU execution
~~~

---

## 13–25 min｜Hands-on：实现 Mini CUDA Scheduler

先定义 operation：

~~~python
from dataclasses import dataclass, field

@dataclass
class Op:
    name: str
    duration: float
    stream: int
    deps: list[str] = field(default_factory=list)
~~~

输入：

~~~python
ops = [
    Op("gemm0", 400, 0),
    Op("comm0", 300, 1, ["gemm0"]),
    Op("gemm1", 450, 0, ["comm0"]),
]
~~~

实现：

~~~python
def schedule(ops):
    ...
~~~

维护：

~~~python
finish_time = {}
stream_available = {}
~~~

每个 op：

~~~python
dep_ready = max(
    finish_time[d]
    for d in op.deps
) if op.deps else 0

stream_ready = stream_available.get(
    op.stream, 0
)

start = max(
    dep_ready,
    stream_ready,
)

end = start + op.duration
~~~

记录：

~~~python
finish_time[op.name] = end
stream_available[op.stream] = end
~~~

应该得到：

~~~
stream0:

gemm0
|████████|
0       400

stream1:

        comm0
        |██████|
       400    700

stream0:

              gemm1
              |█████████|
             700       1150
~~~

即使 comm0.stream != gemm0，仍然没有 overlap，因为 dependency 决定了 critical path。

---

## 25–32 min｜现在真正制造 overlap

把 tensor 拆成 4 chunks：

~~~
G0
G1
G2
G3
~~~

计算：

~~~
gemm_chunk0
gemm_chunk1
gemm_chunk2
gemm_chunk3
~~~

每块：

~~~
100 μs
~~~

通信：

~~~
allreduce_chunk0
allreduce_chunk1
allreduce_chunk2
allreduce_chunk3
~~~

每块：

~~~
80 μs
~~~

依赖：

~~~
G0 → A0
G1 → A1
G2 → A2
G3 → A3
~~~

而不是：

~~~
G0
G1
G2
G3
 │
 ▼
A0
A1
A2
A3
~~~

建立：

~~~python
ops = [
    Op("G0", 100, 0),
    Op("G1", 100, 0),
    Op("G2", 100, 0),
    Op("G3", 100, 0),

    Op("A0", 80, 1, ["G0"]),
    Op("A1", 80, 1, ["G1"]),
    Op("A2", 80, 1, ["G2"]),
    Op("A3", 80, 1, ["G3"]),
]
~~~

但这里有个坑。

如果 simulator 按数组顺序处理：

~~~
G0 G1 G2 G3 A0 A1 A2 A3
~~~

会得到：

~~~
Compute:
G0 | G1 | G2 | G3

Comm:
                    A0 | A1 | A2 | A3
~~~

仍然没有 overlap。

原因不是 GPU，而是你的 host launch order。

改成：

~~~python
ops = [
    G0, A0,
    G1, A1,
    G2, A2,
    G3, A3,
]
~~~

应该接近：

~~~
time →

compute:
|G0|G1|G2|G3|
0 100 200 300 400

comm:
   |A0|A1|A2|A3|
   100 180 260 340 420
~~~

serial：

~~~
400 + 320
= 720 μs
~~~

pipeline：

~~~
≈ 420 μs
~~~

理论 speedup：

~~~
720 / 420
≈ 1.71×
~~~

这一步非常重要：

> **Overlap 不只是 stream 问题，也是 workload decomposition + dependency graph + launch scheduling 问题。**

---

## 32–37 min｜让 simulator 更像 GPU：加入资源冲突

现在给 Op 增加：

~~~python
@dataclass
class Op:
    name: str
    duration: float
    stream: int
    deps: list[str]
    sm_usage: float
    hbm_usage: float
~~~

例如：

~~~
GEMM:

SM  = 0.90
HBM = 0.50


NCCL:

SM  = 0.30
HBM = 0.60
~~~

如果同时执行：

~~~
SM:
0.9 + 0.3 = 1.2

HBM:
0.5 + 0.6 = 1.1
~~~

两者都超过 1.0。

所以虽然 DAG 允许：

~~~
GEMM || NCCL
~~~

硬件未必能以 standalone speed 并行。

第一版可以非常粗暴地定义 slowdown：

~~~python
slowdown = max(
    total_sm_usage,
    total_hbm_usage,
    1.0,
)
~~~

那么：

~~~
GEMM 100 μs
~~~

与 NCCL overlap 后可能变：

~~~
120 μs
~~~

NCCL：

~~~
80 μs
→
96 μs
~~~

现在重新计算：

~~~
serial
vs
ideal overlap
vs
resource-aware overlap
~~~

你应该看到：

~~~
timeline overlap ratio
~~~

可能很高，但：

~~~
critical-path reduction
~~~

小得多。

这就是 #3 中讨论的 **effective overlap**，今天你真正把它实现出来了。

---

## 37–41 min｜接回 NCCL / Tensor Parallel

真实 TP layer 可以抽象为：

~~~
GEMM
 │
 ▼
partial output
 │
 ▼
AllReduce
 │
 ▼
next operator
~~~

如果只有整个 GEMM 完成以后才能：

~~~
ncclAllReduce(full_tensor)
~~~

依赖是：

~~~
████████████ GEMM
            ███████ NCCL
~~~

几乎没有 overlap 空间。

想变成：

~~~
████ G0
    ████ G1
        ████ G2

    ███ A0
        ███ A1
            ███ A2
~~~

必须改变：

~~~
operator decomposition
+
communication granularity
+
CUDA dependency graph
~~~

所以所谓 communication-computation overlap 并不只是 NCCL 的能力。

它跨越：

~~~
Model/runtime
     ↓
operator partitioning
     ↓
CUDA scheduling
     ↓
NCCL chunking
     ↓
GPU resources
     ↓
NVLink / RDMA
~~~

这也是为什么通信库工程和 runtime/operator engineering 经常交叉。

---

## 41–45 min｜Debugging Challenge

给你一段 profiler timeline：

~~~text
stream 7:

GEMM0
0 ───────────── 400


stream 12:

          NCCL0
          250 ───────── 600


stream 7:

                         GEMM1
                         600 ───── 950
~~~

Standalone benchmark：

~~~
GEMM0 = 300 μs
NCCL0 = 220 μs
~~~

观察到：

~~~
timeline overlap:
150 μs
~~~

但 layer latency：

~~~
950 μs
~~~

请按顺序回答：

~~~
1. 为什么 GEMM0 从 300 → 400 μs？

2. NCCL 为什么从 220 → 350 μs？

3. 150 μs overlap 实际减少了多少 critical-path latency？

4. GEMM1 为什么恰好在 NCCL0 end 才开始？
~~~

第四个问题尤其重要。

它暗示 GEMM1 可能存在：

~~~
NCCL0 → GEMM1
~~~

真实数据依赖。

所以即使你进一步让：

~~~
GEMM0 || NCCL0
~~~

高度 overlap，下一层仍然必须等待：

~~~
NCCL completion
~~~

优化目标应该是 critical path，而不是 overlap percentage。

最后给 simulator 增加一个函数：

~~~python
def critical_path(ops):
    ...
~~~

输出：

~~~
G0 → A0 → ... → GEMM1
~~~

以及：

~~~
critical_path_duration
~~~

这会迫使你从“看时间线”升级成真正的 DAG reasoning。

---

## 今日 Takeaway

今天只保留这个判断链：

~~~
different streams
      ↓
只是允许并行

dependency DAG
      ↓
决定能不能并行

resource contention
      ↓
决定并行后是否仍然快

critical path
      ↓
决定 overlap 是否真正改善 latency
~~~

所以以后看到 Nsight Systems 中 GEMM 和 NCCL 重叠，不要首先问 overlap ratio 多高，而应该问：

> **这个 overlap 到底缩短了多少 critical path？**

这条思维可以直接连接你熟悉的多线程、异步 I/O、pipeline 和网络事件驱动模型：CUDA stream 本质上也是在有限执行资源上推进具有依赖关系的异步工作。

### Next-step challenge

把 simulator 升级为 event-driven scheduler：维护 ready queue，而不是依赖输入数组顺序；支持多个 compute/NCCL op、stream ordering、event dependency 和 SM/HBM resource capacity。

然后用它比较三种 Tensor Parallel 执行策略：

- full GEMM → full AllReduce
- 4-way chunk pipeline
- 8-way chunk pipeline

画出 chunk 越细后为什么最终会被 launch/protocol overhead 抵消。

下一次切换到 **inference runtime/source-code reading**：沿 vLLM 的一次 decode iteration，从 scheduler output 开始追进 model runner，再追到 attention backend / custom op，重点训练“第一次面对大型 AI Infra 代码库，如何用 execution path 而不是目录结构读源码”。
