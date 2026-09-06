# 2026-09-06｜第 3 次：Performance Debugging：证明 NCCL 和 GEMM 到底有没有 overlap

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

AI Infra Systems Practice #3｜Performance Debugging：证明 NCCL 和 GEMM 到底有没有 overlap

前两次分别做了 vLLM KV Cache 和 Ring AllReduce。今天继续提高一档：不再推导通信公式，而是站在 performance engineer 的角度分析 CUDA timeline。

目标是掌握一个以后看 vLLM、Megatron、Tensor Parallel 性能都能复用的方法：

不要用 GPU utilization 猜瓶颈；从 dependency → CUDA stream → kernel timeline → hardware resource contention 逐层证明。

0–8 min：建立问题模型

假设 TP=4 的某层执行：

~~~
GPU0 ─ GEMM ─┐
GPU1 ─ GEMM ─┤
GPU2 ─ GEMM ─┼── AllReduce ──> next layer
GPU3 ─ GEMM ─┘
~~~

profiling 得到：

~~~
GEMM       600 μs
AllReduce  400 μs
layer time ≈ 1000 μs
~~~

理想情况下，如果能 overlap：

~~~
Stream Compute:
|████████ GEMM ████████|
          600 us
Stream NCCL:
        |████ AllReduce ████|
             400 us
~~~

可能接近：

~~~
600~700 μs
~~~

但实际 timeline 是：

~~~
Compute Stream
GEMM
|████████████|
0           600
NCCL Stream
            |████████|
           600      1000
~~~

第一个 debugging question：

为什么没有 overlap？

不要马上回答“NCCL 性能不好”。

至少存在四种完全不同的原因：

① Data dependency
② CUDA stream synchronization
③ NCCL launch / framework scheduling
④ GPU resource contention

今天就是区分它们。

⸻

8–18 min：手工分析 CUDA dependency

考虑：

Stream 0:
GEMM(A) → Tensor X
Stream 1:
AllReduce(X)

虽然用了两个 stream：

stream0 != stream1

也不意味着可以并行。

因为：

~~~
GEMM
  │
  │ produces X
  ▼
 X
  │
  │ consumes X
  ▼
AllReduce
~~~

这是 RAW dependency（Read After Write）。

所以：

~~~
GEMM(X)
██████████
AllReduce(X)
          ███████
~~~

是正确行为。

真正可以 overlap 的通常是类似：

~~~
GEMM(chunk 0)
████
GEMM(chunk 1)
    ████
GEMM(chunk 2)
        ████

同时：

AllReduce(chunk 0)
    ███
AllReduce(chunk 1)
        ███
~~~

形成 pipeline：

~~~
time ─────────────────────>
Compute:
| G0 | G1 | G2 | G3 |
Comm:
     | A0 | A1 | A2 | A3 |
~~~

这和 networking 里的 packet pipeline 很像。

你应该立即联想到：

~~~
large tensor
    ↓
chunking
    ↓
pipeline
    ↓
communication/computation overlap
~~~

而不是简单：

~~~
create another CUDA stream
~~~

⸻

18–28 min：今天的核心练习——手写一个 Timeline Analyzer

用 Python。

输入：

~~~python
events = [
    ("GEMM_0",     "compute", 0,   500),
    ("AllReduce_0","comm",    400, 700),
    ("GEMM_1",     "compute", 500, 950),
    ("AllReduce_1","comm",    850, 1150),
]
~~~

单位 μs。

实现：

~~~python
def overlap(a_start, a_end, b_start, b_end):
    ...
~~~

要求计算：

~~~
GEMM_0 vs AllReduce_0
GEMM_0 vs AllReduce_1
GEMM_1 vs AllReduce_0
GEMM_1 vs AllReduce_1
~~~

overlap duration：

~~~
max(
    0,
    min(a_end, b_end)
    -
    max(a_start, b_start)
)
~~~

然后实现：

~~~python
total_compute_time(events)
total_comm_time(events)
total_overlap_time(events)
~~~

最后输出：

~~~
compute = 950 μs
comm    = 600 μs
overlap = ? μs
overlap ratio =
overlap / comm
~~~

注意这里有一个坑。

假设：

~~~
GEMM0:
0 ───────────── 500
GEMM1:
400 ─────────── 900
NCCL:
450 ─── 600
~~~

如果简单：

~~~
sum(pairwise_overlap)
~~~

NCCL 的：

~~~
450~500
~~~

可能被计算两次。

所以真正正确的问题变成：

如何求 interval union？

这已经从 AI Infra 变成你熟悉的数据结构问题。

实现：

~~~python
def merge_intervals(intervals):
    ...
~~~

例如：

~~~
[0,500]
[400,900]
→
[0,900]
~~~

再求：

~~~
compute_union ∩ communication_union
~~~

这就是一个极简版 profiler 后处理器。

⸻

28–35 min：加入 Nsight Systems 思维

以后你真正 profiling 时，看到的层次大致是：

~~~
CPU
│
├── PyTorch / vLLM
│
├── cudaLaunchKernel
│
└── ncclAllReduce
        │
        ▼
CUDA Streams
│
├── Stream 7
│      GEMM
│
├── Stream 12
│      NCCL kernel
│
└── memcpy
        │
        ▼
GPU
│
├── Tensor Core
├── CUDA Core
├── SM
├── L2
└── HBM
~~~

这里必须建立一个非常重要的 debugging discipline：

~~~
CPU API timeline
       ≠
GPU execution timeline
~~~

例如：

~~~
CPU:
ncclAllReduce()
| 20 μs |
GPU:
ncclKernel()
       |████████████ 400 μs ████████████|
~~~

NCCL API 往往只是 enqueue GPU work。

所以看到：

~~~
ncclAllReduce() = 20 μs
~~~

不能说：

~~~
AllReduce took 20 μs
~~~

必须看 GPU kernel。

这和 Linux asynchronous I/O 很类似：

~~~
submit I/O
    ≠
I/O complete
~~~

⸻

35–40 min：更难的一层——Timeline overlap ≠ Effective overlap

假设你看到：

~~~
GEMM:
|████████████████|
NCCL:
    |████████|
~~~

视觉上 overlap 很漂亮。

但：

~~~
GEMM alone:
400 μs
NCCL alone:
300 μs

同时执行：

GEMM:
650 μs
NCCL:
500 μs
~~~

为什么？

因为两个 kernel 可能竞争：

~~~
SM
L2
HBM bandwidth
NVLink
PCIe
~~~

例如 GEMM 如果已经：

~~~
SM utilization ≈ 95%
~~~

NCCL kernel 还需要 SM 做：

~~~
reduce
copy
protocol handling
~~~

那么：

~~~
communication overlap
~~~

可能只是：

~~~
communication contention
~~~

因此真正应该比较：

~~~
T_serial
vs
T_overlap
~~~

定义：

~~~
T_serial = T_compute + T_comm
~~~

例如：

~~~
400 + 300 = 700 μs
~~~

实际 overlap：

~~~
T_overlap = 650 μs
~~~

虽然 timeline 上：

~~~
overlap ratio = 80%
~~~

实际收益只有：

~~~
700 / 650 ≈ 1.08x
~~~

这就是 performance debugging 中很容易误判的一点。

⸻

40–43 min：把它接回 vLLM

现在把视角拉回推理 runtime：

~~~
Request
   │
   ▼
vLLM Scheduler
   │
   ▼
Model Runner
   │
   ▼
Attention / GEMM
   │
   ▼
Tensor Parallel
   │
   ▼
AllReduce
   │
   ▼
NCCL
   │
   ▼
CUDA kernel
   │
   ▼
NVLink / RDMA
~~~

假设 vLLM throughput 很差。

你可能看到：

~~~
GPU util = 60%
~~~

但这一个数字无法告诉你问题在哪里。

正确流程应该类似：

~~~
Step 1
Scheduler 有没有足够 batch？
        ↓ yes
Step 2
GPU timeline 有没有 bubble？
        ↓ yes
Step 3
bubble 是 CPU launch delay
还是 CUDA dependency？
        ↓
Step 4
NCCL 是否成为 critical path？
        ↓
Step 5
NCCL 慢是 topology / bandwidth
还是 synchronization？
        ↓
Step 6
如果跨机：
继续进入 RDMA / NIC / switch
~~~

你原来的网络 troubleshooting 经验可以直接迁移：

~~~
应用慢
不是直接：
"网络慢"
而是：
application
 ↓
socket
 ↓
TCP
 ↓
NIC
 ↓
switch
~~~

AI Infra 只是变成：

~~~
inference
 ↓
scheduler
 ↓
CUDA runtime
 ↓
kernel
 ↓
NCCL
 ↓
NVLink / RDMA
~~~

⸻

43–45 min：Debugging Challenge

给你一组 profiler 数据：

~~~
TP = 8
GEMM:
start = 0 μs
end   = 600 μs
NCCL:
start = 350 μs
end   = 850 μs
next GEMM:
start = 850 μs
end   = 1400 μs
~~~

单独 benchmark：

~~~
GEMM alone = 450 μs
NCCL alone = 300 μs
~~~

回答三个问题：

① Timeline overlap 是多少？
② 为什么 GEMM 从 450 μs
   变成了 600 μs？
③ 下一步你最希望看到哪三个指标？

建议至少考虑：

~~~
SM utilization
HBM bandwidth
NVLink / network throughput
~~~

但不要只列指标。

对每个指标写：

~~~
hypothesis
    ↓
metric
    ↓
expected observation
~~~

例如：

~~~
Hypothesis:
GEMM 和 NCCL 竞争 SM
Metric:
SM active / occupancy
Expected:
overlap 区间 SM 接近 saturation
~~~

这才是 performance debugging，而不是“打开 profiler 看一下”。

今日 Takeaway

记住这个判断链：

~~~
Different streams
      ↓
不代表能 overlap
Timeline overlap
      ↓
不代表有性能收益
GPU utilization 高
      ↓
不代表 GPU compute 是瓶颈
~~~

真正需要证明的是：

Dependency 是否允许并行 → Runtime 是否真正并行 → Hardware 是否能够高效并行 → Critical path 是否因此缩短。

这套方法以后可以直接用于分析 vLLM、Tensor Parallel、NCCL、CUDA kernel 以及跨机 RDMA。

Next-step challenge：自己完成上面的 merge_intervals + overlap analyzer，然后扩展 event 数据结构，增加：

~~~
GPU
stream_id
kernel_type
start
end
~~~

让程序自动输出每个 stream 的 busy ratio、communication overlap ratio 和 GPU idle gaps。下一轮应切换主题，进入 RDMA data path：从 ibv_post_send() 一路追到 NIC DMA / CQE，并实现一个最小 verbs 状态机。
