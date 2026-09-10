# 2026-09-10｜第 5 次：Distributed Systems：设计一个“不会被慢节点拖死”的推理 Worker 调度器

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

AI Infra Systems Practice #5｜Distributed Systems：设计一个“不会被慢节点拖死”的推理 Worker 调度器

这次从上一轮 RDMA data path 切换到分布式系统。目标是训练 inference runtime 很核心的一项能力：

把 GPU worker 看成分布式资源，而不是一个 for worker in workers 循环。

今天的问题：一个推理 runtime 有 4 个 worker，其中一个出现 tail latency，Scheduler 应该如何发现、隔离并恢复，而不把整个 batch 拖死？

0–8 min｜建立故障模型

假设 coordinator 每轮向 4 个 worker 分发任务：

                Scheduler
             /     |      \
            /      |       \
          W0      W1       W2      W3
         12ms     11ms     13ms    80ms

最简单的 barrier：

dispatch_all()
wait_all()

这一轮耗时不是：

avg = 29 ms

而是：

max(worker latency)
≈ 80 ms

这和 Tensor Parallel 特别相关。

如果一个 TP group：

GPU0 ─┐
GPU1 ─┤
GPU2 ─┼── collective
GPU3 ─┘

其中 GPU3 慢：

GPU0: GEMM done ───── wait ─────┐
GPU1: GEMM done ───── wait ─────┤
GPU2: GEMM done ───── wait ─────┼→ AllReduce
GPU3: GEMM ──────────────────────┘

那么：

slow rank
   ↓
collective synchronization
   ↓
all ranks stall

所以 distributed inference 中的 tail latency 会通过 collective 放大。

⸻

8–15 min｜先区分三种“慢”

看到：

worker3 = 80 ms

不能直接判断：

worker3 overload

至少有三类：

① Compute straggler
GEMM / attention 慢
GPU contention
thermal throttling
kernel anomaly
② Communication straggler
NCCL
RDMA
PCIe
NVLink
network congestion
③ Scheduling/runtime straggler
CPU scheduling
Python/runtime
CUDA launch delay
queue backlog

所以首先定义：

T_total =
T_queue
+
T_launch
+
T_compute
+
T_comm
+
T_sync

这是今天后面实现 scheduler 时最重要的数据模型。

⸻

15–28 min｜Hands-on：实现 Straggler Detector

用 Python 建一个 worker latency tracker：

from collections import deque
class WorkerStats:
    def __init__(self, window=20):
        self.samples = deque(maxlen=window)
    def record(self, latency_ms):
        self.samples.append(latency_ms)
    def p50(self):
        ...
    def p95(self):
        ...

不要使用：

average latency

作为唯一指标。

例如：

W0:
10 11 10 12 11
W1:
10 10 11 10 12
W2:
11 12 10 11 10
W3:
10 11 10 80 90

average 会稀释问题。

你需要计算：

median
p95

然后实现：

def is_straggler(worker, cluster_stats):
    ...

第一版规则可以非常简单：

worker_p95 >
2 × cluster_median

例如：

cluster median = 11 ms
threshold =
22 ms
W3 p95 ≈ 90 ms
→ straggler

然后打印：

W0 HEALTHY
W1 HEALTHY
W2 HEALTHY
W3 STRAGGLER

⸻

28–35 min｜升级：不要把“检测”与“处理”混在一起

很多系统设计会写成：

if latency > threshold:
    remove_worker()

这很危险。

正确模型应该至少有：

HEALTHY
   │
   │ repeated slowdown
   ▼
SUSPECT
   │
   │ persistent
   ▼
DEGRADED
   │
   │ recovery
   ▼
HEALTHY

实现：

class Worker:
    state = "HEALTHY"
    bad_windows = 0
    good_windows = 0

规则：

HEALTHY
3 consecutive bad windows
        ↓
SUSPECT
SUSPECT
3 more bad windows
        ↓
DEGRADED

恢复也不要：

one good sample
→ HEALTHY

而使用 hysteresis：

5 consecutive good windows
        ↓
HEALTHY

这和网络设备里的：

link flap suppression

思想非常接近。

否则 worker：

healthy
bad
healthy
bad

会导致 scheduler 不断：

add
remove
add
remove

形成 control-plane oscillation。

⸻

35–39 min｜真正困难的地方：TP worker 能不能直接 remove？

假设：

TP = 4
rank0
rank1
rank2
rank3

rank3 被检测成：

DEGRADED

能不能：

workers.remove(rank3)

？

通常不能这么简单。

因为模型参数已经按照 TP=4 partition：

W =
┌────┬────┬────┬────┐
│ W0 │ W1 │ W2 │ W3 │
└────┴────┴────┴────┘

并且 collective communicator 也是：

world_size = 4

rank3 消失后：

AllReduce(rank0,1,2,3)

无法正常继续。

因此 inference runtime 的 failure domain 往往不是：

single GPU

而可能是：

TP group

例如：

Replica A
GPU0 GPU1 GPU2 GPU3
       TP=4
Replica B
GPU4 GPU5 GPU6 GPU7
       TP=4

如果 A 的 GPU3 出问题，更现实的策略可能是：

Scheduler
Replica A → DEGRADED
Replica B → HEALTHY
new requests
      ↓
Replica B

而不是：

remove GPU3

这就是 distributed systems 中：

failure domain 必须和资源组织方式一致。

⸻

39–42 min｜把它接回通信库

现在假设 profiler 显示：

rank0 NCCL: 82 ms
rank1 NCCL: 81 ms
rank2 NCCL: 83 ms
rank3 NCCL: 80 ms

能不能判断：

NCCL 全部都慢

？

不能。

可能实际是：

rank0:
compute finished @ 10 ms
rank1:
compute finished @ 11 ms
rank2:
compute finished @ 12 ms
rank3:
compute finished @ 70 ms

collective：

rank0 ───────────── wait ──────┐
rank1 ──────────── wait ───────┤
rank2 ─────────── wait ────────┼→ collective
rank3 ─ compute ────────────────┘

于是你看到：

"NCCL latency = 80ms"

但 root cause 可能是：

rank3 compute straggler

而不是 RDMA。

因此 troubleshooting 必须做跨 rank correlation：

rank
 │
 ├── queue timestamp
 ├── CUDA launch
 ├── kernel start/end
 ├── collective enter
 ├── collective exit
 └── completion

这也是为什么 distributed profiling 比单 GPU profiling 难很多：

single timeline
      ↓
multi-rank causal timeline

⸻

42–45 min｜Debugging Challenge

假设 TP=4：

             compute     NCCL
rank0         12ms        45ms
rank1         13ms        44ms
rank2         12ms        45ms
rank3         51ms         6ms

GPU utilization：

rank0  48%
rank1  49%
rank2  47%
rank3  96%

NIC：

rank0  low utilization
rank1  low utilization
rank2  low utilization
rank3  low utilization

你的第一结论不应该是：

"NCCL 在 rank0-2 上有问题"

试着构造 causal timeline：

time ─────────────────────────────>
rank0:
compute
██████
      wait................
                         NCCL
rank1:
compute
██████
      wait................
rank2:
compute
██████
      wait................
rank3:
compute
█████████████████████████
                         NCCL

然后回答：

root cause 在哪一层？
Application?
Scheduler?
CUDA/kernel?
NCCL?
RDMA?

再写出三个验证动作，必须使用：

Hypothesis
→ Metric
→ Expected observation

例如：

Hypothesis
rank3 的 GEMM 是 straggler
Metric
各 rank 同一 GEMM kernel duration
Expected
rank3 kernel duration ≫ rank0-2

不要停在这里。继续问：

为什么 rank3 GEMM 慢？

下一层可能是：

GPU contention
clock throttling
different workload
memory pressure
kernel shape difference

这才形成完整的 root-cause tree。

今日 Takeaway

今天最值得保留的不是 straggler detector 代码，而是这个模型：

request
   ↓
scheduler
   ↓
worker
   ↓
GPU compute
   ↓
collective synchronization
   ↓
NCCL
   ↓
RDMA/NVLink

一个 rank 的局部 slowdown，可以通过 collective synchronization 变成整个 TP group 的全局 slowdown。

因此看到：

NCCL wait 很长

不能自动推出：

communication 慢

你必须区分：

time waiting to enter communication

和：

time actually transferring communication data

这正是从“会看性能指标”走向 AI Infra 系统级 debugging 的关键一步。

Next-step challenge：把今天的 WorkerStats 扩展成一个小型 simulator：4 个 TP workers + 2 个 replicas，随机注入 compute straggler、network straggler 和 worker failure，让 scheduler 根据 rolling P95 和状态机把新请求从 degraded replica 转移到 healthy replica；同时记录 吞吐量、P50/P99 latency、false-positive straggler detection。下一轮适合切回实现/runtime 方向，深入 continuous batching：Scheduler 如何在每个 decode step 动态改变 batch，并最终形成 GPU kernel workload。
