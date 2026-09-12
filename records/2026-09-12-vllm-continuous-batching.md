# 2026-09-12｜第 6 次：vLLM Continuous Batching：一次 schedule() 如何变成一次 GPU Forward

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #6｜vLLM Continuous Batching：一次 `schedule()` 如何变成一次 GPU Forward

前几次已经从 KV Cache、collective、性能分析、RDMA、distributed straggler 分别看了系统的不同层。今天回到 ，但比第一次深入一层：**沿真实执行路径，把 Scheduler → KV allocation → Model Runner → GPU workload 串起来。**

今天只解决一个问题：

> **Continuous batching 到底“continuous”在哪里？为什么它不是简单地把请求凑成一个 batch？**

截至 **2026-09-11** 的 vLLM V1 文档里，scheduler 明确以 iteration 为粒度工作：每个 scheduling step 对应一次 model forward，并输出 `{request_id: num_tokens}`，告诉 model runner 本轮每个请求实际处理多少 token。

源码入口：

[vLLM V1 Scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py?utm_source=chatgpt.com)

### 0–8 min｜先扔掉“固定 batch”思维

传统 offline inference 很容易理解：

```text
Batch 0:

R0 ─────────────┐
R1 ─────────────┼── forward
R2 ─────────────┤
R3 ─────────────┘

全部结束

Batch 1:
...
```

在线 LLM serving 不行，因为请求长度不同：

```text
R0: prompt 1000 → output 10
R1: prompt  100 → output 500
R2: prompt 4000 → output 50
```

如果等待整个 batch：

```text
R0 finished ───────────── waiting
R1 ─────────────────────────────>
R2 finished ───── waiting
```

GPU capacity 被浪费。

Continuous batching 的核心不是：

```text
不断创建 batch
```

而是：

```text
每一个 iteration
        ↓
重新决定
“下一次 forward 处理哪些 request 的多少 token”
```

因此 batch membership 是动态的。

---

## 8–15 min｜理解 vLLM V1 最重要的调度抽象

当前 Scheduler 源码里有一段非常值得你直接阅读的注释：V1 scheduler 并不把请求硬编码成“prefill phase”和“decode phase”；它主要追踪 `num_computed_tokens` 与当前应该计算到的 token 数，并让前者逐步追上后者。这样 chunked prefill、prefix caching 和 speculative decoding 可以落在同一个调度模型里。

把它抽象成：

```text
Request R

tokens that should exist
████████████████████████

tokens already computed
██████████

             ↑
             gap
```

定义：

```text
need =
target_tokens
-
num_computed_tokens
```

Scheduler 的工作实际上类似：

```python
for request in requests:

    need = request.target - request.computed

    n = min(
        need,
        remaining_token_budget,
        available_kv_capacity,
    )

    schedule[request.id] = n
```

于是输出可能是：

```python
{
    "R0": 1,
    "R1": 128,
    "R2": 1,
    "R3": 512,
}
```

注意这非常关键。

**一次 GPU forward 中，不同 request 不一定贡献相同数量的 token。**

当前 scheduler 接口也明确允许 `num_tokens` 是整个新 prompt、单个 autoregressive token，或者 chunked prefill / speculative decoding 等情况下的中间值。

---

# 15–28 min｜Hands-on：实现一个 Mini Continuous Scheduler

今天的主要编码任务约 13 分钟。

```python
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    total_tokens: int
    computed_tokens: int = 0
```

Scheduler：

```python
def schedule(requests, token_budget):
    ...
```

规则：

```text
remaining = token_budget

按 FCFS：

need =
total_tokens - computed_tokens

scheduled =
min(need, remaining)
```

测试：

```python
requests = [
    Request("A", 10, 8),
    Request("B", 100, 20),
    Request("C", 40, 0),
]

token_budget = 32
```

你应该得到类似：

```text
A → 2
B → 30
C → 0
```

执行一次：

```python
for req_id, n in output.items():
    request.computed_tokens += n
```

第二轮重新 schedule。

关键点就在这里：

```text
Iteration N

A B

    ↓ forward


Iteration N+1

B C D

    ↓ forward
```

**batch 是每轮重建的。**

这就是 continuous batching 最基本的 runtime intuition。

---

# 28–34 min｜加入 KV Cache：调度第一次变成真正的系统问题

现在增加：

```text
block_size = 16 tokens
free_blocks = 10
```

如果 B：

```text
computed = 20
```

想再计算：

```text
30 tokens
```

最终：

```text
50 tokens
```

需要的 block 数：

```text
ceil(50 / 16) = 4
```

假设原来已经有：

```text
ceil(20 / 16) = 2
```

那么新增：

```text
2 blocks
```

所以 scheduler 不再只是：

```text
token_budget
```

而变成：

```text
                 ┌─ token budget
                 │
request ────────>├─ KV capacity
                 │
                 ├─ max sequences
                 │
                 └─ model constraints
```

当前 vLLM Scheduler 的实际实现也会创建 `KVCacheManager`，并在调度过程中为请求安排所需的新 KV blocks；调度结束还会检查 scheduled-token budget 和 running-request 数量等约束。

现在修改你的函数：

```python
def schedule(
    requests,
    token_budget,
    free_blocks,
    block_size,
):
    ...
```

要求输出：

```text
request
scheduled_tokens
new_blocks
```

例如：

```text
A    2      0
B   30      2
C    0      0
```

这一步非常重要，因为你已经把第一次练习中的：

```text
KV allocator
```

和今天的：

```text
Scheduler
```

真正连接起来了。

---

# 34–38 min｜SchedulerOutput 如何变成 GPU workload？

现在沿代码继续向下：

```text
Scheduler
     │
     │
     │ {req_id: num_tokens}
     ▼
SchedulerOutput
     │
     ▼
Model Runner
     │
     ├── prepare input
     ├── position
     ├── KV mapping
     └── batch metadata
            │
            ▼
        model forward
            │
      ┌─────┴──────┐
      ▼            ▼
    GEMM        Attention
      │            │
      └─────┬──────┘
            ▼
       TP collective
```

当前 GPU model runner 甚至维护了类似这样的 batch CPU state：

```text
req_ids
num_scheduled_tokens
num_tokens
prefill lengths
is_prefilling
...
```

也就是说，scheduler 做出的“逻辑资源决策”，最终必须被 model runner 翻译成 GPU 可以执行的数据布局。

这形成一个很重要的层次：

```text
Scheduling plane

request
token budget
KV budget
priority
       │
       ▼

Execution plane

tensor
CUDA kernel
attention
GEMM
NCCL
```

Scheduler 本身并不执行 attention。

它产生：

```text
execution plan
```

Model Runner 把 plan materialize 成：

```text
GPU workload
```

这和操作系统很像：

```text
process scheduler
      │
      ▼
execution context
      │
      ▼
CPU
```

---

# 38–42 min｜为什么 continuous batching 会影响 Kernel 性能？

这是今天最值得深入思考的一步。

假设 iteration 1：

```text
128 requests
全部 decode

128 × 1 token
```

iteration 2：

```text
1 × 1024-token prefill
+
32 × decode
```

虽然都是一次 forward：

```text
GPU workload shape
```

完全不同。

可能表现为：

```text
Decode-heavy

small token count/request
large batch
KV bandwidth sensitive
        │
        ▼
attention / memory behavior


Prefill-heavy

large token count/request
matrix dimensions larger
        │
        ▼
GEMM compute efficiency ↑
```

所以：

```text
scheduler decision
      ↓
batch shape
      ↓
operator shape
      ↓
kernel selection
      ↓
SM/HBM utilization
      ↓
latency
```

这就是 **runtime 和 operator 之间真正的连接点**。

你以后做 kernel optimization 时不能只问：

```text
这个 kernel benchmark 多快？
```

还必须问：

```text
生产 scheduler 实际产生什么 shape distribution？
```

否则很可能优化了根本不常出现的 workload。

---

# 42–45 min｜Systems Challenge

现在给你：

```text
max_num_batched_tokens = 512

KV free capacity ≈ 600 tokens
```

队列：

```text
A decode   need=1
B decode   need=1
C decode   need=1

D prefill  need=1000
E prefill  need=300
```

FCFS 顺序：

```text
A B C D E
```

回答时不要只计算 token。

比较两种策略：

```text
Strategy 1

A 1
B 1
C 1
D 509
```

和：

```text
Strategy 2

A 1
B 1
C 1
D 253
E 256
```

分析至少四个维度：

```text
TTFT
TPOT

GPU kernel efficiency

KV pressure
```

然后增加第五个：

```text
fairness
```

你会发现不存在简单的：

```text
maximize batch size
```

策略。

真正的问题是：

```text
Scheduler objective

          throughput
             ▲
             │
TTFT ◄───────┼───────► TPOT
             │
             ▼
          fairness

subject to:

GPU memory
token budget
kernel efficiency
```

这才是 inference scheduler 的本质。

当前 vLLM 甚至已经暴露 scheduler iteration、running/waiting requests、KV-cache usage、prefix-cache 等统计信息，这些指标正是以后验证调度策略效果的入口。

## 今日 Takeaway

把今天所有内容压缩成一条链：

```text
incoming requests
       ↓
running / waiting queues
       ↓
Scheduler
       ↓
token allocation
       +
KV block allocation
       ↓
SchedulerOutput
       ↓
Model Runner
       ↓
dynamic batch
       ↓
Attention / GEMM
       ↓
CUDA kernels
       ↓
NCCL / communication
```

最重要的一句话是：

> **Continuous batching 是一个每次 model iteration 都重新进行的在线资源分配问题；scheduler 的决策最终会改变 GPU operator 的 workload shape。**

这也是为什么 **推理框架、Runtime、算子和通信库不能完全割裂学习**。

### Next-step challenge

把今天的 Mini Scheduler 再增加两个指标：

```text
arrival_time
first_scheduled_time
```

统计：

```text
TTFT proxy
queueing latency
GPU token utilization
KV utilization
```

然后实现两种 policy：

```text
FCFS
vs
decode-priority
```

给它输入随机的：

```text
short decode
long prefill
```

流量，观察 **吞吐量、decode latency 和 prefill starvation** 如何变化。

下一轮不要继续 scheduler；切到 **operator / implementation**：手写一个简化的 tiled matrix multiplication，先建立 `naive GEMM → tiling → memory hierarchy → arithmetic intensity` 的性能模型，再把它连接到 Transformer GEMM 和 Tensor Parallel。
