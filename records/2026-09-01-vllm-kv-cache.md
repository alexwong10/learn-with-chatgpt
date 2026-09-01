# 2026-09-01｜第 1 次：vLLM：把 KV Cache 当成“内存管理系统”来理解

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

已更新。后续 45 分钟练习会增加 **LLM 推理框架 / Runtime** 这条主线，重点包括 vLLM 的执行链路、continuous batching / scheduler、KV Cache 与 PagedAttention、显存管理、TP/PP、NCCL 与通信-计算 overlap，以及 profiling 和真实源码阅读/修改。

我会特别把这些内容和原来的 **RDMA → 通信库 → GPU Runtime → 算子/Kernel → 推理框架** 串起来，而不是只教 vLLM 的 API 用法。原来的周二/周四/周六 20:00 频率不变。

## 第 1 次｜vLLM：把 KV Cache 当成“内存管理系统”来理解

今天不从 API 开始，而是从一个更适合系统工程师的视角进入 ：

> **LLM serving 本质上同时在做 GPU 计算、内存管理和在线调度。**

今天的 45 分钟只解决一个问题：

**为什么 vLLM 需要 block-based KV Cache，以及 Scheduler 和 KV Cache Manager 是怎么协作的？**

当前 vLLM V1 的 Scheduler 会创建 `KVCacheManager`，由它负责 KV block 的分配、缓存命中和回收；调度输出里甚至直接携带新分配的 block IDs 等 KV-cache 元数据。

### 0–10 min：先建立系统模型

假设正在服务三个请求：

```text
R1: prompt=1000 tokens, 已 decode 20
R2: prompt=4000 tokens, 已 decode 300
R3: prompt=500 tokens,  等待运行
```

Transformer 每生成一个 token，都需要访问此前 token 对应的 K/V。

最朴素的实现是：

```text
request
   │
   ▼
[ contiguous KV buffer ]

R1 ── [................]
R2 ── [................................]
R3 ── [...]
```

问题和传统动态内存分配非常类似：

```text
request 生命周期不同
sequence 长度动态增长
request 随时结束
显存容量有限
```

如果提前按照 `max_seq_len` 给每个 request 分一整块连续显存，会产生严重的内部浪费；如果不断 resize/copy，又会引入数据搬移和外部碎片问题。

vLLM 的核心思路可以抽象为：

```text
Logical KV
R1: 0 1 2 3 4 5 ...

        │ mapping

Physical GPU blocks
┌─────┐ ┌─────┐ ┌─────┐
│ B17 │ │ B03 │ │ B91 │
└─────┘ └─────┘ └─────┘
```

因此，不要求一个 request 的 KV 在物理 GPU memory 上连续。

如果你熟悉 Linux，可以先建立这个类比：

```text
Virtual Memory              vLLM KV Cache
------------------------------------------------
virtual address             logical token/block
physical page               physical KV block
page allocator              BlockPool
page mapping                request → blocks
memory reclaim              block eviction/free
```

这个类比并不完全等价，但非常有用。

### 10–20 min：读真实源码

今天只追这条调用链，不要试图理解整个 vLLM：

```text
Scheduler
   │
   ▼
KVCacheManager
   │
   ▼
KVCacheCoordinator
   │
   ▼
BlockPool
   │
   ▼
physical KV blocks
```

从当前 Scheduler 实现开始：

[vLLM Scheduler source/docs](https://docs.vllm.ai/en/v0.27.0/api/vllm/v1/core/sched/scheduler/?utm_source=chatgpt.com)

然后重点寻找这些概念：

```text
KVCacheManager(...)
allocate / free
num_blocks
block_size
request
num_computed_tokens
```

再看测试代码：

[vLLM scheduler tests](https://github.com/vllm-project/vllm/blob/main/tests/v1/core/test_scheduler.py?utm_source=chatgpt.com)

测试特别适合第一次读大型 AI Infra 项目，因为它直接暴露了系统的不变量，例如 scheduler 的 `running` / `waiting` 状态，以及 KV block pool 的 free-block accounting。

你今天需要自己回答三个问题：

```text
1. 谁决定一个 request 这轮可以执行多少 token？

2. 谁决定这些 token 的 KV 放在哪些 block？

3. request 完成以后，谁负责回收 block？
```

不要停留在类名。用 Linux 内核式思维追踪：

```text
state owner
resource owner
allocation
lifetime
reclaim
```

### 20–35 min：实现一个 Mini KV Block Allocator

不用 GPU。

用 Python 写：

```python
class BlockPool:
    def __init__(self, num_blocks: int):
        ...

    def allocate(self, request_id: str, n: int):
        ...

    def free(self, request_id: str):
        ...
```

假设：

```text
num_blocks = 8
block_size = 4 tokens
```

模拟：

```text
t0:
R1 arrives, 6 tokens

t1:
R2 arrives, 9 tokens

t2:
R1 generates 5 more tokens

t3:
R3 arrives, 8 tokens

t4:
R1 finishes
```

你至少维护：

```python
free_blocks

request_blocks = {
    "R1": [...],
    "R2": [...]
}
```

要求每一步打印：

```text
request -> block mapping
free blocks
used blocks
```

例如：

```text
R1 -> [0, 1]
R2 -> [2, 3, 4]

free = [5, 6, 7]
```

然后处理关键情况：

```text
R1 原来 6 tokens
block_size = 4

已有：
B0 = tokens 0..3
B1 = tokens 4..5

现在再生成 5 tokens。

需要几个新 block？
```

先自己算，再写代码。

这里你会第一次碰到 inference runtime 的一个核心问题：

**Scheduler 调度的是 token，而 allocator 管理的是 block。**

二者粒度不同。

这正是 runtime 设计中经常出现的：

```text
execution granularity
        ≠
resource allocation granularity
```

### 35–42 min：加入调度约束

现在 GPU 只有：

```text
8 blocks × 4 tokens
= 32 token capacity
```

假设：

```text
R1 needs 3 more blocks
R2 needs 4 more blocks
free blocks = 5
```

问：

```text
能不能同时 schedule R1 + R2？
```

显然不能。

于是 scheduler 必须做 admission/resource decision：

```text
Scheduler
    │
    │  "我要执行这些 tokens"
    ▼
KVCacheManager
    │
    │  "显存够不够？"
    ▼
BlockPool
```

这已经和通信库开始产生连接。

未来 TP=4 时，一个 request 的执行可能变成：

```text
Scheduler
   │
   ├── GPU0 ─ GEMM ─┐
   ├── GPU1 ─ GEMM ─┤
   ├── GPU2 ─ GEMM ─┼── AllReduce
   └── GPU3 ─ GEMM ─┘
            │
            ▼
         NCCL
            │
      NVLink / RDMA
```

所以最终性能不是简单的：

```text
GPU kernel performance
```

而是：

```text
scheduler
   × batching
   × KV memory
   × kernels
   × collective communication
```

这就是为什么通信库、Runtime 和推理框架实际上属于同一条能力链。

### 42–45 min：系统推理题

最后不用写代码。

假设：

```text
GPU KV capacity = 100 blocks

Request A = 60 blocks
Request B = 50 blocks
Request C = 20 blocks
```

A 正在运行，B、C waiting。

现在 Scheduler 应该：

```text
A + B ?
A + C ?
preempt A ?
delay B ?
```

不要只回答“哪个能装进去”。

考虑：

```text
吞吐量
TTFT
TPOT
公平性
preemption cost
prefix-cache reuse
```

现实实现很快就会复杂起来。例如 2026 年 vLLM 的一个实际 bug 报告展示了一个很典型的系统问题：启用 CPU KV offload 后，在特定 KV-cache pressure 下，多个请求竞争 GPU KV budget 可能让 scheduler 进入 `Running: 0 / Waiting: N` 的永久停滞状态。这个案例很值得以后拿来做一次完整 debugging exercise。

## 今日 Takeaway

今天只需要真正记住一句：

> **vLLM 的 KV Cache 管理，本质上是一个在线 GPU memory allocator；Scheduler 则是在 token execution 和有限 KV blocks 之间做动态资源调度。**

因此你已有的 Linux allocator、resource accounting、queue/state-machine 思维可以直接迁移到 inference runtime。

下一次再把视角下沉一层：从 **KV block allocation → PagedAttention kernel 如何根据 block mapping 真正读取非连续 KV memory**。到那一步，会开始进入 GPU kernel / operator 的边界。
