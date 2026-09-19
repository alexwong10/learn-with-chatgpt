# 2026-09-19｜第 8 次：Collective Implementation：亲手实现 Ring Reduce-Scatter + AllGather

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #8｜Collective Implementation：亲手实现 Ring Reduce-Scatter + AllGather

上一轮做的是 **GEMM / tiling / Roofline**，今天切回通信库，但比 #2 的 Ring AllReduce 成本模型更进一步：**不再把 AllReduce 当成一个 API，而是把它拆成通信库内部真正需要维护的 chunk 状态机。**

今天的目标是：45 分钟后，你能从 `all_reduce(buf)` 一直想到 **chunk → send/recv → reduce → ownership → pipeline → transport**。

核心问题是：

> 为什么高性能 collective library 需要把一个大 tensor 切成 chunk，而不是简单地让每个 rank 一次发送整个 tensor？

### 0–7 min｜先把 AllReduce 拆掉

假设：

```text
4 ranks

R0: [a0 b0 c0 d0]
R1: [a1 b1 c1 d1]
R2: [a2 b2 c2 d2]
R3: [a3 b3 c3 d3]
```

目标：

```text
A = a0+a1+a2+a3
B = b0+b1+b2+b3
C = c0+c1+c2+c3
D = d0+d1+d2+d3
```

最终：

```text
R0: [A B C D]
R1: [A B C D]
R2: [A B C D]
R3: [A B C D]
```

Ring：

```text
R0 → R1
↑     ↓
R3 ← R2
```

拆成：

```text
AllReduce
   │
   ├── Reduce-Scatter
   │
   └── AllGather
```

Reduce-Scatter 的目标不是让所有 rank 得到完整结果，而是：

```text
R0: [A]
R1: [B]
R2: [C]
R3: [D]
```

然后 AllGather 才把：

```text
A B C D
```

传播给所有 rank。

这两个阶段是今天实现的主体。

---

## 7–15 min｜先实现“慢得离谱但绝对正确”的版本

用 Python：

```python
nranks = 4

buffers = [
    [1, 10, 100, 1000],
    [2, 20, 200, 2000],
    [3, 30, 300, 3000],
    [4, 40, 400, 4000],
]
```

先写一个 reference：

```python
def reference_allreduce(buffers):
    nranks = len(buffers)
    nchunks = len(buffers[0])

    result = [
        sum(buffers[r][c] for r in range(nranks))
        for c in range(nchunks)
    ]

    return [result[:] for _ in range(nranks)]
```

应该得到：

```text
[10, 100, 1000, 10000]
```

每个 rank 一份。

这个 reference 非常重要。

以后写通信算法、kernel、allocator 都建议保留：

```text
slow + simple + obviously correct
```

版本作为 oracle。

通信库优化尤其需要：

```text
optimized_result == reference_result
```

否则性能越优化，bug 越隐蔽。

---

# 15–28 min｜核心练习：实现 Ring 状态机

不要直接照抄 NCCL。

自己维护：

```python
state = [
    buffers[0][:],
    buffers[1][:],
    buffers[2][:],
    buffers[3][:],
]
```

定义：

```text
next(rank) = (rank + 1) % N
prev(rank) = (rank - 1 + N) % N
```

每一个 step：

```text
Rank r
    │
    │ send one chunk
    ▼
Rank r+1
    │
    │ reduce
    ▼
local partial result
```

关键约束：

**一个 step 中所有 send 必须基于 step 开始前的状态。**

不能写成：

```python
for rank in range(nranks):
    send()
    recv()
    update()
```

然后让后面的 rank 读取前面刚修改的数据。

这会产生一个非常典型的 simulation bug：

```text
logical parallel operation
        ↓
被错误实现成
        ↓
sequential dependent operation
```

正确做法：

```python
messages = []

for rank in range(nranks):
    messages.append(
        (
            src,
            dst,
            chunk_id,
            value,
        )
    )

# 所有 send 都准备完

for msg in messages:
    apply_receive(msg)
```

也就是：

```text
Phase 1:
prepare communication

Phase 2:
commit communication
```

这其实已经有 distributed systems 的味道：

```text
snapshot
→ transition
→ new state
```

---

## Chunk routing

给每个 chunk 一个 ID：

```text
0 1 2 3
```

你需要自己设计：

```python
def chunk_to_send(rank, step, nranks):
    ...
```

然后打印：

```text
Reduce-Scatter Step 0

R0 --chunk ?--> R1
R1 --chunk ?--> R2
R2 --chunk ?--> R3
R3 --chunk ?--> R0
```

执行三轮：

```text
N - 1
```

之后检查 invariant：

```text
每个 rank 恰好拥有
一个 fully reduced chunk
```

这里不要急着找公式。

如果 chunk routing 写错，拿纸画：

```text
step
rank
chunk owner
```

三维关系。

通信库开发里，这种 indexing/debugging 能力非常重要。

---

# 28–34 min｜实现 AllGather

现在每个 rank 只有一个最终 chunk：

```text
R0: A
R1: B
R2: C
R3: D
```

继续沿 ring：

```text
R0 → R1 → R2 → R3 → R0
```

但这次：

```text
recv
```

以后不做：

```text
reduce
```

而是：

```text
store
```

状态变化类似：

```text
step 0

R0: A D
R1: A B
R2: B C
R3: C D


step 1

R0: A C D
R1: A B D
...
```

再经过：

```text
N - 1
```

轮：

```text
R0: A B C D
R1: A B C D
R2: A B C D
R3: A B C D
```

最后：

```python
assert ring_result == reference_result
```

今天一定要留下这个 assertion。

---

# 34–39 min｜为什么真实 NCCL 还要继续切 chunk？

到这里我们的：

```text
chunk
```

其实还是：

```text
tensor / nranks
```

假设：

```text
tensor = 8 GiB
nranks = 8
```

一个 chunk：

```text
1 GiB
```

如果执行：

```text
receive 1 GiB
→
reduce
→
send 1 GiB
```

pipeline 很差。

更合理的是把它继续切：

```text
1 GiB

↓

64 MiB
64 MiB
64 MiB
...
```

于是：

```text
Rank0:

chunk0 ───────────────>

       chunk1 ───────────────>

              chunk2 ───────────────>
```

邻居可以：

```text
receive chunk0
      ↓
reduce chunk0
      ↓
forward chunk0
```

同时上一 rank 已经在发送：

```text
chunk1
```

形成：

```text
time ───────────────────────>

R0:
SEND C0 | SEND C1 | SEND C2

R1:
        RECV/REDUCE C0
                  RECV/REDUCE C1

R2:
                RECV/REDUCE C0
```

这就是 pipeline。

从系统角度看，与 TCP segmentation、CPU pipeline 都有相似性：

```text
大任务
 ↓
分成多个工作单元
 ↓
让不同 stage 同时工作
```

---

# 39–42 min｜现在连接 RDMA

把今天的：

```text
send(chunk)
```

展开：

```text
Collective
    │
    ▼
Ring state machine
    │
    ▼
chunk
    │
    ▼
transport.send()
    │
    ▼
RDMA
    │
    ▼
WQE
    │
    ▼
SQ
    │
    ▼
Doorbell
    │
    ▼
NIC DMA
```

于是通信库内部真正面对的问题变成：

```text
Ring wants:

send C0
send C1
send C2
send C3
...

Transport has:

SQ_DEPTH
CQ_DEPTH
outstanding WR
registered buffers
NIC bandwidth
```

假设：

```text
SQ_DEPTH = 4
```

但 collective pipeline 想同时：

```text
8 outstanding chunks
```

就必须出现：

```text
backpressure
```

例如：

```text
while outstanding >= SQ_DEPTH:
    poll_completion()
```

这就是 #4 学的 RDMA queue model 和今天 collective state machine 的真正连接点。

上层算法说：

```text
我要继续推进 ring
```

下层 transport 说：

```text
暂时没有 queue credits
```

通信库必须协调两者。

---

# 42–45 min｜Systems Challenge：chunk 是不是越小越好？

假设：

```text
Tensor = 1 GiB

Network bandwidth = 50 GB/s

per-message software/NIC latency = 3 μs
```

比较：

```text
chunk = 256 MiB

vs

chunk = 1 MiB

vs

chunk = 4 KiB
```

256 MiB：

```text
优点：
message 数少

缺点：
pipeline granularity 粗
```

4 KiB：

```text
优点：
pipeline 非常细

缺点：
message 数巨大
doorbell/WQE/CQE overhead
latency cost
```

所以：

```text
chunk size ↓

pipeline parallelism ↑

但是

message overhead ↑
WQE pressure ↑
CQ pressure ↑
```

最终又是一个系统 tradeoff：

```text
             pipeline
                ▲
                │
                │
small chunk ◄───┼───► large chunk
                │
                ▼
          overhead / latency
```

这和上一轮 GEMM 的：

```text
tile size
```

其实非常相似。

GEMM：

```text
tile too small
→ reuse 不够

tile too large
→ register/shared memory pressure
```

Collective：

```text
chunk too small
→ protocol/WQE overhead

chunk too large
→ pipeline 不充分
```

**AI Infra 中很多性能优化，本质上都是在寻找合适的工作粒度。**

## 今日 Takeaway

今天把通信库压缩成这一条路径：

```text
AllReduce
   ↓
Reduce-Scatter
   +
AllGather
   ↓
Ring state machine
   ↓
chunk
   ↓
pipeline
   ↓
transport
   ↓
RDMA WQE
   ↓
NIC
```

最重要的一句话：

> **Collective library 不是简单调用网络发送接口，而是在有限 transport credits、通信延迟和带宽约束下推进一个分布式数据流状态机。**

这也是为什么你已有的 **网络协议栈、状态机、队列、流控和性能分析经验**，与 communication-library 工作有很强的迁移关系。

### Next-step challenge

把今天的 simulator 增加：

```text
chunk_size
link_bandwidth
per_message_latency
SQ_DEPTH
```

每个 `send()` 不再瞬间完成，而产生：

```text
completion_time =
start_time
+ latency
+ bytes / bandwidth
```

只有收到 completion 才释放一个 SQ credit。

最终输出：

```text
total AllReduce latency
link utilization
max outstanding WQE
stall time due to SQ full
```

然后分别测试：

```text
4 KiB
64 KiB
1 MiB
16 MiB
256 MiB
```

你应该会看到一个很重要的现象：**最佳 chunk size 往往既不是最大，也不是最小。**

下一轮切换到 **PagedAttention / operator-runtime boundary**：从 `block_table[logical_block] → physical_block` 开始，手算一次 attention 如何读取非连续 KV blocks，再实现一个 CPU 版 paged KV lookup，最后分析这种间接寻址对 GPU kernel 的 memory-access pattern 意味着什么。
