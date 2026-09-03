# 2026-09-03｜第 2 次：从 Ring AllReduce 推导 NCCL 通信成本

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #2：从 Ring AllReduce 推导 NCCL 通信成本

上一轮是 **vLLM / KV Cache 内存管理**，这次切换到 **Collective Communication**。目标不是记住 NCCL API，而是能够从网络与系统角度解释：**为什么大模型 Tensor Parallel 经常被 AllReduce 卡住，以及 Ring AllReduce 到底在网络上传输了什么。**

 是今天的核心对象。

### 0–8 min｜问题：4 张 GPU 怎么求和？

假设 Tensor Parallel = 4，每张 GPU 得到一个 1 GiB tensor：

```text
GPU0: X0
GPU1: X1
GPU2: X2
GPU3: X3

目标：

GPU0 ─┐
GPU1 ─┼─> Y = X0 + X1 + X2 + X3
GPU2 ─┤
GPU3 ─┘

最终每张 GPU 都需要 Y
```

最直觉的实现可能是：

```text
GPU1 ──┐
GPU2 ──┼──> GPU0: reduce
GPU3 ──┘
              │
              ├──> GPU1
              ├──> GPU2
              └──> GPU3
```

问题是 GPU0 成为热点。

Ring AllReduce 则构造：

```text
GPU0 ──> GPU1
 ▲          │
 │          ▼
GPU3 <── GPU2
```

并把 tensor 切成 `N` 个 chunk。

今天先不要查公式。自己推导它。

---

## 8–20 min｜手算一次 Ring AllReduce

设：

```text
N = 4 GPUs

Tensor:
[A B C D]

每个 chunk = 256 MiB
```

Ring AllReduce 分两阶段：

```text
Reduce-Scatter
      +
AllGather
```

### Reduce-Scatter

每轮每个 GPU：

```text
send one chunk
receive one chunk
reduce received chunk
```

需要：

```text
N - 1 = 3 rounds
```

画出下面这个状态机并补全数据流：

```text
round 0

GPU0 ── ? ──> GPU1
GPU1 ── ? ──> GPU2
GPU2 ── ? ──> GPU3
GPU3 ── ? ──> GPU0


round 1

GPU0 ── ? ──> GPU1
...
```

关键不是把 NCCL 的具体 chunk 编号背下来，而是理解一个 invariant：

> 每轮之后，都有部分 chunk 向“最终 owner”进一步累积。

三轮之后，每张 GPU 拥有一个完成 reduction 的 chunk：

```text
GPU0: [ ? ]
GPU1: [ ? ]
GPU2: [ ? ]
GPU3: [ ? ]
```

然后进入 AllGather。

再经过 `N-1` 轮：

```text
GPU0: [A' B' C' D']
GPU1: [A' B' C' D']
GPU2: [A' B' C' D']
GPU3: [A' B' C' D']
```

其中：

```text
A' = A0+A1+A2+A3
...
```

---

## 20–30 min｜现在推导性能公式

假设 tensor 大小：

```text
S bytes
```

有：

```text
N GPUs
```

每轮传：

```text
S/N
```

Reduce-Scatter：

```text
(N - 1) × S/N
```

AllGather：

```text
(N - 1) × S/N
```

因此每张 GPU 总通信量：

```text
D = 2 × (N - 1)/N × S
```

当 `N → ∞`：

```text
D ≈ 2S
```

这是今天最重要的一个结果。

注意它意味着：

**Ring AllReduce 并不会随着 GPU 数量增加，让每张 GPU 的网络流量变成 N×S。**

例如：

```text
S = 1 GiB
N = 8

D = 2 × 7/8 × 1 GiB
  = 1.75 GiB
```

假设有效链路带宽：

```text
BW = 50 GB/s
```

先忽略 latency：

```text
T ≈ 1.75 GiB / 50 GB/s
```

自己算一下，大约是：

```text
≈ 35 ms
```

---

## 30–38 min｜写一个 15 行性能模型

用 Python 写：

```python
def ring_allreduce_time(
    tensor_bytes,
    nranks,
    bandwidth_GBs,
    latency_us,
):
    ...
```

模型使用：

```text
T =
2 × (N-1) × latency
+
2 × (N-1)/N × S/BW
```

然后跑：

```text
tensor = 1 GiB

N:
2
4
8
16
32
```

分别测试：

```text
BW = 25 GB/s
latency = 2 us
```

再测试：

```text
tensor =
4 KiB
64 KiB
1 MiB
64 MiB
1 GiB
```

观察什么时候：

```text
latency dominated
```

什么时候：

```text
bandwidth dominated
```

你应该得到非常典型的系统规律：

```text
small message
    ↓
latency sensitive

large message
    ↓
bandwidth sensitive
```

这和你熟悉的网络编程完全同构。

TCP/RDMA 中你会考虑：

```text
RTT
bandwidth
message size
packetization
```

Collective library 中只是变成：

```text
collective latency
link bandwidth
tensor size
chunking
topology
```

---

## 38–43 min｜把它接回 vLLM / Tensor Parallel

假设一个 Transformer layer：

```text
       GPU0 GEMM ──┐
       GPU1 GEMM ──┤
Input ─GPU2 GEMM ──┼── AllReduce ──> next op
       GPU3 GEMM ──┘
```

如果：

```text
GEMM = 400 μs
AllReduce = 300 μs
```

单 layer：

```text
compute       communication
|----------|  |--------|
   400us         300us
```

那么增加 GPU 后，即使 GEMM 从：

```text
800 us → 400 us
```

总时间可能只是：

```text
800 us
↓
400 + 300
= 700 us
```

于是：

```text
2× GPU
≠
2× throughput
```

这就是 Tensor Parallel scalability 的核心问题之一。

下一步优化自然会进入：

```text
communication/computation overlap
```

例如：

```text
naive:

GEMM
████████

AllReduce
        ██████


overlap:

GEMM
████████
    ██████
    AllReduce
```

但这又要求你理解：

```text
CUDA streams
events
kernel dependency
NCCL kernels
chunking
SM utilization
```

因此可以看到一条完整的 AI Infra 技术链：

```text
vLLM scheduler
       │
       ▼
Tensor Parallel
       │
       ▼
PyTorch distributed
       │
       ▼
NCCL
       │
       ▼
Ring / Tree algorithms
       │
       ▼
CUDA kernels
       │
       ▼
NVLink / PCIe / RDMA
       │
       ▼
NIC / switch / network
```

这也是你的 Linux + networking 背景真正能产生迁移优势的位置。

---

## 43–45 min｜Debugging Challenge

现在给你一个真实工程风格的问题。

8 GPU Tensor Parallel：

```text
GPU utilization:       45%
SM utilization:        42%

NCCL AllReduce:
tensor size:           512 MiB
duration:              18 ms

NVLink theoretical BW:
50 GB/s
```

第一步不要说：

```text
"NCCL 太慢"
```

先计算：

```text
algorithmic bytes
=
2 × (N-1)/N × S
```

然后计算：

```text
effective bandwidth
=
algorithmic bytes / duration
```

再比较：

```text
effective BW
vs
link theoretical BW
```

最后列出 **三个你下一步会验证的假设**。

建议从下面这些层次组织：

```text
Application
    │
Runtime / scheduling
    │
CUDA stream dependency
    │
NCCL algorithm
    │
GPU topology
    │
NVLink / PCIe / RDMA
```

不要直接跳到“网络有问题”。

### 今日 Takeaway

**Collective performance 的第一原则不是看 GPU utilization，而是先建立 communication cost model。**

以后看到一个 AllReduce 性能问题，你首先应该能在纸上写出：

```text
T ≈ latency cost + data-transfer cost

Ring AllReduce:

T ≈
2(N-1)α
+
2(N-1)/N × S/BW
```

有了这个 baseline，Nsight Systems、NCCL profiler、`NCCL_DEBUG`、RDMA counters 才有意义，否则 profiling 很容易退化成“看图猜瓶颈”。

**Next-step challenge：**在不查资料的情况下思考一个问题——如果 8 张 GPU 分布在 **2 台机器，每台 4 GPU**，机内是 NVLink，机器间是 RDMA，继续使用一个扁平 Ring 是否仍然合理？

下一次可以沿另一条线进入 **Performance Debugging：从 Nsight Systems 时间线判断 GEMM、NCCL 和 CUDA stream 是否真正发生 overlap**。
