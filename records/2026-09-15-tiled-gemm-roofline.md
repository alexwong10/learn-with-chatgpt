# 2026-09-15｜第 7 次：Operator：从 Naive GEMM 到 Tiling，理解 GPU 为什么需要“数据复用”

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

AI Infra Systems Practice #7｜Operator：从 Naive GEMM 到 Tiling，理解 GPU 为什么需要“数据复用”

上一轮是 vLLM continuous batching，这次切到 operator / kernel。今天不要求 CUDA 环境，重点建立一个之后学习 CUDA、Triton、FlashAttention 都会反复使用的性能模型：

算子优化的核心通常不是“少做计算”，而是让昂贵的数据搬运服务更多计算。

今天结束时，你应该能回答：为什么两个都做 2MNK FLOPs 的 GEMM，性能可以差一个数量级？

0–8 min｜先建立 Roofline 思维

计算：

C[M,N] = A[M,K] × B[K,N]

FLOPs 约为：

2MNK

假设：

M=N=K=4096
FP16

计算量约：

2 × 4096³
≈ 137 GFLOPs

问题是 GPU 不只需要算，还需要搬数据：

HBM
 │
 ▼
L2
 │
 ▼
L1 / Shared Memory
 │
 ▼
Registers
 │
 ▼
Tensor Core / CUDA Core

先记住：

Performance ≤ min(
    Peak Compute,
    Memory Bandwidth × Arithmetic Intensity
)

其中：

Arithmetic Intensity
=
FLOPs / bytes moved

这就是 Roofline model 的核心。

⸻

8–15 min｜为什么 naive GEMM 很差？

最朴素的思路：

for (m = 0; m < M; m++)
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++)
            C[m][n] += A[m][k] * B[k][n];

一个 C[m][n]：

A[m,0] × B[0,n]
A[m,1] × B[1,n]
...
A[m,K-1] × B[K-1,n]

问题在于邻近输出：

C[m,n]
C[m,n+1]

都会需要：

A[m,k]

而：

C[m,n]
C[m+1,n]

都会需要：

B[k,n]

如果每次都从 HBM 重新加载：

A ──────┐
A ──────┤
A ──────┤
A ──────┘
B ──────┐
B ──────┤
B ──────┤
B ──────┘

大量 bandwidth 被浪费。

真正需要利用的是：

data reuse

⸻

15–25 min｜核心练习：自己推导 Tiling

不要先写代码。

假设计算一个：

4 × 4

输出 tile：

C tile
┌───────────────┐
│ C00 C01 C02 C03 │
│ C10 C11 C12 C13 │
│ C20 C21 C22 C23 │
│ C30 C31 C32 C33 │
└───────────────┘

对于某一个 K tile：

A tile = 4 × BK
B tile = BK × 4

数据路径变成：

            HBM
        A          B
        │          │
        ▼          ▼
     Shared / Cache
       ┌──────┐
       │A tile│
       └──────┘
            ×
       ┌──────┐
       │B tile│
       └──────┘
            │
            ▼
        registers
        C 4×4 tile

关键是：

一个：

A[m,k]

被复用于：

4 个 C 元素

一个：

B[k,n]

也被复用于：

4 个 C 元素

如果 tile 扩大：

BM × BN

reuse 大约进一步提高。

所以：

larger tile
    ↓
more reuse
    ↓
higher arithmetic intensity

但不能无限增大，因为：

larger tile
    ↓
more shared memory
more registers
    ↓
lower occupancy

于是出现第一个 GPU optimization tradeoff：

data reuse
     ▲
     │
     │
tile size
     │
     ▼
resource pressure

⸻

25–35 min｜Hands-on：实现 CPU Tiled GEMM

今天实际写代码。

先写：

def naive_gemm(A, B):
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C

然后实现：

def tiled_gemm(A, B, BM=32, BN=32, BK=32):
    ...

循环顺序：

for bm
  for bn
    for bk
      for i
        for j
          for k

也就是：

M
│
├── BM
│
N
├── BN
│
K
└── BK

不要期待 Python 版本一定更快。

今天不是 benchmark Python interpreter，而是理解 memory access structure。

你真正要统计的是：

logical memory loads

给程序增加：

a_loads += 1
b_loads += 1

然后比较 naive 与 tiled 的理论数据复用。

更进一步，可以不模拟每次访问，直接建立模型：

naive：

A/B loads ≈ 2MNK

理想 tiled：

A loads ≈ MNK / BN
B loads ≈ MNK / BM

所以：

BM = BN = 32

时，理论 global-memory traffic 可以显著下降。

⸻

35–39 min｜为什么这和 Transformer 直接相关？

Transformer 中大量计算最终落到 GEMM：

X × Wq
X × Wk
X × Wv
X × Wo
X × Wup
X × Wgate
X × Wdown

例如：

[B × tokens, hidden]
        ×
[hidden, 4×hidden]

所以 vLLM Scheduler 决定：

这一轮处理多少 token

最终会改变 GEMM：

M dimension

例如：

decode-heavy iteration
M = 32
K = 4096
N = 11008

而 prefill-heavy：

M = 2048
K = 4096
N = 11008

同一个 operator：

GEMM

workload shape 完全不同。

于是上一轮的：

Scheduler

和今天的：

Kernel

真正连接起来：

request distribution
        ↓
continuous batching
        ↓
token count
        ↓
GEMM shape
        ↓
tile strategy
        ↓
SM utilization
        ↓
throughput

这就是为什么 operator benchmark 不能只测试一个漂亮的：

4096 × 4096 × 4096

shape。

⸻

39–42 min｜再接到 Tensor Parallel

假设 MLP：

X × W

把 W column-wise 分到 4 张 GPU：

              W
┌────────┬────────┬────────┬────────┐
│   W0   │   W1   │   W2   │   W3   │
└────────┴────────┴────────┴────────┘
    │        │        │        │
   GPU0     GPU1     GPU2     GPU3

每张 GPU：

X × Wi

于是 TP 增加后：

N_local = N / TP

单 GPU GEMM 变小。

这产生一个非常重要的现象：

TP ↑
   ↓
communication ↑
同时
GEMM dimensions ↓
   ↓
kernel efficiency 可能 ↓

因此 scaling 不是：

T(TP=8)
≈
T(TP=1)/8

而是：

T =
T_GEMM(shape / TP)
+
T_collective(TP)
+
T_sync

这把前三轮学过的内容全部串起来了：

Scheduler
    ↓
GEMM shape
    ↓
Kernel efficiency
    ↓
Tensor Parallel
    ↓
AllReduce
    ↓
NCCL
    ↓
RDMA / NVLink

⸻

42–45 min｜Performance Challenge

假设某 GPU：

Peak FP16 compute = 200 TFLOP/s
HBM bandwidth =
2 TB/s

于是 machine balance：

200 TFLOP/s
────────────
2 TB/s
= 100 FLOP/byte

现在两个 kernel：

Kernel A
Arithmetic Intensity =
20 FLOP/byte
Kernel B
Arithmetic Intensity =
200 FLOP/byte

先不要跑 profiler。

直接判断：

Kernel A
memory-bound or compute-bound?
Kernel B
memory-bound or compute-bound?

再算理论上限。

A：

2 TB/s × 20 FLOP/byte
=
40 TFLOP/s

所以即使：

Tensor Core utilization

不高，也不应该第一反应去优化 Tensor Core。

因为它的 Roofline 上限本来就是：

40 TFLOP/s

B：

2 TB/s × 200
=
400 TFLOP/s

超过 compute peak：

200 TFLOP/s

因此：

compute-bound

这就是以后看 Nsight Compute 时应该先有的性能模型。

今日 Takeaway

今天只需要真正留下这个公式：

Performance
≤
min(
    Peak Compute,
    Bandwidth × Arithmetic Intensity
)

以及：

Tiling
  ↓
data reuse ↑
  ↓
memory traffic ↓
  ↓
arithmetic intensity ↑

但 tile 越大又会增加：

register pressure
shared-memory pressure

所以 operator optimization 本质上是 memory hierarchy + parallelism + compute utilization 的资源平衡问题。

Next-step challenge

把今天的 tiled_gemm() 扩展成一个纯性能模型，而不是继续优化 Python：

输入：
M, N, K
BM, BN, BK
dtype_bytes
HBM_BW
peak_FLOPS
输出：
FLOPs
estimated HBM bytes
arithmetic intensity
memory-bound / compute-bound
roofline performance

然后比较：

M = 32      ← decode
M = 2048    ← prefill
K = 4096
N = 11008

并分别测试：

TP = 1 / 2 / 4 / 8

观察随着 TP 增加，单卡 GEMM 变小以后 arithmetic intensity 和 kernel efficiency 为什么可能恶化。

下一轮切换到 collective implementation：不再只是推导 Ring AllReduce 公式，而是自己用 Python/C 模拟 Reduce-Scatter + AllGather 的 chunk state machine，再加入 channel/pipeline，理解 NCCL 为什么要把一个大 tensor 拆成多个 chunk 并让通信持续“流动”起来。
