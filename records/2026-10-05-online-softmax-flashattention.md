# 2026-10-05｜第 15 次：Operator：从 Online Softmax 推导 FlashAttention

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #15｜Operator：从 Online Softmax 推导 FlashAttention

上一轮做的是 Prefill–Decode Disaggregation，重点在 distributed runtime 和 KV transfer。今天切到 operator implementation：不调用现成 attention API，而是从普通 Attention 一步步推导出 tiled / online Attention。

今天只解决一个问题：

> **FlashAttention 为什么能显著减少显存访问，却仍然精确计算 softmax(QKᵀ)V？**

重点不是记 FlashAttention 算法，而是理解 operator、HBM traffic、tiling、online reduction 之间的关系。最后再把它接回 PagedAttention 和 vLLM。

---

## 0–7 min｜先定位真正的问题：不是 FLOPs，而是中间矩阵

普通 Attention：

~~~
Q: [N, d]
K: [N, d]
V: [N, d]
S = QKᵀ               [N, N]
P = softmax(S)         [N, N]
O = PV                 [N, d]
~~~

复杂度：

~~~
QKᵀ:   O(N²d)
PV:    O(N²d)
~~~

但今天重点看 memory。

假设：

~~~
N = 8192
FP16
~~~

仅一个 S：

~~~
8192² × 2 bytes
≈ 128 MiB
~~~

如果执行过程是：

~~~
Q,K
 ↓
QKᵀ
 ↓
write S → HBM
read S ← HBM
 ↓
softmax
 ↓
write P → HBM
read P ← HBM
 ↓
PV
~~~

巨大的 N×N 中间结果反复经过 HBM。

因此真正的问题是：

~~~
compute QKᵀ
    ↓
materialize N² matrix
    ↓
HBM traffic
    ↓
softmax
    ↓
another N² matrix
    ↓
HBM traffic
~~~

这和 #7 的 GEMM tiling 是同一种问题：

数据已经在片上时，能不能把后续计算做完，而不是写回 HBM 后再读回来？

---

## 7–14 min｜困难点：Softmax 看起来必须看到整行

对于一行 score：

~~~
x = [x₁, x₂, ... xₙ]
~~~

稳定 softmax：

~~~
m = max(x)
pᵢ =
exp(xᵢ - m)
──────────────
Σ exp(xⱼ - m)
~~~

问题来了。

如果 score 被分块：

~~~
[x₁ ... x₄]
[x₅ ... x₈]
[x₉ ... x₁₂]
~~~

计算第一个 tile 时，只知道 max(x₁ ... x₄)，并不知道后面 x₁₀ 是否更大。

所以不能简单：

~~~
tile
 ↓
softmax(tile)
 ↓
PV(tile)
~~~

因为：

~~~
softmax([A,B])
≠
concat(
    softmax(A),
    softmax(B)
)
~~~

今天真正需要解决的就是：

怎样 streaming 地计算一个 global softmax？

---

## 14–25 min｜Hands-on ①：实现 Online Softmax

假设已经处理过一部分数据，维护两个状态：

~~~
m = 当前最大值
l = Σ exp(xᵢ - m)
~~~

来了新 tile：x_new。

计算：

~~~
m_new =
max(
    m,
    max(x_new)
)
~~~

旧的 l 是基于旧 max：

~~~
exp(xᵢ - m)
~~~

但现在基准变成 m_new，因此必须 rescale：

~~~
l_old_rescaled =
l × exp(m - m_new)
~~~

新 tile：

~~~
l_new_tile =
Σ exp(x_new - m_new)
~~~

最终：

~~~
l_new =
l × exp(m - m_new)
+
Σ exp(x_new - m_new)
~~~

实现：

~~~python
import math

def online_softmax_stats(xs, block_size=4):
    m = float("-inf")
    l = 0.0
    for start in range(0, len(xs), block_size):
        block = xs[start:start + block_size]
        block_max = max(block)
        m_new = max(m, block_max)
        l = (
            l * math.exp(m - m_new)
            + sum(
                math.exp(x - m_new)
                for x in block
            )
        )
        m = m_new
    return m, l
~~~

然后 reference：

~~~python
def reference_stats(xs):
    m = max(xs)
    l = sum(math.exp(x - m) for x in xs)
    return m, l
~~~

测试：

~~~python
xs = [
    1.2, 3.4, -0.7, 2.1,
    8.2, 1.1, 0.2, 4.5,
]

m1, l1 = reference_stats(xs)
m2, l2 = online_softmax_stats(xs)
assert abs(m1 - m2) < 1e-6
assert abs(l1 - l2) < 1e-6
~~~

这里最重要的是理解 m 和 l 就是 streaming reduction state。

它和网络里的 connection state，或者通信算法里的 partial reduction 很相似：不需要保存所有历史输入，只需要保存足以继续计算的状态。

---

## 25–33 min｜Hands-on ②：直接 Online Attention，不再保存 Attention Matrix

真正需要的是：

~~~
O =
softmax(qKᵀ)V
~~~

对于一个 query q，不要保存 scores[0:N]，而维护第三个状态 o。它表示当前未归一化的 weighted V accumulator。

对于新的 K/V tile：

~~~
scores = q @ K_tile.T
~~~

先：

~~~
m_new =
max(m, max(scores))
~~~

然后旧输出也必须 rescale：

~~~
o_old_rescaled =
o × exp(m - m_new)
~~~

新 tile contribution：

~~~
o_tile =
Σ exp(scoreᵢ - m_new) × Vᵢ
~~~

所以：

~~~
o_new =
o × exp(m - m_new)
+
Σ exp(scoreᵢ - m_new)Vᵢ
~~~

最后：

~~~
O = o / l
~~~

也就是同时维护 (m, l, o) 状态。

伪代码：

~~~python
m = -inf
l = 0
o = zeros(d)

for K_tile, V_tile in tiles:
    scores = q @ K_tile.T
    m_new = max(m, max(scores))
    alpha = exp(m - m_new)
    weights = exp(scores - m_new)
    l = alpha * l + sum(weights)
    o = alpha * o + weights @ V_tile
    m = m_new

return o / l
~~~

这一步就是今天的核心。

注意我们从来没有创建 [N, N] 的 attention matrix。

---

## 33–37 min｜从单 Query 扩展成 Tile

真实 kernel 不会只处理一个 q，而是：

~~~
Q tile:
Br × d

K tile:
Bc × d
~~~

片上计算：

~~~
S_tile =
Q_tile × K_tileᵀ

shape:
Br × Bc
~~~

然后：

~~~
S_tile
 ↓
online softmax update
 ↓
P_tile × V_tile
 ↓
update O_tile
~~~

整个过程：

~~~
                 HBM
       Q         K         V
       │         │         │
       ▼         ▼         ▼
       ┌───────────────────┐
       │   SRAM/register   │
       │                   │
       │ Q_tile            │
       │ K_tile            │
       │ V_tile            │
       │                   │
       │ QKᵀ tile          │
       │ ↓                 │
       │ softmax state     │
       │ ↓                 │
       │ output accumulator│
       └───────────────────┘
                │
                ▼
               HBM
                O
~~~

关键区别：

传统：

~~~
QKᵀ
 ↓
HBM
 ↓
softmax
 ↓
HBM
 ↓
PV
~~~

tiled attention：

~~~
QKᵀ tile
 ↓
softmax update
 ↓
PV update
~~~

尽可能都在片上完成。

所以 FlashAttention 的关键不是减少 QK FLOPs，而是减少 HBM IO。

这就是所谓 IO-aware algorithm。

---

## 37–40 min｜现在接回 PagedAttention

这里出现一个有意思的冲突。

FlashAttention 喜欢 K tile、V tile 规则连续地加载。

但 #9 的 Paged KV：

~~~
logical KV:
B0 B1 B2 B3

physical:
17 → 3 → 42 → 8
~~~

所以 decode attention 实际可能是：

~~~
block_table
    │
    ▼
physical block 17
    │
    ▼
load K/V tile
block_table
    │
    ▼
physical block 3
    │
    ▼
load K/V tile
~~~

这意味着 inference attention kernel 同时要解决两个问题：

算法层：

~~~
online softmax
+
tiling
+
IO reduction
~~~

runtime memory 层：

~~~
paged KV
+
indirect addressing
+
block mapping
~~~

最终：

~~~
logical sequence
       ↓
block table
       ↓
physical KV block
       ↓
load K/V tile
       ↓
QKᵀ tile
       ↓
online softmax
       ↓
output accumulator
~~~

这就是 PagedAttention 和 FlashAttention 并不是互相替代的两个概念。

它们主要解决不同层的问题：

~~~
Paged KV:
memory management / addressing

FlashAttention:
attention computation / IO
~~~

而 inference framework 的 attention backend 需要把两者协调起来。

---

## 40–43 min｜Prefill 和 Decode 为什么又不一样？

Prefill：

~~~
Q:
[N, d]

K,V:
[N, d]
~~~

有大量 query：Q tile × K tile。

所以矩阵运算规模大，tiling 和 Tensor Core utilization 很重要。

Decode：

~~~
Q:
[B, 1, d]

KV:
[B, N, d]
~~~

每个 request 只有很少的新 query，却必须读取整个历史 KV。

所以 decode 更接近：

~~~
small Q
+
large KV scan
~~~

这就是为什么 decode attention 往往更受 KV memory bandwidth 影响。

因此 scheduler 又重新进入 picture：

~~~
vLLM Scheduler
       ↓
iteration composition
       ↓
prefill-heavy / decode-heavy
       ↓
attention shape
       ↓
kernel path
       ↓
compute vs memory behavior
~~~

也就是说 #6 的 continuous batching 最终确实会改变今天 operator 的 workload。

---

## 43–45 min｜Performance Challenge

假设：

~~~
N = 8192
d = 128
FP16
~~~

传统 attention 至少 materialize：

~~~
S:
8192 × 8192 × 2
≈ 128 MiB
~~~

如果还 materialize P，又约 128 MiB。

只算：

~~~
write S
read S
write P
read P
~~~

就已经产生大约 512 MiB HBM traffic，还没算 Q、K、V、O。

而 tiled online attention 不需要把完整 S、P 写回 HBM。

现在假设：

~~~
HBM bandwidth =
2 TB/s
~~~

仅这 512 MiB 理论最小时间：

~~~
≈ 0.25 ms
~~~

一个 kernel 看起来不多，但 32 layers 就是约 8 ms；而且实际执行还包括其他数据访问、kernel launch 和非理想 bandwidth。

这就是为什么分析 GPU operator 时，不能只数 FLOPs；必须同时数 bytes。

---

## 今日 Takeaway

今天只需要真正掌握这一条：

普通 Attention：

~~~
QKᵀ
 ↓
materialize N²
 ↓
HBM
 ↓
softmax
 ↓
HBM
 ↓
PV
~~~

Online / tiled Attention：

~~~
QKᵀ tile
 ↓
(m,l,o) streaming state
 ↓
softmax + PV
 ↓
next tile
~~~

FlashAttention 最关键的思想不是一个特殊 CUDA trick，而是：

通过 tiling + online softmax，把原本需要 materialize 的 N×N 中间结果留在片上并立即消费，从而用额外的计算组织复杂度换取大幅减少的 HBM IO。

把最近几次串起来，现在你应该能看到完整链条：

~~~
Request
  ↓
vLLM Scheduler
  ↓
batch / token shape
  ↓
Paged KV block table
  ↓
Attention backend
  ↓
K/V tile load
  ↓
online softmax
  ↓
GPU operator
  ↓
TP projection
  ↓
NCCL
  ↓
RDMA
~~~

这正是从 runtime、operator 一直贯穿到 communication library 的 AI Infra execution path。

### Next-step challenge

把今天的 online attention 真正实现成 NumPy 版本，分别写 naive_attention() 和 tiled_attention(BQ, BK)，随机生成 Q/K/V 后验证最大误差；同时统计两种实现的峰值中间 tensor bytes，不要只比较运行时间。

下一次切换到 communication-library implementation：实现一个简化的 NCCL topology-aware collective planner。输入 GPU/NVLink/PCIe/NIC topology graph，让程序在 Ring 和 Tree 之间选择并估算 AllReduce latency；这样会从“理解 Ring”进一步走到通信库真正需要做的 topology discovery → algorithm selection → channel construction。
