# 2026-09-22｜第 10 次：GPUDirect RDMA：为什么 NIC 可以直接读写 GPU 显存？

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #10｜GPUDirect RDMA：为什么 NIC 可以直接读写 GPU 显存？

上一轮是 **PagedAttention / KV memory → kernel**。这次切到通信 data path，把 #4 的 RDMA 再向 GPU 方向推进一步。

今天的核心问题：

> **跨机 Tensor Parallel 做 AllReduce 时，一块 GPU 上的数据究竟怎样到达另一台机器的 GPU？GPUDirect RDMA 相比 CPU staging 少掉了什么？**

目标不是背 GPUDirect RDMA 技术名词，而是建立一条能用于 NCCL 性能分析的完整 mental model。

### 0–8 min｜先画出两条数据路径

没有 GPUDirect RDMA 时，可以把跨机 GPU 通信简化为：

```text
Node A

GPU Memory
    │
    │ GPU → CPU memory copy
    ▼
Pinned Host Memory
    │
    │ NIC DMA read
    ▼
   NIC
    │
==== Network ====
    │
   NIC
    │
    │ NIC DMA write
    ▼
Pinned Host Memory
    │
    │ CPU memory → GPU copy
    ▼
GPU Memory
```

这里至少涉及：

```text
GPU memory → host memory
network transfer
host memory → GPU memory
```

如果传 1 GiB 数据，即使网络本身很快，host staging 也会增加 PCIe 流量、同步和软件开销。

GPUDirect RDMA 的目标可以抽象成：

```text
Node A                         Node B

GPU Memory                    GPU Memory
    │                              ▲
    │ PCIe DMA                     │ PCIe DMA
    ▼                              │
   NIC =========================> NIC
              RDMA
```

也就是：

```text
GPU memory
    ↕
NIC
```

而不是：

```text
GPU
 ↕
CPU memory
 ↕
NIC
```

关键不是“完全绕过 PCIe”。

真正减少的是 **host-memory staging**。

---

## 8–15 min｜把 #4 的 RDMA memory registration 模型迁移过来

普通 RDMA：

```text
user virtual address
        │
        ▼
ibv_reg_mr()
        │
        ▼
DMA-accessible memory
        │
        ├── lkey
        └── rkey
```

现在 buffer 不再来自：

```c
malloc()
```

而可能来自：

```text
cudaMalloc()
     ↓
GPU virtual address
```

NIC 要访问它，就产生了一个很熟悉的问题：

> NIC 怎么把 GPU virtual address 转换成真正能够 DMA 的 GPU memory？

概念上仍然需要：

```text
GPU VA
   │
   ▼
memory registration
   │
   ▼
DMA mapping
   │
   ▼
NIC
```

所以 GPUDirect RDMA 并没有消灭：

```text
registration
mapping
protection
```

只是把 DMA target/source 从：

```text
host DRAM
```

变成：

```text
GPU memory
```

这也是为什么 registration cache 在 GPU 通信里仍然重要。

---

# 15–25 min｜Hands-on：建立一个通信路径成本模型

今天不需要 GPU。

用 Python 写一个简单模型：

```python
def transfer_time(
    size_bytes,
    network_bw,
    pcie_bw,
    network_latency,
    staged,
):
    ...
```

先假设：

```text
size = 1 GiB

PCIe effective bandwidth =
25 GB/s

Network bandwidth =
50 GB/s

Network latency =
5 μs
```

### CPU staging

粗略模型：

```text
GPU → Host
+
Network
+
Host → GPU
```

因此：

```text
T_staged ≈
S / PCIe_BW
+
S / Network_BW
+
S / PCIe_BW
+
latency
```

代入：

```text
1 / 25
+
1 / 50
+
1 / 25

≈ 100 ms
```

忽略单位换算细节，数量级即可。

GPUDirect：

```text
T_GDR ≈
S / effective_path_BW
+
latency
```

注意不要简单写：

```text
S / network_bw
```

因为真正链路是：

```text
GPU memory
   ↓
PCIe
   ↓
NIC
   ↓
Network
```

所以简单模型可以：

```python
effective_bw = min(
    pcie_bw,
    network_bw,
)
```

于是：

```text
T_GDR ≈
S / min(PCIe_BW, Network_BW)
```

在上面的参数中：

```text
≈ 40 ms
```

然后分别测试：

```text
4 KiB
64 KiB
1 MiB
16 MiB
1 GiB
```

输出：

```text
size
staged_time
gdr_time
speedup
```

你应该观察到：

```text
small message
→ latency / setup 更重要

large message
→ bandwidth / copy elimination 更重要
```

这和 #2 的 collective cost model 完全连接起来了。

---

# 25–32 min｜加入 PCIe topology：为什么“支持 GDR”还不够？

假设服务器：

```text
             CPU

        ┌─────┴─────┐
        │           │
     PCIe SW0     PCIe SW1
      /    \        /    \
   GPU0   GPU1    GPU2   NIC
```

GPU2 → NIC：

```text
GPU2
 ↓
PCIe SW1
 ↓
NIC
```

GPU0 → NIC：

```text
GPU0
 ↓
PCIe SW0
 ↓
CPU / root complex
 ↓
PCIe SW1
 ↓
NIC
```

虽然两者都可能：

```text
GPUDirect RDMA capable
```

实际 bandwidth / latency 却可能不同。

于是：

```text
GPU0 AllReduce slow
GPU2 AllReduce fast
```

不能只检查：

```text
GDR enabled?
```

还要检查：

```text
GPU ↔ NIC affinity
PCIe topology
NUMA
ACS / routing
link width / generation
```

这和普通服务器网络性能调优里的 NUMA affinity 是同一个思维模式：

```text
logical connectivity
≠
physical locality
```

---

# 32–38 min｜把它接到 NCCL

现在展开跨机 Tensor Parallel：

```text
vLLM
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
 ├── topology discovery
 │
 ├── algorithm
 │
 ├── channel/chunk
 │
 └── transport
       │
       ▼
      RDMA
       │
       ▼
GPU memory ←→ NIC
```

假设一次 AllReduce：

```text
8 GPUs
2 nodes
4 GPUs/node
```

这里其实有两个完全不同的 transport domain：

```text
intra-node:

GPU
 ↕
NVLink / PCIe
 ↕
GPU


inter-node:

GPU
 ↕
PCIe
 ↕
NIC
 ↕
RDMA network
 ↕
NIC
 ↕
PCIe
 ↕
GPU
```

所以你在 #2 里写的：

```text
T ≈ 2(N-1)/N × S/BW
```

现在应该升级成：

```text
T_collective =
f(
    topology,
    algorithm,
    intra-node BW,
    inter-node BW,
    chunk size,
    pipeline,
    GDR path
)
```

一个单一的：

```text
BW = 50 GB/s
```

已经不足以描述真实机器。

---

# 38–42 min｜Performance Debugging：看到 NCCL 慢，该看哪里？

假设：

```text
2 nodes
4 GPUs/node

AllReduce size = 512 MiB

GPU0-3:
40 GB/s

GPU4-7:
18 GB/s
```

不要马上说：

```text
RDMA network slow
```

建立 hypothesis tree：

```text
                    NCCL slow
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      algorithm       GPU/NIC       network
                      topology
          │             │             │
      Ring/Tree      PCIe path      congestion
      channels       NUMA           packet loss
      chunk size     affinity       retransmission
                        │
                        ▼
                       GDR
```

然后坚持之前练过的格式：

```text
Hypothesis
→ Metric
→ Expected observation
```

例如：

```text
Hypothesis:
GPU4-7 到 NIC 的 PCIe path 较差

Metric:
GPU/NIC topology + PCIe throughput

Expected:
慢 rank 都位于同一较差 PCIe path
```

另一个：

```text
Hypothesis:
inter-node network 才是瓶颈

Metric:
NIC port throughput

Expected:
NIC 接近 line rate，
而 GPU/NIC PCIe 尚未饱和
```

再一个：

```text
Hypothesis:
通信没有真正走 GPUDirect RDMA

Metric:
NCCL transport/debug information
+ host-memory traffic

Expected:
存在 host staging，
CPU/DRAM/PCIe 出现额外 copy traffic
```

这就是系统级 debugging 和“调 NCCL 参数试试看”的区别。

---

# 42–45 min｜最终 Challenge：找真正瓶颈

现在有：

```text
GPU HBM:
1500 GB/s

GPU ↔ NIC PCIe:
28 GB/s

NIC:
50 GB/s

Network:
45 GB/s effective

Remote NIC ↔ GPU PCIe:
24 GB/s
```

先不要用 profiler。

回答：

```text
GPU0
 ↓
local PCIe
 ↓
NIC
 ↓
network
 ↓
NIC
 ↓
remote PCIe
 ↓
GPU1
```

理论 steady-state bandwidth 上限是多少？

答案应该来自：

```text
min(
    1500,
    28,
    50,
    45,
    50,
    24,
    1500
)
```

也就是：

```text
24 GB/s
```

因此即使：

```text
NIC utilization ≈ 48%
```

也不能得出：

```text
network utilization 不够，
所以网络有问题
```

因为瓶颈实际上可能已经在：

```text
remote PCIe
```

这就是今天最重要的 performance intuition：

> **端到端吞吐量由 data path 上最窄的一段决定。**

### 今日 Takeaway

把今天压缩成：

```text
GPU memory
     │
     │ registration / mapping
     ▼
PCIe
     │
     ▼
NIC
     │
     │ RDMA
     ▼
Network
     │
     ▼
NIC
     │
     ▼
PCIe
     │
     ▼
Remote GPU memory
```

GPUDirect RDMA 的关键价值不是“网络绕过 PCIe”，而是：

**让 NIC 能直接 DMA GPU memory，从跨机 GPU 通信路径中去掉不必要的 host-memory staging。**

然后把最近几次练习串起来，你现在已经可以建立这样一条完整路径：

```text
vLLM Scheduler
      ↓
GPU workload shape
      ↓
GEMM / Attention
      ↓
Tensor Parallel
      ↓
AllReduce
      ↓
NCCL Ring / chunks
      ↓
RDMA WQE
      ↓
GPUDirect RDMA
      ↓
PCIe → NIC → Network
```

这条链就是从 **推理 runtime → operator → communication library → hardware data path** 的完整 AI Infra 视角。

**Next-step challenge：**把今天的 Python 模型升级成一个 topology graph：节点包括 `GPU / PCIe switch / NIC / network`，边具有 `bandwidth + latency`，给定两个 GPU 自动找通信路径，并计算 `bottleneck bandwidth`。然后模拟 **8 GPU × 2 node**，比较“任意 GPU 选 NIC”和“GPU-NIC affinity aware”两种 placement。

下一次切换到 **performance debugging + 实际 profiler reasoning**：给出一组接近真实的 vLLM/NCCL/CUDA timeline、GPU metrics 和网络 counters，从 **TTFT 变差**这个应用层现象开始，逐层定位到底是 scheduler、KV pressure、GEMM、NCCL 还是 RDMA 导致，并训练一次完整的 **application → runtime → GPU → communication → network root-cause analysis**。
