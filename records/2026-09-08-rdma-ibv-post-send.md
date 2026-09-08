# 2026-09-08｜第 4 次：RDMA：从 ibv_post_send() 追到 NIC DMA 与 CQE

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #4｜RDMA：从 `ibv_post_send()` 追到 NIC DMA 与 CQE

前三次已经覆盖了 **vLLM KV Cache → Ring AllReduce → CUDA/NCCL overlap debugging**。今天切到 RDMA，并且不从“RDMA 是什么”开始，而是训练通信库开发最重要的能力之一：

> **看到一次 RDMA Write，能够在脑中还原 CPU、QP、WQE、Doorbell、NIC、PCIe、DMA、CQ/CQE 之间的数据路径。**

今天结束时，你应该能够解释一个很实用的问题：

**为什么 `ibv_post_send()` 返回成功，并不意味着数据已经到达远端？**

### 0–8 min｜先画出 RDMA Write 的完整路径

考虑：

```c
struct ibv_sge sge = {
    .addr   = (uintptr_t)buf,
    .length = len,
    .lkey   = mr->lkey,
};

struct ibv_send_wr wr = {
    .opcode     = IBV_WR_RDMA_WRITE,
    .sg_list    = &sge,
    .num_sge    = 1,
    .wr_id      = 42,
    .send_flags = IBV_SEND_SIGNALED,
};

wr.wr.rdma.remote_addr = remote_addr;
wr.wr.rdma.rkey        = remote_rkey;

ibv_post_send(qp, &wr, &bad_wr);
```

不要把它理解成：

```text
ibv_post_send()
      │
      ▼
send packet
      │
      ▼
remote memory
```

更准确的 mental model 是：

```text
Application
    │
    │ ibv_post_send()
    ▼
libibverbs / provider
    │
    │ build WQE
    ▼
Send Queue
    │
    │ ring doorbell
    ▼
NIC / HCA
    │
    │ DMA read
    ▼
local registered memory
    │
    ▼
packetization
    │
    ▼
NIC ─── network ─── remote NIC
                        │
                        │ DMA write
                        ▼
                 remote registered memory

local NIC
    │
    │ completion
    ▼
CQ
    │
    ▼
CQE
    │
    ▼
ibv_poll_cq()
```

今天要建立的第一个重要概念：

```text
post
≠
execute
≠
complete
```

这和你熟悉的异步网络 I/O 很像：

```text
submit async I/O
≠
I/O finished
```

---

## 8–15 min｜理解 WQE、Doorbell 和 CQE

把 Queue Pair 简化成：

```text
QP
├── SQ: Send Queue
└── RQ: Receive Queue
```

应用调用：

```text
ibv_post_send()
```

本质是在告诉 NIC：

```text
“这里有新的 Work Request 可以执行。”
```

用户态 verbs/provider 会把 WR 转成 NIC 能理解的 **WQE（Work Queue Element）**。

可以抽象为：

```text
WR
│
│ software representation
▼
WQE
│
│ hardware representation
▼
NIC
```

例如：

```text
WQE
--------------------------------
opcode        RDMA_WRITE
local_addr    0x100000
length        4096
lkey          0x123
remote_addr   0x800000
rkey          0x456
--------------------------------
```

然后发生非常关键的一步：

```text
CPU
 │
 │ write doorbell
 ▼
NIC
```

Doorbell 的语义大致是：

> **SQ 中有新的 WQE，你可以开始处理了。**

NIC 随后通过 PCIe DMA 读取本地 buffer。

注意这里的数据路径：

```text
CPU
  │
  │ control
  ▼
NIC

Memory
  │
  │ DMA
  ▼
NIC
```

CPU 不需要：

```text
memcpy(user_buffer, kernel_buffer)
```

这正是 RDMA 高性能的关键之一。

---

## 15–25 min｜今天最重要的系统问题：为什么必须注册内存？

调用：

```c
ibv_reg_mr(pd, buf, size, flags);
```

之后得到：

```text
MR
├── addr
├── length
├── lkey
└── rkey
```

为什么 NIC 不能直接 DMA 任意：

```text
void *buf
```

？

从 Linux virtual memory 的角度思考。

应用看到：

```text
virtual address
0x7f123456...
```

但 NIC 做 DMA 时最终需要能够访问：

```text
physical / DMA mapped memory
```

而用户态页面正常情况下可能：

```text
page fault
swap
migration
unmap
```

NIC 不可能在发送过程中突然处理：

```text
page fault
```

所以注册内存的核心作用可以先理解成：

```text
user virtual memory
       │
       │ registration
       ▼
stable DMA-accessible mapping
       │
       ├── local protection: lkey
       │
       └── remote protection: rkey
```

这和 Linux 内核经验直接连接起来：

```text
virtual memory
page pinning
DMA mapping
IOMMU
device DMA
```

这也是为什么通信库设计经常非常关心：

```text
registration cache
```

因为如果每次发送都：

```text
register
send
deregister
```

registration overhead 会非常明显。

---

# 25–35 min｜实现：Mini RDMA Queue Pair Simulator

今天的 hands-on 不要求 RDMA 网卡。

用 Python 实现一个极简 QP。

数据结构：

```python
from dataclasses import dataclass

@dataclass
class WQE:
    wr_id: int
    opcode: str
    local_addr: int
    remote_addr: int
    length: int

@dataclass
class CQE:
    wr_id: int
    status: str
```

然后：

```python
class QueuePair:

    def __init__(self):
        self.sq = []
        self.cq = []

    def post_send(self, wqe):
        ...

    def nic_progress(self):
        ...

    def poll_cq(self):
        ...
```

要求模拟下面过程：

```python
qp.post_send(
    WQE(
        wr_id=1,
        opcode="WRITE",
        local_addr=0x1000,
        remote_addr=0x8000,
        length=4096,
    )
)
```

此时必须：

```text
SQ:
[WQE1]

CQ:
[]
```

特别注意：

```python
post_send()
```

不能直接产生 CQE。

然后：

```python
qp.nic_progress()
```

模拟 NIC：

```text
fetch WQE
    ↓
DMA read
    ↓
network transfer
    ↓
remote DMA write
    ↓
generate CQE
```

之后：

```text
SQ:
[]

CQ:
[CQE1]
```

最后：

```python
qp.poll_cq()
```

得到：

```text
wr_id = 1
status = SUCCESS
```

---

## 加一个真正像通信库的问题

连续 post：

```text
WQE1
WQE2
WQE3
WQE4
```

但是只有 WQE4：

```text
IBV_SEND_SIGNALED
```

模拟：

```text
WQE1  unsignaled
WQE2  unsignaled
WQE3  unsignaled
WQE4  signaled
```

问：

**为什么高性能通信库不会让每一个 WQE 都产生 CQE？**

想想：

```text
CQE generation
CQ memory traffic
PCIe traffic
CQ polling
CPU processing
```

如果：

```text
1 million messages/s
```

每个 message 都产生 CQE，就意味着：

```text
1 million CQEs/s
```

因此常见优化思路是：

```text
post
post
post
post
post
post
post
SIGNALLED post
        │
        ▼
       CQE
```

这就是：

```text
completion moderation
```

的基本思想。

这已经非常接近真正通信库的 hot path 设计。

---

# 35–40 min｜把它连接到 NCCL

现在把抽象层往上拉：

```text
LLM inference
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
transport
      │
      ├── NVLink
      │
      └── Network
             │
             ▼
            RDMA
             │
             ▼
             QP
             │
             ▼
            WQE
             │
             ▼
            NIC
```

因此以后看到：

```text
NCCL AllReduce latency spike
```

你的 debugging tree 不应该只有：

```text
NCCL
```

而应该继续往下：

```text
collective algorithm
        │
        ▼
chunk / channel
        │
        ▼
transport
        │
        ▼
RDMA QP
        │
        ├── SQ starvation?
        ├── CQ polling?
        ├── WQE backlog?
        ├── retransmission?
        └── NIC congestion?
```

这里正是传统 networking engineer 向 AI communication engineer 转型时最有价值的知识迁移。

---

# 40–43 min｜一个非常重要的区别：SEND vs RDMA WRITE

比较：

```text
SEND
```

和：

```text
RDMA WRITE
```

SEND：

```text
Sender
  │
 SEND
  ▼
Network
  │
  ▼
Receiver RQ
  │
  ▼
posted receive buffer
```

receiver 必须提前：

```c
ibv_post_recv()
```

RDMA WRITE：

```text
Sender
  │
  │ remote_addr + rkey
  ▼
Network
  │
  ▼
Remote NIC
  │
  │ DMA
  ▼
Remote Memory
```

远端 CPU 不需要为每次 WRITE 执行：

```text
recv()
```

因此：

```text
SEND/RECV
```

更像：

```text
message passing
```

而：

```text
RDMA READ/WRITE
```

更像：

```text
remote memory operation
```

理解这个区别以后，再看 NCCL、UCX、NVSHMEM 等通信系统的 transport design 会容易很多。

---

# 43–45 min｜Debugging Challenge

现在假设：

```text
ibv_post_send()      success

ibv_poll_cq()
    ↓
一直没有 CQE
```

不要直接判断：

```text
network packet loss
```

建立 troubleshooting tree：

```text
Application
    │
    ▼
WQE posted?
    │
    ▼
Doorbell rung?
    │
    ▼
NIC fetched WQE?
    │
    ▼
DMA succeeded?
    │
    ▼
Packet transmitted?
    │
    ▼
Remote QP reachable?
    │
    ▼
Remote memory/rkey valid?
    │
    ▼
Transport completed?
    │
    ▼
CQE generated?
    │
    ▼
Application polling correct CQ?
```

给自己 2 分钟，至少列出 **5 个 hypothesis**。

例如其中一个：

```text
Hypothesis:
应用 poll 了错误的 CQ

Evidence:
NIC counters 显示 packets transmitted

Verification:
检查 QP → send_cq binding
```

尽量全部按照：

```text
Hypothesis
    ↓
Evidence
    ↓
Verification
```

来写，而不是罗列“可能原因”。

---

## 今日 Takeaway

今天真正需要留下的是这一条路径：

```text
ibv_post_send()
      ↓
WR
      ↓
WQE in SQ
      ↓
Doorbell
      ↓
NIC fetch
      ↓
PCIe DMA
      ↓
Network
      ↓
Remote NIC DMA
      ↓
Completion
      ↓
CQE
      ↓
ibv_poll_cq()
```

以及三个不等式：

```text
post ≠ execute

execute ≠ complete

CPU return ≠ remote completion
```

当你能够沿着这条链逐层定位问题时，你就开始从“会使用 RDMA API”进入了**通信库工程师的系统视角**。

**Next-step challenge：**给今天的 Mini QP Simulator 增加 `SQ_DEPTH=8`、signaled/unsignaled WQE、CQ depth 和 `nic_progress(max_wqes)`，然后故意制造 **SQ full、CQ overflow、错误 rkey** 三种故障。下一次切换回 inference runtime，可以沿着 **vLLM continuous batching：一次 scheduler step 如何把 running/waiting requests 转换成一次 GPU model execution** 往下追，并把 Scheduler、KV allocation 和实际 CUDA workload 串成完整执行链。
