# 2026-09-19｜第 9 次：PagedAttention：从 Block Table 一路追到 GPU 的 KV Load

> 这份内容来自 ChatGPT“已计划”功能中的学习记录。

## AI Infra Systems Practice #9｜PagedAttention：从 Block Table 一路追到 GPU 的 KV Load

上一轮实现了 Ring collective 的 chunk state machine。今天切回 **inference runtime + operator boundary**，把 #1 的 KV block allocator 和 #6 的 scheduler 继续向下追到 attention kernel。

今天的目标不是“知道 PagedAttention 是分页 KV Cache”，而是能回答：

> **给定一个 token position，PagedAttention 到底怎样找到 GPU 上真正的 K/V 地址？这种间接寻址为什么值得付出？**

当前 vLLM 的 V1 worker 中仍有明确的 `BlockTable` 抽象，其中区分 KV-cache allocation block size 与底层 attention kernel block size，并负责把 scheduled token 映射到 cache slot。

### 0–8 min｜先手算地址翻译

假设：

```text
block_size = 4 tokens

Request A:
logical tokens = 0 ... 9

block_table[A] = [7, 2, 11]
```

逻辑 KV：

```text
token:
0 1 2 3 | 4 5 6 7 | 8 9
    B0        B1       B2
```

实际 GPU KV blocks：

```text
logical B0 → physical block 7
logical B1 → physical block 2
logical B2 → physical block 11
```

所以 token 6：

```text
logical_block = 6 // 4 = 1
offset        = 6 % 4  = 2

physical_block
= block_table[A][1]
= 2
```

最终 KV slot：

```text
slot = physical_block * block_size + offset

     = 2 * 4 + 2
     = 10
```

因此核心地址翻译就是：

```text
token position
      ↓
logical block
      ↓
block_table lookup
      ↓
physical block
      ↓
offset
      ↓
KV slot
```

当前 vLLM 源码中也能直接看到同样的映射形式：先用 `position // page_size` 索引 block table 得到 block ID，再通过 `position % page_size` 得到页内 offset，最后构造实际 slot。

这和你熟悉的虚拟内存很像：

```text
Virtual Memory              Paged KV Cache

virtual page                logical KV block
page table                  block table
physical frame              physical KV block
page offset                 token offset
```

但要注意：**这是很有用的类比，不是说 GPU 在使用 CPU MMU 页表机制。**

---

## 8–15 min｜为什么值得做这层映射？

如果不用 block table，一个 request 最舒服的布局当然是：

```text
Request A:

KV
████████████████████████████
```

整块连续。

问题是生成长度未知：

```text
prompt = 500

output:
10?
100?
1000?
8000?
```

如果预留：

```text
max_seq_len = 8192
```

大量显存没有真正存 token。

如果动态找连续空间，又会面对：

```text
allocation
fragmentation
relocation
copy
```

分页以后：

```text
A → [7, 2, 11]

B → [5, 1]

C → [9, 4, 13, 6]
```

物理 block 可以分散。

Request 增长时：

```text
A:
[7, 2, 11]

需要更多 KV
      ↓

allocate block 3

A:
[7, 2, 11, 3]
```

不需要把已有 KV：

```text
7 → 2 → 11
```

搬到一块新的连续区域。

这就是典型的：

```text
indirection cost
        ↕
allocation flexibility
```

系统设计 tradeoff。

---

# 15–27 min｜Hands-on：实现 CPU Paged KV Cache

今天的主要实现任务。

先建立物理 KV cache：

```python
BLOCK_SIZE = 4

kv_cache = [
    [None] * BLOCK_SIZE
    for _ in range(16)
]
```

block table：

```python
block_tables = {
    "A": [7, 2, 11],
    "B": [5, 1],
}
```

先实现：

```python
def translate(req_id, token_pos):
    logical_block = token_pos // BLOCK_SIZE
    offset = token_pos % BLOCK_SIZE

    physical_block = block_tables[req_id][logical_block]

    return physical_block, offset
```

再实现：

```python
def write_kv(req_id, token_pos, value):
    block, offset = translate(req_id, token_pos)
    kv_cache[block][offset] = value
```

以及：

```python
def read_kv(req_id, token_pos):
    block, offset = translate(req_id, token_pos)
    return kv_cache[block][offset]
```

测试：

```python
for i in range(10):
    write_kv("A", i, f"A_KV_{i}")
```

然后：

```python
for i in range(10):
    print(i, translate("A", i), read_kv("A", i))
```

应该看到：

```text
0 → (7,0)
1 → (7,1)
2 → (7,2)
3 → (7,3)

4 → (2,0)
5 → (2,1)
6 → (2,2)
7 → (2,3)

8 → (11,0)
9 → (11,1)
```

如果这一部分你能不参考资料自己写出来，PagedAttention 的内存模型就已经掌握了一半。

---

# 27–33 min｜把 KV lookup 放进 Attention

现在 query token：

```text
q
```

需要：

```text
K0 K1 K2 ... K9
```

计算：

```text
score_i = q · K_i
```

普通 contiguous cache：

```text
K0 K1 K2 K3 K4 K5 K6 K7 K8 K9
```

Paged cache：

```text
block 7:
K0 K1 K2 K3

block 2:
K4 K5 K6 K7

block 11:
K8 K9
```

kernel 的逻辑变成：

```python
for logical_block in context_blocks:

    physical_block = block_table[logical_block]

    for offset in range(BLOCK_SIZE):

        K = key_cache[physical_block][offset]

        score = dot(q, K)
```

当前 vLLM attention backend 的实际调用也把 `block_table` 与 `key_cache`、`value_cache`、sequence lengths 一起交给 paged-attention 路径。

所以注意：

```text
BlockTable
```

不是单纯 scheduler bookkeeping。

它最终进入：

```text
attention execution
```

这就是 runtime → operator 的边界。

---

# 33–38 min｜现在从 GPU 角度找问题

你已经知道分页的优点。

现在问：

**代价是什么？**

首先是额外 indirection：

```text
logical block
     ↓
load block_table
     ↓
physical block
     ↓
load K/V
```

而且：

```text
physical blocks
```

可能是：

```text
7 → 2 → 11 → 3
```

并不连续。

这对 GPU memory system 意味着需要认真考虑：

```text
memory coalescing
cache locality
block size
thread mapping
vectorized load
```

因此真正的 kernel 不会简单写：

```python
for token:
    lookup()
```

然后让每个 thread 随机访问。

而是让一组 threads 协作处理一个 KV block，使**一个 block 内的数据仍然规则、连续、可向量化**。

vLLM 的 PagedAttention 设计资料描述的典型组织方式就是：KV cache 被分为固定 token 数的 blocks，一个 warp 可以处理一个完整 KV block；thread group 再协作读取一个 token 的 query/key 向量。

这揭示了一个很重要的 kernel-design 思路：

```text
global layout 可以不连续

但

local work unit 内部必须尽量规则
```

这和很多系统设计高度一致。

---

# 38–41 min｜为什么 block size 不是随便选？

假设：

```text
block_size = 1
```

优点：

```text
几乎没有内部碎片
```

但：

```text
block table 巨大
allocation metadata 多
indirection 多
kernel work unit 太细
```

如果：

```text
block_size = 1024
```

metadata 很少，但一个 request 最后只使用：

```text
1 token
```

时可能浪费接近整个 block。

所以：

```text
small block
    │
    ├─ fragmentation ↓
    ├─ metadata ↑
    └─ kernel granularity ↓


large block
    │
    ├─ fragmentation ↑
    ├─ metadata ↓
    └─ locality / regularity ↑
```

这和上一轮 collective 的：

```text
chunk size
```

以及 #7 GEMM 的：

```text
tile size
```

本质上是同一种 systems problem：

> **选择合适的工作粒度。**

而且现实 backend 确实受 kernel 约束。例如当前 vLLM ROCm native paged-attention 路径只支持某些 block sizes（16/32），其他满足条件的尺寸会走不同 kernel 路径；backend 因此需要显式声明支持的 block size。

---

# 41–45 min｜Systems Challenge：你真的减少显存了吗？

假设：

```text
100 requests

max_seq_len = 8192
actual lengths:
[300, 700, 1200, ...]
```

传统静态 allocation：

```text
allocated_tokens
=
100 × 8192
=
819,200 token slots
```

假设实际总 token：

```text
100,000
```

利用率：

```text
≈ 12.2%
```

现在：

```text
block_size = 16
```

每个 request 分配：

```text
ceil(seq_len / 16)
```

个 blocks。

请给今天的程序增加：

```python
def allocated_tokens(length, block_size):
    return (
        (length + block_size - 1)
        // block_size
        * block_size
    )
```

然后计算：

```text
waste =
allocated_tokens
-
actual_tokens
```

分别模拟：

```text
block_size:

1
8
16
32
64
128
```

输出：

```text
block_size
allocated
wasted
utilization
block_table_entries
```

你会看到：

```text
block size ↑
       │
       ├── metadata ↓
       │
       └── internal fragmentation ↑
```

这时再增加一个 GPU/kernel 成本模型：

```text
cost =
metadata_cost
+
fragmentation_cost
+
kernel_efficiency_cost
```

你就会发现：

**最节省显存的 block size，并不一定是最快的 block size。**

## 今日 Takeaway

把今天全部压缩成这一条链：

```text
Scheduler
   ↓
KVCacheManager
   ↓
physical KV blocks
   ↓
BlockTable
   ↓
logical token
   ↓
physical block + offset
   ↓
PagedAttention kernel
   ↓
K/V memory load
   ↓
QKᵀ → softmax → V
```

最值得记住的是：

> **PagedAttention 把 KV cache 的“连续虚拟序列”与“不连续物理存储”解耦；block table 是 runtime 内存管理决策进入 attention kernel 的桥梁。**

这也是一个很典型的 AI Infra 设计：**用一层 indirection 换取更好的资源利用率，然后让 kernel 设计去消化这层 indirection 的性能成本。**

源码阅读建议今天只看两个入口，不要陷进整个仓库：

[vLLM V1 BlockTable implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/block_table.py?utm_source=chatgpt.com)

[vLLM paged-attention operator bindings](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py?utm_source=chatgpt.com)

### Next-step challenge

把今天的 CPU simulator 改成：

```text
Request A
Request B
Request C
        ↓
共享一个 physical block pool
```

支持：

```text
allocate_block()
free_request()
append_token()
translate()
```

然后故意制造：

```text
A finish
→ free [7,2,11]

D arrive
→ reuse [2,11,...]
```

验证 **request 生命周期改变时 block table 可以改变，而已经存活 request 的逻辑 token 空间无需移动**。

下一轮切换到 **RDMA / GPU communication**：从普通 registered host memory 再向下走一步，理解 **GPUDirect RDMA 为什么能绕过 CPU staging、GPU memory 如何暴露给 NIC，以及这条 data path 对 NCCL 跨机通信意味着什么**。
