# learn-with-chatgpt

记录使用 ChatGPT 的“已计划”功能辅助学习的过程。每一次学习记录保存为一个独立的 Markdown 文档，并在这里附上历史记录说明与对应链接。

## 学习记录

| 日期 | 记录 | 说明 |
| --- | --- | --- |
| 2026-09-01 | [第 1 次：vLLM：把 KV Cache 当成“内存管理系统”来理解](records/2026-09-01-vllm-kv-cache.md) | vLLM 执行链路、KV Cache block 分配与 Scheduler 协作 |
| 2026-09-03 | [第 2 次：从 Ring AllReduce 推导 NCCL 通信成本](records/2026-09-03-ring-allreduce-nccl.md) | Ring AllReduce、NCCL 通信成本模型与 Tensor Parallel 扩展性 |
| 2026-09-06 | [第 3 次：Performance Debugging：证明 NCCL 和 GEMM 到底有没有 overlap](records/2026-09-06-performance-debugging-overlap.md) | CUDA timeline、stream dependency、kernel overlap 与硬件资源竞争 |
| 2026-09-08 | [第 4 次：RDMA：从 ibv_post_send() 追到 NIC DMA 与 CQE](records/2026-09-08-rdma-ibv-post-send.md) | RDMA Write 数据路径、WQE/Doorbell/CQE、Mini QP 与故障排查 |
| 2026-09-10 | [第 5 次：Distributed Systems：设计一个“不会被慢节点拖死”的推理 Worker 调度器](records/2026-09-10-straggler-worker-scheduler.md) | Straggler 检测、Worker 状态机、TP failure domain 与跨 rank 根因分析 |
| 2026-09-12 | [第 6 次：vLLM Continuous Batching：一次 schedule() 如何变成一次 GPU Forward](records/2026-09-12-vllm-continuous-batching.md) | Scheduler、KV allocation、Model Runner 与动态 GPU workload |
| 2026-09-15 | [第 7 次：Operator：从 Naive GEMM 到 Tiling，理解 GPU 为什么需要“数据复用”](records/2026-09-15-tiled-gemm-roofline.md) | Roofline、Arithmetic Intensity、Tiling、数据复用与 Tensor Parallel |
| 2026-09-19 | [第 8 次：Collective Implementation：亲手实现 Ring Reduce-Scatter + AllGather](records/2026-09-19-ring-reduce-scatter-allgather.md) | Ring 状态机、chunk pipeline、RDMA transport 与 backpressure |
| 2026-09-19 | [第 9 次：PagedAttention：从 Block Table 一路追到 GPU 的 KV Load](records/2026-09-19-pagedattention-block-table.md) | BlockTable 地址翻译、Paged KV Cache、GPU memory load 与 block size 权衡 |
| 2026-09-22 | [第 10 次：GPUDirect RDMA：为什么 NIC 可以直接读写 GPU 显存？](records/2026-09-22-gpudirect-rdma.md) | GPU-NIC DMA、PCIe topology、NCCL transport 与端到端瓶颈分析 |
| 2026-09-24 | [第 11 次：端到端性能诊断：TTFT 突然升高，到底该怪谁？](records/2026-09-24-end-to-end-performance-debugging.md) | TTFT critical path、Scheduler/KV pressure、NCCL/RDMA 归因与证据链 |
| 2026-09-26 | [第 12 次：CUDA Runtime：用 Stream + Event + DAG 判断“真 overlap”还是假 overlap](records/2026-09-26-cuda-runtime-overlap.md) | CUDA stream/event、依赖 DAG、chunk pipeline、资源冲突与 critical path |
