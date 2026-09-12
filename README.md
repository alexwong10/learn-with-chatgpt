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
