# 实验限制

- E1–E3 使用非流式请求。客户端完整响应延迟不是 TTFT，报告中的 TTFT、Queue 和 Prefill 来自 vLLM Native Metrics。
- Prometheus Histogram 的短时 P95 会受到 bucket 粒度影响，因此短 Queue 的定量分析使用 counter 的 `Δsum/Δcount`。
- `max_tokens` 是生成上限。E4 使用 `ignore_eos=true` 固定实际输出长度；该设置只用于控制变量实验。
- E3B 观察到较高 KV Usage，但没有 Preemption 或 KV allocation failure，因此不能声称制造了 KV exhaustion。
- E4 是固定 40 请求批次，没有测量稳态最大吞吐或完整 Decode saturation plateau。
- E4 request-window throughput 会受到有限请求批次的启动与排空阶段影响，不能直接外推为在线服务容量。
- E1/E3A 没有记录 scheduler-step 级 scheduled tokens、active prefill count 和 batch shape，因此不对 Prefill 差异作唯一内部机制归因。
- Chunked Prefill ON/OFF 会同时改变本实验中的 effective `max_num_batched_tokens`，不能当作纯 Chunking Policy 单变量实验。
- Docker Compose 是参考编排。运行者需要根据模型目录、GPU 显存和镜像环境修改配置。
- 历史模型日志显示 compressed-tensors / Marlin 加载路径，不能保证任意同名 AWQ 权重具有相同结果；未归档完整模型 revision 和依赖锁文件。
- 新整理的脚本未在 4090D 完整重跑；历史结果与新脚本运行验证分开记录。
- 完整原始日志和 Prometheus 快照保存在本地实验档案中，公开仓库仅包含精简汇总、复现代码和关键图表。
