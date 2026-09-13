# 实验限制

- E1–E3 使用非流式请求。客户端记录的是端到端延迟，TTFT、Prefill、Queue 和 TPOT 来自 vLLM 原生指标，不能把客户端端到端延迟直接表述为 TTFT。
- Prometheus Histogram 的 P95 受 bucket 粒度限制。Queue 等短时阶段同时给出窗口内 `Δsum / Δcount` 均值，避免只根据粗粒度分位桶判断差异。
- E3B 的测试范围内没有出现 preemption 或 allocation failure，因此结论仅适用于当前模型、请求形态和并发范围，不能推出更高压力下仍无 KV 容量瓶颈。
- E4 固定发送 40 个请求，吞吐量按首个请求开始到最后一个请求结束的 request window 计算，代表该批 workload 的完成速率，不等同于长时间稳态容量。
- E1/E3A 未采集 scheduler-step 级 batch trace，对调度行为的解释基于请求结果和原生阶段指标，不作逐步调度路径的唯一归因。
- `max_tokens` 是输出上限。E4 使用 `ignore_eos=true` 控制输出长度；其他实验需结合实际输出 token 数解释时延和吞吐。
- Docker Compose 是参考部署拓扑，模型路径、GPU 设备、显存预算和运行参数需要按目标机器修改。
- 早期量化实验中的非流式客户端字段已按其真实含义归类为端到端请求时间和每输出 token 请求时间；真实 TTFT/TPOT 仅使用流式客户端或引擎原生指标。
- GPTQ 校准集由重复的短中英文模板构成；AWQ 使用 64 条 UltraChat 样本并截断到 128 token。当前结果用于验证量化与部署流程，不包含系统性的质量评测。
- llama.cpp 的 Q4_K_M/Q8_0 结果来自同一 RTX 5060 CUDA 环境和同一构建配置，适合比较该测试矩阵内的速度与显存差异，不外推为跨引擎或跨 GPU 的普遍结论。
