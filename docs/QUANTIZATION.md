# 量化模型部署与评测

本项目中的量化工作定位为**量化模型转换/加载、推理引擎适配、资源约束观察与服务性能评测**。它可以证明具备量化模型部署经验，但不等同于设计量化算法、编写量化 CUDA Kernel 或完成严格的模型精度研究。

## 已完成范围

| 格式 | 推理引擎 | 环境 | 工作内容 |
|---|---|---|---|
| FP16 | vLLM 0.22.1 | RTX 4090D 24 GB | 建立基线并完成 72 组并发矩阵 |
| AWQ-Int4 | vLLM 0.22.1 | RTX 4090D 24 GB | 验证量化权重加载、compressed-tensors/Marlin 路径并完成 23 组矩阵 |
| GPTQ-Int4 / AWQ-Int4 | vLLM | RTX 4090D 24 GB | 小样本部署探索，观察格式、模型体积与显存预分配行为 |
| GGUF Q4_K_M / Q8_0 | llama.cpp | RTX 5060 8 GB | 本地低显存部署和 llama.cpp 格式验证 |

FP16 矩阵包含 72 组，每组 30 请求，共 2160 条成功记录；AWQ 矩阵包含 23 组，每组 30 请求，共 690 条成功记录。精简统计见 [`results/quantization-summary.csv`](../results/quantization-summary.csv)，来源文件指纹见 [`results/source-manifest.json`](../results/source-manifest.json)。原始逐请求 CSV 与 GPU 采样日志保留在只读历史档案中，不放入求职展示仓库。

## 可解释的工程认识

- 量化权重减少模型文件和权重工作集，但 vLLM 的 `gpu_memory_utilization` 还会影响 KV Cache 预留，因此不能只看 `nvidia-smi` 总占用判断量化是否生效。
- AWQ/GPTQ 的实际速度取决于量化格式、执行 Kernel、batch/concurrency 和 workload；不能用“INT4 必然比 FP16 快”概括。
- GGUF 适合 llama.cpp 的本地或 CPU/GPU 混合部署；vLLM 的 FP16/AWQ 数据与 llama.cpp 的 GGUF 数据来自不同 GPU，速度不可横向归因。
- 当前项目没有困惑度、任务准确率或人工质量评测，因此只讨论部署与性能，不声称量化后的模型质量等价于 FP16。

## 历史指标修正

旧 W6 benchmark 使用非流式请求，收到完整 JSON 后才结束计时。实现中：

```text
ttft_ms = e2e_latency_ms
tpot_ms = e2e_latency_ms / output_tokens
```

所以这两个字段分别是完整响应 E2E 和平均每输出 token 的请求耗时，不是真实 TTFT 与 Streaming TPOT。旧报告中依赖这些字段的 TTFT/TPOT 表格、因果归因和量化加速比不迁移。历史矩阵仍能证明部署覆盖、请求成功率，并可在相同 workload 下比较非流式 E2E 与 request output rate。

E1–E3 的 TTFT、Queue 和 Prefill 来自 vLLM Native Metrics；E4 的 TPOT 来自 Streaming Client 并与 Native Metrics 交叉验证。两套口径与旧量化矩阵分开呈现。

## 面试表达边界

可以表述为：

> 完成 Qwen2.5-7B FP16、GPTQ、AWQ 与 GGUF 模型的部署验证；在 RTX 4090D 上执行 FP16/AWQ 共 95 组、2850 请求矩阵，并识别、修正早期非流式 benchmark 将 E2E 错当 TTFT/TPOT 的指标问题。

不应表述为：

- 自研 GPTQ/AWQ 量化算法；
- 开发 Marlin 或其他量化 Kernel；
- 已证明量化没有精度损失；
- GGUF 与 vLLM 的跨 GPU 速度对比可以证明引擎优劣。
