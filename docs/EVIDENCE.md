# 证据与复现范围

历史实验来自 AutoDL RTX 4090D；源码阅读基于 vLLM 0.21.0，实验运行时为 0.22.1。原始档案保留在旧 V3 项目，只读核对，不要求读者拥有该目录才能运行新实验。

## 数字映射

README/性能报告中的性能数字按下表定位。CSV 时间列均为毫秒；展示保留 1–2 位小数，核心成果中的秒数为毫秒除以 1000。模型名称、版本、GPU 和配置值来自启动记录，不属于 CSV 性能测量值。

| 展示项 | CSV / case | 字段或计算 | 当前脚本 |
|---|---|---|---|
| E1 并发、Prefill、Queue、TTFT | e1-e3-summary.csv / e1-v2-c1,c4,c8,c16,c32 | concurrency,prefill_time_avg_ms,queue_avg_ms,native_ttft_avg_ms | run_e1_concurrency.sh |
| E2 输入长度与性能 | e1-e3-summary.csv / e2-v3-128,512,1024,2048 | prompt_tokens_mean 及上述时间列 | run_e2_prefill.sh |
| E3A seq32 基线 | e1-e3-summary.csv / e1-v2-c16,c32 | Queue/Prefill/TTFT | run_e1_concurrency.sh |
| E3A seq8 对照 | e1-e3-summary.csv / e3a-v2-c16,c32 | Queue/Prefill/TTFT | run_e3a_scheduler.sh |
| E3B 显存比例对照 | e1-e3-summary.csv / e1-v2-c32,e3b-v2-040-c32,e3b-v2-035-c32 | 时间列及 preemptions_delta；显存比例来自配置记录 | run_e3b_kv_capacity.sh |
| E4 TTFT/TPOT | e4-summary.csv / concurrency=1,4,8,16,32 | client_ttft_p50_ms,client_tpot_p50_ms,native_tpot_avg_ms | run_e4_decode.sh |
| E4 输入129、输出512、40请求 | e4-summary.csv | prompt_tokens,output_tokens_per_request,successful_requests,total_requests | run_e4_decode.sh |
| E4 吞吐 | e4-summary.csv | successful_requests × output_tokens_per_request / request_window_seconds | run_e4_decode.sh |
| FP16/AWQ 量化矩阵规模 | quantization-summary.csv | groups,requests_per_group,total_requests,successful_requests | 历史 w6_fp16_matrix.sh / w6_awq_matrix.sh |
| GPTQ/AWQ/GGUF 量化产物与 llama-bench | quantization-artifacts.csv | artifact_size、quantization_time、pp512、tg128 | 历史量化实战记录与 W3 报告 |

以上 CSV 均位于 [results](../results/)。E1–E3 的 19 行保留字段已逐字段与历史 summary 比较；E4 的 5 行延迟列与历史 full summary 比较，请求窗口取各 case 日志中的 wall_time_s，吞吐重新计算。原始 full summary 的 aggregate_output_tps 使用外层时间，不沿用该值。

## 历史档案映射

- E1–E3：`monitoring/e1-e2-e3-v2-summary.csv`；原始请求与快照为 `monitoring/<case>.csv`、`<case>-before.txt`、`<case>-after.txt`。
- E4：`monitoring/e4/e4-summary-full.csv`、`e4-cN.csv`、`e4-cN.log` 和 before/after 快照。
- 配置：`monitoring/e1-v2-server-config.txt`、`monitoring/e4/vllm-e4-startup.log`；历史分析为 `docs/W9 performance/E1-E4-analysis.md`。
- 架构阅读：`docs/W7/vllm system-architecture.md`，原稿明确标注 0.21.0。迁移文档不宣称已逐函数核对 0.22.1。
- 量化矩阵：旧项目 `docs/W6/data/results_fp16/` 中 72 个结果组、`docs/W6/data/results_awq/` 中 23 个结果组；每组 30 请求，共 2850 条成功记录。公开仓库仅保留汇总和来源文件指纹。
- 量化探索：旧项目 `docs/W3/quantization-comparison.md` 与 `docs/W6/report.md`。早期非流式 benchmark 将 E2E 错标为 TTFT，并用 E2E/output_tokens 计算所谓 TPOT，因此这些字段和由其导出的加速比不进入新仓库结论。
- 量化过程：历史记录保存了 GPTQ/AWQ 参数化脚本、量化完成输出、GGUF 转换日志与 llama-bench 汇总输出。llama-bench 记录 CUDA backend、GPU 型号、VRAM、build 和 `ngl=99`，但未保留逐层 offload 日志。当前仓库将其整理为参考脚本和 `quantization-artifacts.csv`；未迁移重复的安装日志、完整控制台输出和个人绝对路径。
- 文件指纹见 [source-manifest.json](../results/source-manifest.json)。指纹用于识别作者保存的档案，不代替公开原始证据。

## 配置与结论边界

启动日志确认 Prefix Caching OFF、Chunked Prefill ON、max_seq_len=32768；历史分析记录 ON 时有效 token budget=2048。新脚本显式固定这些条件。历史模型目录后缀为 -lcm，运行时 quantization=compressed-tensors，使用 Marlin；缺少模型 revision/hash，不能保证替换任意同名 AWQ 权重后数值一致。

Chunked Prefill ON/OFF、KV 使用率 85%–90% 均为历史补充观察，未作为新的独立复现实验入口。E2 报告的 Prefill KV 列来自历史 summary 的 prefill_kv_avg，未保留在精简 CSV；它不是新测量值。

主实验脚本重建 workload 与采集过程，但尚未在 4090D 完整重跑。请求 token 数应以重跑实际值为准；更换 tokenizer/chat template/权重后不能机械比较历史数据。
