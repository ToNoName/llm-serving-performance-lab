# vLLM Serving Performance Analysis

## 实验目标

E1–E4 使用控制变量方法分析单机 vLLM Serving 中四类问题：并发增加后的 Queue 变化、Prompt Length 对 Prefill 的影响、Scheduler admission capacity 与 KV capacity 的区别，以及 Decode concurrency 下的吞吐—延迟关系。

## 实验环境

```text
GPU: NVIDIA RTX 4090D（AutoDL）
Model: Qwen2.5-7B-Instruct AWQ Int4（历史目录后缀 -lcm）
Runtime quantization: compressed-tensors / Marlin
vLLM: 0.22.1
gpu_memory_utilization: 0.85（E3B 除外）
max_num_seqs: 32（E3A 除外）
Prefix Caching: OFF
Chunked Prefill: ON
max_model_len: 32768
```

关闭 Prefix Caching 是为了让请求的 Prompt Token 实际进入 Prefill。E1–E3 使用非流式请求，客户端 CSV 不提供真实 TTFT；核心指标来自实验窗口前后 vLLM Native Metrics 的差分。

## 数据口径

```text
Native metric average = (after_sum - before_sum) / (after_count - before_count)
```

E1–E3 使用：

- `vllm:time_to_first_token_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:request_prefill_time_seconds`
- `vllm:request_prefill_kv_computed_tokens`
- `vllm:num_preemptions_total`

E4 使用 Streaming Client 记录首个正文事件与后续 token 间隔，并用 Native TPOT 和 ITL 交叉检查。

## E1：Concurrency Scaling

固定 prompt≈1028、max_tokens=256、`max_num_seqs=32`：

| Concurrency | Prefill avg | Queue avg | Native TTFT avg |
|---:|---:|---:|---:|
| 1 | 110.5 ms | 0.03 ms | 119.5 ms |
| 4 | 272.4 ms | 22.6 ms | 326.9 ms |
| 8 | 386.9 ms | 141.0 ms | 559.4 ms |
| 16 | 436.4 ms | 416.0 ms | 944.4 ms |
| 32 | 472.9 ms | 1076.6 ms | 1666.0 ms |

E1 表明并发上升同时增加 Prefill workload 与调度等待。它本身不能证明 Queue 全部由 `max_num_seqs` ceiling 导致，因此使用 E3A 进一步验证。

![E1 concurrency](../results/figures/e1-concurrency.png)

## E2：Prefill Scaling

固定 concurrency=4，仅改变 Prompt Length：

| Prompt tokens | Prefill KV | Prefill avg | Queue avg | Native TTFT avg |
|---:|---:|---:|---:|---:|
| 137 | 137 | 48.4 ms | 0.02 ms | 64.7 ms |
| 515 | 515 | 159.9 ms | 0.02 ms | 180.8 ms |
| 1028 | 1028 | 279.6 ms | 19.1 ms | 326.1 ms |
| 2054 | 2054 | 482.2 ms | 104.8 ms | 672.4 ms |

短 Prompt 阶段，TTFT 的增加主要来自 Prefill。Prompt 进一步变长后，请求占用执行资源更久，Queue 也开始增加，形成次生放大。

![E2 prefill](../results/figures/e2-prefill.png)

## E3A：Scheduler Admission Capacity

保持模型、workload、KV budget、Prefix Caching 与 Chunked Prefill 配置一致，将 `max_num_seqs` 从 32 降至 8。

| Concurrency | `max_num_seqs` | Prefill avg | Queue avg | Native TTFT avg |
|---:|---:|---:|---:|---:|
| 16 | 32 | 436 ms | 416 ms | 944 ms |
| 16 | 8 | 273 ms | 2081 ms | 2438 ms |
| 32 | 32 | 473 ms | 1077 ms | 1666 ms |
| 32 | 8 | 272 ms | 5431 ms | 5809 ms |

当请求压力超过当前 admission capacity 后，新增延迟主要进入 Queue。Prefill 没有同步恶化，因此不能把 TTFT 上升解释为 Prefill compute 变慢。

![E3A scheduler](../results/figures/e3a-scheduler-capacity.png)

### Chunked Prefill 补充观察

此节为历史补充分析，不纳入当前 E1–E4 主实验脚本的复现范围；来源见 [证据映射](EVIDENCE.md)。

在本实验的 vLLM 0.22.1 配置中，没有显式设置 `max_num_batched_tokens`：

```text
Chunked Prefill ON  -> effective max_num_batched_tokens = 2048
Chunked Prefill OFF -> effective max_num_batched_tokens = 32768
```

因此 ON/OFF 对照不是纯 Chunking Policy 单变量实验。effective token budget 改变后，Queue 与 Prefill 之间的 latency distribution 也发生变化。由于没有 scheduler-step trace，本文不对 Prefill 的微观差异作唯一机制归因。

## E3B：KV Capacity Control

固定 prompt≈1028、max_tokens=256、concurrency=32 和 `max_num_seqs=32`：

| `gpu_memory_utilization` | Prefill avg | Queue avg | Native TTFT avg |
|---:|---:|---:|---:|
| 0.85 | 472.9 ms | 1076.6 ms | 1666.0 ms |
| 0.40 | 473.5 ms | 1106.0 ms | 1668.1 ms |
| 0.35 | 467.6 ms | 1089.9 ms | 1674.1 ms |

虽然单独的容量验证运行观察到约 85%–90% KV Usage，但所有列入汇总的实验均为 `preemptions_delta=0`。显著削减 KV budget 也没有明显改变 TTFT。KV Usage 表示压力，不能单独证明 KV exhaustion。

![E3B KV capacity](../results/figures/e3b-kv-capacity.png)

## E4：Decode Scaling

固定 129-token Prompt、512-token 实际输出、`temperature=0`、`ignore_eos=true`，仅改变 concurrency：

| Concurrency | Client TTFT P50 | Client TPOT P50 | Native TPOT | Queue avg | Request-window TPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 33.37 ms | 5.80 ms | 5.80 ms | 0.02 ms | 170.8 |
| 4 | 72.18 ms | 5.83 ms | 5.85 ms | 0.02 ms | 669.7 |
| 8 | 119.66 ms | 5.99 ms | 6.02 ms | 0.02 ms | 1282.5 |
| 16 | 260.56 ms | 6.37 ms | 6.35 ms | 0.05 ms | 1989.0 |
| 32 | 351.34 ms | 7.31 ms | 7.19 ms | 4.69 ms | 2741.8 |

TPOT 随并发升高而恶化，但系统在同一时间处理更多 Decode sequences，因此输出吞吐继续增长。Client 与 Native TPOT 变化方向和数值接近。

请求窗口从 benchmark 开始等待请求到全部请求完成。外层实验时间还包含脚本初始化、指标抓取与收尾，不作为主吞吐分母。

![E4 throughput](../results/figures/e4-throughput.png)
![E4 TPOT](../results/figures/e4-tpot.png)

## 诊断结论

```text
TTFT 上升
├── Prefill 与 Prompt KV 同步上升，Queue 低：检查 Prefill compute
├── Queue 上升，Prefill 未同步恶化：检查 Scheduler/admission
└── KV Usage 上升且伴随 Preemption/分配失败：检查 KV capacity

TPOT 上升
└── 结合 output throughput 判断 Decode batching 的吞吐—延迟权衡
```

单个指标不足以直接定位瓶颈。Waiting 可能由 sequence capacity、token budget、KV block、调度策略或执行能力共同引起；高 KV Usage 也不等于 KV exhaustion。

## 结果文件

- [E1–E3 summary](../results/e1-e3-summary.csv)
- [E4 summary](../results/e4-summary.csv)
- [复现步骤](REPRODUCTION.md)
- [限制](LIMITATIONS.md)
