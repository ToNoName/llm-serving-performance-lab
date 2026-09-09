# 实验复现指南

## 环境

E1–E4 原始结果来自 AutoDL RTX 4090D、vLLM 0.22.1 和 Qwen2.5-7B-Instruct AWQ Int4。本地 RTX 5060 用于 Gateway 与 llama.cpp 开发，不作为 E1–E4 的等价复现环境。

需要 Python 3.10+ 及其 `venv` 支持。在 Ubuntu/WSL 中如果 `python -m venv` 提示缺少 `ensurepip`，先安装发行版提供的 `python3-venv` 包。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/gateway.txt
pip install -r requirements/benchmark.txt

export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct-AWQ-Int4
export TOKENIZER_PATH="$MODEL_PATH"
```

## 启动 vLLM

基线、E1、E2 和 E4：

```bash
./scripts/start_vllm.sh baseline
```

E3A：

```bash
./scripts/start_vllm.sh e3a-seq8
```

E3B：

```bash
./scripts/start_vllm.sh e3b-mem040
./scripts/start_vllm.sh e3b-mem035
```

在独立的 GPU 环境中安装 `vllm==0.22.1`（例如 `pip install vllm==0.22.1`），并确认模型权重及 tokenizer 与历史实验一致。Gateway/benchmark requirements 不安装推理引擎。

启动脚本显式关闭 Prefix Caching、开启 Chunked Prefill，并设置 `max_num_batched_tokens=2048`。该预算依据历史分析记录；原始实验使用版本默认值，重跑时还需核对启动日志的最终配置。新脚本尚未在 4090D 完整重跑。

每次改变服务配置前先停止旧的 vLLM 进程。脚本打印 vLLM 版本、GPU 信息和完整启动命令。

## 运行实验

E1 和 E2 使用 baseline profile：

```bash
./scripts/run_e1_concurrency.sh
./scripts/run_e2_prefill.sh
```

E3A 使用 `e3a-seq8` profile：

```bash
./scripts/run_e3a_scheduler.sh
```

E3B 分别启动对应 profile：

```bash
./scripts/start_vllm.sh e3b-mem040
# 新终端
./scripts/run_e3b_kv_capacity.sh mem040
```

将 profile 和参数替换为 `e3b-mem035`、`mem035` 后运行第二组。

E4 使用 baseline profile：

```bash
./scripts/run_e4_decode.sh
```

默认原始输出写入 `results/raw/`，该目录被 Git 忽略。E1–E3 每个 case 在请求前后抓取 `/metrics`，再通过 `summarize_metrics.py` 计算 Native metric 差分。

## 单独运行 benchmark

非流式 workload：

```bash
python -m src.benchmark.benchmark_native \
  --experiment E1 \
  --tokenizer "$TOKENIZER_PATH" \
  --concurrency 1,4,8,16,32 \
  --input-tokens 1024 \
  --max-tokens 256 \
  --num-requests 60 \
  --output results/raw/e1-client.csv
```

Streaming workload：

```bash
python -m src.benchmark.benchmark_streaming \
  --tokenizer "$TOKENIZER_PATH" \
  --concurrency 32 \
  --prompt-tokens 128 \
  --max-tokens 512 \
  --num-requests 40 \
  --output results/raw/e4-c32.csv
```

## Gateway

```bash
cp .env.example .env
./scripts/start_gateway.sh
curl http://localhost:8000/health
curl http://localhost:8000/admin/backends
```

E1–E4 默认直连 vLLM，以减少 Gateway connection pool 对 workload 的影响。

## 参考 Docker Compose

`deployment/docker-compose.reference.yml` 用于验证 Nginx、Gateway、推理后端和监控的参考拓扑。它不是无需修改即可部署到任意机器的生产配置。

先复制环境文件并修改以下关键项：

```bash
cp .env.example .env
# 编辑 .env：至少设置 MODEL_DIR、模型容器路径和 Grafana 密码
```

`VLLM_BACKEND_URL`/`LLAMA_BACKEND_URL` 用于直接在宿主机运行 Gateway；Compose 使用独立的 `COMPOSE_VLLM_BACKEND_URL`/`COMPOSE_LLAMA_BACKEND_URL`，默认指向服务名。容器中的 `localhost` 不能用于访问另一个容器。

`MODEL_DIR` 是宿主机上的模型目录；`VLLM_MODEL_PATH` 和 `LLAMA_MODEL_PATH` 是该目录挂载到容器 `/models` 后的路径。单后端部署时，`BACKEND_MODE` 必须与启用的 profile 一致。

vLLM 单后端：

```bash
# .env 中设置 BACKEND_MODE=vllm
docker compose -f deployment/docker-compose.reference.yml --profile vllm config -q
docker compose -f deployment/docker-compose.reference.yml --profile vllm up -d --build
```

llama.cpp 单后端：

```bash
# .env 中设置 BACKEND_MODE=llama
docker compose -f deployment/docker-compose.reference.yml --profile llama config -q
docker compose -f deployment/docker-compose.reference.yml --profile llama up -d --build
```

只有两个后端都启动时才使用 `BACKEND_MODE=auto`：

```bash
docker compose -f deployment/docker-compose.reference.yml \
  --profile vllm --profile llama up -d --build
```

启动后检查：

```bash
set -a
source .env
set +a
docker compose -f deployment/docker-compose.reference.yml ps
curl http://localhost:${GATEWAY_PORT:-8000}/health
curl http://localhost:${GATEWAY_PORT:-8000}/admin/backends
curl http://localhost:${PROMETHEUS_PORT:-9090}/-/ready
curl http://localhost:${GRAFANA_PORT:-3000}/api/health
```

停止参考栈：

```bash
docker compose -f deployment/docker-compose.reference.yml \
  --profile vllm --profile llama down
```
