# gateway.py V2
# 双后端网关：input_tokens路由 + 超时/错误码/计时/日志

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

load_dotenv()

# ---------- 配置 ----------

VLLM_BACKEND_URL = os.getenv(
    "VLLM_BACKEND_URL",
    "http://host.docker.internal:8001/v1/chat/completions",
)
LLAMA_BACKEND_URL = os.getenv(
    "LLAMA_BACKEND_URL",
    "http://host.docker.internal:8002/v1/chat/completions",
)

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", "512"))
VLLM_SERVED_MODEL = os.getenv("VLLM_SERVED_MODEL", "qwen")
LLAMA_SERVED_MODEL = os.getenv("LLAMA_SERVED_MODEL", "qwen2.5-7b-instruct-Q4_K_M")

HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=50)


# ---------- Metrics ----------

REQUEST_COUNT = Counter("gateway_requests_total", "Total chat completion requests")
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "E2E latency of chat completions",
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64),
)
BACKEND_SELECTION = Counter(
    "gateway_backend_selection_total",
    "Requests routed per backend",
    ["backend"],
)
ERROR_COUNT = Counter(
    "gateway_errors_total",
    "Errors by type",
    ["error_type"],
)


# ---------- 路由模式 ----------

class BackendMode(str, Enum):
    AUTO = "auto"
    VLLM = "vllm"
    LLAMA = "llama"


current_mode = BackendMode(os.getenv("BACKEND_MODE", "auto"))


# ---------- 请求模型 ----------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen", description="模型名称")
    messages: list[ChatMessage] = Field(..., description="消息列表")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=256, ge=2, le=4096)
    stream: Optional[bool] = Field(default=False)

class SwitchBackendRequest(BaseModel):
    mode: BackendMode


# ---------- URL工具 ----------

def _health_url(full_url: str) -> str:
    """从 chat completions URL 派生健康检查 URL。
    http://host:8001/v1/chat/completions → http://host:8001/health
    http://host:8002/v1/chat/completions → http://host:8002/health
    vLLM 和 llama.cpp 均支持 /health 端点。
    """
    if "/v1/chat/completions" in full_url:
        return full_url.rsplit("/v1/chat/completions", 1)[0] + "/health"
    return full_url.rstrip("/") + "/health"


# ---------- 日志 ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gateway")


def log_request(
    trace_id: str,
    backend: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    stream: bool,
    status_code: int,
    first_chunk_ms: Optional[float],
    stream_chunk_interval_ms: Optional[float],
    e2e_latency_ms: float,
    backend_latency_ms: float,
    error_type: str = "",
):
    gateway_overhead_ms = round(e2e_latency_ms - backend_latency_ms, 2)
    logger.info(
        json.dumps(
            {
                "trace_id": trace_id,
                "backend": backend,
                "model": model,
                "input_tokens": input_tokens,
                "output_count": output_tokens,
                "output_count_unit": "chunks" if stream else "tokens",
                "stream": stream,
                "status_code": status_code,
                "error_type": error_type,
                "first_chunk_ms": (
                    round(first_chunk_ms, 2) if first_chunk_ms is not None else None
                ),
                "stream_chunk_interval_ms": (
                    round(stream_chunk_interval_ms, 2)
                    if stream_chunk_interval_ms is not None
                    else None
                ),
                "e2e_latency_ms": round(e2e_latency_ms, 2),
                "backend_latency_ms": round(backend_latency_ms, 2),
                "gateway_overhead_ms": gateway_overhead_ms,
            },
            ensure_ascii=False,
        )
    )


# ---------- Token估算 ----------

def estimate_tokens(messages: list[ChatMessage]) -> int:
    text = " ".join(msg.content for msg in messages)
    if not text:
        return 0
    cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    cn_ratio = cn_chars / max(len(text), 1)
    if cn_ratio > 0.3:
        return int(len(text) * 0.5)
    return int(len(text) * 0.25)


# ---------- 路由选择 ----------

def select_backend(request: ChatCompletionRequest) -> tuple:
    """返回 (backend_url, backend_name)"""
    if current_mode == BackendMode.VLLM:
        return VLLM_BACKEND_URL, "vllm"
    if current_mode == BackendMode.LLAMA:
        return LLAMA_BACKEND_URL, "llama"
    token_estimate = estimate_tokens(request.messages)
    if token_estimate > TOKEN_THRESHOLD:
        return VLLM_BACKEND_URL, "vllm"
    return LLAMA_BACKEND_URL, "llama"


# ---------- 后端模型名映射 ----------

VLLM_MODEL_MAP = {
    "qwen": VLLM_SERVED_MODEL,
    "qwen2.5-7b": VLLM_SERVED_MODEL,
}

LLAMA_MODEL_MAP = {
    "qwen": LLAMA_SERVED_MODEL,
    "qwen2.5-7b": LLAMA_SERVED_MODEL,
}


def resolve_model_name(request_model: str, backend_name: str) -> str:
    model_lower = request_model.lower().strip()
    if backend_name == "vllm":
        return VLLM_MODEL_MAP.get(model_lower, request_model)
    return LLAMA_MODEL_MAP.get(model_lower, request_model)


# ---------- FastAPI ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        limits=HTTPX_LIMITS,
    )
    logger.info(
        "Gateway started | vllm=%s | llama=%s | threshold=%d",
        VLLM_BACKEND_URL,
        LLAMA_BACKEND_URL,
        TOKEN_THRESHOLD,
    )
    yield
    await app.state.http_client.aclose()
    logger.info("Gateway shutdown")


app = FastAPI(title="Inference Gateway V2", lifespan=lifespan)


# ---------- 健康检查 ----------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "inference-gateway"}


# ---------- Prometheus Metrics ----------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------- 后端状态 ----------

@app.get("/admin/backends")
async def admin_backends():
    results = {}
    for name, url in [("vllm", VLLM_BACKEND_URL), ("llama", LLAMA_BACKEND_URL)]:
        try:
            resp = await app.state.http_client.get(
                _health_url(url), timeout=5.0
            )
            results[name] = {"url": url, "healthy": resp.status_code == 200}
        except Exception:
            results[name] = {"url": url, "healthy": False}
    return {"mode": current_mode.value, "backends": results}


# ---------- 手动切换 ----------

@app.post("/admin/switch-backend")
async def admin_switch_backend(req: SwitchBackendRequest):
    global current_mode
    current_mode = req.mode
    logger.info("Backend mode switched to %s", current_mode.value)
    return {"mode": current_mode.value}


# ---------- 核心路由 ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    trace_id = str(uuid.uuid4())[:8]
    e2e_start = time.monotonic()
    backend_url, backend_name = select_backend(request)

    REQUEST_COUNT.inc()
    BACKEND_SELECTION.labels(backend_name).inc()

    payload = request.model_dump()
    payload["model"] = resolve_model_name(request.model, backend_name)
    stream = payload.get("stream", False)
    input_token_estimate = estimate_tokens(request.messages)

    logger.info(
        "[%s] route=%s model=%s est_tokens=%d stream=%s",
        trace_id,
        backend_name,
        payload["model"],
        input_token_estimate,
        stream,
    )

    client: httpx.AsyncClient = app.state.http_client
    target_url = backend_url

    # ========== 非流式 ==========
    if not stream:
        try:
            backend_start = time.monotonic()
            resp = await client.post(target_url, json=payload)
            backend_latency_ms = (time.monotonic() - backend_start) * 1000
            e2e_latency_ms = (time.monotonic() - e2e_start) * 1000

            if resp.status_code != 200:
                error_type = "backend_error"
                log_request(
                    trace_id, backend_name, payload["model"],
                    input_token_estimate, 0, False,
                    resp.status_code, 0, 0,
                    e2e_latency_ms, backend_latency_ms,
                    error_type=error_type,
                )
                if resp.status_code >= 500:
                    ERROR_COUNT.labels("backend_unavailable").inc()
                    raise HTTPException(
                        status_code=502,
                        detail={"error": "backend_unavailable", "trace_id": trace_id},
                    )
                raise HTTPException(status_code=resp.status_code, detail=resp.json())

            resp_data = resp.json()
            output_tokens = (
                resp_data.get("usage", {}).get("completion_tokens", 0)
                if isinstance(resp_data, dict)
                else 0
            )

            log_request(
                trace_id, backend_name, payload["model"],
                input_token_estimate, output_tokens, False,
                200, None, None,
                e2e_latency_ms, backend_latency_ms,
            )
            REQUEST_LATENCY.observe(e2e_latency_ms / 1000)
            return resp_data

        except httpx.TimeoutException:
            e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
            ERROR_COUNT.labels("backend_timeout").inc()
            log_request(
                trace_id, backend_name, payload["model"],
                input_token_estimate, 0, False,
                504, 0, 0, e2e_latency_ms, 0,
                error_type="backend_timeout",
            )
            raise HTTPException(
                status_code=504,
                detail={"error": "backend_timeout", "trace_id": trace_id},
            )
        except httpx.ConnectError:
            e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
            ERROR_COUNT.labels("backend_unavailable").inc()
            log_request(
                trace_id, backend_name, payload["model"],
                input_token_estimate, 0, False,
                502, 0, 0, e2e_latency_ms, 0,
                error_type="backend_unavailable",
            )
            raise HTTPException(
                status_code=502,
                detail={"error": "backend_unavailable", "trace_id": trace_id},
            )
        except HTTPException:
            raise
        except Exception as e:
            e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
            ERROR_COUNT.labels("internal_error").inc()
            log_request(
                trace_id, backend_name, payload["model"],
                input_token_estimate, 0, False,
                500, 0, 0, e2e_latency_ms, 0,
                error_type="internal_error",
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "internal_error", "trace_id": trace_id, "details": str(e)},
            )

    # ========== 流式 ==========
    else:

        async def stream_generator():
            backend_start = time.monotonic()
            first_chunk_time = None
            chunk_count = 0

            try:
                async with client.stream("POST", target_url, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
                        log_request(
                            trace_id, backend_name, payload["model"],
                            input_token_estimate, 0, True,
                            resp.status_code, 0, 0,
                            e2e_latency_ms, 0, error_type="backend_error",
                        )
                        yield f"data: {error_body.decode()}\n\n"
                        return

                    async for chunk in resp.aiter_bytes():
                        # HTTP byte chunks are transport units, not model tokens.
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic()
                        chunk_count += 1
                        yield chunk

                        if await req.is_disconnected():
                            logger.info(
                                "[%s] client disconnected after %d chunks",
                                trace_id, chunk_count,
                            )
                            return

            except httpx.TimeoutException:
                e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
                ERROR_COUNT.labels("backend_timeout").inc()
                log_request(
                    trace_id, backend_name, payload["model"],
                    input_token_estimate, 0, True,
                    504, 0, 0, e2e_latency_ms, 0, error_type="backend_timeout",
                )
                error_data = json.dumps({"error": "backend_timeout", "trace_id": trace_id})
                yield f"data: {error_data}\n\n"
                return
            except httpx.ConnectError:
                e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
                ERROR_COUNT.labels("backend_unavailable").inc()
                log_request(
                    trace_id, backend_name, payload["model"],
                    input_token_estimate, 0, True,
                    502, 0, 0, e2e_latency_ms, 0, error_type="backend_unavailable",
                )
                error_data = json.dumps({"error": "backend_unavailable", "trace_id": trace_id})
                yield f"data: {error_data}\n\n"
                return
            except Exception as e:
                e2e_latency_ms = (time.monotonic() - e2e_start) * 1000
                ERROR_COUNT.labels("internal_error").inc()
                log_request(
                    trace_id, backend_name, payload["model"],
                    input_token_estimate, 0, True,
                    500, 0, 0, e2e_latency_ms, 0, error_type="internal_error",
                )
                return

            backend_latency_ms = (time.monotonic() - backend_start) * 1000
            e2e_latency_ms = (time.monotonic() - e2e_start) * 1000

            first_chunk_ms = (
                (first_chunk_time - backend_start) * 1000 if first_chunk_time else None
            )
            output_tokens_est = chunk_count
            stream_chunk_interval_ms = (
                (backend_latency_ms - first_chunk_ms) / (chunk_count - 1)
                if first_chunk_ms is not None and chunk_count > 1
                else None
            )

            log_request(
                trace_id, backend_name, payload["model"],
                input_token_estimate, output_tokens_est, True,
                200, first_chunk_ms, stream_chunk_interval_ms,
                e2e_latency_ms, backend_latency_ms,
            )
            REQUEST_LATENCY.observe(e2e_latency_ms / 1000)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
