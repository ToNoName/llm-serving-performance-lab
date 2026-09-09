#!/usr/bin/env python3
"""Run non-streaming workloads for E1-E3.

Client-side results intentionally do not claim TTFT or TPOT. Those metrics are
derived from vLLM Prometheus snapshots with ``summarize_metrics.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


BASE_UNIT = (
    "大语言模型推理系统需要综合考虑计算效率、显存管理、KV Cache、"
    "批处理、调度策略以及服务延迟。"
)


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, fix_mistral_regex=True
    )


def count_chat_tokens(tokenizer: Any, prompt: str) -> int:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)


def make_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Build a deterministic prompt close to the requested chat-token length."""
    repeat = 1
    while count_chat_tokens(tokenizer, BASE_UNIT * repeat) < target_tokens:
        repeat *= 2

    lo, hi = 0, repeat
    best = ""
    best_diff = abs(count_chat_tokens(tokenizer, best) - target_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = BASE_UNIT * mid
        actual = count_chat_tokens(tokenizer, candidate)
        diff = abs(actual - target_tokens)
        if diff < best_diff:
            best, best_diff = candidate, diff
        if actual < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1

    for filler in ("测试", "数据", "模型", "推理", "性能", "AI", "a"):
        while True:
            candidate = best + filler
            diff = abs(count_chat_tokens(tokenizer, candidate) - target_tokens)
            if diff >= best_diff:
                break
            best, best_diff = candidate, diff
    return best


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    request_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return {
                "request_id": request_id,
                "status": "http_error",
                "status_code": response.status_code,
                "e2e_latency_ms": elapsed_ms,
                "error": response.text[:500],
            }

        body = response.json()
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "request_id": request_id,
            "status": "success",
            "status_code": 200,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "e2e_latency_ms": elapsed_ms,
            "request_output_tps": (
                output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            ),
            "error": "",
        }
    except httpx.TimeoutException as exc:
        status, error = "timeout", repr(exc)
    except httpx.ConnectError as exc:
        status, error = "connection_error", repr(exc)
    except Exception as exc:
        status, error = "error", repr(exc)

    return {
        "request_id": request_id,
        "status": status,
        "status_code": "",
        "e2e_latency_ms": (time.perf_counter() - started) * 1000,
        "error": error,
    }


async def run_case(
    args: argparse.Namespace,
    tokenizer: Any,
    input_tokens: int,
    concurrency: int,
):
    prompt = make_prompt(tokenizer, input_tokens)
    local_prompt_tokens = count_chat_tokens(tokenizer, prompt)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True

    limits = httpx.Limits(
        max_connections=max(concurrency + 8, 32),
        max_keepalive_connections=max(concurrency + 8, 32),
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout), limits=limits
    ) as client:
        for index in range(args.warmup):
            await send_request(client, args.url, payload, -(index + 1))

        async def guarded(index: int):
            async with semaphore:
                return await send_request(client, args.url, payload, index)

        started = time.perf_counter()
        rows = await asyncio.gather(
            *(guarded(index) for index in range(args.num_requests))
        )
        request_window_seconds = time.perf_counter() - started

    run_id = f"{args.experiment}_{input_tokens}in_{args.max_tokens}out_c{concurrency}"
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row.update(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "experiment": args.experiment,
                "backend": args.backend,
                "model": args.model,
                "quantization": args.quantization,
                "target_input_tokens": input_tokens,
                "local_prompt_tokens": local_prompt_tokens,
                "max_output_tokens": args.max_tokens,
                "concurrency": concurrency,
                "request_window_seconds": request_window_seconds,
            }
        )
    return rows


FIELDS = [
    "run_id", "timestamp", "experiment", "backend", "model",
    "quantization", "target_input_tokens", "local_prompt_tokens",
    "prompt_tokens", "max_output_tokens", "output_tokens", "concurrency",
    "request_id", "status", "status_code", "e2e_latency_ms",
    "request_output_tps", "request_window_seconds", "error",
]


def print_case_summary(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row["status"] == "success"]
    latencies = [float(row["e2e_latency_ms"]) for row in successful]
    detail = f", e2e_p50_ms={statistics.median(latencies):.2f}" if latencies else ""
    print(f"{rows[0]['run_id']}: success={len(successful)}/{len(rows)}{detail}")


async def run(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.tokenizer)
    all_rows: list[dict[str, Any]] = []
    for input_tokens in args.input_tokens:
        for concurrency in args.concurrency:
            rows = await run_case(args, tokenizer, input_tokens, concurrency)
            all_rows.extend(rows)
            print_case_summary(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def comma_separated_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="E1, E2, E3A, or E3B")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path")
    parser.add_argument("--quantization", default="awq-int4")
    parser.add_argument("--concurrency", type=comma_separated_ints, default=[1])
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--input-tokens", type=comma_separated_ints, default=[1024])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
