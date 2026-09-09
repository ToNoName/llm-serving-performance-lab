#!/usr/bin/env python3
"""Run the E4 streaming workload and measure client TTFT and TPOT."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


BASE_PROMPT = """这是一个大模型推理性能测试。
请连续输出简单的编号文本，例如：item_0001、item_0002、item_0003。
持续生成即可，不需要解释测试目的。"""


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, fix_mistral_regex=True
    )


def count_chat_tokens(tokenizer: Any, prompt: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def make_prompt(tokenizer: Any, target_tokens: int) -> str:
    prompt = BASE_PROMPT
    while count_chat_tokens(tokenizer, prompt) < target_tokens:
        prompt += " 背景信息。"
    return prompt


async def one_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: str,
    request_id: int,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_content_time = None
    last_content_time = None
    text_parts: list[str] = []
    prompt_tokens = None
    output_tokens = None
    status, error = "success", ""

    try:
        async with client.stream("POST", args.url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                choices = event.get("choices") or []
                content = (choices[0].get("delta") or {}).get("content") if choices else None
                if content:
                    now = time.perf_counter()
                    first_content_time = first_content_time or now
                    last_content_time = now
                    text_parts.append(content)
    except Exception as exc:
        status, error = "error", repr(exc)

    ended = time.perf_counter()
    text = "".join(text_parts)
    if output_tokens is None and text:
        output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if prompt_tokens is None:
        prompt_tokens = count_chat_tokens(tokenizer, prompt)

    ttft_ms = (
        (first_content_time - started) * 1000 if first_content_time is not None else None
    )
    tpot_ms = None
    if first_content_time is not None and last_content_time is not None and output_tokens and output_tokens > 1:
        tpot_ms = (last_content_time - first_content_time) * 1000 / (output_tokens - 1)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status": status,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_latency_ms": (ended - started) * 1000,
    }


async def run(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.tokenizer)
    prompt = make_prompt(tokenizer, args.prompt_tokens)
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=max(args.concurrency + 8, 64),
        max_keepalive_connections=max(args.concurrency + 8, 64),
    )
    timeout = httpx.Timeout(connect=30, read=None, write=30, pool=None)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def guarded(index: int):
            async with semaphore:
                return await one_request(client, args, tokenizer, prompt, index)

        started = time.perf_counter()
        rows = await asyncio.gather(*(guarded(i) for i in range(args.num_requests)))
        request_window_seconds = time.perf_counter() - started

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "request_id", "status", "error", "prompt_tokens",
              "output_tokens", "ttft_ms", "tpot_ms", "e2e_latency_ms"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "success"]
    total_output_tokens = sum(int(row["output_tokens"] or 0) for row in successful)
    tpot_values = [float(row["tpot_ms"]) for row in successful if row["tpot_ms"]]
    summary = {
        "experiment": "E4",
        "concurrency": args.concurrency,
        "requests": len(rows),
        "successful_requests": len(successful),
        "prompt_tokens_target": args.prompt_tokens,
        "max_output_tokens": args.max_tokens,
        "total_output_tokens": total_output_tokens,
        "request_window_seconds": request_window_seconds,
        "request_window_output_tps": total_output_tokens / request_window_seconds,
        "client_tpot_p50_ms": statistics.median(tpot_values) if tpot_values else None,
    }
    summary_path = Path(args.summary_output or f"{output}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
