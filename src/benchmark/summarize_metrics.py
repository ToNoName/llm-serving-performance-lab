#!/usr/bin/env python3
"""Summarize before/after vLLM Prometheus snapshots for E1-E3."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


VALUE_RE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+eE.0-9]+)$")


def read_metrics(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VALUE_RE.match(line.strip())
        if not match:
            continue
        value = float(match.group("value"))
        if math.isfinite(value):
            name = match.group("name")
            totals[name] = totals.get(name, 0.0) + value
    return totals


def delta(before: dict[str, float], after: dict[str, float], metric: str):
    if metric not in before or metric not in after:
        return None
    return after[metric] - before[metric]


def average_delta(before: dict[str, float], after: dict[str, float], metric: str):
    total = delta(before, after, f"{metric}_sum")
    count = delta(before, after, f"{metric}_count")
    if total is None or count is None or count <= 0:
        return None
    return total / count


def milliseconds(value):
    return "" if value is None else f"{1000 * value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = read_metrics(args.before)
    after = read_metrics(args.after)
    with args.requests.open(encoding="utf-8") as handle:
        request_rows = list(csv.DictReader(handle))
    successful = [row for row in request_rows if row.get("status") == "success"]
    preemptions = delta(before, after, "vllm:num_preemptions_total")
    row = {
        "experiment": args.experiment,
        "case": args.case,
        "successful_requests": len(successful),
        "total_requests": len(request_rows),
        "native_ttft_avg_ms": milliseconds(
            average_delta(before, after, "vllm:time_to_first_token_seconds")
        ),
        "native_queue_avg_ms": milliseconds(
            average_delta(before, after, "vllm:request_queue_time_seconds")
        ),
        "native_prefill_avg_ms": milliseconds(
            average_delta(before, after, "vllm:request_prefill_time_seconds")
        ),
        "native_tpot_avg_ms": milliseconds(
            average_delta(before, after, "vllm:request_time_per_output_token_seconds")
        ),
        "native_itl_avg_ms": milliseconds(
            average_delta(before, after, "vllm:inter_token_latency_seconds")
        ),
        "native_decode_avg_ms": milliseconds(
            average_delta(before, after, "vllm:request_decode_time_seconds")
        ),
        "generation_tokens_delta": (
            ""
            if delta(before, after, "vllm:generation_tokens_total") is None
            else f"{delta(before, after, 'vllm:generation_tokens_total'):.0f}"
        ),
        "preemptions_delta": "" if preemptions is None else f"{preemptions:.0f}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
