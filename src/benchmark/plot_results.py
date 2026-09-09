#!/usr/bin/env python3
"""Generate the compact E4 figures used by the report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/e4-summary.csv"))
    parser.add_argument("--native-summary", type=Path, default=Path("results/e1-e3-summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.summary.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with args.native_summary.open(encoding="utf-8-sig") as handle:
        native_rows = list(csv.DictReader(handle))

    concurrency = [int(row["concurrency"]) for row in rows]
    throughput = [float(row["request_window_output_tps"]) for row in rows]
    client_tpot = [float(row["client_tpot_p50_ms"]) for row in rows]
    native_tpot = [float(row["native_tpot_avg_ms"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    colors = {"Prefill": "#0ea5e9", "Queue": "#f97316", "TTFT": "#334155"}

    def metric_plot(selected, x_values, x_label, title, output_name):
        selected = list(selected)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for label, field in (
            ("Prefill", "prefill_time_avg_ms"),
            ("Queue", "queue_avg_ms"),
            ("TTFT", "native_ttft_avg_ms"),
        ):
            ax.plot(
                x_values,
                [float(row[field]) for row in selected],
                marker="o",
                linewidth=2,
                label=label,
                color=colors[label],
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.set_xticks(x_values)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / output_name, dpi=180)
        plt.close(fig)

    e1 = sorted(
        (row for row in native_rows if row["experiment"] == "E1"),
        key=lambda row: int(row["concurrency"]),
    )
    metric_plot(
        e1,
        [int(row["concurrency"]) for row in e1],
        "Concurrency",
        "E1 concurrency scaling",
        "e1-concurrency.png",
    )

    e2 = sorted(
        (row for row in native_rows if row["experiment"] == "E2"),
        key=lambda row: float(row["prompt_tokens_mean"]),
    )
    metric_plot(
        e2,
        [int(float(row["prompt_tokens_mean"])) for row in e2],
        "Prompt tokens",
        "E2 prefill scaling",
        "e2-prefill.png",
    )

    by_case = {row["case"]: row for row in native_rows}
    e3a = [
        by_case["e1-v2-c16"], by_case["e3a-v2-c16"],
        by_case["e1-v2-c32"], by_case["e3a-v2-c32"],
    ]
    positions = list(range(len(e3a)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for offset, (label, field) in enumerate((
        ("Prefill", "prefill_time_avg_ms"),
        ("Queue", "queue_avg_ms"),
        ("TTFT", "native_ttft_avg_ms"),
    )):
        ax.bar(
            [position + (offset - 1) * width for position in positions],
            [float(row[field]) for row in e3a],
            width=width,
            label=label,
            color=colors[label],
        )
    ax.set_xticks(positions, ["c16 seq32", "c16 seq8", "c32 seq32", "c32 seq8"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E3A scheduler admission capacity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "e3a-scheduler-capacity.png", dpi=180)
    plt.close(fig)

    e3b = [
        ("0.85", by_case["e1-v2-c32"]),
        ("0.40", by_case["e3b-v2-040-c32"]),
        ("0.35", by_case["e3b-v2-035-c32"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, field in (
        ("Prefill", "prefill_time_avg_ms"),
        ("Queue", "queue_avg_ms"),
        ("TTFT", "native_ttft_avg_ms"),
    ):
        ax.plot(
            [item[0] for item in e3b],
            [float(item[1][field]) for item in e3b],
            marker="o",
            linewidth=2,
            label=label,
            color=colors[label],
        )
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E3B KV capacity control at concurrency 32")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "e3b-kv-capacity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(concurrency, throughput, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output throughput (token/s)")
    ax.set_title("E4 request-window output throughput")
    ax.set_xticks(concurrency)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "e4-throughput.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(concurrency, client_tpot, marker="o", linewidth=2, label="Client TPOT P50")
    ax.plot(concurrency, native_tpot, marker="s", linewidth=2, label="vLLM native TPOT")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("E4 per-request token latency")
    ax.set_xticks(concurrency)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "e4-tpot.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
