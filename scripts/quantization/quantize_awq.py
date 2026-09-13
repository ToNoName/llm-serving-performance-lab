#!/usr/bin/env python3
"""Create an AWQ W4A16 compressed-tensors artifact with llm-compressor."""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    dataset = load_dataset(
        args.dataset, split=f"train_sft[:{args.num_samples}]"
    ).shuffle(seed=args.seed)
    dataset = dataset.map(
        lambda row: {
            "text": tokenizer.apply_chat_template(row["messages"], tokenize=False)
        }
    )
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]
        ),
    ]
    started = time.monotonic()
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_length,
        num_calibration_samples=args.num_samples,
    )
    elapsed = time.monotonic() - started

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "source_model": args.model,
        "method": "AWQ",
        "scheme": "W4A16_ASYM",
        "dataset": args.dataset,
        "num_calibration_samples": args.num_samples,
        "max_sequence_length": args.max_length,
        "seed": args.seed,
        "quantization_time_seconds": round(elapsed, 2),
    }
    (args.output / "experiment-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
