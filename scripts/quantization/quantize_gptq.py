#!/usr/bin/env python3
"""Create a GPTQ-Int4 artifact from a Hugging Face causal language model.

This records the historical recipe. AutoGPTQ compatibility is sensitive to the
PyTorch, Transformers, CUDA, and compiled extension versions.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


CALIBRATION_TEMPLATES = [
    "The development of artificial intelligence has significantly impacted various industries, including",
    "Machine learning algorithms are used in natural language processing to understand and generate",
    "In recent years, deep learning models have achieved remarkable progress in the field of",
    "The transformer architecture revolutionized sequential data processing by introducing",
    "Efficient inference of neural networks requires optimization techniques such as",
    "Quantization reduces the memory footprint of large language models by",
    "Natural language understanding requires models to comprehend context, nuance, and",
    "The key innovation behind attention mechanisms is their ability to focus on relevant",
    "Large language models are trained on vast amounts of text data to learn patterns and",
    "Fine-tuning pretrained models on domain-specific data can significantly improve",
    "深度学习在计算机视觉领域取得了突破性进展，特别是",
    "大语言模型的出现改变了自然语言处理的研究范式，使得",
    "量化技术是压缩神经网络模型的重要手段，它可以",
    "Transformer 架构的核心创新在于自注意力机制，该机制能够",
    "预训练模型的领域微调已经成为提高任务效果的重要方法，因为",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    for index in range(args.num_samples):
        prefix = CALIBRATION_TEMPLATES[index % len(CALIBRATION_TEMPLATES)]
        encoded = tokenizer(
            f"{prefix} Example {index}.",
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        examples.append(
            {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
            }
        )

    config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        damp_percent=0.01,
        sym=True,
    )
    started = time.monotonic()
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.quantize(examples)
    elapsed = time.monotonic() - started

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "source_model": args.model,
        "method": "GPTQ",
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
        "sym": True,
        "damp_percent": 0.01,
        "num_calibration_samples": args.num_samples,
        "max_sequence_length": args.max_length,
        "quantization_time_seconds": round(elapsed, 2),
    }
    (args.output / "experiment-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
