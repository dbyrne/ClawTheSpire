"""Fine-tune Qwen3-8B on STS2 advisor decisions using QLoRA via Unsloth.

Prerequisites:
    pip install unsloth datasets trl

Usage:
    python -m sts2_solver.finetune --data logs/training_data.jsonl
    python -m sts2_solver.finetune --data logs/training_data.jsonl --epochs 3 --export gguf

After training, import into Ollama:
    ollama create sts2-advisor -f Modelfile
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "models" / "sts2-advisor"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "logs" / "training_data.jsonl"


def load_dataset(path: Path, min_score: float = 0.0, require_full_prompt: bool = False):
    """Load training data from JSONL, filtering by quality."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            meta = data.get("_meta", {})

            if meta.get("run_score", 0) < min_score:
                continue
            if require_full_prompt and not meta.get("has_full_prompt"):
                continue

            # Extract just the messages for training
            examples.append({"messages": data["messages"]})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B for STS2 advisor")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help=f"Training data JSONL (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--base-model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base model (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum run quality score for training data (default: 0.0)",
    )
    parser.add_argument(
        "--require-full-prompt", action="store_true",
        help="Only use examples with full user prompts (not placeholders)",
    )
    parser.add_argument(
        "--export", choices=["gguf", "merged", "lora", "none"], default="gguf",
        help="Export format after training (default: gguf)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.data}...")
    examples = load_dataset(args.data, args.min_score, args.require_full_prompt)
    print(f"Training examples: {len(examples)}")

    if len(examples) < 10:
        print("WARNING: Very few training examples. Consider collecting more data first.")
        print("Run: python -m sts2_solver.batch_runner --games 50")
        if len(examples) == 0:
            return

    # Import Unsloth (deferred to avoid import errors if not installed)
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install unsloth datasets trl")
        return

    # Load model with QLoRA
    print(f"\nLoading {args.base_model} with QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Prepare dataset
    def format_example(example):
        """Format messages into the chat template."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_example)
    print(f"Dataset size: {len(dataset)} examples")

    # Train
    args.output.mkdir(parents=True, exist_ok=True)

    training_config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        max_seq_length=2048,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()

    # Export
    if args.export == "gguf":
        print("\nExporting to GGUF (Q4_K_M)...")
        gguf_dir = args.output / "gguf"
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"GGUF saved to {gguf_dir}")

        # Write Ollama Modelfile
        modelfile_path = gguf_dir / "Modelfile"
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            modelfile_path.write_text(
                f"FROM {gguf_files[0].name}\n"
                f"PARAMETER temperature 0.3\n"
                f"PARAMETER num_ctx 2048\n"
            )
            print(f"\nTo import into Ollama:")
            print(f"  cd {gguf_dir}")
            print(f"  ollama create sts2-advisor -f Modelfile")
            print(f"\nThen run with:")
            print(f"  python -m sts2_solver.runner --local --model sts2-advisor")

    elif args.export == "merged":
        print("\nSaving merged model...")
        model.save_pretrained_merged(str(args.output / "merged"), tokenizer)
        print(f"Merged model saved to {args.output / 'merged'}")

    elif args.export == "lora":
        print("\nSaving LoRA adapter...")
        model.save_pretrained(str(args.output / "lora"))
        tokenizer.save_pretrained(str(args.output / "lora"))
        print(f"LoRA saved to {args.output / 'lora'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
