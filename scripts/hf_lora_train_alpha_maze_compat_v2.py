"""Compatibility LoRA/SFT trainer for AlphaMaze course-maze data.

V2 fixes TRL 0.12 collation by allowing unused source columns to be removed.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for Menlo/AlphaMaze-v0.2-1.5B on course-rule mazes.")
    parser.add_argument("--model", default="Menlo/AlphaMaze-v0.2-1.5B")
    parser.add_argument("--data", default="train/hf_sft_messages.jsonl")
    parser.add_argument("--out", default="artifacts/hf_lora/alphamaze-course-lora-main")
    parser.add_argument("--max-samples", type=int, default=3986)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=1536)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"offline={offline} model={args.model}", flush=True)

    ds = load_dataset("json", data_files=str(Path(args.data).resolve()))["train"]
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    print(f"train samples={len(ds)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=offline,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    def format_example(example):
        messages = example["messages"]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return f"User:\n{messages[0]['content']}\nAssistant:\n{messages[1]['content']}"

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    config_params = inspect.signature(SFTConfig.__init__).parameters
    train_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        packing=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=True,
    )
    if "max_length" in config_params:
        train_kwargs["max_length"] = args.seq_len
    elif "max_seq_length" in config_params:
        train_kwargs["max_seq_length"] = args.seq_len
    train_args = SFTConfig(**train_kwargs)

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=ds,
        peft_config=peft_config,
        formatting_func=format_example,
    )
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved LoRA adapter: {args.out}", flush=True)


if __name__ == "__main__":
    main()
