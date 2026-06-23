from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class SFTJsonl(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1536, max_samples: int = 0):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                user = row["messages"][0]["content"]
                target = row["messages"][1]["content"].strip()
                self.rows.append((user, target))
                if max_samples and len(self.rows) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        user, target = self.rows[idx]
        prompt = f"User:\n{user}\nAssistant:\n"
        answer = target + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= self.max_length:
            answer_ids = answer_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
        max_prompt = max(1, self.max_length - len(answer_ids))
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if not any(x != -100 for x in labels):
            raise ValueError("empty supervised labels after truncation")
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual LoRA SFT for AlphaMaze on AGRL 3x3-memory decisions.")
    parser.add_argument("--model", default="Menlo/AlphaMaze-v0.2-1.5B")
    parser.add_argument("--data", default="artifacts/agrl_large/hf_sft_messages.jsonl")
    parser.add_argument("--out", default="artifacts/agrl_large/alphamaze_agrl_lora")
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda={torch.cuda.is_available()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"model_dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=offline,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    ds = SFTJsonl(args.data, tokenizer, args.max_length, args.max_samples)
    print(f"records={len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step = 0
    accum_loss = 0.0
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            raw_loss = out.loss
            valid_labels = int((batch["labels"] != -100).sum().item())
            if valid_labels <= 0 or not torch.isfinite(raw_loss):
                print(f"non_finite_or_empty_loss step={step} loss={raw_loss.detach().float().cpu().item()} valid_labels={valid_labels}", flush=True)
                raise RuntimeError("non-finite loss or empty labels")
            loss = raw_loss / args.grad_accum
            loss.backward()
            accum_loss += float(loss.detach().cpu()) * args.grad_accum
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step={step}/{args.max_steps} loss={accum_loss/10:.4f}", flush=True)
                accum_loss = 0.0
            if step >= args.max_steps:
                break
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = {"steps": step, "records": len(ds), "model": args.model, "device": str(device)}
    Path(args.out, "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    main()
