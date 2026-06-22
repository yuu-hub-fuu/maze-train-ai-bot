from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "ATTACK_BOSS", "STOP"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


class PromptDataset(Dataset):
    def __init__(self, rows, stoi, max_len=1536):
        self.rows = rows
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row["messages"][0]["content"]
        ids = [self.stoi.get(ch, self.stoi["<unk>"]) for ch in text[-self.max_len:]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ACTION_TO_ID[row["messages"][1]["content"]], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = max(lengths).item()
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    return padded, lengths, torch.stack(ys)


class MazeGRUPolicy(nn.Module):
    def __init__(self, vocab_size, num_actions=6, emb=96, hidden=192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, num_actions))

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        feat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(feat)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["messages"][1]["content"] in ACTION_TO_ID:
                    rows.append(row)
    return rows


def build_vocab(rows):
    chars = sorted({ch for row in rows for ch in row["messages"][0]["content"]})
    stoi = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        if ch not in stoi:
            stoi[ch] = len(stoi)
    return stoi


def accuracy(model, loader, device):
    model.eval()
    ok = total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss_sum += ce(logits, y).item() * y.numel()
            ok += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    return {"loss": loss_sum / max(1, total), "accuracy": ok / max(1, total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/train/hf_sft_messages.jsonl")
    ap.add_argument("--out", default="artifacts/torch_policy/maze_gru_policy.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.data)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.9))
    train_rows, val_rows = rows[:split], rows[split:]
    stoi = build_vocab(rows)
    train_ds = PromptDataset(train_rows, stoi)
    val_ds = PromptDataset(val_rows, stoi)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MazeGRUPolicy(len(stoi)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss()
    best = {"accuracy": -1.0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"device": str(device), "cuda": torch.cuda.is_available(), "rows": len(rows), "vocab": len(stoi)}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        val = accuracy(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total), **val}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val["accuracy"] > best["accuracy"]:
            best = val
            torch.save({"model": model.state_dict(), "stoi": stoi, "actions": ACTIONS, "metrics": best}, out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best": best, "rows": len(rows), "vocab": len(stoi)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
