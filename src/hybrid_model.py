"""
Hybrid model: frozen fine-tuned RoBERTa [CLS] embeddings + stylometric features
-> small MLP head.

Why frozen? Extracting embeddings once and training a tiny head on top is
1) much faster to iterate on, 2) less prone to overfitting on a 2-week timeline.
Document this trade-off in the report.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/roberta-fakenews"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
STYLO_DIM = 8
EMB_DIM = 768


@torch.no_grad()
def extract_embeddings(texts, tokenizer, model, max_len=256):
    model.eval()
    embs = []
    for i in range(0, len(texts), BATCH):
        batch = list(texts[i:i + BATCH])
        enc = tokenizer(
            batch, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt"
        ).to(DEVICE)
        out = model.roberta(**enc)          # base encoder of the fine-tuned model
        cls = out.last_hidden_state[:, 0]   # [CLS] token
        embs.append(cls.cpu().numpy())
    return np.vstack(embs)


class HybridHead(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, stylo_dim=STYLO_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + stylo_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, emb, stylo):
        return self.net(torch.cat([emb, stylo], dim=1))


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_preds, all_labels, total_loss = [], [], 0.0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for emb, stylo, y in loader:
            emb, stylo, y = emb.to(DEVICE), stylo.to(DEVICE), y.to(DEVICE)
            logits = model(emb, stylo)
            loss = loss_fn(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds),
        all_preds,
        all_labels,
    )


def make_loader(emb, stylo, labels, shuffle):
    ds = TensorDataset(
        torch.tensor(emb, dtype=torch.float32),
        torch.tensor(stylo, dtype=torch.float32),
        torch.tensor(labels.values, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=256, shuffle=shuffle)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    splits = {}
    for name in ["train", "val", "test", "cross_domain_test"]:
        df = pd.read_csv(f"data/{name}.csv")
        stylo = np.load(f"data/{name}_stylo.npy")
        print(f"Extracting embeddings for {name} ({len(df)} rows)...")
        emb = extract_embeddings(df.text.tolist(), tokenizer, base)
        splits[name] = (df, emb, stylo)

    # normalize stylometric feats with TRAIN stats only
    mean = splits["train"][2].mean(axis=0)
    std = splits["train"][2].std(axis=0) + 1e-8
    for name in splits:
        df, emb, stylo = splits[name]
        splits[name] = (df, emb, (stylo - mean) / std)
    np.save("models/stylo_mean.npy", mean)
    np.save("models/stylo_std.npy", std)

    loaders = {
        name: make_loader(emb, stylo, df.label, shuffle=(name == "train"))
        for name, (df, emb, stylo) in splits.items()
    }

    model = HybridHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0
    for epoch in range(15):
        tr_loss, tr_acc, tr_f1, *_ = run_epoch(model, loaders["train"], opt)
        _, val_acc, val_f1, *_ = run_epoch(model, loaders["val"])
        print(f"epoch {epoch}: train f1={tr_f1:.4f}  val f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "models/hybrid_head.pt")

    model.load_state_dict(torch.load("models/hybrid_head.pt"))
    for name in ["test", "cross_domain_test"]:
        _, acc, f1, preds, labels = run_epoch(model, loaders[name])
        print(f"\n=== Hybrid on {name} ===  acc={acc:.4f} f1={f1:.4f}")
        print(classification_report(labels, preds, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
