"""
Evaluation beyond accuracy:
1. Full transformer eval on in-domain and cross-domain test sets
2. Confidence calibration (reliability diagram + ECE)
3. Error analysis: dump most-confident mistakes for qualitative review
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/roberta-fakenews"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64


@torch.no_grad()
def predict_probs(texts, tokenizer, model, max_len=256):
    model.eval()
    probs = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(
            list(texts[i:i + BATCH]), truncation=True, max_length=max_len,
            padding=True, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: how far predicted confidence is from actual accuracy."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, bin_stats = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
        bin_stats.append((bin_conf, bin_acc, int(mask.sum())))
    return ece, bin_stats


def reliability_diagram(bin_stats, title, path):
    confs = [b[0] for b in bin_stats]
    accs = [b[1] for b in bin_stats]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.plot(confs, accs, "o-", label="model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def analyze(name, df, probs):
    preds = probs.argmax(axis=1)
    labels = df.label.values
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    ece, bin_stats = expected_calibration_error(probs, labels)
    cm = confusion_matrix(labels, preds).tolist()

    print(f"\n=== {name} ===")
    print(f"accuracy={acc:.4f}  f1={f1:.4f}  ECE={ece:.4f}")
    print(f"confusion matrix [ [TN,FP],[FN,TP] ]: {cm}")

    reliability_diagram(bin_stats, f"Reliability: {name}",
                        f"results/reliability_{name}.png")

    # most confident mistakes -> error analysis goldmine
    conf = probs.max(axis=1)
    wrong = df[preds != labels].copy()
    wrong["confidence"] = conf[preds != labels]
    wrong["predicted"] = preds[preds != labels]
    wrong = wrong.sort_values("confidence", ascending=False).head(30)
    wrong.to_csv(f"results/errors_{name}.csv", index=False)

    return {"accuracy": acc, "f1": f1, "ece": ece, "confusion_matrix": cm}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    results = {}
    for name, path in [
        ("kaggle_test", "data/test.csv"),
        ("liar_test", "data/cross_domain_test.csv"),
        ("fnn_test", "data/fnn_test.csv"),
    ]:
        df = pd.read_csv(path)
        probs = predict_probs(df.text.tolist(), tokenizer, model)
        np.save(f"results/probs_{name}.npy", probs)
        results[name] = analyze(name, df, probs)

    with open("results/transformer_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics, reliability diagrams, and error CSVs to results/")


if __name__ == "__main__":
    main()
