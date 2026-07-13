"""
Fine-tune RoBERTa-base for fake news classification.
Designed for Google Colab (T4 GPU). Runtime: ~30-60 min depending on data size.
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "roberta-base"
MAX_LEN = 256          # articles are long; 256 tokens is a speed/quality compromise
OUTPUT_DIR = "models/roberta-fakenews"
SEED = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def to_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LEN, padding=False
        )

    return ds.map(tok, batched=True, remove_columns=["text"])


def main():
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train = to_dataset(pd.read_csv("data/train.csv"), tokenizer)
    val = to_dataset(pd.read_csv("data/val.csv"), tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=300,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",   # set to "wandb" if you configure Weights & Biases
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
