"""
Explainability with SHAP on the fine-tuned transformer.
Produces token-level attributions: which words pushed the prediction
toward "fake" or "real".

Note: SHAP on transformers is slow — explain a small, curated set of
examples (10-20), not the whole test set.
"""

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = "models/roberta-fakenews"
DEVICE = 0 if torch.cuda.is_available() else -1
N_EXAMPLES = 12


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        top_k=None,
        truncation=True,
        max_length=256,
    )

    explainer = shap.Explainer(clf)

    test = pd.read_csv("data/test.csv")
    # curate: a few correct fakes, correct reals + the errors file if it exists
    sample = pd.concat([
        test[test.label == 1].sample(N_EXAMPLES // 2, random_state=42),
        test[test.label == 0].sample(N_EXAMPLES // 2, random_state=42),
    ])
    # truncate to keep SHAP tractable
    texts = [t[:1200] for t in sample.text.tolist()]

    shap_values = explainer(texts)

    # save interactive HTML visualizations
    for i, text in enumerate(texts):
        html = shap.plots.text(shap_values[i, :, "LABEL_1"], display=False)
        with open(f"results/shap_example_{i}.html", "w") as f:
            f.write(html)

    print(f"Saved {len(texts)} SHAP visualizations to results/shap_example_*.html")
    print("Open them in a browser; red tokens push toward 'fake'.")


if __name__ == "__main__":
    main()
