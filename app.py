"""
Gradio demo: paste an article -> get fake probability + stylometric snapshot.
Run: python app.py   (or in Colab: it will print a public share link)
"""

import numpy as np
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.append("src")
from stylometric import extract_features, FEATURE_NAMES  # noqa: E402

MODEL_DIR = "models/roberta-fakenews"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


@torch.no_grad()
def predict(text):
    if not text or len(text.strip()) < 30:
        return {"(paste a longer text)": 1.0}, ""

    enc = tokenizer(
        text, truncation=True, max_length=256, return_tensors="pt"
    ).to(DEVICE)
    probs = torch.softmax(model(**enc).logits, dim=1)[0].cpu().numpy()

    stylo = extract_features(text)
    stylo_str = "\n".join(
        f"{name}: {val:.3f}" for name, val in zip(FEATURE_NAMES, stylo)
    )

    return (
        {"real": float(probs[0]), "fake": float(probs[1])},
        stylo_str,
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, label="Paste a news article or headline"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Stylometric features", lines=8),
    ],
    title="Misinformation Detector",
    description=(
        "Fine-tuned RoBERTa classifier trained on the Kaggle Fake/Real News "
        "corpus. NOTE: this is a research prototype — predictions reflect "
        "linguistic patterns, not verified facts, and accuracy drops on "
        "out-of-domain text (see cross-domain results in the repo)."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
