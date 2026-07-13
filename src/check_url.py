"""
Check a news article by URL: downloads the page, extracts the article text,
runs the fine-tuned model, prints the verdict.

Usage:
    python src/check_url.py https://example.com/some-news-article
or run without arguments to enter a URL interactively.

Requires: pip install trafilatura
"""

import sys
import torch
import trafilatura
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/roberta-fakenews"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fetch_article(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    # trafilatura extracts the main article body, dropping menus/ads/comments
    return trafilatura.extract(downloaded, include_comments=False)


@torch.no_grad()
def classify(text: str, tokenizer, model):
    enc = tokenizer(
        text, truncation=True, max_length=256, return_tensors="pt"
    ).to(DEVICE)
    probs = torch.softmax(model(**enc).logits, dim=1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])  # (real, fake)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else input("Article URL: ").strip()

    print(f"Fetching: {url}")
    text = fetch_article(url)
    if not text or len(text) < 100:
        print("Could not extract article text from this page "
              "(paywall, JS-heavy site, or not an article). "
              "Try copying the text into the Gradio demo instead.")
        return

    print(f"Extracted {len(text)} characters. First 200:\n  {text[:200]}...\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    p_real, p_fake = classify(text, tokenizer, model)
    verdict = "LIKELY FAKE" if p_fake > 0.5 else "LIKELY REAL"

    print(f"=== {verdict} ===")
    print(f"P(real) = {p_real:.3f}   P(fake) = {p_fake:.3f}")
    print()
    print("Caveats: the model judges LINGUISTIC STYLE, not facts. "
          "It was trained on 2016-17 US political news; on other topics, "
          "languages, or writing styles its confidence is not reliable "
          "(see calibration results in the repo).")


if __name__ == "__main__":
    main()
