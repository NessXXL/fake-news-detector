"""
Stylometric feature extraction.
These features capture HOW something is written, not WHAT it says —
useful signal for manipulation/clickbait detection and cheap to compute.
"""

import numpy as np
import pandas as pd
import textstat
from textblob import TextBlob

FEATURE_NAMES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "flesch_reading_ease",
    "exclamation_ratio",
    "question_ratio",
    "caps_ratio",
    "avg_word_length",
    "num_words_log",
]


def extract_features(text: str) -> list:
    text = text if isinstance(text, str) else ""
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = max(len(words), 1)

    blob = TextBlob(text[:5000])  # cap for speed

    return [
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        # clip: textstat can return extreme values on weird inputs
        float(np.clip(textstat.flesch_reading_ease(text), -100, 150)),
        text.count("!") / n_chars,
        text.count("?") / n_chars,
        sum(1 for c in text if c.isupper()) / n_chars,
        sum(len(w) for w in words) / n_words,
        float(np.log1p(n_words)),
    ]


def featurize_dataframe(df: pd.DataFrame, text_col: str = "text") -> np.ndarray:
    feats = np.array([extract_features(t) for t in df[text_col]])
    return feats


def normalize(train_feats, *other):
    """Standardize using train statistics only (no leakage)."""
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0) + 1e-8
    return [(f - mean) / std for f in (train_feats, *other)], (mean, std)


if __name__ == "__main__":
    for split in ["train", "val", "test", "cross_domain_test"]:
        df = pd.read_csv(f"data/{split}.csv")
        feats = featurize_dataframe(df)
        np.save(f"data/{split}_stylo.npy", feats)
        print(f"{split}: {feats.shape}")
