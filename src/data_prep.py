"""
Data preparation v3 — three domains, leakage-cleaned, honest splits.

Domains:
1. Kaggle ISOT  — long news articles (Reuters vs fake-news sites), 2016-17 US politics
2. LIAR         — short fact-checked political claims (official train/valid/test)
3. FakeNewsNet  — headlines only (PolitiFact + GossipCop celebrity news)
                  NOTE: public CSVs contain titles, not full articles. We
                  classify headlines — different granularity, documented.
4. OnionOrNot   — hard examples for style blindness: The Onion headlines
                  (absurd content in professional news style, label=fake)
                  vs r/NotTheOnion (REAL news that sounds absurd, label=real).
                  Teaches the model that polished style != truth and
                  absurd content != fake.

Leakage fixes (verified on raw data):
- strip full datelines ("WASHINGTON (Reuters) -"), not just the agency name
- normalize apostrophes/contractions identically for both classes
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SEED = 42

LIAR_LABEL_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "barely-true": 1, "false": 1, "pants-fire": 1,
}
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true_ct", "false_ct", "half_true_ct",
    "mostly_true_ct", "pants_fire_ct", "context",
]

DATELINE_RE = re.compile(
    r"^[A-Z][A-Za-z .,'/-]{0,60}?\((?:Reuters|AP|AFP)\)\s*[-–—]*\s*")
AGENCY_RE = re.compile(r"\(?(?:Reuters|AP|AFP)\)?\s*[-–—]?")

FNN_FILES = {  # FakeNewsNet public CSVs
    "politifact_fake.csv": 1, "politifact_real.csv": 0,
    "gossipcop_fake.csv": 1, "gossipcop_real.csv": 0,
}


def normalize_apostrophes(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"(\w)'(t|s|re|ve|ll|d|m)\b", r"\1\2", text)
    text = re.sub(r"(\w) (t|s|re|ve|ll|d|m)\b", r"\1\2", text)
    return text.replace("'", " ")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = DATELINE_RE.sub(" ", text)
    text = AGENCY_RE.sub(" ", text)
    text = normalize_apostrophes(text)
    return re.sub(r"\s+", " ", text).strip()


def load_kaggle():
    fake = pd.read_csv(f"{DATA_DIR}/Fake.csv")
    real = pd.read_csv(f"{DATA_DIR}/True.csv")
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna(""))
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]]
    df = df[df.text.str.len() > 30].drop_duplicates("text").reset_index(drop=True)
    df["source"] = "kaggle"
    return df


def load_liar_split(path):
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df["label"] = df["label"].map(LIAR_LABEL_MAP)
    df["text"] = df["statement"].apply(clean_text)
    df = df[["text", "label"]].dropna()
    df = df[df.text.str.len() > 10].drop_duplicates("text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["source"] = "liar"
    return df


def load_fnn():
    parts = []
    for fname, label in FNN_FILES.items():
        path = os.path.join(DATA_DIR, "fnn", fname)
        df = pd.read_csv(path)
        df = df[["title"]].dropna()
        df["label"] = label
        df["text"] = df["title"].apply(clean_text)
        parts.append(df[["text", "label"]])
    df = pd.concat(parts, ignore_index=True)
    df = df[df.text.str.len() > 20].drop_duplicates("text").reset_index(drop=True)

    # GossipCop is heavily imbalanced (16.8k real vs 5.3k fake):
    # downsample the majority class so the model can't win by guessing "real"
    real = df[df.label == 0]
    fake = df[df.label == 1]
    real = real.sample(n=min(len(real), len(fake) * 2), random_state=SEED)
    df = pd.concat([real, fake], ignore_index=True)
    df["source"] = "fnn"
    return df


def load_onion():
    """OnionOrNot.csv from Kaggle (chrisfilo/onion-or-not).
    Columns: text, label (1 = The Onion, 0 = real weird news)."""
    path = os.path.join(DATA_DIR, "OnionOrNot.csv")
    df = pd.read_csv(path)
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]].dropna()
    df = df[df.text.str.len() > 20].drop_duplicates("text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["source"] = "onion"
    return df


def main():
    kaggle = load_kaggle()
    liar_train = load_liar_split(f"{DATA_DIR}/liar/train.tsv")
    liar_val = load_liar_split(f"{DATA_DIR}/liar/valid.tsv")
    liar_test = load_liar_split(f"{DATA_DIR}/liar/test.tsv")
    fnn = load_fnn()
    onion = load_onion()

    assert not kaggle.text.str.contains(r"\(Reuters\)").any(), "Reuters leak!"
    assert not kaggle.text.str.contains("\u2019").any(), "apostrophe leak!"
    print("Leakage sanity checks passed.")

    k_train, k_temp = train_test_split(
        kaggle, test_size=0.30, stratify=kaggle.label, random_state=SEED)
    k_val, k_test = train_test_split(
        k_temp, test_size=0.50, stratify=k_temp.label, random_state=SEED)

    f_train, f_temp = train_test_split(
        fnn, test_size=0.30, stratify=fnn.label, random_state=SEED)
    f_val, f_test = train_test_split(
        f_temp, test_size=0.50, stratify=f_temp.label, random_state=SEED)

    o_train, o_temp = train_test_split(
        onion, test_size=0.30, stratify=onion.label, random_state=SEED)
    o_val, o_test = train_test_split(
        o_temp, test_size=0.50, stratify=o_temp.label, random_state=SEED)

    # LIAR and FNN are much smaller than Kaggle -> upsample so the model
    # doesn't ignore them (x3). Upsampling is applied to TRAIN ONLY.
    train = pd.concat(
        [k_train] + [liar_train] * 3 + [f_train] * 3 + [o_train] * 2,
        ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val = pd.concat([k_val, liar_val, f_val, o_val], ignore_index=True)

    train.to_csv(f"{DATA_DIR}/train.csv", index=False)
    val.to_csv(f"{DATA_DIR}/val.csv", index=False)
    k_test.to_csv(f"{DATA_DIR}/test.csv", index=False)
    liar_test.to_csv(f"{DATA_DIR}/cross_domain_test.csv", index=False)
    f_test.to_csv(f"{DATA_DIR}/fnn_test.csv", index=False)
    o_test.to_csv(f"{DATA_DIR}/onion_test.csv", index=False)

    print(f"train: {len(train)}  (kaggle {len(k_train)}, liar x3 {len(liar_train)*3}, "
          f"fnn x3 {len(f_train)*3}, onion x2 {len(o_train)*2})")
    print(f"val: {len(val)}   tests: kaggle {len(k_test)}, liar {len(liar_test)}, "
          f"fnn {len(f_test)}, onion {len(o_test)}")
    print(f"train fake ratio: {train.label.mean():.2f}")


if __name__ == "__main__":
    main()
