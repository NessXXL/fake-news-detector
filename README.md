# AI Misinformation Detection: Style + Evidence

**Research question:** Can an AI model detect misinformation beyond its training domain — and does it know when it doesn't know?

Built during the [Stanford Pre-Collegiate Summer Institutes], AI major track, 2026.

## Overview

We built a two-signal system for detecting misinformation:
- **STYLE** — a fine-tuned RoBERTa transformer that judges *how* an article is written
- **EVIDENCE** — a web-search module that checks *whether* independent outlets report the same story

Along the way, we found and fixed hidden data leakage that had inflated our first model's accuracy to a misleading 99.97%, then tested the model across four different text genres to measure how well it actually generalizes.

## Key results

| Test set | Genre | Accuracy | F1 | ECE ↓ |
|---|---|---|---|---|
| Kaggle ISOT | long news articles | 99.98% | 0.9998 | 0.000 |
| OnionOrNot | satire headlines | 93.4% | 0.913 | 0.052 |
| FakeNewsNet | headlines | 80.5% | 0.713 | 0.158 |
| LIAR | short political claims | 63.4% | 0.565 | 0.243 |

**The gradient is the finding:** the shorter the text and the more its truth depends on outside facts, the less any style-based model can see. Accuracy drops steadily from long articles → satire → headlines → short claims.

## What we found along the way

- **Our first model "cheated."** It scored 99.97% in-domain but collapsed to F1 = 0.19 on unseen data. We audited the raw data and found: 84% of real articles started with a dateline like "WASHINGTON (Reuters) –", and fake/real texts had different apostrophe encodings — both let the model guess the label without reading the text. We removed both leaks and added automated checks so they can't silently return.
- **The model was blind to professional satire.** The Onion's "NASA Discovers Concerning Lump On Mars" was rated 99.9% real by early versions. We added hard training examples (OnionOrNot dataset) and fixed a bug where our URL-checker wasn't feeding the headline to the model — together this raised P(fake) on that exact article from 0.001 to 0.957.
- **Two signals catch different failure modes.** Professionally written satire fools STYLE but fails EVIDENCE (no independent outlet reports it). A true story with a uniquely phrased headline can fail EVIDENCE's word-matching but passes STYLE. Neither signal alone is enough.

## Pipeline

```bash
# 1. Get the datasets
# Kaggle "Fake and Real News Dataset" -> data/Fake.csv, data/True.csv
# LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -> data/liar/
# FakeNewsNet public CSVs (auto-downloadable, see below)
# OnionOrNot (Kaggle) -> data/OnionOrNot.csv

mkdir -p data/fnn
for f in politifact_fake politifact_real gossipcop_fake gossipcop_real; do
  wget -q -O data/fnn/$f.csv \
  https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/$f.csv
done

pip install -r requirements.txt
python -m textblob.download_corpora

# 2. Prepare data (cleaning, anti-leakage checks, 4-domain splits)
python src/data_prep.py

# 3. Baseline for comparison
python src/baseline.py

# 4. Fine-tune the transformer (~30-60 min on a GPU)
python src/train_transformer.py

# 5. Evaluate: per-domain accuracy, F1, calibration, error analysis
python src/evaluate.py

# 6. Explainability (SHAP token attributions)
python src/explain.py

# 7. Check any live article by URL (style + web evidence)
pip install trafilatura ddgs
python src/check_url_v2.py <article URL>

# 8. Interactive demo
python app.py
```

## Limitations

- The model judges linguistic style and event corroboration — not the truth of individual claims.
- English-only; training topics skew 2016–20 US politics.
- Style detection of satire isn't fully generalized — the model sometimes misses newer articles it wasn't trained on.

## Author

Zhanibek — Stanford Pre-Collegiate Summer Institutes, AI major, 2026.
