# Misinformation Detection: Style Has a Ceiling

Research question: **Can a fine-tuned transformer detect misinformation beyond
its training domain — and does it know when it doesn't know?**

## Key findings

| Test set | Genre | Accuracy | F1 | ECE |
|---|---|---|---|---|
| Kaggle ISOT | long articles | 99.9% | 0.999 | 0.0005 |
| FakeNewsNet | headlines | 80.9% | 0.717 | 0.109 |
| LIAR | short claims | 62.8% | 0.611 | 0.130 |

- Our first (single-domain) model scored 99.97% in-domain but F1 = 0.19 on
  LIAR — it had learned dataset artifacts, not language. We found and removed
  two leakage channels: agency datelines ("WASHINGTON (Reuters) –", 84% of
  real articles) and an apostrophe-encoding gap between classes (47% vs 0%).
- Multi-domain training (3 corpora) raised LIAR F1 from 0.19 to 0.61 and cut
  calibration error from 0.42 to 0.13.
- ~63% on short claims is a ceiling for style-based detection: a claim's
  truth is not visible in its wording. Our live demo confirms it — The Onion
  satire passes the style model with P(real) = 0.999.
- `check_url_v2.py` adds a second, evidence-based signal: web corroboration
  (do independent outlets report the same story?). Style + evidence together
  catch cases neither signal catches alone.

## Pipeline

```
data/Fake.csv, data/True.csv        <- Kaggle "Fake and Real News Dataset"
data/liar/{train,valid,test}.tsv    <- https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
data/fnn/*.csv                      <- FakeNewsNet public CSVs (see below)

pip install -r requirements.txt
python -m textblob.download_corpora

python src/data_prep.py        # cleaning, anti-leakage, mixed-domain splits
python src/baseline.py         # TF-IDF + LogReg reference
python src/stylometric.py      # style features (optional branch)
python src/train_transformer.py  # fine-tune RoBERTa (GPU, ~30 min)
python src/evaluate.py         # per-domain metrics, calibration, error dumps
python src/explain.py          # SHAP token attributions
python app.py                  # Gradio demo
python src/check_url_v2.py <URL>   # style + web-evidence check of a live article
```

FakeNewsNet download:
```
mkdir -p data/fnn
for f in politifact_fake politifact_real gossipcop_fake gossipcop_real; do
  wget -q -O data/fnn/$f.csv \
  https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/$f.csv
done
```

## Limitations

- The classifier judges linguistic style, not facts; well-written falsehoods pass.
- English-only; training topics are 2016–18 US-centric.
- Kaggle 99.9% partly reflects outlet-style separability, not truth detection.
- Binarizing LIAR's 6 labels ("half-true" → real) is a documented judgment call.
- Evidence check verifies that a story is independently reported, not that
  each claim in it is true; fresh news can be under-corroborated.

## Authors

[Your names] — Stanford AI Summer Intensive, 2026.
Code developed with AI assistance (per course policy); experimental design,
debugging, analysis, and interpretation by the authors.
