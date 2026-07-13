"""
Baseline model: TF-IDF + Logistic Regression.
This is the reference point the transformer must beat.
"""

import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


def evaluate(name, model, X, y, results):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    print(f"\n=== {name} ===")
    print(classification_report(y, preds, target_names=["real", "fake"]))
    results[name] = {"accuracy": acc, "f1": f1}


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    cross = pd.read_csv("data/cross_domain_test.csv")

    vec = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vec.fit_transform(train.text)
    X_test = vec.transform(test.text)
    X_cross = vec.transform(cross.text)

    results = {}

    logreg = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    logreg.fit(X_train, train.label)
    evaluate("LogReg (in-domain)", logreg, X_test, test.label, results)
    evaluate("LogReg (cross-domain LIAR)", logreg, X_cross, cross.label, results)

    joblib.dump(vec, "models/tfidf_vectorizer.joblib")
    joblib.dump(logreg, "models/logreg.joblib")
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Top features driving "fake" prediction - free explainability for LogReg
    feats = vec.get_feature_names_out()
    coefs = logreg.coef_[0]
    top_fake = sorted(zip(coefs, feats), reverse=True)[:25]
    top_real = sorted(zip(coefs, feats))[:25]
    print("\nTop FAKE-indicative n-grams:", [f for _, f in top_fake])
    print("\nTop REAL-indicative n-grams:", [f for _, f in top_real])

    print("\nSaved: models/tfidf_vectorizer.joblib, models/logreg.joblib,"
          " results/baseline_metrics.json")


if __name__ == "__main__":
    main()
