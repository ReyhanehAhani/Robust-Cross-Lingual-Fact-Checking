#!/usr/bin/env python3
"""Character n-gram baselines for multilingual claim labeling (toy snapshot)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_claims.csv"
ART = ROOT / "modeling" / "artifacts"


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=0.35,
        random_state=42,
        stratify=df["label"],
    )
    X_train = df.iloc[tr_idx]["text"]
    X_test = df.iloc[te_idx]["text"]
    y_train = df.iloc[tr_idx]["label"]
    y_test = df.iloc[te_idx]["label"]
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12_000),
            ),
            (
                "clf",
                LogisticRegression(max_iter=500),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "holdout_accuracy": float((pred == y_test).mean()),
        "report": classification_report(
            y_test, pred, labels=[0, 1, 2, 3], digits=3, zero_division=0
        ),
    }
    (ART / "classification_report.txt").write_text(metrics["report"], encoding="utf-8")
    (ART / "metrics.json").write_text(
        json.dumps({"holdout_accuracy": metrics["holdout_accuracy"]}, indent=2),
        encoding="utf-8",
    )
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(4), labels=[str(i) for i in range(4)])
    ax.set_yticks(range(4), labels=[str(i) for i in range(4)])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ART / "confusion_matrix.png", dpi=120)
    plt.close()
    # Error analysis by language
    te_df = df.iloc[te_idx].copy()
    te_df["y_true"] = y_test.values
    te_df["y_pred"] = pred
    err = te_df[te_df["y_true"] != te_df["y_pred"]]
    err.groupby("lang").size().to_csv(ART / "errors_by_lang.csv", header=["count"])
    err.head(80).to_csv(ART / "error_sample.csv", index=False)
    print("Wrote baseline artifacts to", ART)
    return 0


if __name__ == "__main__":
    if os.environ.get("MPLBACKEND") is None:
        os.environ["MPLBACKEND"] = "Agg"
    raise SystemExit(main())
