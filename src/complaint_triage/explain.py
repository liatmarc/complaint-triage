"""Explainability and slice analysis for the shipped baseline.

Two questions a reviewer will ask:
  1. Why did the model say that?  -> per-prediction n-gram contributions, and
     the top n-grams per class (global).
  2. Where does it fail?          -> macro-F1 / accuracy by narrative length,
     by company, and by submission channel.

Because the model is linear, contribution = coefficient x TF-IDF value, so the
explanations are exact rather than approximated (SHAP would give the same
ranking here at far higher cost).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from .baseline import MODELS_DIR, TfidfLogReg

PHASE2_DIR = C.REPORTS_DIR / "phase2"


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def feature_names(model: TfidfLogReg) -> np.ndarray:
    return np.concatenate([model.word_vec.get_feature_names_out(),
                           np.char.add("char:", model.char_vec.get_feature_names_out())])


def top_features_per_class(model: TfidfLogReg, k: int = 15) -> pd.DataFrame:
    """Global view: the n-grams with the largest positive weight for each class."""
    names = feature_names(model)
    rows = []
    for i, cls in enumerate(model.classes_):
        coef = model.clf.coef_[i]
        top = np.argsort(coef)[::-1][:k]
        rows.append({"class": cls, "top_features": ", ".join(names[top])})
    return pd.DataFrame(rows)


def explain_one(model: TfidfLogReg, text: str, k: int = 10) -> dict:
    """Local view: which n-grams pushed this prediction, and towards what."""
    X = model._features([text])
    proba = model.clf.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    names = feature_names(model)
    contrib = X.multiply(model.clf.coef_[pred_idx]).toarray()[0]
    nz = np.flatnonzero(contrib)
    order = nz[np.argsort(contrib[nz])[::-1]]
    return {
        "pred": model.classes_[pred_idx],
        "confidence": float(proba[pred_idx]),
        "runner_up": model.classes_[int(np.argsort(proba)[-2])],
        "for": [(names[i], round(float(contrib[i]), 3)) for i in order[:k]],
        "against": [(names[i], round(float(contrib[i]), 3)) for i in order[::-1][:k] if contrib[i] < 0],
    }


def write_explanations(model_name: str = "tfidf_logreg", n_examples: int = 5) -> Path:
    model = TfidfLogReg.load(MODELS_DIR / f"{model_name}.joblib")
    preds = pd.read_parquet(PHASE2_DIR / f"{model_name}_test_predictions.parquet")

    # A few correct and a few confidently-wrong examples.
    correct = preds[preds["correct"]].sample(n_examples, random_state=0)
    wrong = preds[~preds["correct"]].sort_values("confidence", ascending=False).head(n_examples)

    md = [f"# Explainability: {model_name}\n", "## Top n-grams per class\n",
          top_features_per_class(model).to_markdown(index=False), "\n"]
    for title, subset in (("Correct predictions", correct), ("Most confident errors", wrong)):
        md.append(f"## {title}\n")
        for _, r in subset.iterrows():
            e = explain_one(model, r[C.TEXT])
            md.append(f"**True:** {r[C.TARGET]} | **Pred:** {e['pred']} ({e['confidence']:.2f}) | "
                      f"runner-up: {e['runner_up']}\n")
            md.append(f"> {r[C.TEXT][:400]}{'...' if len(r[C.TEXT]) > 400 else ''}\n")
            md.append("Pushed towards prediction: " + ", ".join(f"`{f}` ({v:+.2f})" for f, v in e["for"]) + "\n")
            if e["against"]:
                md.append("Pushed away: " + ", ".join(f"`{f}` ({v:+.2f})" for f, v in e["against"]) + "\n")
    out = PHASE2_DIR / f"{model_name}_explanations.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[explain] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Slice analysis
# ---------------------------------------------------------------------------
def _score(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(g),
        "accuracy": accuracy_score(g[C.TARGET], g["pred"]),
        "macro_f1": f1_score(g[C.TARGET], g["pred"], average="macro", zero_division=0),
    })


def slice_analysis(model_name: str = "tfidf_logreg", top_companies: int = 10) -> Path:
    preds = pd.read_parquet(PHASE2_DIR / f"{model_name}_test_predictions.parquet")
    test = pd.read_parquet(C.TEST_PARQUET)[["complaint_id", "narrative_len", "submitted_via"]]
    df = preds.merge(test, on="complaint_id")

    df["length_bucket"] = pd.cut(df["narrative_len"], bins=[0, 200, 500, 1000, 2000, 5000, 10**7],
                                 labels=["<200", "200-500", "500-1k", "1k-2k", "2k-5k", ">5k"])
    by_len = df.groupby("length_bucket", observed=True).apply(_score, include_groups=False)

    top = df["company"].value_counts().head(top_companies).index
    by_company = df[df["company"].isin(top)].groupby("company").apply(_score, include_groups=False)
    by_company = by_company.sort_values("n", ascending=False)

    by_channel = df.groupby("submitted_via").apply(_score, include_groups=False)

    overall = _score(df)
    md = [f"# Slice analysis: {model_name}\n",
          f"Overall: n={int(overall['n'])}, accuracy={overall['accuracy']:.3f}, macro-F1={overall['macro_f1']:.3f}\n",
          "## By narrative length (chars)\n", by_len.round(3).to_markdown(), "\n",
          f"## By company (top {top_companies} by volume)\n", by_company.round(3).to_markdown(), "\n",
          "## By submission channel\n", by_channel.round(3).to_markdown(), "\n",
          "Note: macro-F1 within a slice is over the classes present in that slice, "
          "so compare accuracy across slices and use macro-F1 within a slice with care.\n"]
    out = PHASE2_DIR / f"{model_name}_slices.md"
    out.write_text("\n".join(md), encoding="utf-8")
    (PHASE2_DIR / f"{model_name}_slices.json").write_text(json.dumps({
        "by_length": by_len.to_dict(orient="index"),
        "by_company": by_company.to_dict(orient="index"),
        "by_channel": by_channel.to_dict(orient="index"),
    }, indent=2, default=str))
    print(f"[slices] wrote {out}")
    return out
