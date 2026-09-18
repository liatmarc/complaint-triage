"""Model-agnostic evaluation.

Every model (baseline, transformer, later an LLM) is scored by this one module
so the numbers are directly comparable. Produces:
  - a metrics dict (macro/weighted F1, accuracy, log loss, per-class F1)
  - confusion-matrix and calibration figures
  - a per-row predictions table for error analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from . import config as C


def compute_metrics(y_true, y_pred, y_proba: np.ndarray | None, classes: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0),
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
        "per_class_support": {c: int(report[c]["support"]) for c in classes},
    }
    if y_proba is not None:
        m["log_loss"] = log_loss(y_true, y_proba, labels=classes)
    return m


def plot_confusion(y_true, y_pred, classes: list[str], out: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=[c[:28] for c in classes])
    disp.plot(ax=ax, xticks_rotation=60, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_calibration(y_true, y_pred, y_proba: np.ndarray, out: Path, n_bins: int = 10) -> Path:
    """Reliability diagram on the predicted class's confidence.

    If a model says 90% it should be right ~90% of the time. Over-confident
    models are dangerous in routing: they hide the cases that need a human.
    """
    conf = y_proba.max(axis=1)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(conf[mask].mean()); ys.append(correct[mask].mean()); ns.append(mask.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def predictions_table(df: pd.DataFrame, y_pred, y_proba: np.ndarray | None, classes: list[str]) -> pd.DataFrame:
    out = df[["complaint_id", "date_received", "company", C.TARGET, C.TEXT]].copy()
    out["pred"] = y_pred
    out["correct"] = out[C.TARGET] == out["pred"]
    if y_proba is not None:
        out["confidence"] = y_proba.max(axis=1)
        # probability assigned to the true class: low values = confidently wrong
        cls_idx = {c: i for i, c in enumerate(classes)}
        out["p_true"] = [y_proba[i, cls_idx[t]] for i, t in enumerate(out[C.TARGET])]
    return out


def evaluate(model_name: str, df: pd.DataFrame, y_pred, y_proba: np.ndarray | None,
             classes: list[str], out_dir: Path) -> tuple[dict, list[Path]]:
    """Score, plot and dump everything for one model on one split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = df[C.TARGET].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_proba, classes)
    artifacts = [plot_confusion(y_true, y_pred, classes, out_dir / f"{model_name}_confusion.png")]
    if y_proba is not None:
        artifacts.append(plot_calibration(y_true, y_pred, y_proba, out_dir / f"{model_name}_calibration.png"))
    preds = predictions_table(df, y_pred, y_proba, classes)
    preds_path = out_dir / f"{model_name}_predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    artifacts.append(preds_path)
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics, artifacts


def error_sample(preds: pd.DataFrame, n: int = 50, seed: int = 0) -> pd.DataFrame:
    """The 50 worst mistakes: wrong AND confident. This is what you read by hand."""
    wrong = preds[~preds["correct"]]
    if "confidence" in wrong:
        wrong = wrong.sort_values("confidence", ascending=False)
        return wrong.head(n)
    return wrong.sample(min(n, len(wrong)), random_state=seed)
