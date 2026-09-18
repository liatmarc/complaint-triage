"""Phase 2 tests: the evaluation module and the baseline wrapper on tiny data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from complaint_triage import config as C
from complaint_triage.baseline import TfidfLogReg
from complaint_triage.evaluate import compute_metrics, error_sample, evaluate, predictions_table

TEXTS = {
    "Credit card": "my credit card was charged twice and the bank refused a refund",
    "Mortgage": "my mortgage escrow payment went up and the servicer never explained why",
    "Student loan": "my student loan servicer lost my income driven repayment application",
}


def toy_df(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        for label, text in TEXTS.items():
            rows.append({"complaint_id": len(rows), "date_received": pd.Timestamp("2024-01-01"),
                         "company": "BANK", C.TARGET: label, C.TEXT: f"{text} ref {i}"})
    return pd.DataFrame(rows)


def test_baseline_learns_separable_classes():
    df = toy_df()
    model = TfidfLogReg(max_word_features=5000, max_char_features=5000).fit(df[C.TEXT], df[C.TARGET])
    pred = model.predict(df[C.TEXT])
    assert (pred == df[C.TARGET]).mean() > 0.95
    proba = model.predict_proba(df[C.TEXT])
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_metrics_and_error_sample():
    classes = ["A", "B"]
    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["A", "B", "B", "B"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    m = compute_metrics(y_true, y_pred, proba, classes)
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] < 1
    df = pd.DataFrame({"complaint_id": range(4), "date_received": pd.Timestamp("2024-01-01"),
                       "company": "X", C.TARGET: y_true, C.TEXT: list("wxyz")})
    preds = predictions_table(df, y_pred, proba, classes)
    errs = error_sample(preds)
    assert len(errs) == 1 and errs.iloc[0]["confidence"] == 0.8


def test_evaluate_writes_artifacts(tmp_path):
    df = toy_df(10)
    classes = list(TEXTS)
    proba = np.full((len(df), 3), 1 / 3)
    metrics, artifacts = evaluate("toy", df, df[C.TARGET].to_numpy(), proba, classes, tmp_path)
    assert metrics["accuracy"] == 1.0
    assert all(p.exists() for p in artifacts)
    assert (tmp_path / "toy_metrics.json").exists()
