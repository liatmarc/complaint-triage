"""Baseline: TF-IDF (word + char n-grams) -> logistic regression.

This is deliberately boring. It trains in under a minute on CPU, is easy to
explain, and sets the bar every later model must clear by a margin that
justifies its extra cost.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from .evaluate import error_sample, evaluate

MODELS_DIR = C.ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_URI = f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}"


class TfidfLogReg:
    """Thin wrapper so the API in Phase 4 can call .predict / .predict_proba."""

    def __init__(self, C_reg: float = 4.0, max_word_features: int = 200_000,
                 max_char_features: int = 200_000, class_weight: str | None = "balanced"):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_word_features,
                                        sublinear_tf=True, dtype=np.float32)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                        max_features=max_char_features, sublinear_tf=True, dtype=np.float32)
        self.clf = LogisticRegression(C=C_reg, max_iter=2000, class_weight=class_weight)
        self.classes_: list[str] = []

    def _features(self, texts, fit: bool = False):
        if fit:
            return hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]).tocsr()
        return hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()

    def fit(self, texts, labels):
        X = self._features(texts, fit=True)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._features(texts))

    def predict(self, texts):
        return self.clf.predict(self._features(texts))

    def save(self, path: Path) -> Path:
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> TfidfLogReg:
        return joblib.load(path)


def train_baseline(run_name: str = "tfidf_logreg", **model_kwargs) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("complaint-triage")

    train = pd.read_parquet(C.TRAIN_PARQUET)
    val = pd.read_parquet(C.VAL_PARQUET)
    test = pd.read_parquet(C.TEST_PARQUET)
    split_meta = json.loads((C.PROCESSED_DIR / "split_meta.json").read_text())
    classes = split_meta["classes"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model": "tfidf_logreg", "n_train": len(train), "n_val": len(val),
                           "n_test": len(test), "n_classes": len(classes), **model_kwargs})
        mlflow.log_param("test_dates", "->".join(split_meta["test_dates"]))

        t0 = time.time()
        model = TfidfLogReg(**model_kwargs).fit(train[C.TEXT], train[C.TARGET])
        train_secs = time.time() - t0
        mlflow.log_metric("train_seconds", train_secs)

        results = {}
        for split_name, df in (("val", val), ("test", test)):
            t0 = time.time()
            proba = model.predict_proba(df[C.TEXT])
            latency_ms = (time.time() - t0) / len(df) * 1000
            pred = np.array(model.classes_)[proba.argmax(axis=1)]
            metrics, artifacts = evaluate(f"{run_name}_{split_name}", df, pred, proba,
                                          model.classes_, C.REPORTS_DIR / "phase2")
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_metric(f"{split_name}_latency_ms_per_row", latency_ms)
            for a in artifacts:
                mlflow.log_artifact(str(a))
            results[split_name] = metrics

        # Error-analysis sample from the test set: the 50 most confident mistakes.
        preds = pd.read_parquet(C.REPORTS_DIR / "phase2" / f"{run_name}_test_predictions.parquet")
        errs = error_sample(preds)
        err_path = C.REPORTS_DIR / "phase2" / f"{run_name}_test_errors.csv"
        errs.to_csv(err_path, index=False)
        mlflow.log_artifact(str(err_path))

        model_path = model.save(MODELS_DIR / f"{run_name}.joblib")
        mlflow.log_artifact(str(model_path))

        print(f"[baseline] run_id={run.info.run_id}")
        print(f"[baseline] train {train_secs:.0f}s | val macro-F1 {results['val']['macro_f1']:.3f} "
              f"| test macro-F1 {results['test']['macro_f1']:.3f} | test acc {results['test']['accuracy']:.3f}")
        print(f"[baseline] read the errors: {err_path}")
        return {"run_id": run.info.run_id, **results}
