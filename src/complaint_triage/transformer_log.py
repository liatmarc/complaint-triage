"""Score transformer predictions produced on Colab and log them to MLflow.

The GPU box only produces probabilities; all scoring happens here with the same
evaluate() used for the baseline, so the two runs are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from . import config as C
from .baseline import MLFLOW_URI
from .evaluate import error_sample, evaluate


def _align(split_df: pd.DataFrame, proba_df: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Join predictions to the local split on complaint_id; fail loudly on mismatch."""
    merged = split_df.merge(proba_df, on="complaint_id", how="inner", validate="one_to_one")
    if len(merged) != len(split_df):
        raise ValueError(f"only {len(merged)}/{len(split_df)} rows matched; splits differ from Colab's")
    proba = merged[classes].to_numpy(dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return merged.drop(columns=classes), proba


def log_transformer(preds_dir: Path, run_name: str = "distilbert") -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("complaint-triage")
    meta = json.loads((preds_dir / "train_meta.json").read_text())
    classes = meta["classes"]
    split_meta = json.loads((C.PROCESSED_DIR / "split_meta.json").read_text())
    if classes != split_meta["classes"]:
        raise ValueError("class list on Colab differs from local split_meta.json")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: meta[k] for k in ("model", "max_len", "epochs", "lr", "batch", "n_train", "device", "gpu")})
        mlflow.log_metric("train_seconds", meta["train_seconds"])
        results = {}
        for split_name, path in (("val", C.VAL_PARQUET), ("test", C.TEST_PARQUET)):
            df = pd.read_parquet(path)
            proba_df = pd.read_parquet(preds_dir / f"{split_name}_proba.parquet")
            df, proba = _align(df, proba_df, classes)
            pred = np.array(classes)[proba.argmax(axis=1)]
            metrics, artifacts = evaluate(f"{run_name}_{split_name}", df, pred, proba, classes,
                                          C.REPORTS_DIR / "phase2")
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metric(f"{split_name}_latency_ms_per_row", meta["latency_ms_per_row"][split_name])
            for a in artifacts:
                mlflow.log_artifact(str(a))
            results[split_name] = metrics

        preds = pd.read_parquet(C.REPORTS_DIR / "phase2" / f"{run_name}_test_predictions.parquet")
        err_path = C.REPORTS_DIR / "phase2" / f"{run_name}_test_errors.csv"
        error_sample(preds).to_csv(err_path, index=False)
        mlflow.log_artifact(str(err_path))

        print(f"[transformer] run_id={run.info.run_id}")
        print(f"[transformer] val macro-F1 {results['val']['macro_f1']:.3f} | "
              f"test macro-F1 {results['test']['macro_f1']:.3f} | test acc {results['test']['accuracy']:.3f}")
        return {"run_id": run.info.run_id, **results}
