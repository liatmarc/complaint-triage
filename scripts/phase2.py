"""Phase 2: train and evaluate models with MLflow tracking.

Usage:
    python scripts/phase2.py baseline                # TF-IDF + logistic regression
    python scripts/phase2.py baseline --c-reg 1.0    # try a different regularisation
    python scripts/phase2.py log-transformer-cmd transformer_out   # import Colab results
    python scripts/phase2.py explain                 # explainability report
    python scripts/phase2.py slices                  # slice analysis
    python scripts/phase2.py compare                 # table of all runs so far
    mlflow ui --backend-store-uri sqlite:///mlflow.db             # then open http://127.0.0.1:5000
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import typer

from complaint_triage.baseline import MLFLOW_URI, train_baseline
from complaint_triage.explain import slice_analysis, write_explanations
from complaint_triage.transformer_log import log_transformer

app = typer.Typer(add_completion=False)


@app.command()
def baseline(c_reg: float = 4.0, run_name: str = "tfidf_logreg", no_class_weight: bool = False):
    train_baseline(run_name=run_name, C_reg=c_reg,
                   class_weight=None if no_class_weight else "balanced")


@app.command()
def log_transformer_cmd(preds_dir: Path, run_name: str = "distilbert"):
    """Score transformer predictions from Colab (transformer_out/) and log to MLflow."""
    log_transformer(preds_dir, run_name=run_name)


@app.command()
def explain(model_name: str = "tfidf_logreg"):
    """Top n-grams per class + per-prediction contributions for sample rows."""
    write_explanations(model_name)


@app.command()
def slices(model_name: str = "tfidf_logreg"):
    """Accuracy / macro-F1 by narrative length, company and channel."""
    slice_analysis(model_name)


@app.command()
def compare():
    """Print every run's headline metrics side by side."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    runs = mlflow.search_runs(experiment_names=["complaint-triage"])
    if runs.empty:
        typer.echo("no runs yet"); return
    cols = ["tags.mlflow.runName", "params.model", "metrics.val_macro_f1", "metrics.test_macro_f1",
            "metrics.test_accuracy", "metrics.test_log_loss", "metrics.test_latency_ms_per_row",
            "metrics.train_seconds"]
    cols = [c for c in cols if c in runs.columns]
    table = runs[cols].rename(columns=lambda c: c.split(".")[-1]).round(3)
    with pd.option_context("display.width", 200):
        typer.echo(table.to_string(index=False))


if __name__ == "__main__":
    app()
