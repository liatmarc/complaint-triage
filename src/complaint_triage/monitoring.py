"""Post-deployment monitoring from the prediction log.

Reads logs/predictions.jsonl (written by the API) and reports:
  - prediction-distribution drift vs. the training label distribution, as a
    Population Stability Index (PSI): <0.1 stable, 0.1-0.25 watch, >0.25 investigate
  - mean confidence and share of requests flagged for review
  - latency p50 / p95
  - citation-check pass rate for /draft requests

Run on a schedule (cron, GitHub Actions) or on demand:
    python scripts/phase4.py monitor
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .api import LOG_PATH

REPORTS = C.REPORTS_DIR / "phase4"


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """Population Stability Index between two categorical distributions (shares)."""
    cats = sorted(set(expected.index) | set(actual.index))
    e = expected.reindex(cats).fillna(0).clip(lower=eps)
    a = actual.reindex(cats).fillna(0).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df


def monitor_report(log: pd.DataFrame | None = None, window_hours: int | None = None) -> dict:
    log = load_log() if log is None else log
    if log.empty:
        return {"n_requests": 0, "note": "no traffic logged yet"}
    if window_hours:
        log = log[log["ts"] >= log["ts"].max() - pd.Timedelta(hours=window_hours)]

    train_dist = pd.read_parquet(C.TRAIN_PARQUET)[C.TARGET].value_counts(normalize=True)
    live_dist = log["product"].value_counts(normalize=True)

    rep = {
        "n_requests": len(log),
        "period": [str(log["ts"].min()), str(log["ts"].max())],
        "psi_prediction_drift": round(psi(train_dist, live_dist), 4),
        "mean_confidence": round(float(log["confidence"].mean()), 4),
        "share_needs_review": round(float(log["needs_review"].mean()), 4),
        "latency_ms_p50": round(float(log["latency_ms"].quantile(0.5)), 2),
        "latency_ms_p95": round(float(log["latency_ms"].quantile(0.95)), 2),
        "live_distribution": live_dist.round(4).to_dict(),
    }
    drafts = log[log["endpoint"] == "draft"] if "endpoint" in log else pd.DataFrame()
    if not drafts.empty:
        rep["n_drafts"] = len(drafts)
        rep["citation_pass_rate"] = round(float(drafts["citation_ok"].mean()), 4)
        rep["draft_latency_ms_p50"] = round(float(drafts["draft_latency_ms"].quantile(0.5)), 1)
        rep["draft_cost_usd_total"] = round(float(drafts["draft_cost_usd"].sum()), 4)
    d = rep["psi_prediction_drift"]
    rep["drift_status"] = "stable" if d < 0.1 else ("watch" if d < 0.25 else "investigate")
    return rep


def write_monitor_report(**kw) -> Path:
    rep = monitor_report(**kw)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "monitor.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return out
