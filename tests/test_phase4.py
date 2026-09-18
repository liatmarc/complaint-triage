"""Phase 4 tests: the API on a tiny model, and monitoring on a synthetic log."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_phase2 import toy_df
from test_phase3 import FakeClient

from complaint_triage import api as api_mod
from complaint_triage import config as C
from complaint_triage.baseline import TfidfLogReg
from complaint_triage.monitoring import monitor_report, psi


@pytest.fixture
def client(tmp_path, monkeypatch):
    df = toy_df(30)
    model_path = tmp_path / "m.joblib"
    TfidfLogReg(max_word_features=5000, max_char_features=5000).fit(df[C.TEXT], df[C.TARGET]).save(model_path)
    monkeypatch.setattr(api_mod, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_mod, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(api_mod.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["n_classes"] == 3
    assert r.json()["drafting_enabled"] is False


def test_predict_and_logging(client):
    r = client.post("/predict", json={"narrative": "my credit card was charged twice and the bank refused a refund, call 410-555-1234"})
    assert r.status_code == 200
    d = r.json()
    assert d["product"] == "Credit card"
    assert 0 < d["confidence"] <= 1 and len(d["top3"]) == 3
    assert d["pii_redacted"] == {"phone": 1}
    log = [json.loads(l) for l in api_mod.LOG_PATH.read_text().splitlines()]
    assert log[-1]["endpoint"] == "predict" and log[-1]["product"] == "Credit card"


def test_predict_validation(client):
    assert client.post("/predict", json={"narrative": "short"}).status_code == 422


def test_draft_disabled_without_key(client):
    r = client.post("/draft", json={"narrative": "my mortgage escrow payment went up and nobody explained why to me"})
    assert r.status_code == 503


def test_draft_with_fake_drafter(client):
    from complaint_triage.drafting import Drafter
    api_mod.STATE["drafter"] = Drafter(client=FakeClient("Dear Customer, we will respond within 30 business days [1]."), k=12)
    r = client.post("/draft", json={"narrative": "my mortgage escrow payment went up and nobody explained why to me"})
    assert r.status_code == 200
    d = r.json()
    assert d["product"] == "Mortgage" and "Dear Customer" in d["draft"]
    assert isinstance(d["flag_for_review"], bool) and len(d["evidence"]) > 0


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Complaint Triage" in r.text


def test_psi_and_monitor(tmp_path, monkeypatch):
    a = pd.Series({"x": 0.5, "y": 0.5}); b = pd.Series({"x": 0.5, "y": 0.5})
    assert psi(a, b) < 1e-6
    assert psi(a, pd.Series({"x": 0.9, "y": 0.1})) > 0.25
    train = toy_df(10); train.to_parquet(tmp_path / "train.parquet")
    monkeypatch.setattr(C, "TRAIN_PARQUET", tmp_path / "train.parquet")
    log = pd.DataFrame({"ts": pd.to_datetime([1, 2, 3], unit="s"), "endpoint": "predict",
                        "product": ["Credit card", "Mortgage", "Student loan"], "confidence": [0.9, 0.5, 0.7],
                        "needs_review": [False, True, False], "latency_ms": [2.0, 3.0, 4.0]})
    rep = monitor_report(log)
    assert rep["n_requests"] == 3 and rep["drift_status"] == "stable"
    assert rep["share_needs_review"] == pytest.approx(1 / 3, abs=1e-3)
