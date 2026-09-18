"""HTTP service: route a complaint and optionally draft a response.

Endpoints
  GET  /health   -> model + corpus loaded, version, uptime
  POST /predict  -> product, confidence, top-3, needs_review flag
  POST /draft    -> /predict + grounded draft + citation check + evidence
  GET  /         -> minimal UI for non-technical viewers

Every request is appended as one JSON line to LOG_PATH (default logs/predictions.jsonl)
so Phase 4 monitoring can compute drift and latency from real traffic.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config as C

load_dotenv(C.ROOT / ".env")
from .baseline import MODELS_DIR, TfidfLogReg
from .guardrails import redact_pii

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "tfidf_logreg.joblib"))
LOG_PATH = Path(os.getenv("LOG_PATH", C.ROOT / "logs" / "predictions.jsonl"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
VERSION = os.getenv("APP_VERSION", "0.1.0")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = TfidfLogReg.load(MODEL_PATH)
    STATE["started"] = time.time()
    STATE["drafter"] = None   # lazy: only if an API key is present and /draft is called
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yield
    STATE.clear()


app = FastAPI(title="Complaint Triage", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ComplaintIn(BaseModel):
    narrative: str = Field(..., min_length=C.MIN_NARRATIVE_CHARS, max_length=20_000)
    case_ref: str | None = None


class Prediction(BaseModel):
    product: str
    confidence: float
    top3: list[tuple[str, float]]
    needs_review: bool
    threshold: float


class PredictOut(Prediction):
    latency_ms: float
    pii_redacted: dict[str, int]


class DraftOut(PredictOut):
    draft: str
    citation_check: dict
    evidence: list[dict]
    draft_model: str
    draft_cost_usd: float
    draft_latency_ms: float
    flag_for_review: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _predict(text: str) -> Prediction:
    model: TfidfLogReg = STATE["model"]
    proba = model.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    top3 = [(model.classes_[i], round(float(proba[i]), 4)) for i in order[:3]]
    conf = float(proba[order[0]])
    return Prediction(product=model.classes_[order[0]], confidence=round(conf, 4), top3=top3,
                      needs_review=conf < CONFIDENCE_THRESHOLD, threshold=CONFIDENCE_THRESHOLD)


def _log(event: dict) -> None:
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **event}) + "\n")
    except OSError:
        pass   # logging must never break a request


def _drafter():
    if STATE.get("drafter") is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "drafting disabled: ANTHROPIC_API_KEY not set")
        from .drafting import Drafter
        STATE["drafter"] = Drafter(k=12)
    return STATE["drafter"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "model": MODEL_PATH.name,
            "n_classes": len(STATE["model"].classes_), "uptime_s": round(time.time() - STATE["started"], 1),
            "drafting_enabled": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/predict", response_model=PredictOut)
def predict(body: ComplaintIn):
    t0 = time.time()
    text, pii = redact_pii(body.narrative)
    p = _predict(text)
    latency = (time.time() - t0) * 1000
    _log({"endpoint": "predict", "product": p.product, "confidence": p.confidence,
          "needs_review": p.needs_review, "latency_ms": round(latency, 2), "narrative_len": len(text)})
    return PredictOut(**p.model_dump(), latency_ms=round(latency, 2), pii_redacted=pii)


@app.post("/draft", response_model=DraftOut)
def draft(body: ComplaintIn):
    t0 = time.time()
    text, pii = redact_pii(body.narrative)
    p = _predict(text)
    pred_latency = (time.time() - t0) * 1000
    res = _drafter().draft(text, p.product, case_ref=body.case_ref or "NB-000000")
    flag = p.needs_review or not res.citations["ok"]
    _log({"endpoint": "draft", "product": p.product, "confidence": p.confidence, "needs_review": p.needs_review,
          "citation_ok": res.citations["ok"], "flag_for_review": flag, "latency_ms": round(pred_latency, 2),
          "draft_latency_ms": round(res.latency_s * 1000, 1), "draft_cost_usd": res.cost_usd,
          "narrative_len": len(text)})
    return DraftOut(**p.model_dump(), latency_ms=round(pred_latency, 2), pii_redacted=pii,
                    draft=res.draft, citation_check=res.citations, evidence=res.chunks,
                    draft_model=res.model, draft_cost_usd=res.cost_usd,
                    draft_latency_ms=round(res.latency_s * 1000, 1), flag_for_review=flag)


@app.get("/", response_class=HTMLResponse)
def ui():
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
