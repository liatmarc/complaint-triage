"""Phase 4: serve, load-test, monitor.

Usage:
    python scripts/phase4.py serve                 # http://127.0.0.1:8000  (UI at /, docs at /docs)
    python scripts/phase4.py smoke                 # hit /health and /predict on a running server
    python scripts/phase4.py load --n 200          # latency under repeated requests
    python scripts/phase4.py monitor               # drift / latency report from the prediction log
"""
from __future__ import annotations

import time

import httpx
import pandas as pd
import typer

from complaint_triage import config as C

app = typer.Typer(add_completion=False)
BASE = "http://127.0.0.1:8000"


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    import uvicorn
    uvicorn.run("complaint_triage.api:app", host=host, port=port, reload=reload)


@app.command()
def smoke(base: str = BASE):
    r = httpx.get(f"{base}/health", timeout=10); typer.echo(r.json())
    body = {"narrative": "Someone withdrew $500 from my checking account that I did not authorize and the bank has not responded in two weeks."}
    r = httpx.post(f"{base}/predict", json=body, timeout=10)
    typer.echo(r.json())


@app.command()
def load(n: int = 200, base: str = BASE):
    """Send n real test complaints and report latency percentiles (client-side)."""
    test = pd.read_parquet(C.TEST_PARQUET).sample(n, random_state=0)
    lat = []
    with httpx.Client(timeout=30) as c:
        for text in test[C.TEXT]:
            t0 = time.time(); c.post(f"{base}/predict", json={"narrative": text}); lat.append((time.time() - t0) * 1000)
    s = pd.Series(lat)
    typer.echo(f"n={n} p50={s.quantile(.5):.1f}ms p95={s.quantile(.95):.1f}ms max={s.max():.1f}ms")


@app.command()
def monitor(window_hours: int = 0):
    from complaint_triage.monitoring import write_monitor_report
    write_monitor_report(window_hours=window_hours or None)


if __name__ == "__main__":
    app()
