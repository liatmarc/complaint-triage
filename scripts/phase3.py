"""Phase 3: RAG response drafting + evaluation.

Usage:
    python scripts/phase3.py check-key                  # confirms .env is set up (one tiny API call)
    python scripts/phase3.py demo                       # draft a response for one test complaint
    python scripts/phase3.py build-eval-set             # 100 stratified test complaints
    python scripts/phase3.py generate                   # drafts with and without retrieval (~$0.30)
    python scripts/phase3.py generate --k 12 --condition rag_k12 --skip-no-rag   # whole product page
    python scripts/phase3.py judge                      # LLM-as-judge on both conditions (~$0.80)
    python scripts/phase3.py summarize                  # reports/phase3/summary.md + spot-check file
"""
from __future__ import annotations

import json

import pandas as pd
import typer

from complaint_triage import config as C
from complaint_triage.drafting import DRAFT_MODEL, JUDGE_MODEL, Drafter
from complaint_triage.llm_eval import (
    build_eval_set,
    generate_drafts,
    judge_drafts,
    spot_check_sample,
    summarize,
)

app = typer.Typer(add_completion=False)


def _client():
    import anthropic
    return anthropic.Anthropic()


@app.command()
def check_key():
    """One minimal API call to confirm the key works."""
    resp = _client().messages.create(model=DRAFT_MODEL, max_tokens=5,
                                     messages=[{"role": "user", "content": "Say OK"}])
    typer.echo(f"key works: model={DRAFT_MODEL} reply={resp.content[0].text!r}")


@app.command()
def demo(index: int = 0, no_retrieval: bool = False):
    """Draft a response for one test complaint and show the evidence."""
    preds = pd.read_parquet(C.REPORTS_DIR / "phase2" / "tfidf_logreg_test_predictions.parquet")
    r = preds.iloc[index]
    res = Drafter(_client()).draft(r[C.TEXT], r["pred"], case_ref=f"NB-{r['complaint_id']:06d}",
                                   use_retrieval=not no_retrieval)
    typer.echo(f"\n=== COMPLAINT (true: {r[C.TARGET]} | pred: {r['pred']}) ===\n{r[C.TEXT][:800]}\n")
    typer.echo("=== RETRIEVED ===")
    for i, c in enumerate(res.chunks, 1):
        typer.echo(f"[{i}] {c['title']}: {c['text'][:120]}...")
    typer.echo(f"\n=== DRAFT ===\n{res.draft}\n")
    typer.echo(f"citations={json.dumps(res.citations)} tokens={res.input_tokens}/{res.output_tokens} "
               f"cost=${res.cost_usd:.4f} latency={res.latency_s:.1f}s pii_redacted={res.pii_redacted}")


@app.command()
def build_eval_set_cmd(n: int = 100):
    build_eval_set(n)


@app.command()
def generate(k: int = 4, condition: str = "", skip_no_rag: bool = False):
    """Generate drafts. k = chunks retrieved (default 4); use a large k (e.g. 12)
    to give the model the whole product page. condition names the output files."""
    d = Drafter(_client(), k=k)
    generate_drafts(d, condition or "rag", use_retrieval=True)
    if not skip_no_rag:
        generate_drafts(d, "no_rag", use_retrieval=False)


@app.command()
def judge(condition: str = "rag", no_rag: bool = True):
    c = _client()
    typer.echo(f"judge model: {JUDGE_MODEL}")
    judge_drafts(c, condition)
    if no_rag:
        judge_drafts(c, "no_rag")


@app.command()
def summarize_cmd(conditions: str = "rag,no_rag", spot_check: str = "rag"):
    table = summarize(tuple(c.strip() for c in conditions.split(",")))
    typer.echo(table.to_string(index=False))
    out = spot_check_sample(spot_check)
    typer.echo(f"\nspot-check file for manual review: {out}")


if __name__ == "__main__":
    app()
