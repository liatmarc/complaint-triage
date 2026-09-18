"""Evaluate the drafting layer.

1. Build a fixed eval set: 100 test complaints, stratified by product.
2. Generate drafts WITH retrieval and WITHOUT (ablation), using the classifier's
   predicted product so the eval reflects the real pipeline.
3. Score every draft with an LLM judge on a rubric, plus the deterministic
   citation check. Judge and drafter are different models to reduce
   self-preference bias.
4. Write a summary table: mean rubric scores, hallucination rate, citation-check
   pass rate, cost and latency, for each condition.

Everything is cached to disk row by row so a crash or rate limit does not lose
paid work; re-running skips completed rows.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from . import config as C
from .drafting import JUDGE_MODEL, Drafter, cost_usd

PHASE3_DIR = C.REPORTS_DIR / "phase3"
EVAL_SET = C.PROCESSED_DIR / "eval_set_phase3.parquet"

JUDGE_PROMPT = """You are auditing a draft response to a consumer complaint. You are given the complaint, the policy excerpts the drafter was allowed to use, and the draft.

Score the draft on three criteria, each 1-5:
- accuracy: every factual claim (timelines, steps, amounts) is correct per the excerpts; 5 = all correct, 1 = mostly wrong.
- grounding: the draft relies only on the excerpts and does not invent policy; 5 = fully grounded, 1 = mostly invented.
- tone: professional, empathetic, does not admit fault, follows the requested structure; 5 = excellent.
Then set hallucination to true if the draft states ANY specific timeline, amount, or policy step that is not supported by the excerpts.

Respond with ONLY a JSON object, no preamble and no code fences:
{"accuracy": n, "grounding": n, "tone": n, "hallucination": true or false, "note": "at most 15 words"}"""


def build_eval_set(n: int = 100, seed: int = 0) -> pd.DataFrame:
    preds = pd.read_parquet(C.REPORTS_DIR / "phase2" / "tfidf_logreg_test_predictions.parquet")
    # stratified by true product, proportional, at least 3 per class where possible
    per_class = (preds[C.TARGET].value_counts(normalize=True) * n).round().clip(lower=3).astype(int)
    parts = [g.sample(min(per_class[k], len(g)), random_state=seed) for k, g in preds.groupby(C.TARGET)]
    df = pd.concat(parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    df["case_ref"] = [f"NB-{i:06d}" for i in df["complaint_id"]]
    df.to_parquet(EVAL_SET, index=False)
    print(f"[eval] wrote {len(df)} rows to {EVAL_SET}")
    return df


def _cache_path(condition: str) -> Path:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    return PHASE3_DIR / f"drafts_{condition}.jsonl"


def _load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {r["complaint_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def generate_drafts(drafter: Drafter, condition: str, use_retrieval: bool, sleep_s: float = 0.2) -> pd.DataFrame:
    df = pd.read_parquet(EVAL_SET)
    path = _cache_path(condition)
    done = _load_cache(path)
    with path.open("a", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            cid = int(r["complaint_id"])
            if cid in done:
                continue
            res = drafter.draft(r[C.TEXT], r["pred"], case_ref=r["case_ref"], use_retrieval=use_retrieval)
            row = {"complaint_id": cid, "condition": condition, "true_product": r[C.TARGET], "pred_product": r["pred"],
                   "narrative": r[C.TEXT], "draft": res.draft, "chunks": res.chunks, "citations": res.citations,
                   "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                   "cost_usd": res.cost_usd, "latency_s": res.latency_s}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            done[cid] = row
            time.sleep(sleep_s)
    print(f"[eval] {condition}: {len(done)} drafts (cost so far ${sum(r['cost_usd'] for r in done.values()):.3f})")
    return pd.DataFrame(done.values())


def judge_drafts(client, condition: str, model: str = JUDGE_MODEL, sleep_s: float = 0.2) -> pd.DataFrame:
    drafts = _load_cache(_cache_path(condition))
    path = PHASE3_DIR / f"judgments_{condition}.jsonl"
    done = _load_cache(path)
    with path.open("a", encoding="utf-8") as fh:
        for cid, r in drafts.items():
            if cid in done:
                continue
            excerpts = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(r["chunks"], 1)) or "(none)"
            content = (f"COMPLAINT:\n{r['narrative']}\n\nPOLICY EXCERPTS:\n{excerpts}\n\nDRAFT:\n{r['draft']}")
            resp = client.messages.create(model=model, max_tokens=2000, system=JUDGE_PROMPT,
                                          messages=[{"role": "user", "content": content}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            try:
                j = json.loads(re.search(r"\{.*?\}", text, re.DOTALL).group(0))
                j["hallucination"] = bool(j["hallucination"])
            except Exception:
                j = {"accuracy": None, "grounding": None, "tone": None, "hallucination": None,
                     "note": f"unparseable ({getattr(resp, 'stop_reason', '?')}): {text[:80]!r}"}
            row = {"complaint_id": cid, **j, "raw": text[:500], "stop_reason": getattr(resp, "stop_reason", None),
                   "judge_cost_usd": cost_usd(model, resp.usage.input_tokens, resp.usage.output_tokens)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            done[cid] = row
            time.sleep(sleep_s)
    print(f"[eval] judged {len(done)} {condition} drafts")
    return pd.DataFrame(done.values())


def summarize(conditions: tuple[str, ...] = ("rag", "no_rag")) -> pd.DataFrame:
    rows = []
    for cond in conditions:
        d = pd.DataFrame(_load_cache(_cache_path(cond)).values())
        jpath = PHASE3_DIR / f"judgments_{cond}.jsonl"
        if d.empty or not jpath.exists():
            continue
        j = pd.DataFrame(_load_cache(jpath).values())
        m = d.merge(j, on="complaint_id")
        rows.append({
            "condition": cond, "n": len(m),
            "accuracy": m["accuracy"].mean(), "grounding": m["grounding"].mean(), "tone": m["tone"].mean(),
            "hallucination_rate": m["hallucination"].astype(float).mean(),
            "citation_check_pass": m["citations"].apply(lambda c: c["ok"]).mean(),
            "cost_per_draft_usd": m["cost_usd"].mean(), "latency_s": m["latency_s"].mean(),
            "words_per_draft": m["draft"].str.split().str.len().mean(),
        })
        m.to_parquet(PHASE3_DIR / f"merged_{cond}.parquet", index=False)
    table = pd.DataFrame(rows).round(3)
    (PHASE3_DIR / "summary.md").write_text("# Phase 3 evaluation summary\n\n" + table.to_markdown(index=False) + "\n",
                                           encoding="utf-8")
    return table


def spot_check_sample(condition: str = "rag", n: int = 20, seed: int = 0) -> Path:
    """A markdown file of drafts to read by hand, alongside the judge's verdict."""
    m = pd.read_parquet(PHASE3_DIR / f"merged_{condition}.parquet").sample(n, random_state=seed)
    md = [f"# Spot check: {condition} ({n} drafts)\n", "Read each draft; mark AGREE/DISAGREE with the judge in the blank line.\n"]
    for _, r in m.iterrows():
        md += [f"---\n## {r['complaint_id']} | true: {r['true_product']} | pred: {r['pred_product']}\n",
               f"**Complaint:** {r['narrative'][:600]}\n", f"**Draft:**\n\n{r['draft']}\n",
               f"**Judge:** accuracy {r['accuracy']}, grounding {r['grounding']}, tone {r['tone']}, "
               f"hallucination {r['hallucination']} — {r['note']}\n", "**Your verdict:** \n"]
    out = PHASE3_DIR / f"spot_check_{condition}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out
