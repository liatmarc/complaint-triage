"""Remove unparseable judge rows from the cache so `judge` re-scores only those."""
import json

from complaint_triage.llm_eval import PHASE3_DIR

for cond in ("rag", "rag_k12", "no_rag"):
    path = PHASE3_DIR / f"judgments_{cond}.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [r for r in rows if r.get("hallucination") is not None]
    path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    print(f"{cond}: kept {len(keep)}, dropped {len(rows) - len(keep)}")
