import json

from complaint_triage.llm_eval import PHASE3_DIR

for cond in ("rag", "rag_k12", "no_rag"):
    path = PHASE3_DIR / f"judgments_{cond}.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = [r for r in rows if r.get("hallucination") is None]
    print(f"{cond}: {len(bad)}/{len(rows)} unparseable")
    for r in bad[:3]:
        print("   ", r.get("note"))
