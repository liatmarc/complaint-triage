import pandas as pd

from complaint_triage.llm_eval import PHASE3_DIR

m = pd.read_parquet(PHASE3_DIR / "merged_rag_k12.parquet")
m["product_correct"] = m["true_product"] == m["pred_product"]
m["cite_ok"] = m["citations"].apply(lambda c: c["ok"])
m["halluc"] = m["hallucination"].astype(float)

print("Hallucination rate by classifier correctness")
print(m.groupby("product_correct")["halluc"].agg(["mean", "size"]).round(3), "\n")
print("Hallucination rate by citation-check result")
print(m.groupby("cite_ok")["halluc"].agg(["mean", "size"]).round(3), "\n")
print("Hallucination rate by product")
print(m.groupby("pred_product")["halluc"].agg(["mean", "size"]).sort_values("mean", ascending=False).round(3), "\n")
print("Most common unsupported timelines")
print(m["citations"].apply(lambda c: c["unsupported_timelines"]).explode().value_counts().head(10))
