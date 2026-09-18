# Complaint Triage — end-to-end Applied AI project

Route incoming consumer complaints to the right product team (ML classification)
and draft a grounded first response (LLM + RAG), deployed as a monitored API.

Data: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
— millions of real complaints, public, no signup.

| Phase | Status | What it proves |
|---|---|---|
| 1. Data & framing | done | data engineering, validation, honest evaluation design |
| 2. Baselines & modeling | done | ML fundamentals, experiment tracking, error analysis |
| 3. LLM / RAG layer | done | LLM engineering and evaluation |
| 4. Deployment | service, Docker, CI done; hosting next | FastAPI, Docker, CI, monitoring |
| 5. Communication | | write-up, architecture, trade-offs |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                               # 4 tests, runs on synthetic data
```

## Phase 1 — run it

```bash
# Quick dev run: download (~1-2 GB), keep the first ~50k usable rows
python scripts/phase1.py all --max-rows 50000

# Full run (all narratives, ~1M+ rows, a few minutes)
python scripts/phase1.py all
```

Or step by step: `download-cmd`, `clean`, `profile`, `split`.

Outputs:

```
data/processed/complaints_clean.parquet   validated, deduplicated, narratives only
data/processed/{train,val,test}.parquet   time-based split
data/processed/split_meta.json            date ranges, class list, dropped classes
reports/data_card.md + *.png              class balance, text length, volume, label drift
```

## Phase 2 — run it

```bash
python scripts/phase2.py baseline            # TF-IDF + logistic regression, ~1 min on CPU
python scripts/phase2.py compare             # all runs side by side
python scripts/phase2.py explain             # top n-grams per class + per-prediction contributions
python scripts/phase2.py slices              # accuracy by narrative length, company, channel
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

Outputs: `models/tfidf_logreg.joblib`, `reports/phase2/*` (confusion matrix,
calibration plot, per-row predictions, the 50 most confident errors as CSV),
and every param/metric/artifact logged to MLflow (`mlflow.db`).

### Transformer (GPU via Google Colab)

No local GPU is needed. In Colab (Runtime -> Change runtime type -> T4 GPU):

1. Upload `scripts/colab_train.py` and, from `data/processed/`, the files
   `train.parquet`, `val.parquet`, `test.parquet`, `split_meta.json`.
2. Run:
   ```
   !pip -q install transformers datasets accelerate pyarrow scikit-learn
   !python colab_train.py --data-dir . --out-dir transformer_out
   !zip -r transformer_out.zip transformer_out
   ```
3. Download `transformer_out.zip`, extract it into the repo, then:
   ```bash
   python scripts/phase2.py log-transformer-cmd transformer_out
   python scripts/phase2.py compare
   ```
Scoring happens locally with the same `evaluate()` as the baseline.

## Phase 3 — run it

Needs an Anthropic API key. Copy `.env.example` to `.env` and fill in the key.

```bash
python scripts/phase3.py check-key          # one tiny call
python scripts/phase3.py demo               # draft one response, show retrieved evidence
python scripts/phase3.py build-eval-set-cmd # 100 stratified test complaints
python scripts/phase3.py generate           # drafts with & without retrieval (~$0.30)
python scripts/phase3.py judge              # LLM-as-judge, different model (~$0.80)
python scripts/phase3.py summarize-cmd      # reports/phase3/summary.md + spot-check file
```

Pipeline: redact PII -> classifier predicts product -> BM25 retrieval over a
policy corpus (`data/corpus/`, a fictional institution's playbook grounded in
FCRA / FDCPA / Reg E / Reg Z / RESPA) filtered to that product -> Claude Haiku
drafts with inline citations -> deterministic citation check (every "N days"
must appear in a cited chunk). Evaluation: 100 drafts with and without
retrieval, scored by Claude Sonnet on accuracy / grounding / tone +
hallucination flag, plus a 20-draft manual spot check.

## Phase 4 — run it

```bash
python scripts/phase4.py serve            # http://127.0.0.1:8000  (UI at /, OpenAPI docs at /docs)
python scripts/phase4.py smoke            # in a second terminal
python scripts/phase4.py load --n 200     # latency percentiles over real test complaints
python scripts/phase4.py monitor          # drift (PSI), confidence, latency from the prediction log

docker build -t complaint-triage .
docker run --rm -p 8000:8000 --env-file .env complaint-triage
```

Endpoints: `GET /health`, `POST /predict` (product, confidence, top-3, `needs_review`
if confidence < 0.6), `POST /draft` (predict + grounded draft + citation check +
`flag_for_review`). PII is redacted before anything is logged or sent to the LLM.
Every request is appended to `logs/predictions.jsonl`; `monitor` computes a
Population Stability Index between live predictions and the training label
distribution (<0.1 stable, >0.25 investigate).

CI (`.github/workflows/ci.yml`): lint + 20 tests on every push, then builds the
Docker image with a stub model and smoke-tests `/health` and `/predict` inside the
container. No real data or API keys are needed in CI.

## Design decisions (Phase 1)

- **Stream, don't load.** The raw CSV is several GB; `ingest.py` reads it in
  200k-row chunks and appends to a zstd-compressed Parquet.
- **Schema as a contract.** `schema.py` (pandera) validates every chunk:
  unique IDs, plausible dates, minimum narrative length, no surprise columns.
- **Temporal split.** Train on the past, validate on the recent past, test on
  the most recent 10%. A random split would leak future vocabulary and category
  trends and overstate accuracy.
- **Rare classes dropped** (< 50 training examples) and recorded in
  `split_meta.json` so the decision is auditable.
- **Data card**, not just a notebook: a reviewer can read one markdown file and
  know the dataset's size, imbalance, drift and caveats.

## Layout

```
src/complaint_triage/
  config.py    paths, URL, column names (single source of truth)
  ingest.py    download + chunked clean -> Parquet
  schema.py    pandera data contract
  profile.py   figures + data card
  split.py     temporal split
  evaluate.py  model-agnostic metrics, confusion/calibration plots, error sample
  baseline.py  TF-IDF + logistic regression with MLflow tracking
scripts/phase1.py   Phase 1 CLI
  transformer_log.py  score Colab predictions locally, log to MLflow
  explain.py   exact linear-model explanations + slice analysis
  retrieval.py chunking + BM25 retrieval with product filter
  guardrails.py PII redaction, citation / timeline check
  drafting.py  LLM drafting with cost + latency tracking
  llm_eval.py  eval set, ablation, LLM-as-judge, summary
data/corpus/        policy playbook (12 markdown files)
  api.py       FastAPI service + request logging
  ui.html      minimal browser UI
  monitoring.py PSI drift, confidence, latency report
scripts/phase3.py   Phase 3 CLI
scripts/phase4.py   serve / smoke / load / monitor
Dockerfile          multi-stage, CPU-only, non-root
.github/workflows/ci.yml
scripts/phase2.py   Phase 2 CLI
scripts/colab_train.py  standalone DistilBERT fine-tune (GPU)
tests/              synthetic-data tests
```
