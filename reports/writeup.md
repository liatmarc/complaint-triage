# Complaint Triage: a one-page write-up

**Repo:** github.com/liatmarc/complaint-triage · **Live:** complaint-triage.onrender.com

## The problem

Financial institutions receive complaints as free text and must (a) route each one
to the team that owns the product and (b) reply within regulatory deadlines. Both
steps are manual, slow, and error-prone. I built a service that does the routing
automatically and gives the agent a grounded draft reply, then measured honestly
where it helps and where it doesn't.

## Data

The CFPB Consumer Complaint Database is public and large (millions of rows, several
GB). I streamed it in chunks into a validated Parquet file, kept complaints with a
consumer narrative, merged five legacy product labels into their current names
(11 classes, 35:1 imbalance), and split **by date** so the test set is the most
recent 10%: the model is scored on the future, the way it would be used. A leakage
guard removes test narratives that also appear in training, because credit-repair
firms file templated complaints in bulk (10% exact duplicates).

## Routing model

A TF-IDF (word + character n-gram) logistic regression reached **0.757 macro-F1 /
81.6% accuracy** on 4,828 held-out complaints, training in under five minutes on a
laptop. A fine-tuned DistilBERT, after tuning, tied it (0.756) at five times the
training cost on a GPU and 2.5× the inference latency. Reading the 50 most confident
errors by hand explained why: 40 were defensible alternative labels (a totalled car
reported as a repossession is both "vehicle loan" and "debt collection"), 2 were
unreadable, and only 8 were plain model mistakes. The ceiling is in the labels, so
I shipped the simpler, faster, better-calibrated model. Slice analysis showed
accuracy climbing with text length (72% → 85%) and fintechs (Block, PayPal) hardest,
because their products straddle the CFPB's categories.

## Drafting with retrieval

For each routed complaint, the service retrieves the policy page for that product
from a playbook (fictional institution, real federal rules), asks Claude Haiku to
draft a reply citing excerpts inline, and then checks deterministically that every
"N days" in the draft appears in a cited excerpt. On 100 test complaints judged by
a different model:

- no retrieval: **98%** of drafts invented policy, while sounding professional;
- top-4 BM25 chunks: 70%; the flagged "unsupported" timelines were correct rules
  from paragraphs that hadn't been retrieved, so recall, not ranking, was the problem;
- whole product page: 51%;
- plus a cite-the-source prompt: **41%**, grounding 3.8/5, $0.003 per draft.

A 20-draft manual spot check agreed with the judge 19 times. An earlier spot check
caught the judge silently returning empty replies on two-thirds of rows (a token
limit); fixing it moved the headline from a flattering 32% to the true 51%. That
correction is in the decision record on purpose.

41% is too high to send unreviewed, so the product is a **draft plus its evidence
plus a flag** for an agent, not an autoresponder. The citation check is a strong
triage signal: 80% hallucination when it fails, 24% when it passes.

## Deployment and operations

The classifier and drafter sit behind a typed FastAPI service with a small UI,
PII redaction on input, and a JSON-lines prediction log. It runs in a multi-stage,
non-root Docker image. Every push runs lint, 21 offline tests, a Docker build and a
container smoke test in GitHub Actions, then auto-deploys to Render. A monitoring
script computes a Population Stability Index between live predictions and the
training distribution, plus latency percentiles and LLM spend; the public demo has
a daily spend cap.

## What I'd do next

Re-run the model comparison on the full 1.5M-row dataset (no code changes needed;
the transformer would likely pull ahead); move to structured per-claim citations
to push hallucination under 20%; wire the drift alert to a retraining job; and
collect agent edits to measure the time the drafts actually save.

## Cost

Under $10 total: ~$5 in API calls, free GPU time on Colab, free hosting.
