\# Phase 3 decision: response drafting



\*\*Shipped configuration:\*\* Claude Haiku 4.5, full-page retrieval for the predicted

product (k=12, BM25 fallback for larger corpora), citation-disciplined prompt (v3),

deterministic citation check, human review before sending.



\## Ablation (100 stratified test complaints, judged by Claude Sonnet 5)



| Condition | Change | Accuracy | Grounding | Tone | Hallucination | Citation check pass | $/draft |

|---|---|---|---|---|---|---|---|

| no\_rag | no policy excerpts | 2.10 | 1.70 | 3.57 | 98% | 0% | 0.002 |

| rag | top-4 BM25 chunks | 3.06 | 3.24 | 4.29 | 70% | 54% | 0.002 |

| rag\_k12 | whole product page | 3.29 | 3.52 | 4.20 | 51% | 75% | 0.003 |

| rag\_v3 | + cite-the-source prompt | 3.51 | 3.80 | 4.29 | 41% | 77% | 0.003 |



Scores 1–5. Hallucination = any timeline, amount or step not supported by the excerpts.



\## What each step showed



\- \*\*Retrieval is necessary.\*\* Without excerpts the model invents policy 98% of the time

&#x20; while sounding professional (tone 3.6), which is why tone cannot be the metric.

\- \*\*Recall mattered more than ranking.\*\* The most-flagged "unsupported" timelines were

&#x20; correct policy from paragraphs BM25 had not retrieved. Giving the model the whole

&#x20; product page (≈1,500 tokens) cut hallucinations by 19 points for $0.001 per draft.

\- \*\*Remaining errors were miscitations.\*\* Correct numbers cited to the wrong excerpt.

&#x20; A prompt instruction to cite the excerpt containing the number removed another 10 points.

\- \*\*The citation check is a usable triage signal:\*\* \~80% hallucination when it fails,

&#x20; \~24% when it passes. It runs with no tokens and flags drafts for closer review.



\## Judge reliability



A manual spot check of 20 drafts agreed with the judge's hallucination verdict on 19

(95%). An earlier spot check exposed that the judge was returning empty or truncated

replies on two-thirds of rows because of a 300-token limit; the summary had been

silently computed on the remaining third and overstated performance (32% vs the true

51%). Fixed by raising the limit and recording the raw reply and stop reason.



\## Decision



41% hallucination is too high for unreviewed sending, so drafts are delivered to an

agent with the citation-check flag and the retrieved excerpts shown alongside. The

draft saves the agent the structure and the policy lookup; the agent verifies the

numbers. Further reductions are available (a stricter judge-in-the-loop rewrite,

structured output with per-claim citations) but were not needed for this deployment.



\## Cost



Whole evaluation: \~500 drafter calls and \~400 judge calls, under $5 total.

