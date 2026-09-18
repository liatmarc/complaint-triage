\# Phase 2 decision: model selection



\*\*Selected model: TF-IDF (word + character n-grams) + logistic regression.\*\*



\## Comparison



Time-based test set: 4,828 complaints, 11 product classes, most recent \~10% of data.



| Run | Val macro-F1 | Test macro-F1 | Test acc | Test log loss | Latency (ms/row) | Train time |

|---|---|---|---|---|---|---|

| tfidf\_logreg (C=4) | 0.754 | \*\*0.757\*\* | 0.816 | \*\*0.599\*\* | \*\*1.7\*\* (CPU) | \*\*285 s\*\* (CPU) |

| tfidf\_logreg\_c1 (C=1) | 0.750 | 0.747 | 0.811 | 0.669 | 3.0 (CPU) | 311 s (CPU) |

| distilbert (384 tok, 2 ep, lr 3e-5) | 0.725 | 0.731 | 0.803 | 0.655 | 3.1 (T4 GPU) | 681 s (T4 GPU) |

| distilbert\_v2 (512 tok, 3 ep, lr 5e-5) | 0.745 | 0.756 | \*\*0.821\*\* | 0.614 | 4.3 (T4 GPU) | 1,394 s (T4 GPU) |



\## Error analysis



The 50 most confident baseline errors on the test set were reviewed by hand:



| Bucket | Count | Meaning |

|---|---|---|

| Label debatable | 40 | Model's prediction is a defensible reading; complaint spans two products |

| Text ambiguous | 2 | Product cannot be determined from the narrative |

| Model wrong | 8 | Text clearly indicates the true label; model missed it |



Example (label: Debt collection, predicted: Vehicle loan or lease): a paid-off,

totaled car reported as a repossession. The label reflects which company was

complained about, which is not visible in the text.



\## Decision



The tuned DistilBERT matched the baseline on macro-F1 (0.756 vs 0.757) and gained

half a point of accuracy, at 5x the training time on a GPU, 2.5x the inference

latency, and worse validation and log-loss scores. With 84% of confident errors

attributable to label ambiguity rather than model capacity, the remaining

headroom is mostly in the labels. The baseline is selected for deployment:

cheaper, faster, better calibrated, and explainable.



\## Caveat



This comparison used a \~51k-row recent slice of the CFPB data. On the full

\~1.5M-row dataset a transformer would likely pull ahead. The pipeline requires

no code changes to re-run the comparison at full scale.

\## Slice Analysis

Accuracy rises with narrative length (72% under 200 chars, 85% above 2,000), so a confidence 

threshold with human review will mostly catch short complaints. By company, fintechs (Block 63%,

 PayPal 65%) are markedly harder than banks (80–89%) because their products straddle several 
 
 CFPB categories; this matches the error review, where 80% of confident errors were defensible
 
alternative labels.
