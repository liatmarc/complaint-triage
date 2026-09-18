"""Train a tiny model on synthetic data so CI can build and smoke-test the image
without the real CFPB download. Never used for real predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_phase1 import make_raw

from complaint_triage import config as C
from complaint_triage.baseline import MODELS_DIR, TfidfLogReg
from complaint_triage.ingest import clean_chunk

df = clean_chunk(make_raw(600))
TfidfLogReg(max_word_features=5000, max_char_features=5000).fit(df[C.TEXT], df[C.TARGET]).save(MODELS_DIR / "tfidf_logreg.joblib")
print("stub model written")
