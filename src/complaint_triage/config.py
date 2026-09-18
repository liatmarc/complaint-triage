"""Single source of truth for paths, URLs and column names.

Every other module imports from here so that renaming a column or moving a
directory is a one-line change.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. ROOT is the repo when running from a checkout; in a container the
# package is installed into site-packages, so PROJECT_ROOT points at /app.
# ---------------------------------------------------------------------------
ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # read-only or foreign filesystem (e.g. inside a container); paths still resolve

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
# Official CFPB bulk download (~1-2 GB zipped, several GB unzipped).
CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
RAW_ZIP = RAW_DIR / "complaints.csv.zip"
RAW_CSV_NAME = "complaints.csv"  # name inside the zip

# Exact column headers as they appear in the CFPB CSV.
class RawCols:
    DATE_RECEIVED = "Date received"
    PRODUCT = "Product"
    SUB_PRODUCT = "Sub-product"
    ISSUE = "Issue"
    SUB_ISSUE = "Sub-issue"
    NARRATIVE = "Consumer complaint narrative"
    COMPANY_RESPONSE_PUBLIC = "Company public response"
    COMPANY = "Company"
    STATE = "State"
    ZIP = "ZIP code"
    TAGS = "Tags"
    CONSENT = "Consumer consent provided?"
    SUBMITTED_VIA = "Submitted via"
    DATE_SENT = "Date sent to company"
    COMPANY_RESPONSE = "Company response to consumer"
    TIMELY = "Timely response?"
    DISPUTED = "Consumer disputed?"
    COMPLAINT_ID = "Complaint ID"


# Clean snake_case names used everywhere downstream.
COLUMN_MAP = {
    RawCols.DATE_RECEIVED: "date_received",
    RawCols.PRODUCT: "product",
    RawCols.SUB_PRODUCT: "sub_product",
    RawCols.ISSUE: "issue",
    RawCols.SUB_ISSUE: "sub_issue",
    RawCols.NARRATIVE: "narrative",
    RawCols.COMPANY: "company",
    RawCols.STATE: "state",
    RawCols.SUBMITTED_VIA: "submitted_via",
    RawCols.COMPANY_RESPONSE: "company_response",
    RawCols.TIMELY: "timely_response",
    RawCols.COMPLAINT_ID: "complaint_id",
}
KEEP_COLS = list(COLUMN_MAP.keys())

# Our prediction target for Phase 2.
TARGET = "product"
TEXT = "narrative"

# ---------------------------------------------------------------------------
# Processed outputs
# ---------------------------------------------------------------------------
CLEAN_PARQUET = PROCESSED_DIR / "complaints_clean.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
VAL_PARQUET = PROCESSED_DIR / "val.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
DATA_CARD = REPORTS_DIR / "data_card.md"

# Minimum narrative length (chars) worth keeping. Very short narratives are
# usually "see attached" or noise.
MIN_NARRATIVE_CHARS = 20

# Legacy CFPB product names -> current name. Applied during cleaning.
PRODUCT_MAP = {
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Credit card or prepaid card": "Credit card",
    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
}
