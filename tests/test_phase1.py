"""Phase 1 tests on a small synthetic CSV shaped exactly like the CFPB export."""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from complaint_triage import config as C
from complaint_triage.ingest import build_clean_parquet, clean_chunk
from complaint_triage.profile import summary_stats, write_data_card
from complaint_triage.split import temporal_split

PRODUCTS = ["Credit reporting", "Debt collection", "Mortgage", "Credit card", "Student loan"]


def make_raw(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2024-12-31", periods=n)
    products = rng.choice(PRODUCTS, size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    narratives = [
        f"Complaint {i}: I have a problem with my {p.lower()} account and XXXX did not respond "
        * rng.integers(1, 6)
        for i, p in enumerate(products)
    ]
    # Inject rows that should be dropped: missing narrative, too short, bad date
    narratives[0] = None
    narratives[1] = "short"
    raw = pd.DataFrame({
        C.RawCols.DATE_RECEIVED: dates.strftime("%Y-%m-%d"),
        C.RawCols.PRODUCT: products,
        C.RawCols.SUB_PRODUCT: "Generic",
        C.RawCols.ISSUE: "Issue",
        C.RawCols.SUB_ISSUE: None,
        C.RawCols.NARRATIVE: narratives,
        C.RawCols.COMPANY_RESPONSE_PUBLIC: None,
        C.RawCols.COMPANY: rng.choice(["BANK A", "BANK B", "BANK C"], size=n),
        C.RawCols.STATE: "MD",
        C.RawCols.ZIP: "210XX",
        C.RawCols.TAGS: None,
        C.RawCols.CONSENT: "Consent provided",
        C.RawCols.SUBMITTED_VIA: "Web",
        C.RawCols.DATE_SENT: dates.strftime("%Y-%m-%d"),
        C.RawCols.COMPANY_RESPONSE: "Closed with explanation",
        C.RawCols.TIMELY: "Yes",
        C.RawCols.DISPUTED: None,
        C.RawCols.COMPLAINT_ID: np.arange(1_000_000, 1_000_000 + n),
    })
    raw.loc[2, C.RawCols.DATE_RECEIVED] = "not a date"
    return raw


@pytest.fixture
def raw_zip(tmp_path):
    raw = make_raw()
    csv_path = tmp_path / C.RAW_CSV_NAME
    raw.to_csv(csv_path, index=False)
    zip_path = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=C.RAW_CSV_NAME)
    return zip_path


def test_clean_chunk_drops_bad_rows():
    df = clean_chunk(make_raw(100))
    assert len(df) == 97                     # 3 injected bad rows removed
    assert (df["narrative_len"] >= C.MIN_NARRATIVE_CHARS).all()
    assert df["date_received"].dtype.kind == "M"
    assert set(df.columns) == set(C.COLUMN_MAP.values()) | {"narrative_len"}


def test_build_clean_parquet_roundtrip(raw_zip, tmp_path):
    out = tmp_path / "clean.parquet"
    build_clean_parquet(raw_zip=raw_zip, out=out, chunksize=500)
    df = pd.read_parquet(out)
    assert len(df) == 1997
    assert df["complaint_id"].is_unique


def test_temporal_split_is_ordered_and_disjoint():
    df = clean_chunk(make_raw(2000))
    train, val, test, meta = temporal_split(df, min_class_count=10)
    assert train["date_received"].max() <= val["date_received"].min()
    assert val["date_received"].max() <= test["date_received"].min()
    assert not set(train["complaint_id"]) & set(test["complaint_id"])
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] <= len(df)
    assert meta["leaked_removed_test"] == 0
    assert meta["unseen_in_test"] == []


def test_data_card_written(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPORTS_DIR", tmp_path)
    df = clean_chunk(make_raw(500))
    out = write_data_card(df, out=tmp_path / "data_card.md")
    text = out.read_text()
    assert "Class distribution" in text
    assert (tmp_path / "class_balance.png").exists()
    s = summary_stats(df)
    assert s["n_classes"] == len(PRODUCTS)
