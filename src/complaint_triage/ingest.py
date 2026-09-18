"""Download the CFPB bulk export and produce a validated Parquet file.

The raw CSV is several GB, so we never load it whole: we stream it in chunks,
keep only rows with a narrative, and append to a Parquet writer.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from . import config as C
from .schema import validate


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download(url: str = C.CFPB_URL, dest: Path = C.RAW_ZIP, force: bool = False) -> Path:
    """Stream the zip to disk with a progress bar. Skips if already present."""
    if dest.exists() and not force:
        print(f"[download] {dest} already exists, skipping")
        return dest
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------
# 2. Clean one chunk
# ---------------------------------------------------------------------------
def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filter to usable rows and normalise types/column names."""
    df = chunk[C.KEEP_COLS].rename(columns=C.COLUMN_MAP)

    # Keep only complaints where the consumer wrote a narrative.
    df = df.dropna(subset=["narrative", "product", "complaint_id"])
    df["narrative"] = df["narrative"].astype(str).str.strip()
    df = df[df["narrative"].str.len() >= C.MIN_NARRATIVE_CHARS]

    df["date_received"] = pd.to_datetime(df["date_received"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date_received"])

    df["complaint_id"] = df["complaint_id"].astype(int)
    df["narrative_len"] = df["narrative"].str.len().astype(int)

    # Normalise free-text categoricals a little.
    for col in ("product", "sub_product", "issue", "sub_issue", "company", "state",
                "submitted_via", "company_response", "timely_response"):
        df[col] = df[col].astype("string").str.strip()

    df["product"] = df["product"].replace(C.PRODUCT_MAP)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Stream CSV -> Parquet
# ---------------------------------------------------------------------------
def build_clean_parquet(
    raw_zip: Path = C.RAW_ZIP,
    out: Path = C.CLEAN_PARQUET,
    chunksize: int = 200_000,
    max_rows: int | None = None,
) -> Path:
    """Read the CSV inside the zip chunk-by-chunk and write a single Parquet.

    max_rows: stop after roughly this many *kept* rows (handy for quick dev runs).
    """
    print(f"[ingest] streaming {raw_zip} -> {out}")
    writer: pq.ParquetWriter | None = None
    kept = 0
    seen_ids: set[int] = set()

    with zipfile.ZipFile(raw_zip) as z, z.open(C.RAW_CSV_NAME) as fh:
        reader = pd.read_csv(fh, chunksize=chunksize, dtype=str, low_memory=False)
        for chunk in tqdm(reader, unit="chunk"):
            df = clean_chunk(chunk)
            # Bulk export occasionally repeats IDs across chunks; drop them.
            df = df[~df["complaint_id"].isin(seen_ids)]
            seen_ids.update(df["complaint_id"].tolist())
            if df.empty:
                continue
            df = validate(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            kept += len(df)
            if max_rows and kept >= max_rows:
                break

    if writer is not None:
        writer.close()
    print(f"[ingest] wrote {kept:,} rows to {out}")
    return out


def load_clean(path: Path = C.CLEAN_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)
