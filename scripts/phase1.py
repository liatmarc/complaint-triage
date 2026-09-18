"""Phase 1 pipeline: download -> clean -> profile -> temporal split.

Usage:
    python scripts/phase1.py all                 # full run (needs the CFPB zip)
    python scripts/phase1.py all --max-rows 50000  # quick dev run
    python scripts/phase1.py download
    python scripts/phase1.py clean --max-rows 50000
    python scripts/phase1.py profile
    python scripts/phase1.py split
"""
from __future__ import annotations

import typer

from complaint_triage import config as C
from complaint_triage.ingest import build_clean_parquet, download, load_clean
from complaint_triage.profile import write_data_card
from complaint_triage.split import write_splits

app = typer.Typer(add_completion=False)


@app.command()
def download_cmd(force: bool = False):
    """Download the CFPB bulk zip."""
    download(force=force)


@app.command()
def clean(max_rows: int | None = None, chunksize: int = 200_000):
    """Stream the raw CSV into a validated Parquet file."""
    build_clean_parquet(chunksize=chunksize, max_rows=max_rows)


@app.command()
def profile():
    """Write reports/data_card.md and figures."""
    write_data_card(load_clean())


@app.command()
def split(val_frac: float = 0.10, test_frac: float = 0.10, min_class_count: int = 50):
    """Write train/val/test Parquet files using a time-based split."""
    write_splits(load_clean(), val_frac=val_frac, test_frac=test_frac,
                 min_class_count=min_class_count)


@app.command()
def all(max_rows: int | None = None):
    """Run every step."""
    download()
    build_clean_parquet(max_rows=max_rows)
    df = load_clean()
    write_data_card(df)
    write_splits(df)
    typer.echo(f"\nDone. Data card: {C.DATA_CARD}")


if __name__ == "__main__":
    app()
