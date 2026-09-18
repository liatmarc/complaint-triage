"""Profile the clean dataset and write a data card (markdown + PNG figures).

The data card answers the questions a reviewer will ask before trusting a
model: how much data, how imbalanced, how long are the texts, and does the
distribution move over time.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def _save(fig: plt.Figure, name: str) -> Path:
    path = C.REPORTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame) -> Path:
    counts = df[C.TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(counts))))
    counts.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("complaints")
    ax.set_title("Class balance: product")
    return _save(fig, "class_balance.png")


def plot_narrative_length(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    df["narrative_len"].clip(upper=5000).plot.hist(bins=60, ax=ax)
    ax.set_xlabel("narrative length (chars, clipped at 5000)")
    ax.set_title("Narrative length distribution")
    return _save(fig, "narrative_length.png")


def plot_volume_over_time(df: pd.DataFrame) -> Path:
    monthly = df.set_index("date_received").resample("MS").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax)
    ax.set_ylabel("complaints / month")
    ax.set_title("Complaint volume over time")
    return _save(fig, "volume_over_time.png")


def plot_class_share_over_time(df: pd.DataFrame, top_k: int = 6) -> Path:
    """Shows label drift: do the top products keep the same share year to year?"""
    top = df[C.TARGET].value_counts().head(top_k).index
    yearly = (
        df.assign(year=df["date_received"].dt.year)
        .query(f"{C.TARGET} in @top")
        .groupby(["year", C.TARGET]).size()
        .unstack(fill_value=0)
    )
    share = yearly.div(yearly.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    share.plot(ax=ax)
    ax.set_ylabel("share of complaints")
    ax.set_title(f"Share of top-{top_k} products by year (label drift)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    return _save(fig, "class_share_over_time.png")


def summary_stats(df: pd.DataFrame) -> dict:
    counts = df[C.TARGET].value_counts()
    return {
        "rows": len(df),
        "date_min": df["date_received"].min().date(),
        "date_max": df["date_received"].max().date(),
        "n_classes": counts.size,
        "majority_class": counts.index[0],
        "majority_share": round(counts.iloc[0] / len(df), 3),
        "minority_class": counts.index[-1],
        "minority_count": int(counts.iloc[-1]),
        "imbalance_ratio": round(counts.iloc[0] / counts.iloc[-1], 1),
        "narrative_len_median": int(df["narrative_len"].median()),
        "narrative_len_p95": int(df["narrative_len"].quantile(0.95)),
        "duplicate_narratives": int(df["narrative"].duplicated().sum()),
        "n_companies": int(df["company"].nunique()),
        "missing_pct": (df.isna().mean() * 100).round(1).to_dict(),
    }


def write_data_card(df: pd.DataFrame, out: Path = C.DATA_CARD) -> Path:
    s = summary_stats(df)
    figs = [
        plot_class_balance(df),
        plot_narrative_length(df),
        plot_volume_over_time(df),
        plot_class_share_over_time(df),
    ]
    class_table = (
        df[C.TARGET].value_counts()
        .rename("count").to_frame()
        .assign(share=lambda t: (t["count"] / t["count"].sum()).round(3))
        .to_markdown()
    )
    missing_table = pd.Series(s["missing_pct"], name="missing %").to_frame().to_markdown()

    md = f"""# Data card: CFPB consumer complaints (narratives only)

**Source:** {C.CFPB_URL}
**Rows after cleaning:** {s['rows']:,}
**Date range:** {s['date_min']} to {s['date_max']}
**Target:** `{C.TARGET}` ({s['n_classes']} classes)

## Key facts
- Majority class: **{s['majority_class']}** ({s['majority_share']:.1%} of rows)
- Rarest class: **{s['minority_class']}** ({s['minority_count']:,} rows) — imbalance ratio {s['imbalance_ratio']}:1
- Narrative length: median {s['narrative_len_median']} chars, p95 {s['narrative_len_p95']} chars
- Exact-duplicate narratives: {s['duplicate_narratives']:,}
- Distinct companies: {s['n_companies']:,}

## Cleaning applied
- Kept only complaints with a consumer narrative of at least {C.MIN_NARRATIVE_CHARS} chars
- Dropped rows with missing product, complaint ID or unparseable date
- De-duplicated on complaint ID
- Validated against `schema.CleanSchema` (pandera)

## Known caveats
- Narratives are only present when the consumer opted in, so this is a biased subset of all complaints.
- CFPB has renamed and merged product categories over the years; see the drift chart below.
- Narratives are redacted by CFPB (`XXXX` tokens) — treat those as a feature, not noise.

## Class distribution
{class_table}

## Missing values
{missing_table}

## Figures
""" + "\n".join(f"![{p.stem}]({p.name})" for p in figs) + "\n"

    out.write_text(md)
    print(f"[profile] wrote {out}")
    return out
