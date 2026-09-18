"""Time-based train / validation / test split.

Why not a random split? In production the model is trained on the past and
scores the future. A random split leaks future wording, company names and
category trends into training and inflates every metric. Splitting on
`date_received` gives an honest estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C


def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    min_class_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Sort by date; the most recent `test_frac` is test, the slice before it is val.

    Classes with fewer than `min_class_count` rows in train are dropped from all
    three splits — a model can't learn them and they distort macro-F1.
    """
    df = df.sort_values("date_received").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))

    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]

    # Drop exact-duplicate narratives within train, and remove any val/test
    # narrative that also appears in train (would be leakage).
    n_val_before, n_test_before = len(val), len(test)
    train = train.drop_duplicates(subset=[C.TEXT])
    seen = set(train[C.TEXT])
    val = val[~val[C.TEXT].isin(seen)]
    test = test[~test[C.TEXT].isin(seen)]
    leaked_val = n_val_before - len(val)
    leaked_test = n_test_before - len(test)

    keep = train[C.TARGET].value_counts()
    keep = keep[keep >= min_class_count].index
    dropped = sorted(set(df[C.TARGET].unique()) - set(keep))
    train, val, test = (s[s[C.TARGET].isin(keep)] for s in (train, val, test))

    meta = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_dates": [str(train["date_received"].min().date()), str(train["date_received"].max().date())],
        "val_dates": [str(val["date_received"].min().date()), str(val["date_received"].max().date())],
        "test_dates": [str(test["date_received"].min().date()), str(test["date_received"].max().date())],
        "classes": sorted(keep.tolist()),
        "dropped_classes": dropped,
        "unseen_in_test": sorted(set(test[C.TARGET]) - set(train[C.TARGET])),
        "leaked_removed_val": int(leaked_val),
        "leaked_removed_test": int(leaked_test),
    }
    return train, val, test, meta


def write_splits(df: pd.DataFrame, out_dir: Path = C.PROCESSED_DIR, **kw) -> dict:
    train, val, test, meta = temporal_split(df, **kw)
    train.to_parquet(C.TRAIN_PARQUET, index=False)
    val.to_parquet(C.VAL_PARQUET, index=False)
    test.to_parquet(C.TEST_PARQUET, index=False)
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[split] train={meta['n_train']:,} val={meta['n_val']:,} test={meta['n_test']:,}")
    print(f"[split] test covers {meta['test_dates'][0]} -> {meta['test_dates'][1]}")
    if meta["dropped_classes"]:
        print(f"[split] dropped rare classes: {meta['dropped_classes']}")
    return meta
