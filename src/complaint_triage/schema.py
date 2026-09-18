"""Data contract for the cleaned complaints table.

If the CFPB changes their export, or a bug slips into cleaning, validation
fails loudly here instead of silently corrupting a model three steps later.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

from .config import MIN_NARRATIVE_CHARS

CleanSchema = pa.DataFrameSchema(
    {
        "complaint_id": Column(int, unique=True, nullable=False),
        "date_received": Column(
            "datetime64[ns]",
            Check(lambda s: s >= pd.Timestamp("2011-01-01"), error="date before CFPB existed"),
            nullable=False,
        ),
        "product": Column(str, nullable=False),
        "sub_product": Column(str, nullable=True),
        "issue": Column(str, nullable=True),
        "sub_issue": Column(str, nullable=True),
        "narrative": Column(
            str,
            Check(lambda s: s.str.len() >= MIN_NARRATIVE_CHARS, error="narrative too short"),
            nullable=False,
        ),
        "company": Column(str, nullable=True),
        "state": Column(str, nullable=True),
        "submitted_via": Column(str, nullable=True),
        "company_response": Column(str, nullable=True),
        "timely_response": Column(str, nullable=True),
        "narrative_len": Column(int, Check.ge(MIN_NARRATIVE_CHARS), nullable=False),
    },
    strict=True,   # no unexpected columns
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the (coerced) DataFrame. Raises SchemaError on failure."""
    return CleanSchema.validate(df, lazy=True)
