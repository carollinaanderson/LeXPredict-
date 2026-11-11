"""Feature definitions, preprocessing and the time-aware split."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from claimvalue.config import SEED

CAT_LOW = ["uf", "vara"]
# High-cardinality ids: encoded with a smoothed, cross-fitted target encoding INSIDE the model
# pipeline. This replaces "judge name -> arbitrary integer id", which made the tree treat the id
# as an ordered magnitude.
CAT_HIGH = ["comarca", "judge_id"]
NUMERIC = [
    "filing_month",
    "judge_n_prior", "judge_avg_value_prior", "judge_unfav_rate_prior",
    "comarca_n_prior", "comarca_avg_value_prior", "comarca_unfav_rate_prior",
]  # fmt: skip
FEATURES = CAT_LOW + CAT_HIGH + NUMERIC


def make_preprocessor(target_type: str) -> ColumnTransformer:
    """target_type: 'continuous' (regression) or 'binary' (classification)."""
    return ColumnTransformer(
        [
            (
                "low",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
                CAT_LOW,
            ),
            (
                "high",
                TargetEncoder(target_type=target_type, smooth="auto", random_state=SEED),
                CAT_HIGH,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )


def temporal_split(
    df: pd.DataFrame, test_frac: float = 0.2, purge: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Train on the past, test on the future (no random shuffle).

    ``purge=True`` additionally drops training cases whose outcome was only known AFTER the cutoff
    (decision_date >= cutoff). Required when the label is the ruling itself.
    """
    df = df.sort_values(["filing_date", "case_id"], kind="stable").reset_index(drop=True)
    cutoff = df.loc[int(len(df) * (1 - test_frac)), "filing_date"]
    train = df[df["filing_date"] < cutoff]
    if purge:
        train = train[train["decision_date"] < cutoff]
    test = df[df["filing_date"] >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
