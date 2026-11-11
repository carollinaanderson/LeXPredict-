"""Feature set for the ABJ book dataset (``abjData::consumo``).

The book's methodology separates what is known when an appeal is filed from what is
only known after the decision. Every feature below is available at filing time:

* ``assunto``, ``comarca`` and ``tipo_litigio`` are case metadata;
* ``filing_month`` is the filing date;
* the ``*_prior`` features are built by ``03_book_abt.sql`` from cases whose decision
  was already published before this filing;
* ``claim_value_log`` is the value of the cause, known at filing, and is used only by
  the classifier (it is the regression target, never a regressor input).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from claimvalue.config import SEED

CAT_LOW = ["assunto", "tipo_litigio"]
CAT_HIGH = ["comarca"]
NUMERIC = [
    "filing_month",
    "filing_year",
    "comarca_n_prior",
    "comarca_avg_value_prior",
    "comarca_reform_rate_prior",
    "assunto_n_prior",
    "assunto_avg_value_prior",
    "assunto_reform_rate_prior",
]
FEATURES = {
    "regression": CAT_LOW + CAT_HIGH + NUMERIC,
    "classification": CAT_LOW + CAT_HIGH + NUMERIC + ["claim_value_log"],
}
TARGETS = {"regression": "claim_value", "classification": "reformed"}


def make_preprocessor(task: str) -> ColumnTransformer:
    """Encoder fitted inside each CV fold, exactly as in the synthetic pipeline."""
    target_type = "continuous" if task == "regression" else "binary"
    return ColumnTransformer(
        [
            (
                "low",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
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
    """Train on past filings, test on the future; optionally purge by decision date.

    ``purge`` drops training appeals whose decision date falls inside the test window,
    because their label was only known after the cutoff. Required for classification.
    """
    out = df.sort_values(["filing_date", "case_id"], kind="stable").reset_index(drop=True)
    cutoff = out.loc[int(len(out) * (1 - test_frac)), "filing_date"]
    train = out[out["filing_date"] < cutoff]
    if purge:
        train = train[train["decision_date"] < cutoff]
    test = out[out["filing_date"] >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
