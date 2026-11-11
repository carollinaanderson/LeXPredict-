"""SQLite layer: schema, load, data-quality report and the ABT query."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

import pandas as pd

UNFAVORABLE_OUTCOMES = {"Procedente", "Parcialmente procedente"}
FAVORABLE_OUTCOMES = {"Improcedente", "Extinção sem resolução de mérito"}
TABLE_COLUMNS = [
    "case_id", "judge_id", "uf", "comarca", "vara", "filing_date", "decision_date",
    "claim_value", "sentence_outcome", "unfavorable", "sentence_text",
]  # fmt: skip


def _sql(name: str) -> str:
    return resources.files("claimvalue").joinpath("sql", name).read_text(encoding="utf-8")


def build_db(df: pd.DataFrame, db_path: Path) -> int:
    """(Re)create the cases table from an anonymized frame. Returns the number of rows loaded."""
    unknown = set(df["sentence_outcome"].unique()) - UNFAVORABLE_OUTCOMES - FAVORABLE_OUTCOMES
    if unknown:
        raise ValueError(f"unmapped sentence outcomes: {sorted(unknown)}")
    clean = (
        df.drop_duplicates("case_id")
        .assign(unfavorable=lambda d: d["sentence_outcome"].isin(UNFAVORABLE_OUTCOMES).astype(int))
        .loc[:, TABLE_COLUMNS]
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(_sql("01_schema.sql"))
        clean.to_sql("cases", con, if_exists="append", index=False)
    return len(clean)


def data_quality(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(_sql("02_data_quality.sql"), con)


def load_abt(db_path: Path, window_days: int = 180) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        abt = pd.read_sql_query(_sql("03_abt.sql"), con, params={"window_days": window_days})
    for col in ("filing_date", "decision_date"):
        abt[col] = pd.to_datetime(abt[col])
    return abt
