"""SQLite layer for the ABJ book dataset (``abjData::consumo``)."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from claimvalue.book_data import COLUMNS, validate


def _sql(name: str) -> str:
    return resources.files("claimvalue").joinpath("sql", name).read_text(encoding="utf-8")


def build_book_db(df: pd.DataFrame, db_path: Path) -> int:
    """(Re)create ``book_cases`` from the prepared frame. Returns rows loaded."""
    validate(df)
    clean = df.drop_duplicates("case_id").loc[:, COLUMNS].copy()
    clean["reformed"] = clean["reformed"].astype("Int64")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(_sql("01_book_schema.sql"))
        clean.to_sql("book_cases", con, if_exists="append", index=False)
    return len(clean)


def book_data_quality(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(_sql("02_book_quality.sql"), con)


def load_book_abt(db_path: Path, window_days: int = 365) -> pd.DataFrame:
    """Load the leakage-safe ABT and add the log claim value used by the classifier."""
    with sqlite3.connect(db_path) as con:
        abt = pd.read_sql_query(_sql("03_book_abt.sql"), con, params={"window_days": window_days})
    for col in ("filing_date", "decision_date"):
        abt[col] = pd.to_datetime(abt[col])
    abt["filing_year"] = abt["filing_date"].dt.year
    abt["claim_value_log"] = np.log1p(abt["claim_value"].astype(float))
    abt["reformed"] = abt["reformed"].astype("Int64")
    return abt
