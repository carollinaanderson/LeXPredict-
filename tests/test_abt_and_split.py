import sqlite3

import numpy as np
import pandas as pd
import pytest
from pandas.errors import DatabaseError

from claimvalue.db import build_db, data_quality, load_abt
from claimvalue.features import temporal_split


def test_duplicates_are_dropped_on_load(raw, anon, db_path):
    assert len(raw) > raw["process_number"].nunique()  # synthetic export has duplicated rows
    with sqlite3.connect(db_path) as con:
        n = con.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    assert n == anon["case_id"].nunique()
    assert data_quality(db_path).loc[0, "n_cases"] == n


def test_rolling_features_use_only_past_decisions(db_path):
    """Recompute the window features with brute-force pandas and compare to the SQL ABT."""
    abt = load_abt(db_path, window_days=180)
    with sqlite3.connect(db_path) as con:
        cases = pd.read_sql_query(
            "SELECT * FROM cases", con, parse_dates=["filing_date", "decision_date"]
        )
    for _, row in abt.sample(40, random_state=1).iterrows():
        hist = cases[
            (cases["judge_id"] == row["judge_id"])
            & (cases["decision_date"] < row["filing_date"])
            & (cases["decision_date"] >= row["filing_date"] - pd.Timedelta(days=180))
        ]
        assert row["judge_n_prior"] == len(hist)
        if len(hist) and hist["claim_value"].notna().any():
            assert row["judge_avg_value_prior"] == pytest.approx(hist["claim_value"].mean())
        # a case never sees itself or anything decided after its filing date
        assert row["case_id"] not in set(hist["case_id"])


def test_window_changes_the_features(db_path):
    short = load_abt(db_path, 30)["judge_n_prior"].sum()
    long = load_abt(db_path, 365)["judge_n_prior"].sum()
    assert long > short


def test_temporal_split_has_no_overlap_and_no_look_ahead(db_path):
    abt = load_abt(db_path)
    train, test, cutoff = temporal_split(abt, 0.2, purge=True)
    assert train["filing_date"].max() < cutoff <= test["filing_date"].min()
    assert (train["decision_date"] < cutoff).all()  # no label that was unknown at the cutoff
    assert set(train["case_id"]).isdisjoint(test["case_id"])
    assert 0.1 < len(test) / len(abt) < 0.3


def test_schema_rejects_bad_rows(anon, tmp_path):
    bad = anon.copy()
    bad.loc[0, "claim_value"] = -5
    with pytest.raises(DatabaseError) as err:  # pandas wraps the sqlite3 error
        build_db(bad, tmp_path / "bad.db")
    assert isinstance(err.value.__cause__, sqlite3.IntegrityError)


def test_unknown_outcome_is_rejected(anon, tmp_path):
    bad = anon.copy()
    bad.loc[0, "sentence_outcome"] = "Desconhecido"
    with pytest.raises(ValueError):
        build_db(bad, tmp_path / "bad2.db")


def test_no_nan_leak_in_target_columns(db_path):
    abt = load_abt(db_path)
    assert abt["unfavorable"].notna().all()
    assert np.isfinite(abt["filing_month"]).all()
