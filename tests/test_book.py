"""Tests for the ABJ book data path (``abjData::consumo``).

The tests use a small **fictional** table with the same schema. They never read
the real ABJ file: the real data is downloaded at runtime into a gitignored
directory and is not part of the repository.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from claimvalue.anonymize import assert_no_pii
from claimvalue.book_data import (
    COLUMNS,
    fetch_book_data,
    load_prepared,
    prepare_book_frame,
    validate,
)
from claimvalue.book_db import book_data_quality, build_book_db, load_book_abt
from claimvalue.book_features import FEATURES, TARGETS, temporal_split
from claimvalue.book_train import paired_baseline_test
from claimvalue.train import _candidates


@pytest.fixture
def prepared() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2020-01-01")
    for i in range(24):
        filing = start + pd.Timedelta(days=30 * i)
        decision = filing + pd.Timedelta(days=60)
        if i == 23:
            outcome, reformed = "Desistência", pd.NA
        elif i % 3 == 0:
            outcome, reformed = "Reformou", 1
        elif i % 3 == 1:
            outcome, reformed = "Parcial", 1
        else:
            outcome, reformed = "Não reformou", 0
        rows.append(
            {
                "case_id": f"BOOK_{i:016X}",
                "assunto": ["Bancários", "Telefonia", "Fornecimento de Energia"][i % 3],
                "comarca": ["SAO PAULO", "CAMPINAS", "SANTOS"][i % 3],
                "tipo_litigio": ["PF-nPF", "nPF-PF", "PF-PF"][i % 3],
                "filing_date": filing.strftime("%Y-%m-%d"),
                "decision_date": decision.strftime("%Y-%m-%d"),
                "claim_value": float(1000 + 500 * i),
                "decision_outcome": outcome,
                "reformed": reformed,
                "unanimity": "Unânime",
                "case_days": 60,
            }
        )
    df = pd.DataFrame(rows).loc[:, COLUMNS]
    df["reformed"] = df["reformed"].astype("Int64")
    validate(df)
    return df


@pytest.fixture
def book_db(prepared, tmp_path):
    path = tmp_path / "book.db"
    assert build_book_db(prepared, path) == 24
    return path


def test_prepared_schema_and_privacy(prepared):
    assert list(prepared.columns) == COLUMNS
    assert prepared["case_id"].is_unique
    assert prepared["case_id"].str.startswith("BOOK_").all()
    assert_no_pii(prepared)


def test_labels_and_dates(prepared):
    assert prepared["reformed"].notna().sum() == 23
    assert (prepared["reformed"] == 1).sum() == 16
    assert (prepared["reformed"] == 0).sum() == 7
    assert prepared["claim_value"].gt(0).all()
    filing = pd.to_datetime(prepared["filing_date"])
    decision = pd.to_datetime(prepared["decision_date"])
    assert (decision >= filing).all()
    assert (prepared["case_days"] >= 0).all()


def test_book_sql_and_abt(book_db):
    checks = book_data_quality(book_db).set_index("check_name")["value"]
    assert checks["rows"] == "24"
    assert checks["binary_label_rows"] == "23"
    assert checks["decision_before_filing"] == "0"

    abt = load_book_abt(book_db, window_days=365)
    assert len(abt) == 23
    assert abt["reformed"].notna().all()
    assert {"filing_year", "claim_value_log", "comarca_n_prior", "assunto_n_prior"} <= set(
        abt.columns
    )
    assert (abt["comarca_n_prior"] >= 0).all()
    assert (abt["assunto_n_prior"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(abt["filing_date"])


def test_temporal_split_and_purge(book_db):
    abt = load_book_abt(book_db)
    train, test, cutoff = temporal_split(abt, test_frac=0.2)
    assert len(train) + len(test) == len(abt)
    assert train["filing_date"].max() <= cutoff
    assert test["filing_date"].min() >= cutoff

    train_p, test_p, cutoff_p = temporal_split(abt, test_frac=0.2, purge=True)
    assert len(test_p) == len(test)
    assert len(train_p) <= len(train)
    assert train_p["decision_date"].max() < cutoff_p


def test_features_never_leak_the_label_or_post_decision_fields():
    for task, features in FEATURES.items():
        assert TARGETS[task] not in features
        assert "case_days" not in features
        assert "unanimity" not in features
    assert "claim_value_log" in FEATURES["classification"]
    assert "claim_value_log" not in FEATURES["regression"]


def test_prepare_book_frame_maps_outcomes_and_pseudonymises_ids():
    raw = pd.DataFrame(
        {
            "id_processo": ["10245108320208260576", "10768308320208260100", "10000000000000000001"],
            "assunto": ["Bancários", "Telefonia", "Bancários"],
            "comarca": ["SAO PAULO", "CAMPINAS", "SANTOS"],
            "valor": [1000.0, 2000.0, 3000.0],
            "tipo_litigio": ["PF-nPF", "nPF-PF", "PF-PF"],
            "dec_val": ["Reformou", "Não reformou", "Desistência"],
            "dec_unanime": ["Unânime", "Maioria", "Unânime"],
            "dec_date": ["2021-01-10", "2021-02-10", "2021-03-10"],
            "tempo": [10, 20, 30],
        }
    )
    out = prepare_book_frame(raw)

    assert len(out) == 3
    assert out["case_id"].is_unique
    assert out["case_id"].str.startswith("BOOK_").all()
    assert not set(out["case_id"]) & set(raw["id_processo"])
    assert out["reformed"].tolist()[0] == 1
    assert out["reformed"].tolist()[1] == 0
    assert pd.isna(out["reformed"].tolist()[2])
    assert out.loc[0, "filing_date"] == "2020-12-31"
    assert out.loc[0, "decision_date"] == "2021-01-10"


def test_fetch_uses_cached_prepared_file(prepared, tmp_path):
    out = tmp_path / "book"
    out.mkdir()
    prepared.to_csv(out / "consumo_prepared.csv", index=False)
    result = fetch_book_data(out_dir=out)
    assert result["cached"] is True
    assert result["rows"] == len(prepared)


def test_load_prepared_requires_a_runtime_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "raw"))
    with pytest.raises(FileNotFoundError, match="book-fetch"):
        load_prepared()


def test_candidate_set_includes_lightgbm_and_xgboost():
    for task in ("regression", "classification"):
        names = _candidates(task)
        assert "lightgbm" in names
        if importlib.util.find_spec("xgboost"):
            assert "xgboost" in names


def test_paired_baseline_test_returns_a_bounded_p_value():
    y = pd.Series([0, 1, 0, 1])
    pred = np.array([0.1, 0.9, 0.2, 0.8])
    baseline = np.array([0.5, 0.5, 0.5, 0.5])
    result = paired_baseline_test("classification", y, pred, baseline)
    assert result["test"] == "wilcoxon_signed_rank"
    assert result["metric"] == "brier_score_per_case"
    assert result["n_pairs"] == 4
    assert 0.0 <= result["p_value"] <= 1.0
