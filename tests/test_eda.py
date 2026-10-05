import pandas as pd

from claimvalue.db import load_abt
from claimvalue.eda import (
    id_like_columns,
    missingness,
    overview,
    variance_explained,
    write_eda_report,
)


def test_id_like_columns_catches_identifiers(db_path):
    abt = load_abt(db_path)
    assert "case_id" in id_like_columns(abt)  # would let a model memorise rows
    assert "uf" not in id_like_columns(abt)


def test_variance_explained_is_a_share(db_path):
    ve = variance_explained(load_abt(db_path), "claim_value", ["judge_id", "uf"])
    assert ve["eta_squared"].between(0, 1).all()
    assert ve.iloc[0]["column"] == "judge_id"  # finer grouping always explains at least as much


def test_overview_and_missingness(db_path):
    abt = load_abt(db_path)
    assert overview(abt).loc["claim_value", "missing_pct"] > 0
    assert isinstance(missingness(abt, "claim_value", "uf"), pd.DataFrame)


def test_eda_report_is_written(db_path, tmp_path):
    out = write_eda_report(load_abt(db_path), tmp_path / "eda")
    assert (out / "EDA_REPORT.md").read_text().startswith("# EDA report")
    assert (out / "target_distribution.png").stat().st_size > 1000
