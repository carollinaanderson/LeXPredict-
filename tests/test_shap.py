"""Tests for the SHAP explanation layer.

Strategy: keep the models tiny (a 30-tree forest on ~200 rows) so the exact TreeExplainer
runs in milliseconds. The assertions check the two things that make SHAP trustworthy:

* additivity: base + sum(contributions) == model output for every case;
* artefacts: every promised file exists and the per-case CSV parses back.

Skipped cleanly when the ``explain`` extra (shap) is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from claimvalue.features import FEATURES, temporal_split
from claimvalue.shap_explain import TARGET, explain_model, unwrap
from claimvalue.train import _wrap

shap = pytest.importorskip("shap")

N = 240


@pytest.fixture()
def cases() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "case_id": np.arange(N),
            "uf": rng.choice(["SP", "RJ", "MG"], N),
            "vara": rng.choice(["cível", "tributário"], N),
            "comarca": rng.choice(["capital", "interior"], N),
            "judge_id": rng.choice([f"j{i}" for i in range(8)], N),
            "filing_date": pd.date_range("2021-01-01", periods=N, freq="D"),
            "decision_date": pd.date_range("2021-02-01", periods=N, freq="D"),
            "filing_month": rng.integers(1, 13, N),
            "judge_n_prior": rng.integers(0, 40, N).astype(float),
            "judge_avg_value_prior": rng.gamma(2, 3000, N),
            "judge_unfav_rate_prior": rng.uniform(0.2, 0.6, N),
            "comarca_n_prior": rng.integers(0, 90, N).astype(float),
            "comarca_avg_value_prior": rng.gamma(2, 3000, N),
            "comarca_unfav_rate_prior": rng.uniform(0.2, 0.6, N),
        }
    )
    signal = (df["judge_avg_value_prior"] / 6000) + (df["uf"] == "SP") * 0.5
    df["claim_value"] = 5000 * (1 + signal) * rng.lognormal(0, 0.1, N)
    df["unfavorable"] = (signal + rng.normal(0, 0.4, N) > 0.8).astype(int)
    return df


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_unwrap_exposes_encoder_and_tree(task: str, cases: pd.DataFrame) -> None:
    model = _wrap(
        task,
        RandomForestRegressor(n_estimators=5, random_state=0)
        if task == "regression"
        else RandomForestClassifier(n_estimators=5, random_state=0),
    )
    model.fit(cases[FEATURES], cases[TARGET[task]])
    pre, tree, names = unwrap(task, model)
    assert tree is model.regressor_.named_steps["model"] if task == "regression" else tree is model.named_steps["model"]  # fmt: skip
    assert pre is model.regressor_.named_steps["pre"] if task == "regression" else pre is model.named_steps["pre"]  # fmt: skip
    # one-hot expands uf/vara; target encoding and numerics stay one column each
    assert len(names) > len(FEATURES)
    assert "judge_id" in names and any(n.startswith("uf_") for n in names)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_explain_model_is_additive_and_writes_artefacts(
    task: str, cases: pd.DataFrame, tmp_path: Path
) -> None:
    train, test, _ = temporal_split(cases, test_frac=0.25)
    forest = (
        RandomForestRegressor(n_estimators=30, random_state=0)
        if task == "regression"
        else RandomForestClassifier(n_estimators=30, random_state=0)
    )
    model = _wrap(task, forest).fit(train[FEATURES], train[TARGET[task]])

    summary = explain_model(
        task, model, test[FEATURES], test[TARGET[task]], tmp_path, max_cases=40,
        top_k_dependence=2,
    )  # fmt: skip

    assert summary["additivity_ok"], f"contributions must add up: {summary}"
    assert summary["explainer"].startswith("TreeExplainer")
    assert len(summary["top5_global"]) == 5

    out = tmp_path / task
    for name in (
        "global_importance.csv", "beeswarm.png", "bar_mean_abs_shap.png",
        "waterfall_highest.png", "waterfall_median.png", "waterfall_lowest.png",
        "case_explanations.csv", "shap_summary.json",
    ):  # fmt: skip
        assert (out / name).exists(), f"missing artefact {name}"
    dependences = list(out.glob("dependence_*.png"))
    assert len(dependences) == 2, "one dependence plot per top feature"

    shap_summary = json.loads((out / "shap_summary.json").read_text())
    assert shap_summary["task"] == task

    long = pd.read_csv(out / "case_explanations.csv")
    assert {"case_index", "feature", "feature_value", "shap"}.issubset(long.columns)
    # base + sum(contributions) must equal the model's own raw-space output for every case
    per_case = long.groupby("case_index")["shap"].sum()
    bases = long.groupby("case_index")["base_value_raw_space"].first()
    # note: the model's raw space is log1p(BRL) for regression, class-1 probability for
    # sklearn forests (their per-tree output *is* a probability)
    expected = (
        model.regressor_.predict(test[FEATURES].loc[per_case.index])  # inner pipeline: log1p
        if task == "regression"
        else model.predict_proba(test[FEATURES].loc[per_case.index])[:, 1]
    )
    assert np.abs((per_case + bases).to_numpy() - expected).max() < 1e-6
