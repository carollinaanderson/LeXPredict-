import json

import pytest

from claimvalue.anomaly import flag_anomalies
from claimvalue.db import load_abt
from claimvalue.train import run_training


@pytest.fixture(scope="session")
def trained(db_path, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    mlflow_db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    out = {
        t: run_training(t, db_path, model_dir, n_iter=2, mlflow_uri=f"sqlite:///{mlflow_db}")
        for t in ("regression", "classification")
    }
    return model_dir, out


def test_regression_beats_the_dummy_baseline(trained):
    _, out = trained
    board = out["regression"]["leaderboard"].set_index("model")
    assert board.loc["hist_gbm", "test_mae"] < board.loc["dummy_median", "test_mae"]
    assert out["regression"]["meta"]["model"] != "dummy_median"


def test_classifier_has_signal_above_chance(trained):
    _, out = trained
    board = out["classification"]["leaderboard"].set_index("model")
    assert board["test_roc_auc"].drop("dummy_prior").max() > 0.55


def test_best_model_is_chosen_by_cv_not_by_test(trained):
    _, out = trained
    board = out["regression"]["leaderboard"]
    assert board.iloc[0]["cv_score"] == board["cv_score"].max()


def test_artifacts_and_metadata_are_written(trained):
    model_dir, out = trained
    for task in ("regression", "classification"):
        assert (model_dir / f"{task}.joblib").exists()
        meta = json.loads((model_dir / f"{task}.json").read_text())
        assert meta["data_fingerprint"] and meta["versions"]["sklearn"]
        lo, hi = meta["primary_metric_ci95"]
        assert lo <= hi
    assert (model_dir / "reports" / "regression_importance.csv").exists()


def test_unknown_task_raises(db_path, tmp_path):
    with pytest.raises(ValueError):
        run_training("clustering", db_path, tmp_path)


def test_anomaly_detector_flags_injected_outlier(db_path):
    abt = load_abt(db_path)
    abt.loc[abt.index[10], "claim_value"] = abt["claim_value"].max() * 500
    flagged = flag_anomalies(abt, contamination=0.01)
    assert flagged.iloc[0]["case_id"] == abt.loc[abt.index[10], "case_id"]
    assert flagged.iloc[0]["is_anomaly"]
