import pytest

pytest.importorskip("torch")

from claimvalue.dl import train_dl_regression  # noqa: E402


def test_dl_trains_and_reports_metrics(db_path, tmp_path):
    res = train_dl_regression(db_path, tmp_path, epochs=3, patience=2)
    assert res["test_metrics"]["mae"] > 0 and res["epochs_run"] >= 1
    assert (tmp_path / "dl_regression.pt").exists()
