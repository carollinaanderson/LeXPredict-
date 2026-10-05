"""Model selection and training for both tasks.

Design rules (each one fixes a problem of the original notebooks):
* time-ordered split and TimeSeriesSplit CV: train on the past, test on the future
* encoders live INSIDE the Pipeline, so they are fitted per CV fold (no leakage)
* hyper-parameters are chosen by cross-validation on TRAIN only; the test set is touched once
* every model is compared against a dummy baseline, with a bootstrap CI on the final error
* skewed money target is modelled on a log scale (TransformedTargetRegressor)
* artefacts are saved with metadata (data fingerprint, library versions, metrics) and
  every run is logged to MLflow
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from claimvalue.config import SEED
from claimvalue.db import load_abt
from claimvalue.evaluate import (
    bootstrap_ci,
    classification_metrics,
    regression_metrics,
    slice_report,
)
from claimvalue.features import FEATURES, make_preprocessor, temporal_split

TASKS = ("regression", "classification")


def _candidates(task: str) -> dict[str, tuple[object, dict]]:
    if task == "regression":
        return {
            "dummy_median": (DummyRegressor(strategy="median"), {}),
            "decision_tree": (
                DecisionTreeRegressor(random_state=SEED),
                {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [10, 30, 100, 300]},
            ),
            "random_forest": (
                RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=SEED),
                {"max_depth": [6, 10, None], "min_samples_leaf": [10, 30, 100]},
            ),
            "hist_gbm": (
                HistGradientBoostingRegressor(max_iter=250, random_state=SEED),
                {
                    "learning_rate": [0.03, 0.06, 0.1],
                    "max_depth": [3, 5, None],
                    "min_samples_leaf": [20, 50, 100],
                    "l2_regularization": [0.0, 1.0, 10.0],
                },
            ),
        }
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), {}),
        "decision_tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50, 150]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED),
            {"max_depth": [6, 10, None], "min_samples_leaf": [20, 50, 150]},
        ),
        "hist_gbm": (
            HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
            {
                "learning_rate": [0.03, 0.06, 0.1],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 1.0, 10.0],
            },
        ),
    }


def _wrap(task: str, estimator):
    pipe = Pipeline(
        [
            ("pre", make_preprocessor("continuous" if task == "regression" else "binary")),
            ("model", estimator),
        ]
    )
    if task == "regression":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def _grid(task: str, grid: dict) -> dict:
    prefix = "regressor__model__" if task == "regression" else "model__"
    return {f"{prefix}{k}": v for k, v in grid.items()}


def _predict(task: str, model, X):
    return model.predict(X) if task == "regression" else model.predict_proba(X)[:, 1]


def _fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    ).hexdigest()[:16]


@contextlib.contextmanager
def _mlflow_run(uri: str | None, experiment: str, run_name: str):
    try:
        import mlflow
    except ImportError:  # tracking is optional; training must not depend on it
        yield None
        return
    mlflow.set_tracking_uri(uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield mlflow


def run_training(
    task: str,
    db_path: Path,
    model_dir: Path,
    *,
    n_iter: int = 8,
    window_days: int = 180,
    test_frac: float = 0.2,
    mlflow_uri: str | None = None,
) -> dict:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    abt = load_abt(db_path, window_days)
    target = "claim_value" if task == "regression" else "unfavorable"
    data = abt.dropna(subset=[target]) if task == "regression" else abt
    train, test, cutoff = temporal_split(data, test_frac, purge=(task == "classification"))
    X_tr, y_tr = train[FEATURES], train[target]
    X_te, y_te = test[FEATURES], test[target]
    scoring = "neg_mean_absolute_error" if task == "regression" else "roc_auc"
    cv = TimeSeriesSplit(n_splits=4)

    leaderboard, fitted = [], {}
    for name, (estimator, grid) in _candidates(task).items():
        pipe = _wrap(task, estimator)
        if grid:
            n = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
            search = RandomizedSearchCV(
                pipe, _grid(task, grid), n_iter=n, cv=cv, scoring=scoring, random_state=SEED,
                error_score="raise",
            )  # fmt: skip
        else:
            search = GridSearchCV(pipe, {}, cv=cv, scoring=scoring, error_score="raise")
        search.fit(X_tr, y_tr)
        pred = _predict(task, search.best_estimator_, X_te)
        metrics = (
            regression_metrics(y_te, pred)
            if task == "regression"
            else classification_metrics(y_te, pred)
        )
        leaderboard.append(
            {
                "model": name,
                "cv_score": float(search.best_score_),
                **{f"test_{k}": v for k, v in metrics.items()},
            }
        )
        fitted[name] = (search, pred)

    board = (
        pd.DataFrame(leaderboard).sort_values("cv_score", ascending=False).reset_index(drop=True)
    )
    best_name = board.loc[0, "model"]  # chosen by CV on train, NOT by test score
    best_search, best_pred = fitted[best_name]
    best = best_search.best_estimator_

    if task == "regression":
        ci = bootstrap_ci(y_te, best_pred, mean_absolute_error)
        slices = slice_report(test, y_te, best_pred, "uf")
        pi_scoring = "neg_mean_absolute_error"
    else:
        from sklearn.metrics import roc_auc_score

        ci = bootstrap_ci(y_te, best_pred, roc_auc_score)
        slices = slice_report(test, y_te, best_pred, "uf")
        pi_scoring = "roc_auc"
    sample = test.sample(min(len(test), 1500), random_state=SEED)
    imp = permutation_importance(
        best, sample[FEATURES], sample[target], scoring=pi_scoring, n_repeats=5, random_state=SEED
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    meta = {
        "task": task,
        "model": best_name,
        "best_params": {
            k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
            for k, v in best_search.best_params_.items()
        },
        "test_metrics": board.loc[0].filter(like="test_").to_dict(),
        "primary_metric_ci95": ci,
        "n_train": len(train),
        "n_test": len(test),
        "cutoff_date": str(cutoff.date()),
        "window_days": window_days,
        "features": FEATURES,
        "data_fingerprint": _fingerprint(data),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    reports = model_dir / "reports"
    reports.mkdir(exist_ok=True)
    # NOTE: joblib/pickle executes code on load. Only ever load artefacts you produced yourself.
    joblib.dump(best, model_dir / f"{task}.joblib")
    (model_dir / f"{task}.json").write_text(json.dumps(meta, indent=2, default=str))
    board.to_csv(reports / f"{task}_leaderboard.csv", index=False)
    importance.to_csv(reports / f"{task}_importance.csv", index=False)
    slices.to_csv(reports / f"{task}_by_uf.csv", index=False)

    with _mlflow_run(mlflow_uri, "claimvalue", f"{task}-{best_name}") as mlf:
        if mlf is not None:
            mlf.log_params(
                {
                    "task": task,
                    "model": best_name,
                    "window_days": window_days,
                    **meta["best_params"],
                }
            )
            mlf.log_metrics({k: v for k, v in meta["test_metrics"].items()})
            mlf.log_metric("cv_score", float(board.loc[0, "cv_score"]))
            mlf.set_tags({"data_fingerprint": meta["data_fingerprint"], "git_sha": meta["git_sha"]})
            mlf.log_artifact(str(model_dir / f"{task}.json"))
            mlf.log_artifact(str(reports / f"{task}_leaderboard.csv"))

    return {"meta": meta, "leaderboard": board, "importance": importance, "by_uf": slices}
