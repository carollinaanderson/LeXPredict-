"""Model selection and training on the ABJ book dataset.

This is the book-data counterpart of ``train.py``. It reuses the same candidates,
leakage-safe split, baselines, bootstrap CIs and permutation importance, but the
features and targets come from ``abjData::consumo`` (see ``book_features.py``).

The public results in the README and docs are produced by this module.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from claimvalue.book_db import load_book_abt
from claimvalue.book_features import FEATURES, TARGETS, make_preprocessor, temporal_split
from claimvalue.config import SEED
from claimvalue.evaluate import (
    bootstrap_ci,
    classification_metrics,
    regression_metrics,
    slice_report,
)
from claimvalue.train import _candidates, _fingerprint, _grid, _mlflow_run, _predict

TASKS = ("regression", "classification")
DATASET = "abjData::consumo"
CITATION = (
    "Feliz, R. & Trecenti, J. (2022). Metodologia de Pesquisa Jurimétrica. ABJ. "
    "https://livro.abj.org.br; data: abjData::consumo, https://github.com/abjur/abjData"
)


def load_book_task_data(
    task: str, db_path: Path, window_days: int = 365, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Load the book ABT and apply the deterministic future-window split."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    abt = load_book_abt(db_path, window_days)
    target = TARGETS[task]
    data = abt.dropna(subset=[target])
    return temporal_split(data, test_frac, purge=(task == "classification"))


def _wrap(task: str, estimator):
    pipe = Pipeline(
        [
            ("pre", make_preprocessor(task)),
            ("model", estimator),
        ]
    )
    if task == "regression":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def run_book_training(
    task: str,
    db_path: Path,
    model_dir: Path,
    *,
    n_iter: int = 4,
    window_days: int = 365,
    test_frac: float = 0.2,
    mlflow_uri: str | None = None,
) -> dict:
    train, test, cutoff = load_book_task_data(task, db_path, window_days, test_frac)
    features = FEATURES[task]
    target = TARGETS[task]
    X_tr, y_tr = train[features], train[target]
    X_te, y_te = test[features], test[target]
    scoring = "neg_mean_absolute_error" if task == "regression" else "neg_brier_score"
    cv = TimeSeriesSplit(n_splits=4)

    leaderboard, fitted = [], {}
    for name, (estimator, grid) in _candidates(task).items():
        pipe = _wrap(task, estimator)
        if grid:
            n = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
            search = RandomizedSearchCV(
                pipe, _grid(task, grid), n_iter=n, cv=cv, scoring=scoring,
                random_state=SEED, error_score="raise",
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
                "cv_score_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                **{f"test_{k}": v for k, v in metrics.items()},
            }
        )
        fitted[name] = (search, pred)

    board = (
        pd.DataFrame(leaderboard).sort_values("cv_score", ascending=False).reset_index(drop=True)
    )
    best_name = board.loc[0, "model"]  # chosen by CV on train, never by test
    best_search, best_pred = fitted[best_name]
    best = best_search.best_estimator_

    baseline_name = "dummy_median" if task == "regression" else "dummy_prior"
    _, baseline_pred = fitted[baseline_name]
    baseline_metrics = (
        regression_metrics(y_te, baseline_pred)
        if task == "regression"
        else classification_metrics(y_te, baseline_pred)
    )
    baseline_test = paired_baseline_test(task, y_te, best_pred, baseline_pred)

    if task == "regression":
        ci = bootstrap_ci(y_te, best_pred, mean_absolute_error)
        secondary_ci = None
        primary_metric = "mae_brl"
        slices = slice_report(test, y_te, best_pred, "assunto")
        pi_scoring = "neg_mean_absolute_error"
    else:
        ci = bootstrap_ci(y_te, best_pred, brier_score_loss)
        secondary_ci = bootstrap_ci(y_te, best_pred, roc_auc_score)
        primary_metric = "brier"
        slices = slice_report(test, y_te, best_pred, "assunto")
        pi_scoring = "neg_brier_score"

    sample = test.sample(min(len(test), 800), random_state=SEED)
    imp = permutation_importance_compat(best, sample[features], sample[target], pi_scoring)
    importance = (
        pd.DataFrame({"feature": features, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    meta = {
        "task": task,
        "dataset": DATASET,
        "citation": CITATION,
        "model": best_name,
        "target": target,
        "best_params": {
            k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
            for k, v in best_search.best_params_.items()
        },
        "test_metrics": board.loc[0].filter(like="test_").to_dict(),
        "baseline_model": baseline_name,
        "baseline_test_metrics": baseline_metrics,
        "p_value_vs_baseline": baseline_test,
        "primary_metric": primary_metric,
        "primary_metric_ci95": ci,
        "secondary_metric": "roc_auc" if task == "classification" else None,
        "secondary_metric_ci95": secondary_ci,
        "n_train": len(train),
        "n_test": len(test),
        "cutoff_date": str(cutoff.date()),
        "window_days": window_days,
        "features": features,
        "data_fingerprint": _fingerprint(pd.concat([train, test], ignore_index=True)),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    for package in ("xgboost", "lightgbm", "shap"):
        try:
            meta["versions"][package] = __import__(package).__version__
        except ImportError:
            pass
    model_dir.mkdir(parents=True, exist_ok=True)
    reports = model_dir / "reports"
    reports.mkdir(exist_ok=True)
    joblib.dump(best, model_dir / f"{task}.joblib")
    (model_dir / f"{task}.json").write_text(json.dumps(meta, indent=2, default=str))
    board.to_csv(reports / f"{task}_leaderboard.csv", index=False)
    importance.to_csv(reports / f"{task}_importance.csv", index=False)
    slices.to_csv(reports / f"{task}_by_assunto.csv", index=False)

    with _mlflow_run(mlflow_uri, "claimvalue-book", f"{task}-{best_name}") as mlf:
        if mlf is not None:
            mlf.log_params(
                {"task": task, "model": best_name, "dataset": DATASET, **meta["best_params"]}
            )
            mlf.log_metrics({k: v for k, v in meta["test_metrics"].items()})
            mlf.log_metric("cv_score", float(board.loc[0, "cv_score"]))
            mlf.set_tags({"data_fingerprint": meta["data_fingerprint"], "dataset": DATASET})
            mlf.log_artifact(str(model_dir / f"{task}.json"))
            mlf.log_artifact(str(reports / f"{task}_leaderboard.csv"))

    return {"meta": meta, "leaderboard": board, "importance": importance, "by_assunto": slices}


def permutation_importance_compat(model, X, y, scoring):
    """Wrapper so the permutation importance import stays in one place."""
    from sklearn.inspection import permutation_importance

    return permutation_importance(model, X, y, scoring=scoring, n_repeats=5, random_state=SEED)


def paired_baseline_test(task: str, y_true, pred, baseline_pred) -> dict:
    """Two-sided Wilcoxon signed-rank test on per-case loss vs the no-skill baseline.

    This is a *model-vs-baseline* test: it asks whether the model's errors differ from
    the dummy model's errors. It is not a test of whether a feature causes the outcome;
    that is the role of the SHAP analysis. A non-significant result is a valid finding
    and must be reported as such.
    """
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    if task == "regression":
        loss = np.abs(y - pred)
        base = np.abs(y - baseline_pred)
        metric = "absolute_error_brl"
    else:
        loss = (y - pred) ** 2
        base = (y - baseline_pred) ** 2
        metric = "brier_score_per_case"
    diff = loss - base
    if np.allclose(diff, 0.0):
        statistic, p_value = float("nan"), 1.0
    else:
        result = wilcoxon(loss, base, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "test": "wilcoxon_signed_rank",
        "alternative": "two-sided",
        "metric": metric,
        "statistic": statistic,
        "p_value": p_value,
        "n_pairs": int(len(y)),
        "median_loss_model": float(np.median(loss)),
        "median_loss_baseline": float(np.median(base)),
        "median_loss_difference": float(np.median(diff)),
        "interpretation": (
            "A small p-value means the per-case loss distribution differs from the no-skill "
            "baseline. It does not establish causation, calibration or practical value."
        ),
    }
