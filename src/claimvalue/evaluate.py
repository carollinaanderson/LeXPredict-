"""Metrics, bootstrap confidence intervals and error slicing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "rmsle": float(root_mean_squared_log_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mdape": float(np.median(np.abs(y_true - y_pred) / y_true)),
    }


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "base_rate": float(y_true.mean()),
    }


def bootstrap_ci(y_true, y_pred, metric, n: int = 500, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI: how much of a metric difference is just sampling noise?"""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = [
        metric(y_true[idx], y_pred[idx])
        for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))
    ]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def slice_report(df: pd.DataFrame, y_true, y_pred, by: str) -> pd.DataFrame:
    """Error per group (e.g. per UF): shows where the model is weak, not just the average."""
    tmp = pd.DataFrame(
        {by: df[by].to_numpy(), "abs_err": np.abs(np.asarray(y_true) - np.asarray(y_pred))}
    )
    return (
        tmp.groupby(by)["abs_err"]
        .agg(n="size", mae="mean", medae="median")
        .reset_index()
        .sort_values("mae", ascending=False)
    )
