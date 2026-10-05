"""Unsupervised anomaly flags on claim values (Isolation Forest).

A review queue / data-quality tool, not a predictor: it flags claims that are unusually large or
small for their (comarca, vara) peer group. Peer statistics use the whole table, which is fine for
an after-the-fact audit but must not be fed back into the supervised models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from claimvalue.config import SEED

MIN_GROUP = 20


def _robust_z(log_claim: pd.Series, groups: pd.Series) -> pd.Series:
    """Median/MAD z-score inside each peer group; small groups fall back to the global stats."""

    def stats(s: pd.Series) -> tuple[float, float]:
        med = s.median()
        return med, max(1.4826 * (s - med).abs().median(), 1e-6)

    g_med, g_mad = stats(log_claim)
    per_group = log_claim.groupby(groups).agg(["size", "median"])
    mad = (log_claim - groups.map(per_group["median"])).abs().groupby(groups).median()
    med = groups.map(per_group["median"]).where(groups.map(per_group["size"]) >= MIN_GROUP, g_med)
    scale = (1.4826 * groups.map(mad)).clip(lower=1e-6)
    scale = scale.where(groups.map(per_group["size"]) >= MIN_GROUP, g_mad)
    return (log_claim - med) / scale


def flag_anomalies(abt: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Return case_id, ``anomaly_score`` (higher = stranger) and ``is_anomaly``."""
    df = abt.dropna(subset=["claim_value"]).copy()
    df["log_claim"] = np.log1p(df["claim_value"])
    peer = df["comarca"].astype(str) + "|" + df["vara"].astype(str)
    df["peer_z"] = _robust_z(df["log_claim"], peer)
    X = df[["log_claim", "peer_z"]]
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED)
    labels = forest.fit_predict(X)
    out = df[["case_id"]].copy()
    out["anomaly_score"] = -forest.score_samples(X)
    out["is_anomaly"] = labels == -1
    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
