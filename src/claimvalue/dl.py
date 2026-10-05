"""Deep learning: embedding MLP for tabular claim-value regression (PyTorch).

Categorical ids (UF, vara, comarca, judge) get learned embeddings; numeric rolling-window features
are median-imputed and standardized with TRAIN statistics only. The target is modelled on a
standardized log scale. Early stopping uses the most recent 15% of the training period, so model
selection never looks at the test set and respects time order.

Install the extra first:  pip install -e ".[dl]"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from claimvalue.config import SEED
from claimvalue.db import load_abt
from claimvalue.evaluate import regression_metrics
from claimvalue.features import CAT_HIGH, CAT_LOW, NUMERIC, temporal_split

CATS = CAT_LOW + CAT_HIGH


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError('PyTorch is required: pip install -e ".[dl]"') from exc
    return torch, nn


class Prep:
    """Vocabulary + scalers fitted on training data only. Unknown categories map to index 0."""

    def __init__(self, fit_df: pd.DataFrame):
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(fit_df[c].unique()))} for c in CATS}
        self.median = fit_df[NUMERIC].median().to_dict()
        filled = fit_df[NUMERIC].fillna(self.median)
        self.mean = filled.mean().to_dict()
        self.std = filled.std().replace(0, 1).fillna(1).to_dict()
        log_y = np.log1p(fit_df["claim_value"].to_numpy())
        self.y_mean, self.y_std = float(log_y.mean()), float(log_y.std() or 1.0)

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in CATS]

    def x(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat = np.stack([df[c].map(self.vocab[c]).fillna(0).astype(int).to_numpy() for c in CATS], 1)
        num = df[NUMERIC].fillna(self.median)
        num = (num - pd.Series(self.mean)) / pd.Series(self.std)
        return cat.astype(np.int64), num.to_numpy(dtype=np.float32)

    def y(self, df: pd.DataFrame) -> np.ndarray:
        return ((np.log1p(df["claim_value"].to_numpy()) - self.y_mean) / self.y_std).astype(
            np.float32
        )

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        return np.expm1(z * self.y_std + self.y_mean)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in ("vocab", "median", "mean", "std", "y_mean", "y_std")}


def _build_net(cardinalities: list[int], n_num: int, hidden=(128, 64), dropout=0.2):
    torch, nn = _torch()

    class EmbeddingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList(
                nn.Embedding(c, min(16, (c + 1) // 2 + 1), padding_idx=0) for c in cardinalities
            )
            width = sum(e.embedding_dim for e in self.embs) + n_num
            layers: list[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
                width = h
            layers.append(nn.Linear(width, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, xc, xn):
            z = torch.cat([e(xc[:, i]) for i, e in enumerate(self.embs)] + [xn], dim=1)
            return self.mlp(z).squeeze(1)

    return EmbeddingMLP()


def train_dl_regression(
    db_path: Path,
    model_dir: Path,
    *,
    window_days: int = 180,
    epochs: int = 60,
    patience: int = 6,
    batch_size: int = 256,
    test_frac: float = 0.2,
) -> dict:
    torch, nn = _torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    abt = load_abt(db_path, window_days).dropna(subset=["claim_value"])
    train, test, cutoff = temporal_split(abt, test_frac)
    split_at = int(len(train) * 0.85)
    fit, val = train.iloc[:split_at], train.iloc[split_at:]

    prep = Prep(fit)
    xc, xn = (torch.from_numpy(a) for a in prep.x(fit))
    y = torch.from_numpy(prep.y(fit))
    vxc, vxn = (torch.from_numpy(a) for a in prep.x(val))
    vy = torch.from_numpy(prep.y(val))

    net = _build_net(prep.cardinalities(), len(NUMERIC))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    gen = torch.Generator().manual_seed(SEED)

    best_val, best_state, bad, history = float("inf"), None, 0, []
    for _epoch in range(epochs):
        net.train()
        order = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad()
            loss_fn(net(xc[idx], xn[idx]), y[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(vxc, vxn), vy))
        history.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)

    net.eval()
    txc, txn = (torch.from_numpy(a) for a in prep.x(test))
    with torch.no_grad():
        pred = prep.inverse_y(net(txc, txn).numpy())
    metrics = regression_metrics(test["claim_value"], pred)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_dir / "dl_regression.pt")  # load with weights_only=True
    (model_dir / "dl_regression.json").write_text(
        json.dumps(
            {
                "test_metrics": metrics,
                "epochs_run": len(history),
                "best_val_loss": best_val,
                "cutoff_date": str(cutoff.date()),
                "prep": prep.to_json(),
            },
            indent=2,
        )
    )
    return {"test_metrics": metrics, "epochs_run": len(history)}
