"""SHAP explanations for the trained models.

Why this module exists
----------------------
Permutation importance answers "which features does the model use *overall*?". It cannot answer
"why was *this* case predicted at 78%?", which is the per-prediction question stakeholders ask.
SHAP (SHapley Additive exPlanations) answers both: it decomposes one prediction into a signed
contribution per feature, and the contributions add up exactly to the model output.

Why the artefact needs unwrapping
-----------------------------------------------------
Our saved artefact is not a bare tree. It is:

* classification:  Pipeline( ColumnTransformer(pre) -> tree model )
* regression:      TransformedTargetRegressor( log1p / expm1  around  Pipeline(pre -> tree) )

``shap.TreeExplainer`` needs the tree and rows in the tree's own input space (the *encoded*
space: one-hot + target-encoded + numeric), not raw rows with judge names. So this module:

1. unwraps the artefact to get the fitted ``ColumnTransformer`` and the fitted tree model;
2. transforms the test rows once into the encoded space (a DataFrame, so feature names survive);
3. runs ``TreeExplainer`` (exact for tree ensembles) with a permutation-explainer fallback for
   models shap cannot walk tree-by-tree (e.g. sklearn ``HistGradientBoosting``);
4. verifies additivity: base_value + sum(contributions) == model output, for every case.

A note on units: SHAP explains the model's *own output space*.
* regression  -> contributions are in ``log1p(BRL)`` (because the model is trained on log money);
* classification -> contributions are in log-odds (or probability space for sklearn forests).
Converting to "reais pushed up" is not additive, so we keep the honest unit and document it.

Outputs (under ``reports/shap/<task>/``)
---------------------------------------
* ``global_importance.csv``      mean |SHAP| per feature (the model-level ranking)
* ``beeswarm_*.png`` / ``bar_*.png``   whole-model views
* ``waterfall_*_*.png``          one-case views (highest / median / lowest prediction)
* ``dependence_*.png``           how a top feature moves predictions
* ``case_explanations.csv``      every (case, feature) contribution, machine-readable
* ``shap_summary.json``          additivity check, explainer used, sample sizes
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from claimvalue.config import SEED
from claimvalue.features import FEATURES

TARGET = {"regression": "claim_value", "classification": "unfavorable"}


# ---------------------------------------------------------------------------
# 1. Unwrapping: artefact -> (preprocessor, tree model, feature names)
# ---------------------------------------------------------------------------
def unwrap(task: str, fitted) -> tuple:
    """Split a saved artefact into the encoder and the tree model SHAP can walk.

    Returns ``(preprocessor, model, feature_names)`` where ``feature_names`` are the
    *encoded* columns (one-hot expands ``uf``/``vara``, target-encoding keeps one column
    per high-cardinality id).
    """
    if task == "regression":
        # TransformedTargetRegressor(regressor=Pipeline(pre -> model), log1p/expm1)
        pipe = fitted.regressor_
    else:
        pipe = fitted
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return pre, model, names


def encode_features(
    task: str, fitted, X: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Raw rows -> the matrix the tree actually sees (keeps feature names as columns)."""
    pre, _, names = unwrap(task, fitted)
    cols = features or FEATURES
    return pd.DataFrame(pre.transform(X[cols]), columns=names, index=X.index)


def output_space(task: str, model) -> str:
    """Name the unit the SHAP values are in, for the summary and the API response."""
    if task == "regression":
        return "log1p(BRL)"
    if hasattr(model, "get_booster"):
        return "log-odds (XGBoost margin)"
    if hasattr(model, "decision_function"):
        return "log-odds (booster decision function)"
    return "class-1 probability (sklearn forest)"


def raw_output(task: str, model) -> Callable:
    """The model function SHAP decomposes: predict (regression) or probability/margin.

    For sklearn forests the per-tree output *is* a class probability, so the additive raw
    output equals ``predict_proba[:, 1]``. Boosted models expose a margin (log-odds),
    which is what TreeExplainer decomposes.
    """
    if task == "regression":
        return model.predict  # note: the inner model predicts log1p(BRL)
    if hasattr(model, "get_booster"):
        return lambda X: model.predict(X, output_margin=True)
    if hasattr(model, "decision_function"):
        return model.decision_function
    return lambda X: model.predict_proba(X)[:, 1]


def _output_candidates(task: str, model, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """All plausible model-output spaces, so additivity is checked against the right one.

    SHAP's TreeExplainer returns margins for boosted classifiers but probabilities for
    sklearn forests. Rather than guess, compare the reconstruction with every output the
    model exposes and keep the one that actually matches.
    """
    Xn = X.to_numpy()
    candidates: dict[str, np.ndarray] = {}
    if task == "regression":
        candidates["log1p(BRL)"] = np.asarray(model.predict(Xn), dtype=float).ravel()
        return candidates
    if hasattr(model, "get_booster"):
        try:
            candidates["log-odds (XGBoost margin)"] = np.asarray(
                model.predict(Xn, output_margin=True), dtype=float
            ).ravel()
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            candidates["log-odds (decision function)"] = np.asarray(
                model.decision_function(Xn), dtype=float
            ).ravel()
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            candidates["class-1 probability"] = np.asarray(
                model.predict_proba(Xn)[:, 1], dtype=float
            ).ravel()
        except Exception:
            pass
    return candidates


# ---------------------------------------------------------------------------
# 2. The explanation itself
# ---------------------------------------------------------------------------
def explain_model(
    task: str,
    fitted,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    max_cases: int = 800,
    top_k_dependence: int = 4,
    features: list[str] | None = None,
    target: str | None = None,
) -> dict:
    """Produce every SHAP artefact for one task. Returns a small summary dict.

    ``max_cases`` caps the explained sample (deterministic: first rows of the future test
    window). TreeExplainer is exact, so the sample size only affects plot readability and
    fallback-explainer runtime — not correctness.

    ``features`` and ``target`` let the book pipeline reuse this module with its own
    column set. They default to the synthetic pipeline's definitions.
    """
    import matplotlib

    matplotlib.use("Agg")  # CI/headless: produce files, never open a window
    import shap

    features = features or FEATURES
    target = target or TARGET[task]
    rng = np.random.default_rng(SEED)
    out_dir = Path(out_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    take = min(max_cases, len(X_test))
    idx = rng.choice(len(X_test), size=take, replace=False)
    idx.sort()  # keep the temporal order of the future window
    Xs, ys = X_test.iloc[idx], y_test.iloc[idx]
    X_enc = encode_features(task, fitted, Xs, features)
    _, model, feature_names = unwrap(task, fitted)

    # --- explainer: exact for trees, permutation fallback otherwise -------------
    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_enc)
        explainer_kind = "TreeExplainer (exact, tree-specific)"
    except Exception:  # e.g. HistGradientBoosting has no per-tree SHAP walker
        margin = raw_output(task, model)
        background = shap.sample(X_enc, min(100, len(X_enc)), random_state=SEED)
        explainer = shap.Explainer(margin, background, feature_names=feature_names)
        explanation = explainer(X_enc)
        explainer_kind = "Permutation explainer (model-agnostic fallback)"

    values = np.asarray(explanation.values)
    if values.ndim == 3:  # binary classifiers in some shap versions: (n, f, 2)
        values = values[:, :, 1]  # keep the positive-class channel
    base = np.asarray(explanation.base_values)
    base = base[:, 1] if base.ndim == 2 else base
    expl = shap.Explanation(values=values, base_values=base, data=X_enc.to_numpy(),
                            feature_names=feature_names)  # fmt: skip

    # --- additivity check: the property that makes SHAP trustworthy -------------
    reconstructed = base + values.sum(axis=1)
    candidates = _output_candidates(task, model, X_enc)
    if candidates:
        space_label, model_out, additivity_err = min(
            (
                (label, output, float(np.abs(output - reconstructed).max()))
                for label, output in candidates.items()
            ),
            key=lambda item: item[2],
        )
    else:  # pragma: no cover - defensive: every supported model exposes one of the above
        space_label = output_space(task, model)
        model_out = np.asarray(raw_output(task, model)(X_enc.to_numpy()), dtype=float).ravel()
        additivity_err = float(np.abs(model_out - reconstructed).max())

    # --- global importance ------------------------------------------------------
    mean_abs = np.abs(values).mean(axis=0)
    signed = values.mean(axis=0)
    global_imp = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap_signed": signed,
                "share_pct": 100 * mean_abs / mean_abs.sum(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    global_imp.to_csv(out_dir / "global_importance.csv", index=False)

    # --- plots ------------------------------------------------------------------
    plt_kwargs = {"show": False}
    shap.plots.beeswarm(expl, max_display=15, **plt_kwargs)
    matplotlib.pyplot.savefig(out_dir / "beeswarm.png", bbox_inches="tight", dpi=150)
    matplotlib.pyplot.close()

    shap.plots.bar(expl, max_display=15, **plt_kwargs)
    matplotlib.pyplot.savefig(out_dir / "bar_mean_abs_shap.png", bbox_inches="tight", dpi=150)
    matplotlib.pyplot.close()

    preds = _predict_original_space(task, fitted, Xs, features)
    order = np.argsort(preds)
    picks = {
        "highest": int(order[-1]),
        "median": int(order[len(order) // 2]),
        "lowest": int(order[0]),
    }
    for label, j in picks.items():
        shap.plots.waterfall(expl[j], max_display=12, **plt_kwargs)
        matplotlib.pyplot.savefig(out_dir / f"waterfall_{label}.png", bbox_inches="tight", dpi=150)
        matplotlib.pyplot.close()

    for feat in global_imp["feature"].head(top_k_dependence):
        shap.plots.scatter(expl[:, feat], color=expl[:, feat], **plt_kwargs)
        matplotlib.pyplot.savefig(out_dir / f"dependence_{feat}.png", bbox_inches="tight", dpi=150)
        matplotlib.pyplot.close()

    # --- machine-readable per-case explanations ---------------------------------
    rows = []
    Xs_index = Xs.index.to_numpy()
    for i in range(take):
        for f, feat in enumerate(feature_names):
            if abs(values[i, f]) < 1e-9:
                continue  # drop exact-zero contributions to keep the file readable
            rows.append(
                {
                    "case_index": Xs_index[i],
                    "y_true": ys.iloc[i],
                    "predicted_original_space": float(preds[i]),
                    "base_value_raw_space": float(base[i]),
                    "feature": feat,
                    "feature_value": X_enc.iloc[i, f],
                    "shap": float(values[i, f]),
                }
            )
    case_explanations = pd.DataFrame(rows)
    case_explanations.to_csv(out_dir / "case_explanations.csv", index=False)

    summary = {
        "task": task,
        "explainer": explainer_kind,
        "n_cases_explained": take,
        "n_features_encoded": len(feature_names),
        "additivity_max_error": additivity_err,
        "additivity_ok": bool(additivity_err < 1e-4),
        "output_space": space_label,
        "top5_global": global_imp.head(5)[["feature", "mean_abs_shap"]].to_dict("records"),
        "one_case_story": _case_story(expl, picks, global_imp, feature_names),
    }
    (out_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _predict_original_space(
    task: str, fitted, X: pd.DataFrame, features: list[str] | None = None
) -> np.ndarray:
    """Predictions in human units: BRL for regression, probability for classification."""
    cols = features or FEATURES
    if task == "regression":
        return np.asarray(fitted.predict(X[cols]), dtype=float)  # TTR maps back to BRL
    return np.asarray(fitted.predict_proba(X[cols])[:, 1], dtype=float)


def _case_story(expl, picks: dict, global_imp: pd.DataFrame, feature_names) -> dict:
    """Narrative for the flagship case: the top features that pushed it up or down."""
    j = picks["highest"]
    sv = expl[j]
    contribs = sorted(
        ({"feature": f, "shap": float(v)} for f, v in zip(feature_names, sv.values, strict=True)),
        key=lambda d: abs(d["shap"]),
        reverse=True,
    )
    return {
        "case_row": int(j),
        "predicted_raw": float(expl[j].base_values + sv.values.sum()),
        "top_up": [c for c in contribs if c["shap"] > 0][:5],
        "top_down": [c for c in contribs if c["shap"] < 0][:5],
        "global_top_feature": str(global_imp.loc[0, "feature"]),
    }
