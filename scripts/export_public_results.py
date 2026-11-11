#!/usr/bin/env python
"""Export a privacy-safe aggregate snapshot of the book-model results.

The repository never commits raw or case-level data. This script reads the
locally generated aggregate artefacts under ``models/book/`` and writes only:

* ``reports/book/results.json``: leaderboards, metrics, p-values, permutation
  importance and SHAP summaries;
* two aggregate SHAP bar charts (mean absolute SHAP, no case rows).

It deliberately does **not** copy ``case_explanations.csv``, the SQLite
database, the prepared CSV, the joblib models, ``.env`` or any raw rows.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "book"
OUT_DIR = ROOT / "reports" / "book"
TASKS = ("regression", "classification")


def _records(path: Path, limit: int | None = None) -> list[dict]:
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict(orient="records")


def _task(task: str) -> dict:
    meta = json.loads((MODEL_DIR / f"{task}.json").read_text(encoding="utf-8"))
    shap_dir = MODEL_DIR / "reports" / "shap" / task
    shap_summary = json.loads((shap_dir / "shap_summary.json").read_text(encoding="utf-8"))
    return {
        "task": task,
        "target": meta["target"],
        "selected_model": meta["model"],
        "test_metrics": meta["test_metrics"],
        "primary_metric": meta["primary_metric"],
        "primary_metric_ci95": meta["primary_metric_ci95"],
        "secondary_metric": meta.get("secondary_metric"),
        "secondary_metric_ci95": meta.get("secondary_metric_ci95"),
        "baseline_model": meta["baseline_model"],
        "baseline_test_metrics": meta["baseline_test_metrics"],
        "p_value_vs_baseline": meta["p_value_vs_baseline"],
        "best_params": meta["best_params"],
        "n_train": meta["n_train"],
        "n_test": meta["n_test"],
        "cutoff_date": meta["cutoff_date"],
        "features": meta["features"],
        "versions": meta["versions"],
        "leaderboard": _records(MODEL_DIR / "reports" / f"{task}_leaderboard.csv"),
        "permutation_importance_top10": _records(
            MODEL_DIR / "reports" / f"{task}_importance.csv", limit=10
        ),
        "shap": {
            "explainer": shap_summary["explainer"],
            "n_cases_explained": shap_summary["n_cases_explained"],
            "additivity_max_error": shap_summary["additivity_max_error"],
            "additivity_ok": shap_summary["additivity_ok"],
            "output_space": shap_summary["output_space"],
            "top5_global": shap_summary["top5_global"],
            "global_top10": _records(shap_dir / "global_importance.csv", limit=10),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = {task: _task(task) for task in TASKS}
    candidates = sorted({row["model"] for task in tasks.values() for row in task["leaderboard"]})
    report = {
        "dataset": "abjData::consumo",
        "book": "Feliz & Trecenti (2022), Metodologia de Pesquisa Jurimétrica, ABJ",
        "licence": "MIT",
        "run_date": "2026-11-11",
        "candidates": candidates,
        "privacy": (
            "Aggregate metrics only. This file contains no case-level rows, raw case "
            "numbers, names, CPFs, CNPJs, addresses, .env values, database files, "
            "prepared CSV files or model binaries."
        ),
        "reproduce": (
            "Run `claimvalue book-pipeline --n-iter 4`, then "
            "`python scripts/export_public_results.py`."
        ),
        "tasks": tasks,
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for task in TASKS:
        source = MODEL_DIR / "reports" / "shap" / task / "bar_mean_abs_shap.png"
        if source.exists():
            shutil.copy2(source, OUT_DIR / f"{task}_shap_bar.png")
    print(f"wrote aggregate results to {OUT_DIR}")


if __name__ == "__main__":
    main()
