# Aggregate public results

This directory contains a **privacy-safe aggregate snapshot** of the ABJ book
results. It exists so the model comparison, p-values and SHAP summaries are
committed without committing data.

| File | Content |
|---|---|
| `results.json` | Leaderboards for both tasks, test metrics, confidence intervals, paired Wilcoxon p-values, permutation importance and SHAP summaries |
| `regression_shap_bar.png` | Aggregate mean absolute SHAP for the selected regression tree |
| `classification_shap_bar.png` | Aggregate mean absolute SHAP for the selected LightGBM |

## What is committed

- Model names and aggregate metrics, including **LightGBM** and **XGBoost**.
- The model-vs-baseline **p-value** (Wilcoxon signed-rank on per-case loss).
- **SHAP** additivity checks, top features and aggregate bar charts.
- The full interpretation is in [`docs/10_book_results.md`](../docs/10_book_results.md).

## What is not committed

- No raw or prepared case-level data.
- No `case_explanations.csv` or per-case SHAP tables.
- No SQLite database, `.env` file, token, credential or model binary (`.joblib`).

