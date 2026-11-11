# 10 · Results on the ABJ book dataset

> Run date: 2026-11-11. Dataset: `abjData::consumo`, prepared file fingerprint
> `d3e26c735e09d0c6` (regression) and `19b37000e6ed184a` (classification).
> Source and licence: see `docs/09_book_dataset.md` and `CREDITS.md`.
>
> The candidate set includes LightGBM, XGBoost, random forest, histogram gradient
> boosting, a decision tree and a dummy baseline. Section 4 adds paired
> model-vs-baseline tests (p-values) next to the SHAP analysis.
>
> **Read this first.** The public sample is small and the filing-time features
> carry little signal. The selected models do **not** beat the simple baselines
> by a reliable margin. That is the result, not a bug. The value of the project
> is the method: a time-ordered split, leakage-safe history, explicit baselines,
> confidence intervals and per-case explanations. Do not read these numbers as
> production accuracy or as evidence about real legal outcomes.

## 1. Run setup

| Item | Regression | Classification |
|---|---|---|
| Target | `claim_value` (BRL, modelled as `log1p`) | `reformed` (1 = Reformou/Parcial, 0 = Não reformou) |
| Rows with a usable label | 952 | 952 |
| Training rows | 761 | 623 |
| Test rows | 191 | 191 |
| Split | Last 20% of filing dates | Last 20% of filing dates, training labels purged at the cutoff |
| Cutoff | 2021-04-26 | 2021-04-26 |
| History window | 365 days | 365 days |
| CV | 4-fold `TimeSeriesSplit` | 4-fold `TimeSeriesSplit` |
| Search | 4-iteration random search, train only | 4-iteration random search, train only |
| Selection metric | MAE | Brier score |
| Test set | Touched once, after selection | Touched once, after selection |

Candidates: random forest, histogram gradient boosting, decision tree, LightGBM,
XGBoost and a dummy baseline. Encoders (`OneHotEncoder`, `TargetEncoder`) live
inside the pipeline and are fitted per CV fold.

The classification selection metric is Brier score, not AUC. The output is a
probability, so calibration matters. AUC is reported as a secondary metric.

## 2. Claim value (regression)

| Model | CV MAE (R$) | CV SD | Test MAE | Test MedAE | Test RMSE | Test RMSLE | Test R² | Test MdAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decision tree (selected) | 34,717 | 14,968 | 14,862 | 6,170 | 35,594 | 1.101 | -0.096 | 0.580 |
| random forest | 35,074 | 14,664 | 14,259 | 6,518 | 34,444 | 1.060 | -0.026 | 0.542 |
| LightGBM | 35,134 | 14,692 | 14,432 | 7,026 | 32,875 | 1.066 | 0.065 | 0.582 |
| histogram GBM | 35,376 | 14,627 | 15,165 | 7,592 | 33,220 | 1.106 | 0.045 | 0.642 |
| dummy median | 35,494 | 14,640 | 14,959 | 9,624 | 34,447 | 1.139 | -0.026 | 0.564 |
| XGBoost | 36,185 | 15,049 | 16,066 | 9,201 | 33,802 | 1.176 | 0.012 | 0.653 |

Selected model: decision tree, `max_depth=3`, `min_samples_leaf=30`.

- Test MAE: **R$ 14,862**, 95% bootstrap CI **R$ 10,782 to R$ 19,709**.
- Dummy median MAE: **R$ 14,959**.
- Median absolute percentage error: **0.580** for the model, **0.564** for the
  dummy.
- Test R² is negative, so the model is worse than predicting the test mean.
- Paired Wilcoxon signed-rank test on 191 per-case absolute errors versus the
  median baseline: **p = 0.280** (two-sided). Median absolute error is
  R$ 6,170 for the tree and R$ 9,624 for the baseline (median paired difference
  R$ -1,682). The lower median is not statistically distinguishable from the
  baseline at the 5% level in this sample.

The confidence interval is wider than the gap between the model and the dummy.
The random forest has a lower test MAE (R$ 14,259), but it was not selected by
the training CV. Picking it after seeing the test set would be test-set tuning.
The honest conclusion is that this sample does not support a reliable claim-value
model from filing-time metadata.

The shallow tree explains why many SHAP values are zero in section 5: with
`max_depth=3`, the model never splits on the municipality or the filing year.

## 3. Appeal reform (classification)

| Model | CV Brier | CV SD | Test Brier | Test AUC | Test PR-AUC | Test F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (selected) | 0.247 | 0.004 | 0.255 | 0.530 | 0.521 | 0.611 | 0.513 |
| random forest | 0.249 | 0.005 | 0.249 | 0.535 | 0.529 | 0.446 | 0.518 |
| dummy prior | 0.250 | 0.007 | 0.252 | 0.500 | 0.503 | 0.000 | 0.497 |
| decision tree | 0.255 | 0.009 | 0.253 | 0.536 | 0.517 | 0.427 | 0.508 |
| histogram GBM | 0.257 | 0.009 | 0.260 | 0.507 | 0.519 | 0.574 | 0.503 |
| XGBoost | 0.264 | 0.011 | 0.275 | 0.491 | 0.507 | 0.610 | 0.492 |

Test base rate: **0.503**.

- Test Brier: **0.255**, 95% CI **0.241 to 0.270**.
- Dummy prior Brier: **0.252**.
- Test AUC: **0.530**, 95% CI **0.448 to 0.620**.
- The AUC interval includes 0.500.
- Paired Wilcoxon signed-rank test on 191 per-case Brier scores versus the prior
  baseline: **p = 0.846** (two-sided). Median per-case Brier is 0.2446 for the
  model and 0.2961 for the baseline (median paired difference -0.0214). The
  difference is not statistically significant at the 5% level.

The selected LightGBM has a better CV Brier than the dummy (0.247 vs 0.250), but
its test Brier is slightly worse (0.255 vs 0.252). The difference is well inside
the CV dispersion and the test interval. The random forest and decision tree have
slightly better test AUC, but the selection rule chose LightGBM on training data.
Again, switching after seeing the test set would be test-set tuning.

Conclusion: the file supports a **weak, unstable** signal at best. It does not
support a production risk score.

## 4. Model-vs-baseline tests (p-values)

The p-value answers a narrow question: **is the selected model's per-case loss
different from the no-skill baseline?** It is not a test of whether a feature
causes the outcome, and it is not a substitute for SHAP. The test is a two-sided
Wilcoxon signed-rank test on paired per-case losses from the untouched test
window.

| Task | Model | Baseline | Loss | p-value | Median loss (model) | Median loss (baseline) |
|---|---|---|---|---:|---:|---:|
| Regression | decision tree | dummy median | absolute error (R$) | 0.280 | 6,170 | 9,624 |
| Classification | LightGBM | dummy prior | Brier score | 0.846 | 0.2446 | 0.2961 |

Both tests are non-significant at the 5% level. For regression, the tree has a
lower median absolute error but a heavier tail, so the paired test does not
separate it from the baseline. For classification, the test Brier is slightly
worse than the dummy and the p-value is close to 1. This is consistent with the
confidence intervals and with the weak-signal conclusion.

The p-values and the SHAP values are complementary:

- **p-value:** does the model's error distribution differ from the baseline?
- **SHAP:** how does the selected model use each feature, for the whole test set
  and for one case?

A non-significant p-value does not invalidate the SHAP analysis. It means the
selected model is not demonstrably better than the baseline on this sample, so
the SHAP attributions describe a model that should not be used for decisions.

## 5. SHAP on the book models

The exact TreeExplainer was run on all 191 test cases for each task. Additivity
held on every case. The same explanation path was also smoke-tested on an
XGBoost classifier (not selected): additivity max error 8.3e-7 in XGBoost margin
space, so the pipeline can explain XGBoost if it wins the CV search.

| Check | Regression | Classification |
|---|---|---|
| Cases explained | 191 | 191 |
| Encoded features | 18 | 18 |
| Additivity max error | 1.78e-15 | 1.11e-15 |
| Output space | `log1p(BRL)` | log-odds (decision function) |

### 5.1 Regression: what the selected tree uses

| Feature | Share of mean absolute SHAP | Mean signed SHAP |
|---|---:|---:|
| `assunto_Fornecimento de Energia` | 48.9% | -0.122 |
| `assunto_n_prior` | 24.5% | -0.200 |
| `tipo_litigio_nPF-nPF` | 15.5% | +0.011 |
| `assunto_avg_value_prior` | 11.1% | +0.069 |
| all other encoded features | 0.0% | 0.000 |

The model is a three-level tree. It never splits on `comarca`, `filing_year` or
the municipality history, so those features receive zero attribution. This is a
clear example of a general rule: **SHAP explains the model that was trained, not
the data generating process**. A feature with zero SHAP here is not proven
irrelevant; the selected model simply did not use it.

Per-case example, regression row 82:

- Up: `tipo_litigio_nPF-nPF` +0.904, `assunto_Fornecimento de Energia` +0.328,
  `assunto_n_prior` +0.051 (log1p units).
- Down: `assunto_avg_value_prior` -0.017.
- Predicted value in log1p space: 10.801.

### 5.2 Classification: what the selected LightGBM uses

| Feature | Share of mean absolute SHAP | Mean signed SHAP |
|---|---:|---:|
| `tipo_litigio_PF-nPF` | 19.9% | -0.007 |
| `filing_month` | 14.0% | +0.108 |
| `assunto_avg_value_prior` | 12.4% | +0.073 |
| `comarca_reform_rate_prior` | 11.7% | +0.041 |
| `assunto_n_prior` | 10.5% | +0.097 |
| `comarca` | 10.1% | -0.013 |
| `assunto_reform_rate_prior` | 10.0% | +0.083 |
| `comarca_n_prior` | 5.7% | +0.040 |
| `tipo_litigio_nPF-PF` | 3.2% | -0.001 |
| `comarca_avg_value_prior` | 1.6% | +0.010 |
| `assunto_Bancários` | 0.9% | -0.001 |

Per-case example, classification row 182:

- Up: `comarca_reform_rate_prior` +0.350, `filing_month` +0.231,
  `tipo_litigio_PF-nPF` +0.219, `assunto_reform_rate_prior` +0.199,
  `assunto_n_prior` +0.172 (log-odds).
- Down: `comarca` -0.076, `assunto_Bancários` -0.008.
- Raw model output: 1.126 in log-odds, which is a probability above 0.5 after the
  logistic link.

The classification model uses more features than the regression tree, but the
global pattern is still weak and the test AUC is close to chance. The per-case
story is a description of the model's reasoning, not evidence that the reason is
correct.

### 5.3 Artefacts

```
models/book/reports/shap/regression/
  bar_mean_abs_shap.png
  beeswarm.png
  waterfall_highest.png
  waterfall_median.png
  waterfall_lowest.png
  dependence_*.png
  case_explanations.csv
  global_importance.csv
  shap_summary.json

models/book/reports/shap/classification/
  (same layout)
```

## 6. Why the scores are weak

1. **Sample size.** 1,000 rows, 952 labels, 191 test rows. The confidence
   intervals are wide by construction.
2. **Features.** Only subject, municipality, party configuration, filing date and
   prior aggregates are available at filing. The facts that decide an appeal are
   mostly absent.
3. **Municipality sparsity.** 181 municipalities for 952 labelled rows. Target
   encoding helps, but many municipalities have very little history.
4. **Outcome noise.** Whether an appeal is partially reformed depends on the
   record, the arguments and the panel. Those are not in `consumo`.
5. **Temporal drift.** Training on 2015-2021 filings and testing on the last 20%
   is the right design, but it leaves little data for the most recent period.
6. **Baseline strength.** The median claim value and the base reform rate are
   hard to beat when the signal is weak. A baseline-first process exposes that
   instead of hiding it.

## 7. What would improve the analysis

- Use the full `tjsp2inst` source, not the 1,000-row sample, subject to a licence
  and privacy review.
- Add first-instance features: decision type, judge, panel, subject subclass,
  procedural posture, legal grounds.
- Model claim value by subject with a hierarchical or mixed-effects model, so
  municipalities with little history borrow strength.
- Add text, but only with a privacy review and a proper de-identification step.
- Calibrate the classifier (isotonic or Platt) and report calibration curves.
- Validate on a later, external period. Do not tune on the test window.

## 8. Reproduce

```bash
uv sync --extra dev --extra api --extra eda --extra dl
export PYTHONPATH="$PWD/src"        # only needed if the editable .pth is hidden by macOS/iCloud
uv pip install pyreadr              # needed only to read the upstream .rda file
claimvalue book-fetch               # download, verify SHA-256, prepare (gitignored runtime data)
claimvalue book-prepare
claimvalue book-build-db
claimvalue book-eda
claimvalue book-train --n-iter 4
claimvalue book-explain --max-cases 191
claimvalue book-check-pii
```

One command runs the whole book path:

```bash
claimvalue book-pipeline --n-iter 4
```

Artefacts:

| Path | Content |
|---|---|
| `RAW_DATA_DIR/book/consumo_prepared.csv` | Downloaded/verified prepared table (gitignored; default `data/raw/book/`) |
| `RAW_DATA_DIR/book/SOURCE.json` | Runtime provenance: source URL, SHA-256, row counts |
| `data/processed/book_cases.csv` | Validated pipeline copy (gitignored) |
| `data/processed/book_privacy_report.json` | Privacy gate result (gitignored) |
| `data/book.db` | SQLite schema, quality checks and ABT (gitignored) |
| `models/book/regression.joblib` | Selected regression model (gitignored) |
| `models/book/classification.joblib` | Selected classification model (gitignored) |
| `models/book/*.json` | Metrics, CI, p-value, features, fingerprint, citation (gitignored) |
| `models/book/reports/eda/` | EDA report and figures (gitignored) |
| `models/book/reports/` | Leaderboards, importance, slices by subject (gitignored) |
| `models/book/reports/shap/` | SHAP plots, per-case CSV, summary (gitignored) |

No dataset or model artefact is committed. The repository contains only code,
documentation and the provenance metadata in
`src/claimvalue/data/book/SOURCE.md` and `SOURCE.json`.

## 9. Credits

The dataset and the method come from ABJ:

> Feliz, Ricardo; Trecenti, Julio. **Metodologia de Pesquisa Jurimétrica**.
> ABJ, 2022. <https://livro.abj.org.br>. MIT licence.

> Trecenti, Julio; Hirota, Renata; Witkoski, Katerine; Associação Brasileira de
> Jurimetria. **abjData: Databases Used Routinely by the Brazilian Jurimetrics
> Association**. <https://github.com/abjur/abjData>. MIT licence.

This project is not affiliated with or endorsed by ABJ. See `CREDITS.md` for the
full MIT notice.
