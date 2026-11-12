# claimvalue-ml

> **Personal study project.** The public analysis uses the MIT-licensed dataset
> `abjData::consumo` from the Brazilian Jurimetrics Association (ABJ), published
> with the book *Metodologia de Pesquisa Jurimétrica* by Ricardo Feliz and Julio
> Trecenti (ABJ, 2022). The project is not connected to any employer, client or
> real legal case, and contains no confidential or NDA-protected material. See
> [`DISCLAIMER.md`](DISCLAIMER.md), [`CREDITS.md`](CREDITS.md) and
> [`docs/09_book_dataset.md`](docs/09_book_dataset.md).
>
> **Data status.** The public analysis uses real, public court metadata from the
> Court of Justice of São Paulo (TJSP), distributed by ABJ in `abjData::consumo`.
> It is not synthetic and not employer data. The synthetic records in this
> repository are test fixtures only and are never used for the public results.

The project studies two questions on second-degree consumer appeals from São
Paulo:

1. **Claim value (regression).** Given what is known when an appeal is filed,
   what claim value should we expect?
2. **Appeal reform (classification).** Will the second-instance decision change
   the first-instance decision?

The repository covers the full path: runtime-downloaded public data, SQL schema and
quality checks, a leakage-safe analytical base table, EDA, temporal validation,
baselines, confidence intervals, SHAP explanations, an API and CI.

## The honest headline

On this 1,000-row public sample, the selected models do **not** beat the simple
baselines by a reliable margin.

| Task | Selected by CV | Test result | Baseline | p vs baseline | 95% CI | Reading |
|---|---|---|---|---|---|---|
| Claim value | decision tree | MAE R$ 14,862 (MdAPE 0.58) | median: MAE R$ 14,959 (MdAPE 0.56) | 0.280 | R$ 10,782 to R$ 19,709 | no reliable gain |
| Appeal reform | LightGBM | Brier 0.255 (AUC 0.530) | prior: Brier 0.252 (AUC 0.500) | 0.846 | 0.241 to 0.270 | no reliable gain; AUC CI includes 0.5 |

That is the result. The sample is small, the filing-time features are thin, and
appeal outcomes are noisy. The project reports it instead of tuning on the test
set to manufacture a better number. The value is the method: a time-ordered
split, leakage-safe history, explicit baselines, bootstrap intervals and
per-case explanations.

Full tables, SHAP attribution and limitations: [`docs/10_book_results.md`](docs/10_book_results.md).
The same result, the jargon and the privacy boundary explained without statistics:
[`docs/11_stakeholder_brief.md`](docs/11_stakeholder_brief.md).

Privacy-safe aggregate snapshot (metrics, p-values, SHAP summaries and two
aggregate bar charts, with no case-level data): [`reports/book/`](reports/book/).

## Data and credits

The main dataset is `consumo`, a 1,000-row sample of **real, public second-degree
consumer cases from the Court of Justice of São Paulo (TJSP)**, distributed in the
`abjData` R package. It is the data behind every public result in this repository.
It is not synthetic, not fabricated and not employer data.

- No names, CPFs, CNPJs, addresses or free-text case descriptions.
- The public case number is replaced by a stable SHA-256 pseudonym.
- The raw case number is not stored in the prepared file.
- The prepared table is downloaded and verified at runtime into a gitignored
  directory (`RAW_DATA_DIR/book/`, default `data/raw/book/`); **no data file is
  committed**.
- The data is used under the MIT licence, with attribution.
- Filing dates run from 2015-06-17 to 2021-12-07; decision dates run from
  2020-01-07 to 2021-12-21.

The synthetic generator in `src/claimvalue/synthetic.py` remains only for
automated tests and for a controlled benchmark. It is not the public analysis and
none of its records appear in the published numbers.

Citations:

> Feliz, Ricardo; Trecenti, Julio. **Metodologia de Pesquisa Jurimétrica**.
> ABJ, 2022. <https://livro.abj.org.br>. MIT licence.

> Trecenti, Julio; Hirota, Renata; Witkoski, Katerine; Associação Brasileira de
> Jurimetria. **abjData: Databases Used Routinely by the Brazilian Jurimetrics
> Association**. <https://github.com/abjur/abjData>. MIT licence.

If you reuse the data or the method, cite both the book and the `abjData` package.
The dataset is real public TJSP court metadata. The synthetic fixtures are
separate and are not used in any public number.

The exact upstream commit, file checksum and transformations are in
[`src/claimvalue/data/book/SOURCE.md`](src/claimvalue/data/book/SOURCE.md) and
[`CREDITS.md`](CREDITS.md). This project is not affiliated with or endorsed by
ABJ.

## What is in the repository

| Area | Files | What it does |
|---|---|---|
| Data | `book_data.py`, `scripts/build_book_dataset.py`, `src/claimvalue/data/book/` | Downloads the pinned ABJ table at runtime (never committed), validates it, runs the PII gate, writes the pipeline copy and a privacy report |
| SQL | `sql/01_book_schema.sql`, `02_book_quality.sql`, `03_book_abt.sql` | Constraints, quality checks and a leakage-safe ABT built only from decisions published before each filing date |
| Features | `book_features.py` | Subject, municipality, party configuration, filing date and prior history; encoders fitted inside each CV fold |
| ML | `book_train.py`, `evaluate.py` | Temporal split with label purge, `TimeSeriesSplit`, dummy baselines, bootstrap CIs, permutation importance and error by subject |
| Explainability | `shap_explain.py` | Exact TreeSHAP for both tasks, beeswarm, waterfall, dependence, per-case CSV and additivity checks |
| EDA | `book_eda.py` | Descriptive report and figures for the public sample |
| API | `api.py` | Predictions and per-case SHAP explanations behind an API key (synthetic schema) |
| Synthetic testbed | `synthetic.py`, `anonymize.py`, `train.py` | Fictional records for tests and a known-signal benchmark; not the public results |
| DevOps | `Dockerfile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, MLflow | Lint, tests, end-to-end smoke run, secret scan, image build and experiment tracking |

## Quick start

```bash
uv sync --extra dev --extra api --extra eda --extra dl
cp .env.example .env                  # needed for the API and the synthetic pipeline
uv pip install pyreadr                 # only needed to read the upstream .rda file
claimvalue book-pipeline --n-iter 4   # fetch -> prepare -> SQL -> EDA -> train -> SHAP -> PII
pytest
```

If the editable install is hidden by macOS or iCloud, run the same commands with
`PYTHONPATH="$PWD/src"` and the module form:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m claimvalue.cli book-pipeline --n-iter 4
```

Individual book steps:

```bash
claimvalue book-fetch
claimvalue book-prepare
claimvalue book-build-db
claimvalue book-eda
claimvalue book-train --task both --n-iter 4
claimvalue book-explain --task both --max-cases 191
claimvalue book-check-pii
```

Synthetic testbed commands (kept for tests and comparison):

```bash
claimvalue pipeline --n 20000
claimvalue generate-synthetic --n 20000
claimvalue anonymize
claimvalue build-db
claimvalue train --task both
claimvalue explain --task both
claimvalue check-pii
```

## Results

Run date 2026-11-11. Last 20% of filing dates as the test window, cutoff
2021-04-26. The test set was touched once, after model selection.

### Claim value

| Model | CV MAE (R$) | CV SD | Test MAE | Test MdAPE | Test R² |
|---|---:|---:|---:|---:|---:|
| decision tree (selected) | 34,717 | 14,968 | 14,862 | 0.580 | -0.096 |
| random forest | 35,074 | 14,664 | 14,259 | 0.542 | -0.026 |
| LightGBM | 35,134 | 14,692 | 14,432 | 0.582 | 0.065 |
| histogram GBM | 35,376 | 14,627 | 15,165 | 0.642 | 0.045 |
| dummy median | 35,494 | 14,640 | 14,959 | 0.564 | -0.026 |

The selected tree has `max_depth=3` and `min_samples_leaf=30`. The gap to the
median baseline is smaller than the confidence interval. The random forest has a
lower test MAE, but it was not selected on training CV; switching after seeing
the test set would be test-set tuning.

### Appeal reform

| Model | CV Brier | CV SD | Test Brier | Test AUC | Test PR-AUC |
|---|---:|---:|---:|---:|---:|
| LightGBM (selected) | 0.247 | 0.004 | 0.255 | 0.530 | 0.521 |
| random forest | 0.249 | 0.005 | 0.249 | 0.535 | 0.529 |
| dummy prior | 0.250 | 0.007 | 0.252 | 0.500 | 0.503 |
| decision tree | 0.255 | 0.009 | 0.253 | 0.536 | 0.517 |
| histogram GBM | 0.257 | 0.009 | 0.260 | 0.507 | 0.519 |

Brier is the selection metric because the output is a probability. The selected
model's test Brier is slightly worse than the dummy, and the AUC interval
includes 0.5. The result does not support a production risk score.

## SHAP results

191 test cases per task, exact TreeExplainer, additivity checked on every case.

| Check | Regression | Classification |
|---|---|---|
| Additivity max error | 1.78e-15 | 1.11e-15 |
| Output space | `log1p(BRL)` | log-odds |
| Largest share of mean absolute SHAP | `assunto_Fornecimento de Energia` 48.9% | `tipo_litigio_PF-nPF` 19.9% |
| Second | `assunto_n_prior` 24.5% | `filing_month` 14.0% |
| Third | `tipo_litigio_nPF-nPF` 15.5% | `assunto_avg_value_prior` 12.4% |

The regression model is a shallow tree, so several features receive zero SHAP.
That is not proof they are irrelevant; the selected model never split on them.
SHAP explains the trained model, not the world. Per-case stories and plots are in
`models/book/reports/shap/` and [`docs/10_book_results.md`](docs/10_book_results.md).

## The synthetic testbed

The synthetic pipeline is still useful: it has a known signal, so it can test the
whole chain (anonymisation, PII gate, leakage-safe ABT, training, SHAP, API) and
serve as a regression test for the code. Its old scores (MAE R$ 3,396, AUC 0.682)
belong to that controlled setting. They are **not** evidence about real courts.
The documents that describe it carry a synthetic-testbed banner.

## Documentation

| Doc | Topic |
|---|---|
| [`docs/00_data_lifecycle.md`](docs/00_data_lifecycle.md) | Map of every stage, plus the vocabulary reviewers ask about |
| [`docs/09_book_dataset.md`](docs/09_book_dataset.md) | The ABJ dataset, the two tasks, the leakage-safe ABT, privacy and licence |
| [`docs/10_book_results.md`](docs/10_book_results.md) | Full results, SHAP attribution, why the signal is weak, reproduction |
| [`docs/01_etl.md`](docs/01_etl.md) | Extract, transform, load; layers, reconciliation, idempotency, scaling |
| [`docs/02_cleaning_and_munging.md`](docs/02_cleaning_and_munging.md) | Quality dimensions, cleaning decisions, missing-data types, join traps |
| [`docs/03_eda.md`](docs/03_eda.md) | Four-stage EDA playbook, findings, pitfalls (synthetic testbed examples) |
| [`docs/04_ml.md`](docs/04_ml.md) | Framing, leakage taxonomy, validation, metrics, overfitting, MLOps (synthetic testbed) |
| [`docs/05_strategy.md`](docs/05_strategy.md) | Decision chain, maturity ladder, value and risk, learning plan |
| [`docs/06_shap.md`](docs/06_shap.md) | SHAP run report for the synthetic testbed |
| [`docs/07_shap.md`](docs/07_shap.md) | SHAP method, worked example, units, plots, p-value vs SHAP |
| [`docs/08_executive_summary.md`](docs/08_executive_summary.md) | Plain-language guide for non-technical readers (synthetic testbed) |
| [`docs/11_stakeholder_brief.md`](docs/11_stakeholder_brief.md) | Plain-language guide to the **public** run: results, jargon, privacy boundary |
| [`docs/LGPD.md`](docs/LGPD.md) | Pseudonymisation vs anonymisation, principles to code, risk assessment |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every field: type, LGPD class, treatment |
| [`DISCLAIMER.md`](DISCLAIMER.md) | Personal-study notice, data provenance and NDA declaration |
| [`CREDITS.md`](CREDITS.md) | Book and dataset citations, MIT notice |

## Check status

| Item | Status |
|---|---|
| Book pipeline, fetch to PII check | Run end to end on 2026-11-11 |
| Runtime data | Downloaded into `RAW_DATA_DIR/book/`; not committed; SHA-256 verified |
| SHAP on both book tasks, 191 cases each | Run; additivity 1.78e-15 and 1.11e-15 |
| Model-vs-baseline p-values | Run; Wilcoxon p=0.280 (regression) and p=0.846 (classification) |
| Test suite | 59 tests passed on 2026-11-11; covers the synthetic chain and a synthetic book-schema fixture |
| LightGBM in the model search | Run; selected for classification |
| XGBoost in the model search | Run; included in both leaderboards |
| Dockerfile, GitHub Actions, pre-commit hooks, gitleaks | Written, not executed in this environment |
| Legal sufficiency of the anonymisation | Not assessed. Needs a DPO or legal review; see `docs/LGPD.md` |

## Known limits

- The public sample is a 1,000-row sample of TJSP consumer appeals with decisions
  in 2020 and 2021. It is not a national or current sample. The prepared data is
  downloaded at runtime and is not in the repository.
- The features are limited to filing-time metadata. The file has no judge, panel,
  first-instance facts, legal arguments or full text.
- The selected models do not beat the baselines reliably. Do not use them for
  provisioning, settlement or risk decisions.
- Temporal validation reduces leakage but does not remove distribution shift.
- The API serves the synthetic schema. The book models are CLI artefacts for now.
- Nothing here is legal advice.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: claimvalue` after an editable install | macOS or iCloud marks the venv `.pth` files as hidden and Python skips them | `chflags nohidden .venv/lib/python*/site-packages/*.pth`, or run with `PYTHONPATH="$PWD/src"` |
| `lib_lightgbm.dylib ... libomp.dylib` on import | Homebrew OpenMP runtime missing | `brew install libomp` |
| `claimvalue book-fetch` says `pyreadr` is missing | The optional R-data reader is not installed | `uv pip install pyreadr` |
| `claimvalue book-build-db` says the prepared file is missing | `book-fetch` / `book-prepare` has not run | `claimvalue book-fetch && claimvalue book-prepare` |
| `book-train` is slow | Six candidates times four CV folds times the random search | Use `--n-iter 1` for a smoke run; keep `4` for the documented result |

## Licence

The project code is MIT licensed; see [`LICENSE`](LICENSE). The book and the
`abjData` package are MIT licensed by ABJ; see [`CREDITS.md`](CREDITS.md).
