# README.md 
### claimvalue-ml

Leakage-safe ML pipeline that predicts **claim value** (regression) and **ruling outcome** (classification) from
court-case data, with LGPD-oriented anonymization, a SQL feature layer, EDA as code, and MLOps basics.

> **Synthetic data only.** The repo contains no real cases. `claimvalue generate-synthetic` creates fictional raw
> records (fake names, deliberately invalid CPFs, random case numbers) so the anonymization step runs exactly as
> it would on a real export. Real data stays outside the repo (`RAW_DATA_DIR` in `.env`). See `docs/LGPD.md`.

## What it demonstrates

| Area | Where | Highlights |
|---|---|---|
| **Python** | `src/claimvalue/` | typed modules, CLI, config from environment, 43 tests, ruff |
| **SQL** | `src/claimvalue/sql/` | constrained schema, data-quality report, rolling-window ABT (history strictly before the filing date) |
| **ML** | `train.py`, `features.py`, `evaluate.py`, `anomaly.py` | temporal split + label purge, `TimeSeriesSplit` CV, encoders inside the Pipeline, dummy baselines, bootstrap CIs, permutation importance, error by UF, Isolation Forest |
| **DL** | `dl.py` | PyTorch embedding MLP with time-aware early stopping (see status below) |
| **DevOps** | `Dockerfile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, MLflow, FastAPI | lint + tests + end-to-end smoke + gitleaks + image build; API key, validation, non-root container |
| **Privacy / LGPD** | `anonymize.py`, `docs/LGPD.md` | keyed pseudonyms, date shift, amount generalisation, text scrubbing and art. 11 redaction, secrecy exclusion, k-anonymity / l-diversity, PII scan that fails closed |

## Quick start

```bash
pip install -e ".[dev,api,eda]"          # add ,dl for PyTorch
cp .env.example .env                      # set ANONYMIZATION_PEPPER (openssl rand -hex 32) and API_KEY
claimvalue pipeline --n 20000             # generate -> anonymize -> SQL -> EDA -> train both tasks -> anomalies -> PII check
uvicorn claimvalue.api:app                # then POST /predict/claim-value with header X-API-Key
pytest
```

Individual steps: `generate-synthetic`, `anonymize [--suppress-k 5] [--claim-sig-digits 2] [--delete-raw]`, `build-db`, `eda`,
`train --task regression|classification|both`, `train-dl`, `anomalies`, `check-pii`.

## Results (synthetic data, seed 42, future-period test set)

| Task | Best (chosen by CV) | Metric | Dummy baseline | 95% CI |
|---|---|---|---|---|
| Claim value | random forest | MAE R$3,395 (MdAPE 0.39) | R$4,910 (MdAPE 0.59) | 3,234 to 3,558 |
| Ruling outcome | random forest | ROC-AUC 0.681 | 0.500 | 0.665 to 0.698 |

The three tree models are statistically indistinguishable; the point is the *method*, not the number. See `docs/04_ml.md`.

## Documentation (read in order)

| Doc | Topic |
|---|---|
| `docs/00_data_lifecycle.md` | map of every stage, vocabulary (ETL, munging, cleaning, ABT, EDA, leakage) |
| `docs/01_etl.md` | extract/transform/load, layers, row reconciliation, idempotency, scaling path |
| `docs/02_cleaning_and_munging.md` | quality dimensions, cleaning decision log, missing-data types, join traps |
| `docs/03_eda.md` | four-stage EDA playbook, findings and what each means, pitfalls |
| `docs/04_ml.md` | framing, leakage taxonomy, validation, metrics, results, overfitting, DL, MLOps, lessons from the original code |
| `docs/05_strategy.md` | decision chain, maturity ladder, management template, value/risk, consulting lens, STAR story, learning roadmap |
| `docs/LGPD.md` | pseudonymization vs anonymization, principles to code, risk assessment, checklist |
| `docs/data_dictionary.md` | every field: type, LGPD class, treatment |

## Verification status (what was actually run)

| Item | Status |
|---|---|
| Lint (ruff), 43 tests passing, 70% coverage | run |
| Full pipeline on 20,000 synthetic cases, including PII check | run (stages executed separately) |
| `dl.py` and `tests/test_dl.py` | **not executed**: PyTorch did not fit in the authoring sandbox. Run `pip install -e ".[dl]" && pytest tests/test_dl.py` |
| Dockerfile, GitHub Actions workflow, pre-commit hooks, gitleaks | **written, not executed** |
| Legal sufficiency of the anonymization | **not assessed**: needs your DPO/legal team (see `docs/LGPD.md`) |

## Known limits

Regex scrubbing cannot find unknown names in free text (add NER and review before any real text leaves a controlled
environment); date shifting and amount rounding reduce but do not eliminate linkage risk (2.73% of synthetic records sit in
groups below k=5 for an outsider with public dockets); SQLite and a full rebuild are fine for a portfolio and wrong for
production volume (see the scaling table in `docs/01_etl.md`).
