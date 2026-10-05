# 06 · Anonymization audit & best-practice notes 

> Purpose: answer "is the sensitive data anonymized?" with evidence, then map the *method* to the
> Capgemini Data Scientist role (contact center, workforce management, customer experience). This is
> a portfolio narrative, not legal advice — see `docs/LGPD.md` for the LGPD engineering detail.

## 1. The audit verdict

**Yes — the repository contains no real personal data and no secrets.** It is anonymized *by
design*, not by a one-off cleanup. Evidence:

| Check | Result | Where |
|---|---|---|
| Real case data present? | No — synthetic only (fake names, invalid CPFs, random case numbers) | `synthetic.py`, `README.md` |
| Data/model artifacts committed? | None (`data/`, `models/`, `*.csv`, `*.db`, `*.pkl`, `*.pt`) | `.gitignore`, pre-commit `no-data-files` |
| Hardcoded secrets? | None — pepper & API key from env only, `.env.example` ships empty | `config.py`, `.env.example` |
| Secret scanning in CI? | Yes — gitleaks + `detect-private-key` | `.pre-commit-config.yaml`, `ci.yml` |
| Anonymization gate? | Drops identifiers, pseudonymizes, generalizes, scrubs, redacts, fails closed | `anonymize.py` |

The anonymization pipeline (`anonymize.py`) applies, in order:

1. **Drop direct identifiers** — CPF, party name, judge name, raw process number.
2. **Keyed pseudonyms** — HMAC-SHA256 with a secret pepper (a plain hash is brute-forceable on
   low-entropy ids like CPF).
3. **Generalization** — dates shifted by a secret whole-year offset (preserves order/season);
   claim values rounded to 2 significant digits.
4. **Free-text scrubbing** — CPF/CNPJ/CNJ/email/phone/CEP/OAB + known names → neutral tokens.
5. **Sensitive-data redaction** — any text with art. 11 vocabulary (health, religion, politics,
   race, genetic…) is replaced wholesale.
6. **Secrecy exclusion** — cases under judicial secrecy never leave the source.
7. **Risk measurement** — k-anonymity and l-diversity quantify linkage/homogeneity risk.
8. **Fail-closed PII scan** — `assert_no_pii` raises before any write if a pattern survives.

## 2. Why this maps to the Data Scientist JD

The JD asks for: **Python + SQL, large/complex datasets, predictive modeling & forecasting,
supervised learning & classification, and turning findings into recommendations for technical and
non-technical stakeholders.** This repo is a direct, demonstrable match:

| JD requirement | Where it is demonstrated |
|---|---|
| Strong **Python** | typed modules, CLI, config-from-env, 43 tests, ruff, 70% coverage |
| Strong **SQL** | constrained schema, data-quality report, leakage-safe rolling-window ABT (`sql/03_abt.sql`) |
| **Large datasets** | pipeline runs on 20k+ rows; scaling path documented in `docs/01_etl.md` |
| **Predictive modeling** | regression (claim value) + classification (ruling outcome), dummy baselines, bootstrap CIs |
| **Forecasting** | time-ordered split, `TimeSeriesSplit` CV, label purge — the same discipline as contact-center volume forecasting |
| **Supervised learning / classification** | RF / HistGBM / DT with encoders inside the pipeline (no leakage) |
| **Business impact** | every metric is benchmarked against a dummy baseline; error sliced by UF to show *where* the model is weak |
| **Communication** | `docs/05_strategy.md` has a management template + STAR story; EDA report is auto-generated prose, not just charts |

The single most transferable idea: **leakage-safe, time-aware modeling**. Contact-center
forecasting has the same trap this repo fixes — using information in training that would not exist
at prediction time (e.g. a "days to resolution" feature that is only known *after* the call ends).
`features.py:temporal_split` and `sql/03_abt.sql` encode exactly that discipline.

## 3. Best practices this repo demonstrates (and you can cite)

1. **Raw is immutable; every step is idempotent and rerunnable** — derive, never edit.
2. **Fail loudly and early** — CHECK constraints and `assert_no_pii` stop bad data at the door.
3. **Encoders live inside the pipeline** — fitted per CV fold, so no target leakage.
4. **Measure against a baseline** — a number without a dummy model next to it means nothing.
5. **Time is a first-class citizen** — split, features and labels all respect "what was known when".
6. **Secrets never touch code or git** — env + secrets manager, gitleaks, non-root container.
7. **Document decisions, not just code** — every doc has a "why" column.

## 4. Honest residual risks (say these in an interview — it reads as seniority)

- **Pseudonymization ≠ anonymization.** While the pepper exists, the data is still personal under
  LGPD. The `privacy_report.json` labels the dataset "PSEUDONYMIZED" until a documented risk
  assessment concludes otherwise.
- **Regex can't find unknown names** (lawyers, witnesses, relatives). Real free text needs NER
  (e.g. a spaCy `pt` model) plus a human spot-check of ≥200 samples.
- **Linkage risk is not zero.** On synthetic data, 2.73% of records sit in groups below k=5 for an
  outsider with public dockets. Mitigations: bigger generalization bands, `--suppress-k 5`, or
  releasing only aggregates.
- **Judge profiling is ethically sensitive** — treat as a legal/ethics review item, not just a
  feature (`docs/LGPD.md` §6).

## 5. One-line answer for the interview

> "The repo ships synthetic data only, and the anonymization is a *gate* in the pipeline — direct
> identifiers are dropped, the rest is pseudonymized with a keyed HMAC, dates and amounts are
> generalized, free text is scrubbed and art. 11 vocabulary redacted, and a fail-closed PII scan
> blocks any write that still matches an identifier pattern. I also measure re-identification risk
> with k-anonymity and l-diversity, and I'm explicit that pseudonymization is not anonymization
> while the key exists."
