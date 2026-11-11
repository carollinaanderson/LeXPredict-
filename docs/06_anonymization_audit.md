# 06 · Anonymization audit & best-practice notes 

> Purpose: answer "is the sensitive data anonymized?" with evidence, then show which engineering
> practices the pipeline demonstrates. This is a personal study project. The public analysis uses
> the public, MIT-licensed `abjData::consumo` sample; the synthetic pipeline is a testbed. Not
> legal advice. See `docs/09_book_dataset.md`, `docs/LGPD.md` and `DISCLAIMER.md` for provenance.

## 1. The audit verdict

**Yes, for direct identifiers and secrets.** The repository contains no employer or client data,
no names, CPFs or CNPJs of real parties, and no secrets. The public analysis uses case-level
metadata from ABJ with the public case number replaced by a SHA-256 pseudonym and dropped. The
synthetic records are fictional. Evidence:

| Check | Result | Where |
|---|---|---|
| Real case data present? | No employer/client data. Public ABJ sample with no direct identifiers; synthetic fixtures for tests. The ABJ table is downloaded at runtime and is not committed | `src/claimvalue/data/book/` (provenance only), `synthetic.py` |
| Data/model artifacts committed? | No. Runtime data and generated artefacts are gitignored; the pre-commit `no-data-files` hook blocks data files with no exceptions | `.gitignore`, `.pre-commit-config.yaml` |
| Hardcoded secrets? | None — pepper & API key from env only, `.env.example` ships empty | `config.py`, `.env.example` |
| Secret scanning in CI? | Yes — gitleaks + `detect-private-key` | `.pre-commit-config.yaml`, `ci.yml` |
| Anonymization gate? | Book path: SHA-256 case pseudonym, schema/date/label checks, `assert_no_pii`. Synthetic path: drops identifiers, pseudonymizes, generalizes, scrubs, redacts, fails closed | `book_data.py`, `anonymize.py` |

The book path uses a separate, smaller gate because the source has no names, CPFs, CNPJs,
addresses or free text:

1. `scripts/build_book_dataset.py` / `claimvalue book-fetch` replaces the public
   case number with a stable SHA-256 pseudonym, does not store the raw number,
   and writes the prepared table only to a gitignored runtime directory.
2. `book_data.py` validates the schema, unique IDs, positive claim values, date order, target
   values and the absence of CPF/CNPJ-shaped strings.
3. `claimvalue book-prepare` runs `assert_no_pii` and writes
   `data/processed/book_privacy_report.json`.

The synthetic/raw-export gate in `anonymize.py` applies, in order:

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

## 2. What the pipeline demonstrates

The project exercises the skills a data-science role typically asks for: **Python and SQL, work on
large and complex datasets, predictive modelling and forecasting, supervised learning and
classification, and communication of findings to technical and non-technical readers.** The table
maps each skill to evidence in the repository.

| Skill | Where it is demonstrated |
|---|---|
| Strong **Python** | typed modules, CLI, config-from-env, the full test suite, ruff, coverage gate |
| Strong **SQL** | constrained schema, data-quality report, leakage-safe rolling-window ABT (`sql/03_abt.sql`) |
| **Large datasets** | pipeline runs on 20k+ rows; scaling path documented in `docs/01_etl.md` |
| **Predictive modeling** | regression (claim value) + classification (ruling outcome), dummy baselines, bootstrap CIs |
| **Forecasting** | time-ordered split, `TimeSeriesSplit` CV, label purge — the same discipline as demand forecasting |
| **Supervised learning / classification** | RF / HistGBM / DT with encoders inside the pipeline (no leakage) |
| **Business impact** | every metric is benchmarked against a dummy baseline; error sliced by UF to show *where* the model is weak |
| **Communication** | `docs/05_strategy.md` has a management template and a STAR project story; EDA report is auto-generated prose, not just charts |

The most transferable idea: **leakage-safe, time-aware modeling**. Demand forecasting has the same
trap this repo fixes: using information in training that would not exist at prediction time (e.g. a
"days to resolution" feature that is only known after the case ends). `features.py:temporal_split`
and `sql/03_abt.sql` encode exactly that discipline.

## 3. Best practices this repo demonstrates

1. **Raw is immutable; every step is idempotent and rerunnable** — derive, never edit.
2. **Fail loudly and early** — CHECK constraints and `assert_no_pii` stop bad data at the door.
3. **Encoders live inside the pipeline** — fitted per CV fold, so no target leakage.
4. **Measure against a baseline** — a number without a dummy model next to it means nothing.
5. **Time is a first-class citizen** — split, features and labels all respect "what was known when".
6. **Secrets never touch code or git** — env + secrets manager, gitleaks, non-root container.
7. **Document decisions, not just code** — every doc has a "why" column.

## 4. Honest residual risks

- **Pseudonymization ≠ anonymization.** While the pepper exists, the data is still personal under
  LGPD. The `privacy_report.json` labels the dataset "PSEUDONYMIZED" until a documented risk
  assessment concludes otherwise.
- **Regex can't find unknown names** (lawyers, witnesses, relatives). Real free text needs NER
  (e.g. a spaCy `pt` model) plus a human spot-check of ≥200 samples.
- **Linkage risk is not zero.** On the synthetic testbed, 2.73% of records sit in groups below k=5 for an
  outsider with public dockets. Mitigations: bigger generalization bands, `--suppress-k 5`, or
  releasing only aggregates.
- **Judge profiling is ethically sensitive** — treat as a legal/ethics review item, not just a
  feature (`docs/LGPD.md` §6).

## 5. One-line summary

> "The public analysis ships a small, public, MIT-licensed ABJ sample with no direct identifiers:
> the raw case number is replaced by a SHA-256 pseudonym, the schema and dates are validated, and a
> fail-closed PII scan runs before the prepared file is used. The synthetic testbed is fictional.
> For a raw identifiable export, the anonymization gate drops direct identifiers, pseudonymizes with
> a keyed HMAC, generalizes dates and amounts, scrubs free text, redacts art. 11 vocabulary and
> blocks any write that still matches an identifier pattern. I also measure re-identification risk
> with k-anonymity and l-diversity, and I'm explicit that pseudonymization is not anonymization
> while the key exists."
