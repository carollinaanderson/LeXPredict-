# 01 · ETL: Extract, Transform, Load

## What each letter means in this repo

| Letter | Decision | Implementation | Why |
|---|---|---|---|
| **E** | Source is a flat export read from `RAW_DATA_DIR` | `cli.py` reads `raw_cases.csv` | The raw file is the only place identifiers exist; keeping it outside the repo shrinks the blast radius |
| **T** | Pseudonymize, generalize, type, deduplicate, map outcomes | `anonymize.py`, `db.build_db` | Privacy and quality fixes happen *before* analysis so no downstream artefact ever holds identifiers |
| **L** | Load into SQL with constraints | `sql/01_schema.sql` | A database with rules rejects bad rows; a CSV accepts anything |

## Layers (medallion view)

| Layer | Meaning | Here | Typical enterprise tool |
|---|---|---|---|
| Bronze | raw, as received | `raw_cases.csv` (identifiable, restricted) | landing zone / Delta bronze |
| Silver | cleaned, conformed, typed | `cases` table (anonymized, constrained) | Delta silver, dbt staging |
| Gold | business-ready features/marts | `03_abt.sql` result | Delta gold, dbt marts, feature store |

## Row reconciliation: always account for every row

| Step | Rows | Delta | Reason |
|---|---|---|---|
| Raw export | 20,200 | | 20,000 cases + 200 duplicated rows |
| After secrecy filter | 20,005 | -195 | cases under judicial secrecy never leave the source |
| After dedupe on `case_id` | 19,808 | -197 | duplicates from the export |
| Usable for regression | about 19.2k | -3.03% | target (`claim_value`) missing |

If these numbers surprise you in a real run, **stop and investigate before modelling**. Silent row loss
is the most common source of wrong results.

## Schema contract (what `01_schema.sql` enforces)

| Rule | Prevents |
|---|---|
| `case_id` PRIMARY KEY | double counting after joins |
| `claim_value > 0` or NULL | negative / zero amounts breaking log transforms |
| `decision_date >= filing_date` | impossible timelines (and leakage bugs in window features) |
| `unfavorable IN (0,1)` | label drift |
| `length(uf) = 2` | free-text states |
| unknown `sentence_outcome` raises in `build_db` | a new ruling category silently becoming "favorable" |
| indexes on (judge_id, decision_date), (comarca, decision_date) | the self-join in the ABT turning quadratic |

## Idempotency, incrementality, backfills

* **Idempotent**: `build_db` drops and rebuilds the table, so reruns are safe. Cost: full reload.
* **Incremental (production)**: keep a watermark (`max(updated_at)` loaded), upsert by `case_id`, and
  recompute window features only for affected judges/comarcas. Required once volume makes full reloads slow.
* **Backfill**: because features are defined *as of each case's filing date*, a historic rebuild gives the
  same values the model would have seen then. That property is what makes retraining trustworthy.
* **Late-arriving data**: a ruling recorded days after the decision changes `unfavorable` history. Decide
  explicitly whether features use *decision date* or *registration date*; here: decision date.

## When SQLite stops being enough

| Signal | Move to |
|---|---|
| multiple writers, concurrent users | PostgreSQL |
| tens of millions of rows, window joins slow | Spark / Databricks (Delta), partition by month |
| many transformations with dependencies | dbt for SQL models + tests |
| schedules, retries, alerting | Airflow / Databricks Workflows |
| repeated features across models | feature store |

A sketch of the orchestration (what you would build next):

```
extract_raw  ->  anonymize  ->  pii_scan  ->  load_silver  ->  dq_checks  ->  build_abt  ->  train  ->  evaluate  ->  register
                                  (fail = stop)                  (fail = stop)                            (gate: beats champion?)
```

## ETL failure modes to be able to discuss

| Failure | Symptom | Defence |
|---|---|---|
| Join fan-out | row count grows after a join | aggregate in CTEs before joining (see `03_abt.sql`); assert row counts |
| Silent type coercion | `errors="coerce"` turns bad dates into NaT | count NaT after parsing; fail above a threshold |
| Timezone / date-format mix | off-by-one windows | store ISO-8601, parse once |
| Schema drift | new column / renamed field | schema contract + test on load |
| Partial load | half a table after a crash | transaction (`with sqlite3.connect`) |
