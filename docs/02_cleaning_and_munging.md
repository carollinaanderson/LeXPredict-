# 02 · Data cleaning and munging

Cleaning = making data *correct*. Munging = making data *usable* (shape, join, window). Do both
**reproducibly**: no manual edits, every decision in code and in this log.

## Data quality dimensions and the check for each

| Dimension | Question | Check here | Result (synthetic run) |
|---|---|---|---|
| Completeness | what is missing? | `pct_missing_claim_value` in `02_data_quality.sql` | 3.03% |
| Uniqueness | duplicate records? | `drop_duplicates("case_id")` + PK | 197 duplicates removed |
| Validity | values in allowed domain? | CHECK constraints, outcome mapping | 0 violations (bad rows raise) |
| Consistency | do fields agree? | `decision_date >= filing_date` | enforced |
| Timeliness | is the period what you think? | `first_filing` / `last_filing`, volume plot | check edges for truncation |
| Plausibility | outliers/errors? | `anomaly.py`, p99 vs max | 193 flagged for review |

## Cleaning decision log

| Issue | Detection | Decision | Why | Risk to watch |
|---|---|---|---|---|
| Missing `claim_value` | quality report, `missingness(by uf)` | drop for regression only; keep for classification | the target cannot be imputed honestly | missing not at random (e.g. high-value cases unrecorded) biases the model |
| Missing numeric features (no judge history) | `judge_n_prior = 0` | leave as NaN | tree models handle NaN natively; zero is a different fact from "unknown" | do not fill with the mean: it invents history |
| Duplicates | row count vs unique ids | keep first, count removed | exports re-send records | different versions of the same case: prefer the latest by `updated_at` in production |
| Right-skewed amounts | skew 3.09 raw vs -0.23 on log | model `log1p(y)`, report errors in BRL | squared-error models chase outliers otherwise | back-transform bias: report medians too |
| Extreme values | p99 vs max, anomaly flags | **keep** and flag, do not clip silently | an outlier may be the most valuable case | clipping the target hides risk |
| Unknown categories at prediction time | new judge or comarca | encoders use `handle_unknown` / smoothing toward the global mean | models must not crash on new ids | cold start for new judges |
| Judicial secrecy | `segredo_justica` flag | exclude from the pipeline entirely | legal, not statistical | verify the flag exists in real exports |

## Missing data: three mechanisms

| Type | Meaning | Example | Handling |
|---|---|---|---|
| MCAR | missing completely at random | random export glitch | dropping is unbiased |
| MAR | depends on observed data | one comarca's clerks skip the field | model the pattern or stratify; check `missingness(by=...)` |
| MNAR | depends on the missing value itself | very large claims left blank | cannot be fixed with statistics; fix at the source |

## Munging patterns and their traps

| Task | Pattern | Trap | Guard |
|---|---|---|---|
| Join tables | `pd.merge(..., validate="one_to_one")` | fan-out inflates rows | validate + row-count assertion |
| Rolling history | SQL self-join with `h.decision_date < c.filing_date` | using the case's own or future outcomes | test `test_rolling_features_use_only_past_decisions` recomputes it in pandas |
| Group aggregates | CTE per grain, then join | two joins at different grains double count | one CTE per grain |
| Encode categories | encoder inside the Pipeline | fitting encoders on all data leaks the target | `TargetEncoder` is cross-fitted inside CV |
| Dates | parse once, ISO-8601 | `errors="coerce"` hides failures | count NaT |
| Text | scrub, then keep | regex cannot find unknown names | NER + human review for real text |

## Cleaning checklist (reuse on any dataset)

1. Row count and column types vs the data dictionary
2. Duplicates by business key
3. Missing % per column and per group
4. Domain checks (ranges, categories, date order)
5. Distribution of every numeric (skew, zeros, p99/max)
6. Cross-field consistency
7. Reconcile counts after each step
8. Write each decision down with its reason
