# Data dictionary

Classification: **D** = direct identifier, **Q** = quasi-identifier, **S** = sensitive-risk text, **N** = non-identifying.

## Raw export (identifiable, restricted, never in the repo)

| Field | Type | Class | Treatment |
|---|---|---|---|
| `process_number` (CNJ format) | text | D | -> `case_id` (HMAC pseudonym over digits only); original dropped |
| `cpf` | text | D | dropped |
| `party_name` | text | D | dropped; also scrubbed from text |
| `judge_name` | text | D | -> `judge_id` (HMAC pseudonym); original dropped; scrubbed from text |
| `uf`, `comarca`, `vara` | text | Q (public geography) | kept: they carry the signal |
| `filing_date`, `decision_date` | ISO date | Q | shifted by a secret whole-year offset |
| `claim_value` | BRL float | Q | rounded to 2 significant digits |
| `sentence_outcome` | category | N (attribute of interest) | kept |
| `sentence_text` | text | D/S risk | identifiers scrubbed; redacted if art. 11 vocabulary |
| `segredo_justica` (optional) | bool | control | rows with True are excluded |

## `cases` table (silver)

| Column | Type | Constraint |
|---|---|---|
| `case_id` | TEXT | PRIMARY KEY, `CASE_` + 10 hex |
| `judge_id` | TEXT | NOT NULL, `JUDGE_` + 10 hex |
| `uf` / `comarca` / `vara` | TEXT | NOT NULL, `length(uf)=2` |
| `filing_date` / `decision_date` | TEXT ISO | NOT NULL, `decision_date >= filing_date` |
| `claim_value` | REAL | NULL or > 0 |
| `sentence_outcome` | TEXT | one of 4 known values (else load fails) |
| `unfavorable` | INTEGER | 0/1, derived from the outcome |
| `sentence_text` | TEXT | scrubbed |

## ABT (gold): one row per case, features as of the filing date

| Feature | Meaning | Type | Encoder |
|---|---|---|---|
| `uf`, `vara` | place and court type | categorical, low cardinality | one-hot |
| `comarca`, `judge_id` | district, pseudonymized judge | categorical, high cardinality | smoothed target encoding (inside CV) |
| `filing_month` | month of filing | int 1..12 | passthrough |
| `judge_n_prior` | judge's decisions in the previous `window_days` | count | passthrough (0 = no history) |
| `judge_avg_value_prior` | their mean claim value | float, NaN if no history | passthrough |
| `judge_unfav_rate_prior` | their unfavorable rate | float 0..1 | passthrough |
| `comarca_*_prior` | same three, at district level | | passthrough |

Targets: `claim_value` (regression, modelled as `log1p`), `unfavorable` (classification).
Not features: `case_id`, `filing_date`, `decision_date` (the last is used only to purge labels at the split).
