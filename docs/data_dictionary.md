# Data dictionary

Classification: **D** = direct identifier, **Q** = quasi-identifier, **S** = sensitive-risk text, **N** = non-identifying.

## Book dataset (public analysis, `abjData::consumo`)

Source: Feliz and Trecenti, *Metodologia de Pesquisa Jurimétrica* (ABJ, 2022),
dataset `abjData::consumo`, MIT licence. See `docs/09_book_dataset.md` and
`CREDITS.md`. The table is downloaded and prepared at runtime; no data file is
committed.

| Field | Type | Class | Treatment |
|---|---|---|---|
| `case_id` | text | pseudonym | Stable SHA-256 pseudonym of the public case number; raw number dropped |
| `assunto` | category | Q (public metadata) | Kept as a feature |
| `comarca` | category | Q (public geography) | Kept as a feature; target-encoded inside the pipeline |
| `tipo_litigio` | category | Q (public metadata) | Party configuration; kept as a feature |
| `filing_date` | ISO date | Q | Derived as `decision_date - case_days`; used for the temporal split |
| `decision_date` | ISO date | Q | Used to order history and to purge training labels at the cutoff |
| `claim_value` | BRL float | Q | Regression target; `log1p` for modelling; `claim_value_log` is a classification feature |
| `decision_outcome` | category | N | Reformou, Parcial, Não reformou, or a procedural outcome |
| `reformed` | 0/1/null | N (label) | 1 for Reformou/Parcial, 0 for Não reformou, null for procedural outcomes |
| `unanimity` | category | N | Unânime or Maioria; **not** a feature (known only after the decision) |
| `case_days` | integer | Q | Case duration; used only to derive `filing_date`, **not** a feature |

No names, CPFs, CNPJs, addresses or free-text case descriptions are present in
the prepared file.

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
