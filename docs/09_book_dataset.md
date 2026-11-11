# 09 · The ABJ book dataset and how it maps to this project

> **Source.** Ricardo Feliz and Julio Trecenti, *Metodologia de Pesquisa Jurimétrica*
> (ABJ, 2022), <https://livro.abj.org.br>. Dataset: `abjData::consumo`, from the
> `abjData` package, <https://github.com/abjur/abjData>. Both are MIT licensed.
> This project is not affiliated with, endorsed by, or sponsored by ABJ. Full notice:
> [`CREDITS.md`](../CREDITS.md) and
> [`src/claimvalue/data/book/SOURCE.md`](../src/claimvalue/data/book/SOURCE.md).

This is the dataset behind the public analysis. The synthetic generator in
`src/claimvalue/synthetic.py` is still in the repository, but only for automated
tests and for a controlled benchmark. It is not used for the results in
`docs/10_book_results.md`.

## 1. Why this dataset

The previous version of the project ran on synthetic records. That made the
privacy story simple, but it could not answer a real jurimetric question. The
ABJ book supplies both a public dataset and a research method:

- the dataset is a **1,000-row sample of second-degree consumer cases from the
  Court of Justice of São Paulo (TJSP)**;
- the book treats planning, the data generating process, descriptive statistics
  and visualisation as one chain, with explicit attention to what is known when;
- both the book and the package are MIT licensed, so the data can be used and
  redistributed with attribution.

The dataset contains no names, CPFs, CNPJs, addresses or free-text case
descriptions. The public case number is replaced by a stable SHA-256 pseudonym
in the prepared table. The prepared table is written only to a gitignored
runtime directory and is never committed to the repository.

## 2. What `consumo` contains

| Column in the source | Meaning | Use here |
|---|---|---|
| `id_processo` | Public case number | Dropped; replaced by `case_id` (SHA-256 pseudonym) |
| `assunto` | Case subject | Feature |
| `comarca` | Municipality | Feature |
| `valor` | Value of the cause, BRL | Regression target; feature for classification |
| `tipo_litigio` | Party configuration (PF-nPF, nPF-PF, nPF-nPF, PF-PF) | Feature |
| `dec_val` | Second-instance outcome | Label source |
| `dec_unanime` | Unânime or Maioria | Not used as a feature (known only after the decision) |
| `dec_date` | Decision date | Used to derive the filing date and to order history |
| `tempo` | Case duration in days | Used only to derive `filing_date`; not a feature |

Descriptive facts from the prepared file:

| Fact | Value |
|---|---|
| Rows | 1,000 |
| Subjects | 5 (Bancários, Fornecimento de Energia, Transporte Aéreo, Planos de Saúde, Telefonia) |
| Municipalities | 181 |
| Party configurations | 4 |
| Filing dates | 2015-06-17 to 2021-12-07 |
| Decision dates | 2020-01-07 to 2021-12-21 |
| Rows with a binary reform label | 952 |
| Reformed (Reformou or Parcial) | 441 (46.3% of labelled rows) |
| Not reformed (Não reformou) | 511 |
| Procedural outcomes (no binary label) | 48 |
| Claim value, median | R$ 13,784.78 |
| Claim value, mean | R$ 92,393.09 |
| Claim value, p95 | R$ 100,083.17 |
| Claim value, maximum | R$ 53,440,459.63 |

The claim value is strongly right-skewed. The model therefore learns on
`log1p(claim_value)` and converts predictions back to BRL.

## 3. The two tasks

### 3.1 Claim value (regression)

- Target: `claim_value`, the value of the cause.
- Features known at filing: `assunto`, `comarca`, `tipo_litigio`, `filing_month`,
  `filing_year`, and prior history by subject and municipality.
- Not used: `decision_outcome`, `case_days`, `unanimity`, or anything else that
  is only known after the decision.

The question is: **given what is known when the appeal is filed, what claim
value should we expect?** This is a planning and provisioning question, not a
prediction of the final award.

### 3.2 Appeal reform (classification)

- Target: `reformed = 1` for `Reformou` or `Parcial`; `0` for `Não reformou`.
- Procedural outcomes (`Não Conheceram / Prejudicado / Diligência`,
  `Desistência`) have no binary label and are excluded from the modelling table.
- Features: the same pre-filing set, plus `claim_value_log`.

The question is: **will the second-instance decision change the first-instance
decision?** This is a different question from "will the ruling go against the
company". `consumo` records the party configuration, but not which side
appealed, so "against the company" cannot be identified without assuming who
filed the appeal. The project therefore uses the target the data can actually
support.

## 4. Time, leakage and the ABT

`consumo` has no filing date. It has the decision date and the case duration, so:

```text
filing_date = decision_date - case_days
```

The split is temporal: the last 20% of filings by date form the test window. For
classification, training rows whose decision falls after the cutoff are purged,
because their label was not known at the cutoff. The test set is touched once.

`src/claimvalue/sql/03_book_abt.sql` builds the historical features:

- group by `comarca` and by `assunto`;
- use only cases whose **decision date is before the current filing date**;
- restrict to the last `window_days` (365 by default);
- compute prior counts, prior mean claim value, and prior reform rate.

There is no judge identifier in `consumo`, so the groups are subject and
municipality. The idea is the same one used by the synthetic pipeline's judge
and comarca history: every row is a snapshot at its own filing date. Nothing
that was only known after that date can enter a feature.

## 5. Privacy and licensing

- `case_id` is a stable SHA-256 pseudonym of the public case number. The raw
  number is not stored in the prepared file.
- No names, CPFs, CNPJs, addresses or free-text case descriptions are present.
- `claimvalue book-prepare` runs the same `assert_no_pii` gate used by the
  synthetic pipeline and writes `data/processed/book_privacy_report.json`.
- The data is public and MIT licensed. Attribution is required and is included
  in `CREDITS.md`, `README.md`, the provenance files in
  `src/claimvalue/data/book/`, and the metadata of every trained model.
- The runtime data is never committed and is not part of any release artefact.

## 6. How the runtime download works

The repository does **not** contain `consumo_prepared.csv`. The ready-to-use
table is downloaded, verified and prepared at runtime, then written under
`RAW_DATA_DIR/book/` (default `data/raw/book/`), which is gitignored.

| Item | Value |
|---|---|
| `abjData` commit | `afa83b359917b6e974fbe7281f340a66b2a86cfe` |
| Source URL | `https://raw.githubusercontent.com/abjur/abjData/afa83b359917b6e974fbe7281f340a66b2a86cfe/data/consumo.rda` |
| Source SHA-256 | `3a90cd4c05ed1aa206abe8ffc59b68d0b0c629bfaaeb16bd12f9a0079008fad8` |
| Command | `claimvalue book-fetch` |
| Output | `RAW_DATA_DIR/book/consumo_prepared.csv` and `SOURCE.json` |

```bash
# pyreadr is needed only to read the upstream .rda file.
pip install pyreadr

# Download the pinned file, verify its SHA-256 and prepare the table.
claimvalue book-fetch

# Or use a local copy of consumo.rda.
python scripts/build_book_dataset.py --rda /path/to/consumo.rda --force
```

`claimvalue book-pipeline` runs `book-fetch` first automatically. The script and
the CLI run the same validation: row count, unique case IDs, date order, target
values, positive claim values and the absence of CPF/CNPJ-shaped strings before
writing anything. If the file is already cached and valid, `book-fetch` reuses
it; `--force` re-downloads it.

## 7. Limits

- The file is a **1,000-row sample**, not the full TJSP universe. Confidence
  intervals are wide.
- It covers second-degree consumer cases in São Paulo with decisions in
  2020 and 2021. It is not a national or current sample.
- It has no judge, panel, first-instance facts, legal arguments or full text.
- The outcome is the appeal result, not the final payment or the first-instance
  decision.
- A weak model here is a finding about this sample and these features, not a
  claim that appeal outcomes are unpredictable in general.
- Nothing in this repository is legal advice.
