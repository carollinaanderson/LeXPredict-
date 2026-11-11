# Disclaimer and statement of provenance

## Personal study project

This repository is a personal study and portfolio project. It is not connected to,
commissioned by, endorsed by, or derived from the work of any current or former
employer, client, law firm, court, or public body. It is unrelated to the author's
previous professional work.

## Data provenance

The public analysis uses **`consumo`**, a dataset distributed in the `abjData` R
package maintained by the Brazilian Jurimetrics Association (ABJ). It is a
1,000-row sample of second-degree consumer cases from the Court of Justice of São
Paulo (TJSP), used in the book:

> Feliz, Ricardo; Trecenti, Julio. **Metodologia de Pesquisa Jurimétrica**.
> ABJ, 2022. <https://livro.abj.org.br>. MIT licence.

The dataset and the book are MIT licensed. Full attribution and the MIT notice are
in [`CREDITS.md`](CREDITS.md). This project is not affiliated with or endorsed by
ABJ.

What the prepared dataset contains:

- case-level metadata: subject, municipality, party configuration, filing and
  decision dates, claim value, second-instance outcome and unanimity;
- **no names, CPFs, CNPJs, addresses or free-text case descriptions**;
- a stable SHA-256 pseudonym in place of the public case number. The raw case
  number is not stored in the prepared file.

The data is public court metadata, not confidential material. The project still
treats it with care: the pipeline runs a PII gate, the raw case number is dropped,
and the prepared table is written only to a gitignored runtime directory. No
dataset or model artefact is committed to the repository.

## Synthetic test data

The repository also contains a synthetic generator,
`src/claimvalue/synthetic.py`. It is used **only** by the automated test suite and
the controlled benchmark. Those records are fictional:

- names are placeholders such as `MARIA DA SILVA` and `JOAO DA SILVA`;
- CPF and CNPJ values are invalid, obviously fictional placeholders used only to
  exercise the redaction tests;
- case numbers, courts, amounts and outcomes are fabricated.

The synthetic records are not used for the public results in
`docs/10_book_results.md` and are not evidence about real courts.

## No confidential or NDA-protected material

To the best of the author's knowledge, this repository contains:

- no confidential information from any employer, client, or former employer;
- no material covered by a non-disclosure agreement (NDA);
- no trade secrets, internal documents, proprietary source code, or business data
  from any organisation;
- no personal data of any identifiable individual.

The project was built from public knowledge, a public MIT-licensed dataset and
general open-source tooling.

## Educational purpose and limits

The code and documentation are provided for learning and demonstration. Nothing
here is legal advice, a production system, or a statement about how any real court
or organisation operates. The public sample is small and the selected models do
not beat the simple baselines reliably. The reported metrics must not be read as
evidence about real-world legal outcomes or used for provisioning, settlement or
risk decisions.

## Author's declaration

I declare that I did not use, copy, or disclose any NDA-protected, confidential, or
proprietary information in creating this repository. The project is an independent
study exercise. The public analysis uses the MIT-licensed ABJ dataset described
above; the synthetic fixtures are fictional.

Author: Carollina Anderson
Date: 2026-11-11
