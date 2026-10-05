# LGPD (Lei 13.709/2018): how this repository handles personal data

> This is engineering documentation, **not legal advice**. For real data, involve the organisation's
> encarregado (DPO) / legal team and complete the checklist at the bottom.

## 1. The one distinction that matters

| | Pseudonymized | Anonymized |
|---|---|---|
| LGPD basis | art. 13 par. 4 (written for health research, but the reference definition): link to the person exists only via separately kept additional information | art. 5 III and XI, art. 12: person cannot be identified with reasonable means, judged by objective factors (cost, time, available technology) |
| Still personal data? | **Yes** | **No**, but only if the anonymization is effective and not reversible by reasonable effort |
| In this repo | `case_id`, `judge_id` (HMAC with a secret pepper) | the *released/generalized* output **if** the risk assessment supports it |

A keyed hash is pseudonymization: while the pepper exists, anyone holding it can recompute the links, and with
auxiliary data (public dockets) outsiders may link records without it. **`privacy_report.json` therefore labels the
dataset "PSEUDONYMIZED"** until a documented assessment concludes otherwise. Anonymization is a risk judgement, not a
function call.

Regulatory state (searched October 2026; verify, it may have moved): ANPD's draft *Guia de Anonimização e Pseudonimização* (public consultation
opened 30 Jan 2024) takes a **risk-based approach**, says anonymization should **not be fully automated** (human
participation), and lists pseudonymization techniques including substitution, tokenization, encryption, masking and
salting. The standards promised by art. 12 par. 3 appear on ANPD's 2025-2026 regulatory agenda, and I did not find a final
binding text in my search. Check gov.br/anpd for the current version before citing a requirement.

## 2. LGPD principles (art. 6) mapped to code

| Principle | What it demands | Implementation | Evidence |
|---|---|---|---|
| Finalidade / adequação | declared, compatible purpose | purpose stated in README; only fields the models need survive | `KEEP_COLUMNS`, `DROPPED_COLUMNS` |
| **Necessidade** (minimisation) | the minimum data | CPF, party name, judge name, raw case number dropped; amounts rounded; dates shifted | `test_anonymized_frame_has_no_identifiers` |
| Livre acesso / transparência | subjects can know what is done | `privacy_report.json` records every action | `test_report_describes_actions_without_leaking_data` |
| Qualidade dos dados | accurate, relevant | quality SQL, constraints, dedupe | `docs/02_cleaning_and_munging.md` |
| **Segurança** (art. 46) | technical measures | keyed pseudonyms, secrets in `.env`, gitleaks, pre-commit blocks data files, non-root container, API key | CI + `.pre-commit-config.yaml` |
| Prevenção | avoid damage | `assert_no_pii` fails closed before writing; PII scan in CI | `test_assert_no_pii_fails_closed_without_leaking_values` |
| Não discriminação | no unlawful/abusive discrimination | slice error reports; judge profiling flagged as sensitive (section 6) | `slice_report` |
| Responsabilização | demonstrate compliance | this document, report JSON, key fingerprint, tests | |

## 3. Techniques applied, and what each does not do

| Technique | Protects against | Residual risk | Code |
|---|---|---|---|
| Drop direct identifiers (CPF, names, case number) | direct identification | none from those fields | `anonymize_cases` |
| HMAC-SHA256 pseudonyms with secret pepper | dictionary attack on low-entropy ids (CPF has about 10^9 values; a plain hash is brute-forceable) | key theft; linkage via other columns | `pseudonymize` |
| Whole-year **date shift** (secret, 1 to 5 x 52 weeks) | matching dates against public dockets | order and intervals preserved by design; weak if the shift leaks | `date_shift_days` |
| **Generalisation** of amounts (2 significant digits) | exact-amount linkage | ~5% relative resolution loss; negligible for a model with MAE of thousands | `round_significant` |
| Free-text scrubbing (CPF, CNPJ, case no., e-mail, phone, CEP, OAB, known names) | identifiers in the ruling | **unknown names** (lawyers, witnesses, relatives) | `scrub_text` |
| Redaction of texts with art. 11 vocabulary (health, religion, union, politics, race, genetic...) | sensitive-data exposure | vocabulary list is finite; broad on purpose | `has_sensitive_terms` |
| **Judicial secrecy exclusion** | exposing sealed cases | depends on the source flag being correct | `segredo_justica` filter |
| Suppression of small groups (`--suppress-k`) | rare combinations that single out a record | utility loss; k-anonymity alone is not sufficient | `suppress_small_groups` |
| k-anonymity / l-diversity measurement | quantifies linkage and homogeneity risk | models an attacker; the real one may know more | `reidentification_risk`, `l_diversity` |

## 4. Risk assessment (attacker models)

| Attacker | Knows | Quasi-identifiers | Synthetic run |
|---|---|---|---|
| Internal analyst | the modelling table | uf, comarca, vara, judge | min k = 83, 0% of records below k=5 |
| **Outsider with public dockets** | where, which year, roughly how much | uf, comarca, vara, filing year, claim band | min k = 1; **2.73%** of records in groups below k=5; max risk 1.0 |
| Homogeneity | group membership only | same as above | 0.88% of records in groups with a single outcome |

Reading it: the table is fine for an analyst environment, but **2.73% of records are potentially unique to an outsider**.
Options: raise generalisation (bigger bands, month->year), `claimvalue anonymize --suppress-k 5` (drops those rows),
or keep the data inside the controlled environment and release only aggregates. Pick a threshold *before* looking at the
numbers (a common starting rule: at least 95% of records in groups of 5 or more, then justify it in the RIPD).

## 5. Roles, keys and lifecycle

| Topic | Practice |
|---|---|
| Separation of duties | whoever holds the pepper and the raw export should not be the person publishing the dataset |
| Key management | pepper in a secrets manager, never in git; `pepper_fingerprint` in the report identifies the key without revealing it; rotate on suspicion of exposure (rotation changes every pseudonym, so plan a re-run) |
| Storage | raw export on an encrypted volume outside the repo; derived data in `DATA_DIR` (gitignored) |
| Retention / elimination (arts. 15, 16) | delete the raw export once the anonymized table exists (`--delete-raw`); SSD deletion is not guaranteed, so rely on disk encryption and crypto-erase |
| Access | least privilege; the API has a key and returns only predictions |
| Incidents (art. 48) | if the raw file or pepper leaks, treat pseudonymized data as identifiable; notify per the incident procedure |
| Subject rights (art. 18) | effective anonymization falls outside them; pseudonymized data does not, so you must be able to locate and delete a person's records (keep a secured lookup if the business needs one) |
| Impact assessment (art. 38) | prepare an RIPD when processing is large-scale or may affect fundamental rights; this project's profiling of judges argues for one |
| Automated decisions (art. 20) | people may request review of decisions made *solely* by automated processing that affect their interests; keep a human in the loop for case handling |

## 6. Judge profiling: the sensitive design choice

Judges are people, and ranking them by outcomes is contentious (some jurisdictions restrict it by law). Treat this as an ethics and legal
review item, not only a feature. Mitigations, in order of preference: (1) use court-level features (comarca/vara) only;
(2) aggregate judge data over long windows and large n; (3) never publish judge-level scores; (4) document the legitimate-interest
assessment (art. 10) and keep the model for internal exposure estimation, not for pressuring or targeting individuals.

## 7. Legal basis and employer data

* Court records being public does not remove LGPD duties; later use must respect purpose, good faith and the data
  subject's rights (art. 7 par. 3).
* Data and code from a former employer are usually also covered by **confidentiality/contract** and often
  by IP clauses, independent of LGPD. Anonymizing helps privacy; it does not by itself grant permission. This repository
  ships **only synthetic data**. Before describing real results in an interview, check your contract or ask for written permission,
  and describe the *method*, not client numbers.

## 8. Checklist before touching real data

- [ ] Written authorisation from the data owner; legal basis identified (arts. 7 / 11) with the DPO
- [ ] Raw export on an encrypted volume outside the repo; `RAW_DATA_DIR` set in `.env`
- [ ] `ANONYMIZATION_PEPPER` generated (`openssl rand -hex 32`) and stored in a secrets manager
- [ ] Secrecy flag present and verified; those rows excluded
- [ ] `claimvalue anonymize` run; `privacy_report.json` reviewed by a human
- [ ] Free text spot-checked (sample of at least 200) and NER added if text will leave the environment
- [ ] k threshold agreed in advance; `--suppress-k` or more generalisation applied if the linkage view fails
- [ ] `claimvalue check-pii` passes; gitleaks clean; no data files staged
- [ ] RIPD drafted if profiling or large scale; retention date set; raw export deleted when no longer needed

## 9. What this repository does NOT do

Differential privacy, formal re-identification testing by a third party, NER for names in text, automated
rotation of the pepper, access logging, or legal sign-off. Each is a reasonable next step.
