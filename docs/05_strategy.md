# 05 · Strategic vision: from data to decision

Technical skill gets a model built; strategic vision decides **whether it should exist, who uses it, and how
you prove it paid off**. Use this as a lens on every step in docs 00 to 04.

## 1. The decision chain (start from the right end)

```
Decision  ->  Metric  ->  Data  ->  Model  ->  Action  ->  Measured outcome
 (who does what differently?)                         (did it change money/risk/time?)
```

| Link | Question | This project |
|---|---|---|
| Decision | What will someone do differently? | provision more for predicted high-exposure cases; prioritise likely-unfavorable cases for settlement review |
| Metric | How do we know it helps? | provisioning error vs the current method; share of unfavorable cases caught in the top decile |
| Data | Do we have it, legally, at prediction time? | LGPD controls; features only from before the filing date |
| Model | Simplest thing that beats the baseline | tree/forest vs dummy, with CIs |
| Action | How does the output reach a person? | API / report, human in the loop |
| Outcome | How is value measured after launch? | A/B or before-after on provisioning accuracy |

If you cannot name the decision, do not build the model.

## 2. Maturity ladder

| Level | Question | Example here | Typical effort |
|---|---|---|---|
| Descriptive | What happened? | EDA tables | days |
| Diagnostic | Why? | variance explained by judge/court | days to weeks |
| Predictive | What will happen? | claim value, outcome probability | weeks |
| Prescriptive | What should we do? | settle vs litigate, with cost model | months, needs business rules |

Most business value arrives at the descriptive-to-diagnostic step; do not skip it for modelling glamour.

## 3. Explaining results to management (template)

| Section | Content | Example from the synthetic run |
|---|---|---|
| Question | one sentence | "Can we estimate a new case's claim value better than using the median?" |
| Answer | plain numbers | "Typical error drops from about R$4,900 to about R$3,400 (-31%); 95% interval R$3,230 to R$3,560" |
| Confidence | what it rests on | "tested on the most recent 20% of cases the model never saw" |
| Limits | honest caveats | "ranking of rulings is modest (AUC 0.68); do not automate decisions" |
| Risk | what could go wrong | "judge-level profiling needs legal review; accuracy may drift as rulings change" |
| Ask | the decision you need | "approve a 3-month pilot on one comarca with weekly monitoring" |

Rules: lead with the answer, give one number in business units (reais), show the baseline next to it,
never present a metric alone, state what you would do next.

## 4. Value and risk sketch (quantify before you build)

```
annual value = cases/year x share acted on x improvement per case (R$) - run cost - risk cost
```
Estimate each term with a range. If the optimistic case does not pay, stop. Risk terms: legal/privacy exposure,
model drift, over-reliance on automation, reputational cost of judge profiling.

## 5. Risks by lifecycle stage

| Stage | Typical failure | Early signal | Mitigation |
|---|---|---|---|
| Scoping | no owner for the decision | vague success metric | write the decision and metric first |
| Data | access, quality, legality | missing fields, slow approvals | LGPD checklist, quality SQL, synthetic data to prototype |
| Modelling | leakage, optimistic metrics | too-good results | temporal CV, baselines, CIs |
| Delivery | no adoption | users ignore outputs | co-design, explain in reais, human in the loop |
| Operations | silent drift | rising errors, feature shift | monitoring, retraining policy |

## 6. Consulting lens (for a role like Data Scientist at a services firm)

| Phase | Your deliverable | Evidence in this repo |
|---|---|---|
| Discovery | decision, metric, data inventory, risks | sections 1 and 5; data dictionary |
| Proof of concept | baseline vs model on held-out data | `docs/04_ml.md`, leaderboards |
| MVP | pipeline, tests, API | CLI, CI, FastAPI, Docker |
| Production | monitoring, governance, handover | MLflow, privacy report, runbook-style docs |

Clients buy outcomes, governance and speed to value. Be ready to say what you would change for a cloud stack
(Azure/AWS/GCP equivalents: object storage = raw layer, Spark/Databricks = transform, managed MLflow/registry,
Kubernetes or serverless for serving, secrets manager for the pepper). Map the repo to the *actual* job description, because
requirements differ by team.

## 7. Interview story (STAR) for this project

| | |
|---|---|
| **Situation** | Court-case data, a need to estimate claim value and ruling outcome; first model overfit (train MAE about 2 vs test about 150) |
| **Task** | Make it trustworthy and reproducible, without exposing personal data |
| **Action** | Diagnosed identifier leakage and non-temporal validation; rebuilt with a leakage-safe SQL ABT, temporal CV, baselines and CIs; added LGPD-oriented anonymization with risk metrics; tests, CI, Docker, API, MLflow |
| **Result** | A reproducible pipeline where the model beats the baseline by about 31% on held-out future data, with a documented privacy and risk posture. Differences between top models are within noise, so the simplest is preferred |
| **Lesson** | Validation design and data governance matter more than the algorithm |

Questions you should be able to answer: why temporal split; why target encoding inside the pipeline; difference between
pseudonymization and anonymization; why log the target; what the CI tells you; what you would monitor in production.

## 8. Learning essentials if you are lookiong for a Data Enginnering roadmap 

| Gap (from your backlog) | Why it matters for this project | First step |
|---|---|---|
| Airflow | schedule and retry the pipeline | wrap each CLI command as a task in a DAG |
| Spark / Databricks / Delta | the ABT self-join at scale | port `03_abt.sql` to Spark SQL with window functions |
| dbt | tested, documented SQL models | turn the three SQL files into dbt models with tests |
| Docker / Kubernetes | deployment | run the image locally, then deploy to a managed service |
| Terraform | reproducible infrastructure | codify bucket + database + container service |
| MLOps | monitoring and promotion | add a PSI drift report and a champion/challenger check |
| Cloud (AWS/Azure) | what clients run | deploy the image to a managed container service |
