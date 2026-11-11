"""EDA report for the ABJ book dataset.

The report is deliberately descriptive. It documents the data generating process,
the class balance and the distributions the modelling choices depend on, and it
credits the source. It does not make causal claims.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # tabulate is optional
        return "```\n" + df.to_string(index=False) + "\n```"


def write_book_eda(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(df["claim_value"]), bins=40, color="#102C68")
    ax.set_xlabel("log1p(claim value, BRL)")
    ax.set_ylabel("cases")
    ax.set_title("Claim value is right-skewed, so the model learns on a log scale")
    fig.tight_layout()
    fig.savefig(out_dir / "claim_value_log_distribution.png", dpi=150)
    plt.close(fig)

    outcome = df["decision_outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    outcome.sort_values().plot.barh(ax=ax, color="#102C68")
    ax.set_xlabel("cases")
    ax.set_ylabel("")
    ax.set_title("Second-instance outcome")
    fig.tight_layout()
    fig.savefig(out_dir / "decision_outcome.png", dpi=150)
    plt.close(fig)

    for col, name in (("assunto", "subject"), ("tipo_litigio", "party configuration")):
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.sort_values().plot.barh(ax=ax, color="#AFCA0A")
        ax.set_xlabel("cases")
        ax.set_ylabel("")
        ax.set_title(f"Cases by {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=150)
        plt.close(fig)

    top = df["comarca"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top.sort_values().plot.barh(ax=ax, color="#575756")
    ax.set_xlabel("cases")
    ax.set_ylabel("")
    ax.set_title("Top 15 municipalities")
    fig.tight_layout()
    fig.savefig(out_dir / "top_comarcas.png", dpi=150)
    plt.close(fig)

    monthly = df.set_index("filing_date").resample("QE").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(ax=ax, color="#102C68", marker="o")
    ax.set_xlabel("filing quarter")
    ax.set_ylabel("cases")
    ax.set_title("Filings over time")
    fig.tight_layout()
    fig.savefig(out_dir / "filing_volume.png", dpi=150)
    plt.close(fig)

    # --- tables --------------------------------------------------------------
    value = df["claim_value"]
    summary = pd.DataFrame(
        {
            "statistic": ["count", "mean", "median", "p95", "max", "missing"],
            "value": [
                len(value),
                round(value.mean(), 2),
                round(value.median(), 2),
                round(value.quantile(0.95), 2),
                round(value.max(), 2),
                int(value.isna().sum()),
            ],
        }
    )
    outcomes = (
        df["decision_outcome"]
        .value_counts()
        .rename_axis("decision_outcome")
        .reset_index(name="cases")
    )
    outcomes["share"] = (outcomes["cases"] / len(df)).round(3)
    subjects = df["assunto"].value_counts().rename_axis("assunto").reset_index(name="cases")
    party = df["tipo_litigio"].value_counts().rename_axis("tipo_litigio").reset_index(name="cases")
    top_comarcas = (
        df["comarca"].value_counts().head(15).rename_axis("comarca").reset_index(name="cases")
    )
    missing = df.isna().sum().rename_axis("column").reset_index(name="missing").query("missing > 0")
    if missing.empty:
        missing = pd.DataFrame({"column": ["none"], "missing": [0]})

    report = f"""# EDA: ABJ book dataset (`abjData::consumo`)

> Source: Feliz, Ricardo; Trecenti, Julio. *Metodologia de Pesquisa Jurimétrica*.
> ABJ, 2022. <https://livro.abj.org.br>. Data: `abjData::consumo`,
> <https://github.com/abjur/abjData>. MIT licence. See `CREDITS.md`.

## Shape and period

- Rows: **{len(df)}**
- Columns: **{df.shape[1]}**
- Filing dates: **{df.filing_date.min().date()}** to **{df.filing_date.max().date()}**
- Decision dates: **{df.decision_date.min().date()}** to **{df.decision_date.max().date()}**
- Rows with a binary reform label: **{int(df.reformed.notna().sum())}**
- Reform rate among labelled rows: **{df.reformed.mean():.3f}**

## Claim value (BRL)

{_table(summary)}

![Claim value distribution](claim_value_log_distribution.png)

## Outcome

{_table(outcomes)}

![Outcome counts](decision_outcome.png)

## Subject

{_table(subjects)}

![Cases by subject](assunto.png)

## Party configuration

{_table(party)}

![Cases by party configuration](tipo_litigio.png)

## Municipalities

{_table(top_comarcas)}

![Top municipalities](top_comarcas.png)

## Filings over time

![Filings per quarter](filing_volume.png)

## Missing values

{_table(missing)}

## What this means for the model

- `assunto`, `comarca`, `tipo_litigio` and the filing date are known when the appeal is filed.
- `claim_value` is known at filing and is the regression target; it is a feature only for
  the classification task.
- The `*_prior` features in the ABT only use cases whose decision was published before the
  current filing. This is the book's discipline applied to the public data.
- The outcome label is **reformed**: 1 for Reformou or Parcial, 0 for Não reformou.
  Procedural outcomes (Não Conheceram, Desistência) have no binary label and are excluded
  from the modelling table.
- The data is a 1,000-row sample. Confidence intervals are wide and the results should be
  read as a demonstration of method, not as a production estimate.
"""
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return out_dir
