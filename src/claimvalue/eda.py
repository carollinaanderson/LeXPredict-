"""Exploratory data analysis as code: every table and figure is reproducible and testable.

EDA is not decoration. Each function answers a decision-relevant question:
  * What does the target look like?                -> target_summary / fig_target
  * Where is data missing, and could it be biased? -> missingness / fig_missing
  * Which columns are identifiers, not features?   -> id_like_columns (the original PJ/ID bug)
  * Who drives the outcome: judge, place, court?   -> variance_explained (eta squared)
  * Is volume or outcome drifting over time?       -> monthly / fig_time
  * Do the engineered features relate to target?   -> spearman_with_target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Schema profile: dtype, missing %, distinct values, distinct ratio."""
    n = len(df)
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
            "unique_ratio": (df.nunique() / n).round(4),
        }
    ).rename_axis("column")


def id_like_columns(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Columns where (almost) every row has its own value: identifiers, never features.
    One-hot encoding such a column lets a model memorise rows (tiny train error, huge test error)."""
    return [c for c in df.columns if df[c].nunique() / max(len(df), 1) >= threshold]


def target_summary(values: pd.Series) -> pd.DataFrame:
    """Money targets are right-skewed: compare mean vs median and skew before and after log."""
    v = values.dropna()
    logv = np.log1p(v)
    return pd.DataFrame(
        {
            "raw": [v.mean(), v.median(), v.std(), v.skew(), v.quantile(0.99), v.max()],
            "log1p": [
                logv.mean(),
                logv.median(),
                logv.std(),
                logv.skew(),
                logv.quantile(0.99),
                logv.max(),
            ],
        },
        index=["mean", "median", "std", "skew", "p99", "max"],
    ).round(3)


def missingness(df: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
    """Is the target missing at random? A missing rate that varies by group is a red flag (MAR/MNAR)."""
    return (
        df.assign(missing=df[target].isna())
        .groupby(by)["missing"]
        .agg(n="size", missing_pct=lambda s: round(s.mean() * 100, 2))
        .reset_index()
        .sort_values("missing_pct", ascending=False)
    )


def variance_explained(df: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    """Eta squared of log(value) per categorical column: share of variance a column can explain.
    Columns are nested (judge sits in one comarca/vara), so shares overlap: read as upper bounds."""
    y = np.log1p(df[value].dropna())
    total = ((y - y.mean()) ** 2).sum()
    rows = []
    for g in groups:
        means = y.groupby(df.loc[y.index, g]).transform("mean")
        rows.append(
            {
                "column": g,
                "n_levels": df[g].nunique(),
                "eta_squared": round(float(1 - ((y - means) ** 2).sum() / total), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(month=df["filing_date"].dt.to_period("M").dt.to_timestamp()).groupby("month")
    out = g.agg(
        n_cases=("case_id", "size"),
        unfav_rate=("unfavorable", "mean"),
        median_claim=("claim_value", "median"),
    ).reset_index()
    out[["unfav_rate", "median_claim"]] = out[["unfav_rate", "median_claim"]].round(3)
    return out


def spearman_with_target(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Rank correlation is robust to skew and monotone non-linearity (Pearson on raw money is not)."""
    sub = df[[*features, target]].dropna()
    corr = sub.corr(method="spearman")[target].drop(target)
    return (
        corr.round(3)
        .rename("spearman")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("spearman", key=abs, ascending=False)
    )


def by_group(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(
            n=("case_id", "size"),
            median_claim=("claim_value", "median"),
            unfav_rate=("unfavorable", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("n", ascending=False)
    )


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_target(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    v = df["claim_value"].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(v.clip(upper=v.quantile(0.99)), bins=50, color="#4c78a8")
    ax[0].set(title="Claim value (clipped at p99): right-skewed", xlabel="BRL")
    ax[1].hist(np.log1p(v), bins=50, color="#59a14f")
    ax[1].set(title="log(1 + claim value): roughly symmetric", xlabel="log BRL")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_by_uf(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    g = by_group(df, "uf").sort_values("unfav_rate")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(g["uf"], g["unfav_rate"] * 100, color="#e15759")
    ax.axvline(df["unfavorable"].mean() * 100, color="k", ls="--", lw=1, label="overall")
    ax.set(title="Unfavorable ruling rate by UF", xlabel="% unfavorable")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_time(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    m = monthly(df)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(m["month"], m["n_cases"], color="#4c78a8")
    ax.set(title="Cases filed per month (check ramp-up / truncation at the edges)", ylabel="cases")
    ax2 = ax.twinx()
    ax2.plot(m["month"], m["unfav_rate"] * 100, color="#e15759", alpha=0.7)
    ax2.set_ylabel("% unfavorable", color="#e15759")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_judges(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    d = df.dropna(subset=["claim_value"]).assign(log=lambda x: np.log1p(x["claim_value"]))
    g = d.groupby("judge_id")["log"].agg(["mean", "sem", "size"]).sort_values("mean").reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.errorbar(
        range(len(g)),
        g["mean"],
        yerr=1.96 * g["sem"],
        fmt="o",
        ms=3,
        color="#76b7b2",
        ecolor="#bbb",
    )
    ax.axhline(d["log"].mean(), color="k", ls="--", lw=1)
    ax.set(
        title="Mean log claim value per judge (95% CI): spread = judge/location effect",
        xlabel="judges (sorted)",
        ylabel="log BRL",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_eda_report(abt: pd.DataFrame, out_dir: Path) -> Path:
    """Write tables (CSV), figures (PNG) and a Markdown summary with the computed numbers."""
    from claimvalue.features import NUMERIC

    out_dir.mkdir(parents=True, exist_ok=True)
    abt = abt.copy()
    abt["log_claim"] = np.log1p(abt["claim_value"])
    tables = {
        "overview": overview(abt.drop(columns=["log_claim"])),
        "target_summary": target_summary(abt["claim_value"]),
        "missing_by_uf": missingness(abt, "claim_value", "uf"),
        "variance_explained": variance_explained(
            abt, "claim_value", ["judge_id", "comarca", "uf", "vara"]
        ),
        "spearman_with_log_claim": spearman_with_target(abt, NUMERIC, "log_claim"),
        "by_uf": by_group(abt, "uf"),
        "by_vara": by_group(abt, "vara"),
        "monthly": monthly(abt),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")
    fig_target(abt, out_dir / "target_distribution.png")
    fig_by_uf(abt, out_dir / "unfavorable_by_uf.png")
    fig_time(abt, out_dir / "volume_over_time.png")
    fig_judges(abt, out_dir / "judge_effect.png")

    ids = id_like_columns(abt.drop(columns=["log_claim"]))
    ts, ve = tables["target_summary"], tables["variance_explained"]
    lines = [
        "# EDA report (auto-generated)",
        "",
        f"* rows: **{len(abt):,}**, period: {abt['filing_date'].min().date()} to {abt['filing_date'].max().date()}",
        f"* target missing: **{abt['claim_value'].isna().mean() * 100:.1f}%** of rows (dropped for regression only)",
        f"* unfavorable ruling rate: **{abt['unfavorable'].mean() * 100:.1f}%**",
        f"* identifier-like columns (never use as features): `{ids}`",
        f"* claim value skew: raw {ts.loc['skew', 'raw']} -> log {ts.loc['skew', 'log1p']}; "
        f"mean {ts.loc['mean', 'raw']:,.0f} vs median {ts.loc['median', 'raw']:,.0f}",
        "",
        "## Share of log-claim variance explained by each column (eta squared, nested -> upper bounds)",
        "",
        ve.to_markdown(index=False),
        "",
        "## Spearman correlation of engineered features with log claim value",
        "",
        tables["spearman_with_log_claim"].to_markdown(index=False),
        "",
        "Figures: `target_distribution.png`, `unfavorable_by_uf.png`, `volume_over_time.png`, `judge_effect.png`.",
    ]
    (out_dir / "EDA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir
