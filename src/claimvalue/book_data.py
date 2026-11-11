"""The ABJ book dataset (`abjData::consumo`) and its prepared analysis table.

The dataset is **never committed**. ``claimvalue book-fetch`` downloads the pinned
upstream file, verifies its SHA-256, converts it with ``pyreadr`` and writes the
prepared table under ``RAW_DATA_DIR/book/`` (default ``data/raw/book/``), which is
gitignored. The repository keeps only provenance metadata.

Privacy note: the prepared file has no names, CPFs, CNPJs, addresses or free
text. The public case number is replaced by a stable SHA-256 pseudonym.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from claimvalue.anonymize import assert_no_pii
from claimvalue.config import get_settings

COLUMNS = [
    "case_id",
    "assunto",
    "comarca",
    "tipo_litigio",
    "filing_date",
    "decision_date",
    "claim_value",
    "decision_outcome",
    "reformed",
    "unanimity",
    "case_days",
]
TARGETS = {"regression": "claim_value", "classification": "reformed"}
OUTCOME_REFORMED = {"Reformou", "Parcial"}
OUTCOME_NOT_REFORMED = {"Não reformou"}
OUTCOME_PROCEDURAL = {"Não Conheceram / Prejudicado / Diligência", "Desistência"}

BOOK = {
    "title": "Metodologia de Pesquisa Jurimétrica",
    "authors": ["Ricardo Feliz", "Julio Trecenti"],
    "publisher": "ABJ",
    "year": 2022,
    "url": "https://livro.abj.org.br",
    "license": "MIT",
}
PACKAGE = {
    "name": "abjData",
    "title": "Databases Used Routinely by the Brazilian Jurimetrics Association",
    "authors": ["Julio Trecenti", "Renata Hirota", "Katerine Witkoski", "ABJ"],
    "url": "https://github.com/abjur/abjData",
    "license": "MIT",
    "commit": "afa83b359917b6e974fbe7281f340a66b2a86cfe",
}
SOURCE_URL = f"https://raw.githubusercontent.com/abjur/abjData/{PACKAGE['commit']}/data/consumo.rda"
SOURCE_SHA256 = "3a90cd4c05ed1aa206abe8ffc59b68d0b0c629bfaaeb16bd12f9a0079008fad8"


def case_id(raw: str) -> str:
    """Stable pseudonym for a public case number. Not keyed: the source is public."""
    digest = hashlib.sha256(f"abjData::consumo:{raw}".encode()).hexdigest()[:16]
    return f"BOOK_{digest.upper()}"


def prepare_book_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the source ``consumo`` columns to the project's prepared table."""
    required = {
        "id_processo",
        "assunto",
        "comarca",
        "valor",
        "tipo_litigio",
        "dec_val",
        "dec_unanime",
        "dec_date",
        "tempo",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"consumo is missing columns: {sorted(missing)}")

    df = raw.copy()
    df["decision_date"] = pd.to_datetime(df["dec_date"], errors="raise")
    df["filing_date"] = df["decision_date"] - pd.to_timedelta(df["tempo"], unit="D")

    outcome = df["dec_val"].astype(str)
    unknown = set(outcome.unique()) - OUTCOME_REFORMED - OUTCOME_NOT_REFORMED - OUTCOME_PROCEDURAL
    if unknown:
        raise ValueError(f"unmapped dec_val values: {sorted(unknown)}")

    reformed = pd.Series(pd.NA, index=df.index, dtype="Int64")
    reformed[outcome.isin(OUTCOME_REFORMED)] = 1
    reformed[outcome.isin(OUTCOME_NOT_REFORMED)] = 0

    out = pd.DataFrame(
        {
            "case_id": df["id_processo"].astype(str).map(case_id),
            "assunto": df["assunto"].astype(str),
            "comarca": df["comarca"].astype(str),
            "tipo_litigio": df["tipo_litigio"].astype(str),
            "filing_date": df["filing_date"].dt.strftime("%Y-%m-%d"),
            "decision_date": df["decision_date"].dt.strftime("%Y-%m-%d"),
            "claim_value": df["valor"].astype(float),
            "decision_outcome": outcome,
            "reformed": reformed,
            "unanimity": df["dec_unanime"].astype(str),
            "case_days": df["tempo"].astype(int),
        }
    ).loc[:, COLUMNS]
    return out.sort_values(["filing_date", "case_id"], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"book dataset is missing columns: {sorted(missing)}")
    if df["case_id"].duplicated().any():
        raise ValueError("case_id is not unique")
    if not df["case_id"].astype(str).str.startswith("BOOK_").all():
        raise ValueError("case_id must be a BOOK_ pseudonym")
    if df["claim_value"].isna().any() or (df["claim_value"] <= 0).any():
        raise ValueError("claim_value must be positive and non-null")
    if pd.to_datetime(df["decision_date"]).lt(pd.to_datetime(df["filing_date"])).any():
        raise ValueError("decision_date precedes filing_date")
    if set(df["reformed"].dropna().unique()) - {0, 1}:
        raise ValueError("reformed must be 0/1/null")
    unknown = set(df["decision_outcome"].unique()) - (
        OUTCOME_REFORMED | OUTCOME_NOT_REFORMED | OUTCOME_PROCEDURAL
    )
    if unknown:
        raise ValueError(f"unmapped decision outcomes: {sorted(unknown)}")


def load_prepared() -> pd.DataFrame:
    """Read the runtime-prepared table. It is never shipped with the package."""
    path = get_settings().book_csv
    if not path.exists():
        raise FileNotFoundError(
            f"book dataset not found: {path}\n"
            "Run `claimvalue book-fetch` first (it needs `pip install pyreadr`)."
        )
    return pd.read_csv(path, dtype={"case_id": "string", "reformed": "Int64"})


def write_processed() -> dict:
    """Validate the runtime dataset and write the pipeline copy + privacy report."""
    settings = get_settings()
    df = load_prepared()
    validate(df)
    assert_no_pii(df)

    out = settings.book_processed_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report = {
        "dataset": "abjData::consumo",
        "source": "https://github.com/abjur/abjData",
        "book": "Feliz & Trecenti (2022), Metodologia de Pesquisa Jurimétrica, ABJ",
        "licence": "MIT",
        "rows": int(len(df)),
        "rows_with_binary_label": int(df["reformed"].notna().sum()),
        "rows_regression": int(df["claim_value"].notna().sum()),
        "date_range": [
            str(pd.to_datetime(df["filing_date"]).min().date()),
            str(pd.to_datetime(df["decision_date"]).max().date()),
        ],
        "direct_identifiers": ["raw case number (not stored; replaced by SHA-256 pseudonym)"],
        "pii_scan": "assert_no_pii: passed",
        "label_balance": {
            str(k): int(v)
            for k, v in df["reformed"].value_counts(dropna=False).sort_index().items()
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out.parent / "book_privacy_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def load_processed() -> pd.DataFrame:
    path: Path = get_settings().book_processed_csv
    if not path.exists():
        raise FileNotFoundError(f"run `claimvalue book-prepare` first: {path}")
    return pd.read_csv(path, dtype={"case_id": "string", "reformed": "Int64"})


def fetch_book_data(
    rda_path: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Download, verify, convert and validate the pinned ABJ table.

    Writes ``consumo_prepared.csv`` and ``SOURCE.json`` under ``data/raw/book/``.
    Nothing is written to the repository tree, so the data cannot be committed by
    accident. Set ``force=True`` to bypass a cached prepared file.
    """
    settings = get_settings()
    target_dir = Path(out_dir) if out_dir else settings.book_raw_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "consumo_prepared.csv"

    if csv_path.exists() and not force:
        cached = pd.read_csv(csv_path, dtype={"case_id": "string", "reformed": "Int64"})
        validate(cached)
        return {
            "csv": str(csv_path),
            "rows": int(len(cached)),
            "cached": True,
            "sha256": SOURCE_SHA256,
        }

    rda = Path(rda_path) if rda_path else target_dir / "consumo.rda"
    if not rda.exists():
        try:
            urllib.request.urlretrieve(SOURCE_URL, rda)
        except Exception as exc:  # pragma: no cover - network failure path
            raise RuntimeError(
                "could not download the ABJ dataset. Download it manually from "
                f"{SOURCE_URL} and pass --rda /path/to/consumo.rda"
            ) from exc

    digest = hashlib.sha256(rda.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"source checksum mismatch: expected {SOURCE_SHA256}, got {digest}")

    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "book-fetch needs `pyreadr` to read the upstream .rda file. "
            "Install it with `pip install pyreadr` (or `uv pip install pyreadr`)."
        ) from exc

    raw = pyreadr.read_r(str(rda))["consumo"]
    prepared = prepare_book_frame(raw)
    validate(prepared)
    prepared.to_csv(csv_path, index=False)

    source = {
        "book": BOOK,
        "package": PACKAGE,
        "dataset": "consumo",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "data_committed": False,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows_raw": int(len(raw)),
        "rows_prepared": int(len(prepared)),
        "rows_modelling": int(prepared["reformed"].notna().sum()),
        "columns": COLUMNS,
        "notes": (
            "case_id is a stable SHA-256 pseudonym of the public case number; the raw "
            "case number is not stored. Filing date is decision_date minus case_days. "
            "reformed = 1 for Reformou/Parcial, 0 for Não reformou, null for procedural "
            "outcomes. No names, CPFs or CNPJs are present. This file is runtime data "
            "under data/raw/book/ and is gitignored."
        ),
    }
    (target_dir / "SOURCE.json").write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
    return {
        "csv": str(csv_path),
        "rows": int(len(prepared)),
        "cached": False,
        "sha256": SOURCE_SHA256,
    }


def source_metadata() -> dict:
    """Return the runtime provenance record, falling back to the committed note."""
    runtime = get_settings().book_csv.parent / "SOURCE.json"
    path = (
        runtime
        if runtime.exists()
        else Path(__file__).resolve().parent / "data" / "book" / "SOURCE.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
