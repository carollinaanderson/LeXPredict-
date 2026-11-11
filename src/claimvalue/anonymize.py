"""Pseudonymization, anonymization techniques and PII guard rails (LGPD-oriented).

IMPORTANT vocabulary (LGPD, Lei 13.709/2018)
--------------------------------------------
* Pseudonymized data (art. 13 par. 4) is STILL personal data: the link to the person survives as
  long as the secret (here: the HMAC pepper) or any auxiliary information exists.
* Anonymized data (art. 5 III, art. 12) is outside the law only if re-identification is not
  possible with reasonable means, judged by objective factors (cost, time, available technology).
  That is a RISK judgement, not a property of a script. See docs/LGPD.md.

What this module does
---------------------
Direct identifiers  : names -> keyed pseudonyms; CPF / party name / raw process number -> dropped.
Free text           : CPF, CNPJ, CNJ number, e-mail, phone, CEP, OAB and known names scrubbed;
                      text with art. 11 (sensitive-data) vocabulary is redacted entirely.
Quasi-identifiers   : all dates shifted by one secret whole-year offset (keeps intervals, order,
                      weekday and season); claim value rounded to N significant digits.
Controls            : cases under judicial secrecy are dropped; ``assert_no_pii`` fails closed;
                      ``reidentification_risk`` measures linkage risk; ``suppress_small_groups``
                      removes rare combinations; a privacy report records what was done.

Limits: regex and known-name scrubbing cannot find names it was not told about (lawyers,
witnesses). For real free text add NER (e.g. a spaCy pt model) AND a human spot check.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata

import numpy as np
import pandas as pd

CNJ_RE = re.compile(r"\b\d{7}-?\d{2}\.?\d{4}\.?\d\.?\d{2}\.?\d{4}\b")
CNPJ_RE = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}-\d{4}(?!\d)")
CEP_RE = re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)")
OAB_RE = re.compile(r"\bOAB[/\s-]*[A-Z]{2}\s*(?:n[ºo°.]?\s*)?\d{1,3}\.?\d{3}\b", re.IGNORECASE)
# Order matters: longest numeric pattern first so a CPF-sized run inside a case number is not
# matched on its own.
SCRUB_ORDER = [
    ("[PROCESSO]", CNJ_RE),
    ("[CNPJ]", CNPJ_RE),
    ("[CPF]", CPF_RE),
    ("[EMAIL]", EMAIL_RE),
    ("[TELEFONE]", PHONE_RE),
    ("[CEP]", CEP_RE),
    ("[OAB]", OAB_RE),
]
PII_PATTERNS = {
    "process_number": CNJ_RE,
    "cnpj": CNPJ_RE,
    "cpf": CPF_RE,
    "email": EMAIL_RE,
    "phone": PHONE_RE,
    "cep": CEP_RE,
    "oab": OAB_RE,
}
# Art. 11 vocabulary (health, religion, politics, union, sexual life, genetic/biometric, race).
# Deliberately broad: a redacted ruling text is cheap, a leaked sensitive datum is not.
SENSITIVE_RE = re.compile(
    r"\b(hiv|c[âa]ncer|cid[- ]?10|laudo m[ée]dico|diagn[óo]stic\w*|doen[çc]a\w*|defici[êe]ncia|"
    r"psiqui[áa]tric\w*|depress[ãa]o|religi\w+|igreja|sindicat\w+|partido pol[íi]tic\w*|"
    r"orienta[çc][ãa]o sexual|ra[çc]a|biometri\w*|gen[ée]tic\w*)\b",
    re.IGNORECASE,
)
REDACTED_TEXT = "[TEXTO_REMOVIDO_DADO_SENSIVEL_ART_11]"

RAW_COLUMNS = [
    "process_number",
    "cpf",
    "party_name",
    "judge_name",
    "uf",
    "comarca",
    "vara",
    "filing_date",
    "decision_date",
    "claim_value",
    "sentence_outcome",
    "sentence_text",
]
KEEP_COLUMNS = ["uf", "comarca", "vara", "sentence_outcome"]
DROPPED_COLUMNS = ["cpf", "party_name", "judge_name", "process_number"]


class PIIError(ValueError):
    """Raised when identifiable data is found where it must not be."""


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


def pepper_fingerprint(pepper: str) -> str:
    """Identifies WHICH key produced a dataset without revealing it (for key rotation audits)."""
    return hashlib.sha256(pepper.encode()).hexdigest()[:8]


def pseudonymize(
    value: object, pepper: str, prefix: str, *, digits_only: bool = False
) -> str | None:
    """Deterministic keyed pseudonym. Plain hashes are NOT enough for low-entropy ids like CPF."""
    if value is None or pd.isna(value):
        return None
    norm = re.sub(r"\D", "", str(value)) if digits_only else _norm(value)
    digest = hmac.new(pepper.encode(), norm.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:10]}"


def scrub_text(text: object, known_names: tuple[str, ...] | list[str] = ()) -> object:
    """Replace identifiers in free text with neutral tokens."""
    if not isinstance(text, str):
        return text
    out = text
    for token, pattern in SCRUB_ORDER:
        out = pattern.sub(token, out)
    for name in sorted({n for n in known_names if isinstance(n, str) and n}, key=len, reverse=True):
        out = re.sub(re.escape(name), "[NOME]", out, flags=re.IGNORECASE)
    return out


def has_sensitive_terms(text: object) -> bool:
    return isinstance(text, str) and SENSITIVE_RE.search(text) is not None


def date_shift_days(pepper: str) -> int:
    """Secret shift of 1..5 whole 52-week blocks into the past (keeps weekday and season)."""
    digest = hmac.new(pepper.encode(), b"date-shift", hashlib.sha256).digest()
    return -364 * (1 + int.from_bytes(digest[:4], "big") % 5)


def round_significant(values: pd.Series, digits: int) -> pd.Series:
    """Round to ``digits`` significant digits (generalizes an exact, linkable amount)."""
    x = values.astype(float)
    magnitude = np.floor(np.log10(x.abs().where(x != 0)))
    decimals = (digits - 1 - magnitude).fillna(0).astype(int)
    rounded = [
        np.round(v, int(d)) if pd.notna(v) else np.nan for v, d in zip(x, decimals, strict=True)
    ]
    return pd.Series(rounded, index=values.index)


def assert_no_pii(df: pd.DataFrame) -> None:
    """Fail closed if any text column still matches a PII pattern. Never prints the values."""
    findings = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        for label, pattern in PII_PATTERNS.items():
            hits = int(series.str.contains(pattern).sum())
            if hits:
                findings.append(f"{col}: {hits} rows match {label}")
    if findings:
        raise PIIError("PII patterns found -> " + "; ".join(findings))


def anonymize_cases(
    raw: pd.DataFrame,
    pepper: str,
    *,
    shift_dates: bool = True,
    claim_sig_digits: int | None = 2,
    redact_sensitive: bool = True,
    report: dict | None = None,
) -> pd.DataFrame:
    """Raw export -> pseudonymized, generalized table. Pass ``report={}`` to collect statistics."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    n_in = len(raw)
    if "segredo_justica" in raw.columns:  # judicial secrecy: never leaves the source system
        raw = raw[~raw["segredo_justica"].fillna(False).astype(bool)]
    n_secret = n_in - len(raw)
    raw = raw.reset_index(drop=True)

    out = pd.DataFrame(
        {
            "case_id": [
                pseudonymize(v, pepper, "CASE", digits_only=True) for v in raw["process_number"]
            ],
            "judge_id": [pseudonymize(v, pepper, "JUDGE") for v in raw["judge_name"]],
        }
    )
    for col in KEEP_COLUMNS:
        out[col] = raw[col].to_numpy()

    shift = date_shift_days(pepper) if shift_dates else 0
    for col in ("filing_date", "decision_date"):
        dates = pd.to_datetime(raw[col]) + pd.Timedelta(days=shift)
        out[col] = dates.dt.strftime("%Y-%m-%d").to_numpy()

    claim = raw["claim_value"].astype(float)
    out["claim_value"] = (
        round_significant(claim, claim_sig_digits).to_numpy() if claim_sig_digits else claim
    )

    texts = [
        scrub_text(text, (judge, party))
        for text, judge, party in zip(
            raw["sentence_text"], raw["judge_name"], raw["party_name"], strict=True
        )
    ]
    sensitive = [has_sensitive_terms(t) for t in raw["sentence_text"]]
    if redact_sensitive:
        texts = [REDACTED_TEXT if s else t for t, s in zip(texts, sensitive, strict=True)]
    out["sentence_text"] = texts
    assert_no_pii(out)

    if report is not None:
        report.update(
            {
                "rows_in": n_in,
                "rows_dropped_judicial_secrecy": n_secret,
                "rows_out": len(out),
                "dropped_columns": DROPPED_COLUMNS,
                "pseudonymized_columns": ["case_id", "judge_id"],
                "pseudonymization": "HMAC-SHA256 keyed with secret pepper (key id below)",
                "pepper_fingerprint": pepper_fingerprint(pepper),
                "dates_shifted": bool(shift_dates),
                "claim_value_significant_digits": claim_sig_digits,
                "texts_with_art11_vocabulary": int(sum(sensitive)),
                "texts_redacted": int(sum(sensitive)) if redact_sensitive else 0,
                "pii_scan": "passed",
            }
        )
    return out


def _linkage_view(df: pd.DataFrame) -> pd.DataFrame:
    """Quasi-identifiers a motivated outsider could realistically know from a public docket:
    where (uf, comarca, vara), roughly when (year) and roughly how much (half-decade band)."""
    claim = pd.to_numeric(df["claim_value"], errors="coerce")
    band = np.floor(2 * np.log10(claim.where(claim > 0))).astype("Int64").astype(str)
    return pd.DataFrame(
        {
            "uf": df["uf"],
            "comarca": df["comarca"],
            "vara": df["vara"],
            "filing_year": pd.to_datetime(df["filing_date"]).dt.year,
            "claim_band": band,
        }
    )


LINKAGE_QI = ["uf", "comarca", "vara", "filing_year", "claim_band"]


def reidentification_risk(df: pd.DataFrame, k: int = 5, *, linkage: bool = True) -> dict:
    """k-anonymity style risk. ``linkage=True`` uses what an outsider could know; otherwise the
    internal-analytics view (uf, comarca, vara, judge_id)."""
    if linkage:
        view, qi = _linkage_view(df), LINKAGE_QI
    else:
        qi = ["uf", "comarca", "vara", "judge_id"]
        view = df[qi]
    size = view.groupby(qi, dropna=False)[qi[0]].transform("size")
    return {
        "quasi_identifiers": qi,
        "k_threshold": k,
        "n_groups": int(view.groupby(qi, dropna=False).ngroups),
        "min_k": int(size.min()),
        "pct_records_in_groups_below_k": round(float((size < k).mean() * 100), 2),
        "avg_prosecutor_risk": round(float((1 / size).mean()), 4),
        "max_prosecutor_risk": round(float(1 / size.min()), 4),
    }


def l_diversity(df: pd.DataFrame, sensitive: str = "sentence_outcome", l_min: int = 2) -> float:
    """% of records whose linkage group has fewer than ``l_min`` distinct sensitive values
    (homogeneity attack: knowing the group reveals the outcome even if k is large)."""
    view = _linkage_view(df).assign(_s=df[sensitive].to_numpy())
    distinct = view.groupby(LINKAGE_QI, dropna=False)["_s"].transform("nunique")
    return round(float((distinct < l_min).mean() * 100), 2)


def suppress_small_groups(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, int]:
    """Drop records in linkage groups smaller than ``k``. Returns (frame, n_suppressed)."""
    size = _linkage_view(df).groupby(LINKAGE_QI, dropna=False)["uf"].transform("size")
    keep = (size >= k).to_numpy()
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str], k: int = 5) -> pd.DataFrame:
    """Quasi-identifier combinations shared by fewer than ``k`` rows."""
    sizes = df.groupby(quasi_identifiers, dropna=False).size().rename("n").reset_index()
    return sizes[sizes["n"] < k].sort_values("n").reset_index(drop=True)
