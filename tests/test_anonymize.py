import pytest

from claimvalue.anonymize import (
    PIIError,
    anonymize_cases,
    assert_no_pii,
    k_anonymity,
    pseudonymize,
    scrub_text,
)
from tests.conftest import PEPPER


def test_pseudonym_is_deterministic_and_keyed():
    a = pseudonymize("Juiz Ana Dias", PEPPER, "JUDGE")
    assert a == pseudonymize("  juiz ana  DIAS ", PEPPER, "JUDGE")  # case/space-insensitive
    assert a != pseudonymize("Juiz Ana Dias", PEPPER + "other", "JUDGE")  # depends on the secret
    assert a.startswith("JUDGE_") and "Ana" not in a


def test_case_id_ignores_number_formatting():
    fmt = pseudonymize("0001234-56.2022.8.26.0100", PEPPER, "CASE", digits_only=True)
    plain = pseudonymize("00012345620228260100", PEPPER, "CASE", digits_only=True)
    assert fmt == plain


@pytest.mark.parametrize(
    "text",
    [
        "CPF 123.456.789-09 consta nos autos",
        "CPF 12345678909 consta nos autos",
        "CNPJ 12.345.678/0001-95 consta nos autos",
        "Processo 0001234-56.2022.8.26.0100 consta nos autos",
    ],
)
def test_scrub_removes_identifiers(text):
    out = scrub_text(text)
    assert not any(
        ch.isdigit() for ch in out.split(" consta")[0].replace("CPF", "").replace("CNPJ", "")
    )
    assert_no_pii(__import__("pandas").DataFrame({"t": [out]}))


def test_scrub_removes_known_names_case_insensitively():
    assert scrub_text("Autor MARIA Souza venceu", ["Maria Souza"]) == "Autor [NOME] venceu"


def test_assert_no_pii_fails_closed_without_leaking_values():
    import pandas as pd

    with pytest.raises(PIIError) as err:
        assert_no_pii(pd.DataFrame({"t": ["ok", "CPF 123.456.789-09"]}))
    assert "123.456" not in str(err.value)


def test_anonymized_frame_has_no_identifiers(raw, anon):
    assert_no_pii(anon)
    for col in ("cpf", "party_name", "judge_name", "process_number"):
        assert col not in anon.columns
    blob = " ".join(anon["sentence_text"].dropna())
    assert not any(name in blob for name in raw["judge_name"].unique()[:20])
    assert not any(name in blob for name in raw["party_name"].unique()[:50])
    # the signal survives: location and the ruling are still there
    assert {"uf", "comarca", "vara", "sentence_outcome", "sentence_text"} <= set(anon.columns)
    assert "JULGO" in blob and "Comarca de" in blob


def test_missing_columns_raise(raw):
    with pytest.raises(ValueError):
        anonymize_cases(raw.drop(columns=["cpf"]), PEPPER)


def test_k_anonymity_reports_small_groups(anon):
    risky = k_anonymity(anon, ["uf", "comarca", "vara", "judge_id"], k=10**6)
    assert len(risky) == anon.groupby(["uf", "comarca", "vara", "judge_id"]).ngroups


# ---- LGPD-oriented controls -------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from claimvalue.anonymize import (  # noqa: E402
    REDACTED_TEXT,
    date_shift_days,
    has_sensitive_terms,
    l_diversity,
    pepper_fingerprint,
    reidentification_risk,
    round_significant,
    suppress_small_groups,
)


@pytest.mark.parametrize(
    "text,token",
    [
        ("contato maria@exemplo.com.br hoje", "[EMAIL]"),
        ("telefone (67) 99999-1234 residencial", "[TELEFONE]"),
        ("CEP 79002-100 centro", "[CEP]"),
        ("advogado OAB/SP 123.456 constituido", "[OAB]"),
    ],
)
def test_scrub_extra_identifiers(text, token):
    assert token in scrub_text(text)
    assert_no_pii(pd.DataFrame({"t": [scrub_text(text)]}))


def test_sensitive_text_is_redacted_entirely(raw):
    sample = raw.head(3).copy()
    sample.loc[sample.index[0], "sentence_text"] = "Autor portador de HIV e depressão"
    out = anonymize_cases(sample, PEPPER)
    assert out["sentence_text"].iloc[0] == REDACTED_TEXT
    assert out["sentence_text"].iloc[1] != REDACTED_TEXT
    assert has_sensitive_terms("laudo médico anexo") and not has_sensitive_terms("JULGO PROCEDENTE")


def test_dates_are_shifted_but_intervals_and_weekdays_survive(raw):
    sample = raw.head(200)
    out = anonymize_cases(sample, PEPPER)
    shift = date_shift_days(PEPPER)
    assert shift < 0 and shift % 364 == 0
    f_raw, f_out = (
        pd.to_datetime(sample["filing_date"]).reset_index(drop=True),
        pd.to_datetime(out["filing_date"]),
    )
    d_raw, d_out = (
        pd.to_datetime(sample["decision_date"]).reset_index(drop=True),
        pd.to_datetime(out["decision_date"]),
    )
    assert (f_raw != f_out).all()
    assert ((d_raw - f_raw) == (d_out - f_out)).all()  # durations preserved
    assert (f_raw.dt.weekday == f_out.dt.weekday).all()  # weekday preserved
    assert date_shift_days(PEPPER) == date_shift_days(PEPPER)


def test_claim_value_is_generalized_with_bounded_error():
    s = pd.Series([4812.35, 129.99, 1_250_000.0, np.nan, 15.5])
    r = round_significant(s, 2)
    assert r.iloc[0] == 4800 and r.iloc[2] == 1_200_000 and np.isnan(r.iloc[3])
    rel = ((r - s).abs() / s).dropna()
    assert rel.max() <= 0.051


def test_judicial_secrecy_cases_are_dropped(raw):
    sample = raw.head(50).copy()
    sample["segredo_justica"] = False
    sample.loc[sample.index[:5], "segredo_justica"] = True
    rep = {}
    out = anonymize_cases(sample, PEPPER, report=rep)
    assert len(out) == 45 and rep["rows_dropped_judicial_secrecy"] == 5


def test_report_describes_actions_without_leaking_data(raw):
    rep = {}
    anonymize_cases(raw.head(100), PEPPER, report=rep)
    assert rep["pepper_fingerprint"] == pepper_fingerprint(PEPPER) and rep["pii_scan"] == "passed"
    blob = str(rep)
    assert PEPPER not in blob and raw["judge_name"].iloc[0] not in blob


def test_reidentification_risk_and_suppression(anon):
    risk = reidentification_risk(anon, k=5, linkage=True)
    assert risk["min_k"] >= 1 and 0 <= risk["pct_records_in_groups_below_k"] <= 100
    assert risk["max_prosecutor_risk"] == pytest.approx(1 / risk["min_k"], rel=1e-3)
    kept, n = suppress_small_groups(anon, k=5)
    assert n == len(anon) - len(kept)
    if len(kept):
        assert reidentification_risk(kept, k=5, linkage=True)["min_k"] >= 5
    assert 0 <= l_diversity(anon) <= 100
