"""Synthetic raw court cases.

Everything here is fictional: names are random combinations from short lists, CPFs are generated
with deliberately WRONG check digits (so they can never belong to a real person) and case numbers
are random. The raw table contains fake PII on purpose so the anonymization step is exercised
exactly as it would be on a real export. Geography (UF / comarca) is public information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIRST = [
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Fabio", "Gisele", "Hugo", "Irene", "Jorge",
    "Karen", "Lucas", "Marta", "Nuno", "Olga", "Paulo", "Rita", "Sergio", "Tania", "Vitor",
]  # fmt: skip
LAST = [
    "Almeida", "Barros", "Cardoso", "Dias", "Esteves", "Farias", "Gomes", "Heleno", "Ibrahim",
    "Junqueira", "Klein", "Lopes", "Moreira", "Nogueira", "Oliveira", "Pinto", "Queiroz",
    "Ramos", "Siqueira", "Tavares",
]  # fmt: skip
LOCATIONS = [
    ("SP", "São Paulo"), ("SP", "Campinas"), ("SP", "Santos"), ("RJ", "Rio de Janeiro"),
    ("RJ", "Niterói"), ("MG", "Belo Horizonte"), ("MG", "Uberlândia"), ("PR", "Curitiba"),
    ("PR", "Londrina"), ("BA", "Salvador"), ("BA", "Feira de Santana"), ("GO", "Goiânia"),
    ("MS", "Campo Grande"), ("MS", "Dourados"), ("RS", "Porto Alegre"), ("RS", "Caxias do Sul"),
]  # fmt: skip
UF_EFFECT = {
    "SP": 0.35,
    "RJ": 0.2,
    "MG": 0.0,
    "PR": 0.05,
    "BA": -0.25,
    "GO": -0.1,
    "MS": -0.2,
    "RS": 0.1,
}
VARAS = {
    "1ª Vara Cível": 0.0,
    "2ª Vara Cível": 0.05,
    "Vara Empresarial": 0.6,
    "Juizado Especial Cível": -1.2,
}
FAV = (["Improcedente", "Extinção sem resolução de mérito"], [0.75, 0.25])
UNFAV = (["Procedente", "Parcialmente procedente"], [0.4, 0.6])
PHRASE = {
    "Procedente": "JULGO PROCEDENTE o pedido",
    "Parcialmente procedente": "JULGO PARCIALMENTE PROCEDENTE o pedido",
    "Improcedente": "JULGO IMPROCEDENTE o pedido",
    "Extinção sem resolução de mérito": "EXTINGO o processo sem resolução de mérito",
}


def _names(rng: np.random.Generator, n: int) -> list[str]:
    return [f"{a} {b}" for a, b in zip(rng.choice(FIRST, n), rng.choice(LAST, n), strict=True)]


def _fake_cpf(rng: np.random.Generator) -> str:
    d = [int(x) for x in rng.integers(0, 10, 9)]
    for _ in range(2):  # compute the valid check digits ...
        s = sum(v * w for v, w in zip(d, range(len(d) + 1, 1, -1), strict=True))
        d.append((s * 10 % 11) % 10)
    d[-2] = (d[-2] + 1) % 10  # ... then corrupt one so the CPF is invalid by construction
    t = "".join(map(str, d))
    return f"{t[:3]}.{t[3:6]}.{t[6:9]}-{t[9:]}"


def generate_raw_cases(n: int = 20000, n_judges: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    judge_names = list(dict.fromkeys(f"Juiz {x}" for x in _names(rng, n_judges * 3)))[:n_judges]
    loc = rng.integers(0, len(LOCATIONS), n_judges)
    varas = rng.choice(list(VARAS), n_judges)
    bias_value = rng.normal(0, 0.35, n_judges)
    bias_unfav = rng.normal(0, 0.8, n_judges)

    j = rng.choice(n_judges, n, p=rng.dirichlet(np.full(n_judges, 5.0)))
    uf = np.array([LOCATIONS[i][0] for i in loc])[j]
    comarca = np.array([LOCATIONS[i][1] for i in loc])[j]
    vara = varas[j]
    log_value = (
        8.6
        + np.array([UF_EFFECT[u] for u in uf])
        + np.array([VARAS[v] for v in vara])
        + bias_value[j]
        + rng.normal(0, 0.6, n)
    )
    claim = np.round(np.exp(log_value), 2)
    claim[rng.random(n) < 0.03] = np.nan  # realistic missing values

    filing = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1280, n), unit="D")
    decision = filing + pd.to_timedelta(30 + rng.gamma(3, 70, n).astype(int), unit="D")
    p_unfav = 1 / (1 + np.exp(-(-0.3 + 0.9 * bias_unfav[j] + 0.25 * (log_value - 9))))
    unfav = rng.random(n) < p_unfav
    outcome = np.where(
        unfav,
        rng.choice(UNFAV[0], n, p=UNFAV[1]),
        rng.choice(FAV[0], n, p=FAV[1]),
    )

    seq = rng.permutation(10_000_000)[:n]
    process = [
        f"{s:07d}-{rng.integers(0, 100):02d}.{ts.year}.8."
        f"{rng.integers(1, 28):02d}.{rng.integers(1, 9999):04d}"
        for s, ts in zip(seq, filing, strict=True)
    ]
    cpf = [_fake_cpf(rng) for _ in range(n)]
    party = _names(rng, n)
    judge_col = np.array(judge_names)[j]
    text = [
        f"Processo {p}. Autor(a) {a}, CPF {c}, em face do réu. Comarca de {cm}/{u}. "
        f"{PHRASE[o]}, nos termos do art. 487 do CPC. Juiz(a) {jn}."
        for p, a, c, cm, u, o, jn in zip(
            process, party, cpf, comarca, uf, outcome, judge_col, strict=True
        )
    ]
    raw = pd.DataFrame(
        {
            "process_number": process,
            "cpf": cpf,
            "party_name": party,
            "judge_name": judge_col,
            "uf": uf,
            "comarca": comarca,
            "vara": vara,
            "filing_date": filing.strftime("%Y-%m-%d"),
            "decision_date": decision.strftime("%Y-%m-%d"),
            "claim_value": claim,
            "sentence_outcome": outcome,
            "sentence_text": text,
            "segredo_justica": rng.random(n) < 0.01,  # cases under judicial secrecy
        }
    )
    dupes = raw.sample(frac=0.01, random_state=seed)  # duplicated rows: a real data-quality issue
    return pd.concat([raw, dupes], ignore_index=True)
