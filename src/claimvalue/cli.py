"""Command line entry point:  claimvalue <command>"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from claimvalue.anomaly import flag_anomalies
from claimvalue.anonymize import (
    anonymize_cases,
    assert_no_pii,
    l_diversity,
    reidentification_risk,
    suppress_small_groups,
)
from claimvalue.config import get_settings
from claimvalue.db import build_db, data_quality, load_abt
from claimvalue.synthetic import generate_raw_cases
from claimvalue.train import TASKS, run_training


def _generate(n: int) -> None:
    s = get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    generate_raw_cases(n).to_csv(s.raw_csv, index=False)
    print(f"[generate] {n} SYNTHETIC raw cases -> {s.raw_csv}")


def _anonymize(sig_digits: int, suppress_k: int, delete_raw: bool) -> None:
    s = get_settings()
    raw = pd.read_csv(s.raw_csv)
    report: dict = {}
    anon = anonymize_cases(
        raw, s.require_pepper(), claim_sig_digits=sig_digits or None, report=report
    )
    if suppress_k:
        anon, n_suppressed = suppress_small_groups(anon, suppress_k)
        report.update({"suppression_k": suppress_k, "rows_suppressed": n_suppressed})
        report["rows_out"] = len(anon)
    report["risk_internal_view"] = reidentification_risk(anon, linkage=False)
    report["risk_public_linkage_view"] = reidentification_risk(anon, linkage=True)
    report["pct_records_homogeneous_outcome"] = l_diversity(anon)
    report["legal_status"] = (
        "PSEUDONYMIZED: still personal data under LGPD while the key exists. Treat as such "
        "until a documented re-identification risk assessment concludes otherwise."
    )
    report["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    s.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    anon.to_csv(s.processed_csv, index=False)
    (s.processed_csv.parent / "privacy_report.json").write_text(json.dumps(report, indent=2))
    link = report["risk_public_linkage_view"]
    print(f"[anonymize] {len(raw)} raw -> {len(anon)} rows -> {s.processed_csv}")
    print(f"[anonymize] judicial-secrecy rows dropped: {report['rows_dropped_judicial_secrecy']}")
    print(
        f"[anonymize] linkage risk: min k={link['min_k']}, "
        f"{link['pct_records_in_groups_below_k']}% of records in groups below k={link['k_threshold']}"
    )
    if delete_raw:
        s.raw_csv.unlink()
        print(
            f"[anonymize] raw file deleted: {s.raw_csv} (use an encrypted volume; see docs/LGPD.md)"
        )


def _build_db() -> None:
    s = get_settings()
    n = build_db(pd.read_csv(s.processed_csv), s.db_path)
    print(f"[build-db] {n} cases -> {s.db_path}")
    print(data_quality(s.db_path).T.to_string(header=False))


def _train(task: str, n_iter: int) -> None:
    s = get_settings()
    for t in TASKS if task == "both" else (task,):
        res = run_training(t, s.db_path, s.model_dir, n_iter=n_iter, mlflow_uri=s.mlflow_uri)
        print(f"\n[train:{t}] best={res['meta']['model']} (selected by time-series CV)")
        print(res["leaderboard"].round(3).to_string(index=False))
        print(
            "95% CI of primary test metric:",
            [round(x, 3) for x in res["meta"]["primary_metric_ci95"]],
        )


def _train_dl() -> None:
    from claimvalue.dl import train_dl_regression

    s = get_settings()
    res = train_dl_regression(s.db_path, s.model_dir)
    print(
        "[train-dl]",
        {k: round(v, 3) for k, v in res["test_metrics"].items()},
        res["epochs_run"],
        "epochs",
    )


def _eda() -> None:
    from claimvalue.eda import write_eda_report

    s = get_settings()
    out = write_eda_report(load_abt(s.db_path), s.model_dir / "reports" / "eda")
    print(f"[eda] report + figures -> {out}")


def _anomalies() -> None:
    s = get_settings()
    flagged = flag_anomalies(load_abt(s.db_path))
    out = s.model_dir / "reports" / "anomalies.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out, index=False)
    print(f"[anomalies] {int(flagged['is_anomaly'].sum())} flagged -> {out}")


def _check_pii() -> None:
    s = get_settings()
    assert_no_pii(pd.read_csv(s.processed_csv))
    print("[check-pii] OK: no CPF / CNPJ / process-number patterns in processed data")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="claimvalue")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate-synthetic", "pipeline"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=20000)
        if name == "pipeline":
            sp.add_argument("--n-iter", type=int, default=8)
    tr = sub.add_parser("train")
    tr.add_argument("--task", choices=[*TASKS, "both"], default="both")
    tr.add_argument("--n-iter", type=int, default=8)
    an = sub.add_parser("anonymize")
    an.add_argument("--claim-sig-digits", type=int, default=2, help="0 = keep exact claim value")
    an.add_argument("--suppress-k", type=int, default=0, help="drop linkage groups smaller than k")
    an.add_argument("--delete-raw", action="store_true", help="delete the raw export afterwards")
    for name in ("build-db", "train-dl", "anomalies", "check-pii", "eda"):
        sub.add_parser(name)
    a = p.parse_args(argv)

    if a.cmd == "generate-synthetic":
        _generate(a.n)
    elif a.cmd == "anonymize":
        _anonymize(a.claim_sig_digits, a.suppress_k, a.delete_raw)
    elif a.cmd == "eda":
        _eda()
    elif a.cmd == "build-db":
        _build_db()
    elif a.cmd == "train":
        _train(a.task, a.n_iter)
    elif a.cmd == "train-dl":
        _train_dl()
    elif a.cmd == "anomalies":
        _anomalies()
    elif a.cmd == "check-pii":
        _check_pii()
    elif a.cmd == "pipeline":
        _generate(a.n)
        _anonymize(2, 0, False)
        _build_db()
        _eda()
        _train("both", a.n_iter)
        _anomalies()
        _check_pii()


if __name__ == "__main__":
    main()
