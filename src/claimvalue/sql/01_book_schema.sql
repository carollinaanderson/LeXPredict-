-- Book dataset (abjData::consumo). Prepared table, one row per appeal.
DROP TABLE IF EXISTS book_cases;

CREATE TABLE book_cases (
    case_id          TEXT PRIMARY KEY,
    assunto          TEXT NOT NULL,
    comarca          TEXT NOT NULL,
    tipo_litigio     TEXT NOT NULL CHECK (tipo_litigio IN ('PF-nPF', 'nPF-PF', 'nPF-nPF', 'PF-PF')),
    filing_date      TEXT NOT NULL,
    decision_date    TEXT NOT NULL,
    claim_value      REAL NOT NULL CHECK (claim_value > 0),
    decision_outcome TEXT NOT NULL,
    reformed         INTEGER CHECK (reformed IS NULL OR reformed IN (0, 1)),
    unanimity        TEXT NOT NULL,
    case_days        INTEGER NOT NULL CHECK (case_days >= 0),
    CHECK (decision_date >= filing_date)
);

-- Supports the leakage-safe self-joins in 03_book_abt.sql.
CREATE INDEX idx_book_comarca_decision ON book_cases (comarca, decision_date);
CREATE INDEX idx_book_assunto_decision ON book_cases (assunto, decision_date);
CREATE INDEX idx_book_filing ON book_cases (filing_date);
