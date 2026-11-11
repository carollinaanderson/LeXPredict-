-- Anonymized case table. Constraints make bad data fail loudly at load time.
DROP TABLE IF EXISTS cases;

CREATE TABLE cases (
    case_id          TEXT PRIMARY KEY,
    judge_id         TEXT NOT NULL,
    uf               TEXT NOT NULL CHECK (length(uf) = 2),
    comarca          TEXT NOT NULL,
    vara             TEXT NOT NULL,
    filing_date      TEXT NOT NULL,            -- ISO-8601 (YYYY-MM-DD)
    decision_date    TEXT NOT NULL,
    claim_value      REAL CHECK (claim_value IS NULL OR claim_value > 0),
    sentence_outcome TEXT NOT NULL,
    unfavorable      INTEGER NOT NULL CHECK (unfavorable IN (0, 1)),
    sentence_text    TEXT,
    CHECK (decision_date >= filing_date)
);

-- Supports the rolling-window self-joins in 03_abt.sql
CREATE INDEX idx_cases_judge_decision   ON cases (judge_id, decision_date);
CREATE INDEX idx_cases_comarca_decision ON cases (comarca, decision_date);
