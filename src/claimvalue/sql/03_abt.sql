-- Analytical Base Table with leakage-safe rolling-window features.
--
-- For each case c (filed on c.filing_date) we only use history h that was ALREADY DECIDED
-- before that date and inside the last :window_days days:
--     h.decision_date <  c.filing_date
--     h.decision_date >= c.filing_date - :window_days
-- Nothing known only after the reference date can reach a feature, which is the "janela movel"
-- idea: every row is a snapshot as of its own filing date.
WITH judge_hist AS (
    SELECT c.case_id,
           COUNT(h.case_id)       AS judge_n_prior,
           AVG(h.claim_value)     AS judge_avg_value_prior,
           AVG(h.unfavorable)     AS judge_unfav_rate_prior
    FROM cases c
    LEFT JOIN cases h
           ON h.judge_id = c.judge_id
          AND h.decision_date <  c.filing_date
          AND h.decision_date >= date(c.filing_date, '-' || :window_days || ' days')
    GROUP BY c.case_id
),
comarca_hist AS (
    SELECT c.case_id,
           COUNT(h.case_id)       AS comarca_n_prior,
           AVG(h.claim_value)     AS comarca_avg_value_prior,
           AVG(h.unfavorable)     AS comarca_unfav_rate_prior
    FROM cases c
    LEFT JOIN cases h
           ON h.comarca = c.comarca
          AND h.decision_date <  c.filing_date
          AND h.decision_date >= date(c.filing_date, '-' || :window_days || ' days')
    GROUP BY c.case_id
)
SELECT c.case_id, c.judge_id, c.uf, c.comarca, c.vara,
       c.filing_date, c.decision_date,
       CAST(strftime('%m', c.filing_date) AS INTEGER) AS filing_month,
       c.claim_value, c.unfavorable,
       j.judge_n_prior, j.judge_avg_value_prior, j.judge_unfav_rate_prior,
       k.comarca_n_prior, k.comarca_avg_value_prior, k.comarca_unfav_rate_prior
FROM cases c
JOIN judge_hist   j USING (case_id)
JOIN comarca_hist k USING (case_id)
ORDER BY c.filing_date, c.case_id;
