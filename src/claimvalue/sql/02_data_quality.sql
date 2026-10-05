-- One-row data-quality report, run after every load.
SELECT
    COUNT(*)                                                        AS n_cases,
    COUNT(DISTINCT judge_id)                                        AS n_judges,
    COUNT(DISTINCT comarca)                                         AS n_comarcas,
    ROUND(100.0 * SUM(claim_value IS NULL) / COUNT(*), 2)           AS pct_missing_claim_value,
    ROUND(100.0 * AVG(unfavorable), 2)                              AS pct_unfavorable,
    MIN(filing_date)                                                AS first_filing,
    MAX(filing_date)                                                AS last_filing,
    ROUND(AVG(julianday(decision_date) - julianday(filing_date)), 1) AS avg_days_to_decision,
    ROUND(MAX(claim_value), 2)                                      AS max_claim_value
FROM cases;
