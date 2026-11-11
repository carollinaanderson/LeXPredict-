-- Analytical Base Table for the ABJ book dataset.
--
-- Translation of the book's methodology to this project:
--   * the unit is an appeal, filed on c.filing_date;
--   * every history feature only uses cases whose DECISION was already known before
--     that filing date, inside the last :window_days days;
--   * claim value is known at filing, but the same conservative window is used;
--   * the target `reformed` is 1 for Reformou/Parcial and 0 for Não reformou.
--     Procedural outcomes have a NULL label and are excluded from the modelling table.
--
-- There is no judge identifier in `consumo`, so the reference groups are the case
-- subject (assunto) and the municipality (comarca). This is the same leakage-safe
-- rolling-window idea used by the synthetic pipeline, with the groups the public
-- data actually supports.
WITH base AS (
    SELECT *
    FROM book_cases
    WHERE reformed IS NOT NULL
),
comarca_hist AS (
    SELECT c.case_id,
           COUNT(h.case_id)       AS comarca_n_prior,
           AVG(h.claim_value)     AS comarca_avg_value_prior,
           AVG(h.reformed)        AS comarca_reform_rate_prior
    FROM base c
    LEFT JOIN base h
           ON h.comarca = c.comarca
          AND h.decision_date <  c.filing_date
          AND h.decision_date >= date(c.filing_date, '-' || :window_days || ' days')
    GROUP BY c.case_id
),
assunto_hist AS (
    SELECT c.case_id,
           COUNT(h.case_id)       AS assunto_n_prior,
           AVG(h.claim_value)     AS assunto_avg_value_prior,
           AVG(h.reformed)        AS assunto_reform_rate_prior
    FROM base c
    LEFT JOIN base h
           ON h.assunto = c.assunto
          AND h.decision_date <  c.filing_date
          AND h.decision_date >= date(c.filing_date, '-' || :window_days || ' days')
    GROUP BY c.case_id
)
SELECT c.case_id,
       c.assunto,
       c.comarca,
       c.tipo_litigio,
       c.filing_date,
       c.decision_date,
       CAST(strftime('%m', c.filing_date) AS INTEGER) AS filing_month,
       c.claim_value,
       c.reformed,
       COALESCE(k.comarca_n_prior, 0)         AS comarca_n_prior,
       COALESCE(k.comarca_avg_value_prior, 0) AS comarca_avg_value_prior,
       COALESCE(k.comarca_reform_rate_prior, 0) AS comarca_reform_rate_prior,
       COALESCE(a.assunto_n_prior, 0)         AS assunto_n_prior,
       COALESCE(a.assunto_avg_value_prior, 0) AS assunto_avg_value_prior,
       COALESCE(a.assunto_reform_rate_prior, 0) AS assunto_reform_rate_prior
FROM base c
JOIN comarca_hist k USING (case_id)
JOIN assunto_hist a USING (case_id)
ORDER BY c.filing_date, c.case_id;
