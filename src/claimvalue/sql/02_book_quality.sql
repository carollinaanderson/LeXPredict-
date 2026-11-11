-- Data-quality checks for the book dataset. One row per check, easy to print.
SELECT 'rows' AS check_name, CAST(COUNT(*) AS TEXT) AS value FROM book_cases
UNION ALL
SELECT 'distinct_case_id', CAST(COUNT(DISTINCT case_id) AS TEXT) FROM book_cases
UNION ALL
SELECT 'missing_claim_value', CAST(SUM(claim_value IS NULL) AS TEXT) FROM book_cases
UNION ALL
SELECT 'nonpositive_claim_value', CAST(SUM(claim_value <= 0) AS TEXT) FROM book_cases
UNION ALL
SELECT 'missing_filing_date', CAST(SUM(filing_date IS NULL) AS TEXT) FROM book_cases
UNION ALL
SELECT 'decision_before_filing', CAST(SUM(decision_date < filing_date) AS TEXT) FROM book_cases
UNION ALL
SELECT 'binary_label_rows', CAST(SUM(reformed IS NOT NULL) AS TEXT) FROM book_cases
UNION ALL
SELECT 'reformed_rate',
       CAST(ROUND(AVG(reformed), 4) AS TEXT) FROM book_cases WHERE reformed IS NOT NULL
UNION ALL
SELECT 'first_filing_date', MIN(filing_date) FROM book_cases
UNION ALL
SELECT 'last_decision_date', MAX(decision_date) FROM book_cases
ORDER BY check_name;
