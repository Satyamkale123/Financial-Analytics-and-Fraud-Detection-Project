-- Daily average price
SELECT date, AVG(close) AS avg_close
FROM stocks_raw
GROUP BY date
ORDER BY date;

-- Fraud counts by hour
SELECT EXTRACT(HOUR FROM transaction_time) AS hr, COUNT(*)
FROM fraud_transactions_raw
GROUP BY hr;
