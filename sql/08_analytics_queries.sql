-- Bullish/bearish trend detection
SELECT symbol,
       SUM(CASE WHEN pct_change > 0 THEN 1 ELSE 0 END) AS up_days,
       SUM(CASE WHEN pct_change < 0 THEN 1 ELSE 0 END) AS down_days
FROM model_stock_data
GROUP BY symbol;

-- Fraud hotspots by location
SELECT location, COUNT(*) AS fraud_count
FROM fraud_transactions_raw
WHERE is_fraud = 1
GROUP BY location
ORDER BY fraud_count DESC;
