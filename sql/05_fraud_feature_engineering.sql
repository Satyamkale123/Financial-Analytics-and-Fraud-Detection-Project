CREATE TABLE fraud_features AS
SELECT
    t.*,
    AVG(amount) OVER (PARTITION BY customer_id) AS avg_customer_spend,
    COUNT(*) OVER (PARTITION BY customer_id ORDER BY transaction_time RANGE BETWEEN INTERVAL '10 minutes' PRECEDING AND CURRENT ROW) AS tx_count_10min,
    CASE WHEN amount > AVG(amount) OVER (PARTITION BY customer_id) * 2 THEN 1 ELSE 0 END AS high_risk_amount,
    EXTRACT(HOUR FROM transaction_time) AS hour,
    CASE WHEN EXTRACT(HOUR FROM transaction_time) BETWEEN 0 AND 5 THEN 1 ELSE 0 END AS night_time_flag,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_time) AS purchase_seq
FROM fraud_transactions_raw t;
