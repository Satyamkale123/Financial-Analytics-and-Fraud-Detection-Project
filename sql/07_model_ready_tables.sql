CREATE TABLE model_stock_data AS
SELECT symbol, date, close, pct_change, ma_50, ma_200, volatility_30
FROM stock_features;

CREATE TABLE model_fraud_data AS
SELECT transaction_id, customer_id, amount, avg_customer_spend,
       high_risk_amount, night_time_flag, tx_count_10min,
       hour, is_fraud
FROM fraud_features;
