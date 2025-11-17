CREATE TABLE stocks_raw (
    symbol VARCHAR(10),
    date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT
);

CREATE TABLE fraud_transactions_raw (
    transaction_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20),
    amount NUMERIC,
    transaction_time TIMESTAMP,
    merchant_category VARCHAR(50),
    device_id VARCHAR(50),
    location VARCHAR(50),
    is_fraud INT
);
