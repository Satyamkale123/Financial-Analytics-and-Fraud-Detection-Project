CREATE TABLE stock_features AS
SELECT
    symbol,
    date,
    close,
    LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) AS close_lag_1,
    close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) AS price_diff,
    (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) AS pct_change,
    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50,
    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma_200,
    STDDEV(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS volatility_30
FROM stocks_raw;
