COPY stocks_raw (symbol, date, open, high, low, close, volume)
FROM 's3://your-bucket/stocks/AAPL.csv'
IAM_ROLE 'arn:aws:iam::12345:role/MyRedshiftRole'
CSV IGNOREHEADER 1;
