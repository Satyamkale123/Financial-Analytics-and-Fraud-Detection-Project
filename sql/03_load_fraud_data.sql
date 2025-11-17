COPY fraud_transactions_raw
FROM 's3://your-bucket/fraud/fraud_dataset.csv'
IAM_ROLE 'arn:aws:iam::12345:role/MyRedshiftRole'
CSV IGNOREHEADER 1;
