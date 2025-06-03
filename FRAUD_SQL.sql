CREATE DATABASE fraud_detection;
USE fraud_detection;
CREATE TABLE transactions (
  step INT,
  type VARCHAR(20),
  amount FLOAT,
  nameOrig VARCHAR(30),
  oldbalanceOrg FLOAT,
  newbalanceOrig FLOAT,
  nameDest VARCHAR(30),
  oldbalanceDest FLOAT,
  newbalanceDest FLOAT,
  isFraud INT,
  isFlaggedFraud INT
);

SELECT 
  COUNT(*) AS Total_Transactions,
  SUM(isFraud) AS Fraudulent_Transactions,
  ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS Fraud_Percentage
FROM transactions;

SELECT amount, type, nameOrig, nameDest
FROM transactions
WHERE isFraud = 1
ORDER BY amount DESC
LIMIT 5;

SELECT type, COUNT(*) AS Count
FROM transactions
WHERE isFraud = 1
GROUP BY type
ORDER BY Count DESC;
