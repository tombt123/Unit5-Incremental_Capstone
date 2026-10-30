# Churn data insights

* **Class balance:** 20.4% of the 10,000 customers churned, so the classes are imbalanced.
  Accuracy alone would be misleading (always predicting "stays" scores 79.6%); models are
  judged on F1, precision and recall, and class weights are used where available.
* **Geography:** Germany has the highest churn rate (32.4%); France the lowest (16.2%).
* **Gender:** Female customers churn more (25.1% vs 16.5%).
* **Age:** churned customers are older on average (44.8 vs 37.4 years);
  the 51-60 group churns the most (56.2%). Age has the strongest
  correlation with churn among the numeric columns (r = 0.29).
* **Activity:** inactive members churn at 26.9% vs 14.3% for active members.
* **Products:** customers with 4 products churn at 100.0%;
  customers with 2 products churn least (7.6%).
* **Balance:** churned customers hold higher balances on average (91,108 vs
  72,745); 36.2% of customers have a zero balance.
* **Data quality:** no missing values or duplicates. The first column ('CustomerId') is really a row number and the real id is in
  'CustomerId.1'; both ids and Surname are dropped before modelling.

## Implications for modelling
* Use stratified train/test splits and report F1, precision and recall, not accuracy alone.
* One-hot encode Geography and Gender; scale numeric features; add `HasBalance`, `BalanceSalaryRatio` and `TenureByAge`.
* Expect Age, NumOfProducts, IsActiveMember and Geography to be the most useful predictors.
