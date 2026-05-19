## Metric: WRMSSE
The competition is scored using a Weighted Root Mean Squared Scaled Error (WRMSSE) computed per SKU and aggregated by profit weight. Lower is better.

### 1 - RMSSE

$$RMSSE = \sqrt{\frac{\frac{1}{h} \sum_{t=n+1}^{n+h} (Y_t - \hat{Y}_t)^2}{\frac{1}{n-1} \sum_{t=2}^n (Y_t - Y_{t-1})^2}}$$

where:

- Y_t is the actual value of the time series at time t
- Ŷ_t is the forecast at time t
- n is the length of the training sample (number of historical observations)
- h is the forecasting horizon

The denominator is the mean squared error of the naive one-step forecast (Ŷt = Y{t-1}) computed on the SKU's training data. Interpretation:

- RMSSE = 1 → your model ties the naive baseline
- RMSSE < 1 → better than the naive baseline
- RMSSE > 1 → worse than the naive baseline

### 2 - Aggregate by profit weight
The final score is the weighted average of per-SKU RMSSE, where the weight is each product's share of total profit on the training set:

$$WRMSSE = \sum_{i=1}^{M} w_i \times \text{RMSSE}_i$$

1. For each SKU i, compute cumulative profit over the entire training set (2020-11-17 → 2025-09-05): profit_i = Σ_train ( SalesAmount − CostAmount )

2. A product with negative profit is treated as having weight 0 (profit_i < 0 → profit_i := 0).