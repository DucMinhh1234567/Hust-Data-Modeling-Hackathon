## Goal

Predict daily sales quantity for the next 56 days across roughly 15,972 SKUs of a Vietnamese Auto Parts distributor based on nearly 5 years of transaction history (2020-11-17 → 2025-09-05).

This is a classic retail demand forecasting problem: the SKU distribution is long-tailed a small number of SKUs contribute most of the profit, while many sell sparsely. A good model must capture seasonality / trend for the best-selling SKUs, avoid over-predicting sparse SKUs and handle returns transactions appropriately.

## Leaderboard

The Public score (Validation) is computed on the first 28 days of the horizon (F1..F28 = 2025-09-06 → 2025-10-03) and the Private score (Evaluation) is computed on the next 28 days (F29..F56 = 2025-10-04 → 2025-10-31).

## Competition flow

The top teams (lowest WRMSSE on the Private leaderboard) will advance to the next round. Lower score is better.