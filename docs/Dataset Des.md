## Files provided

| File | Description |
| :--- | :--- |
| `train.csv` | Detailed transaction history, **711,980 rows**, 2020-11-17 → 2025-09-05. Each row is a transaction line. |
| `sample_submission.csv` | Submission template (**31,944 rows × 29 cols**): each SKU has two rows `<SKU>_validation` (Public window) and `<SKU>_evaluation` (Private window), each with 28 forecast columns `F1..F28`. All zeros is a placeholder. |

## Columns of `train.csv`

| Column | Type | Meaning |
| :--- | :--- | :--- |
| **Date** | YYYY-MM-DD | Date of the transaction. |
| **Stt** | int | Internal row sequence number. |
| **ItemCode** | string | SKU code, e.g. `SKU-08063`. |
| **Quantity** | int | Quantity. **Positive** for sales, **negative** for returns. |
| **UnitPrice** | string (decimal `,`) | Unit selling price (VND). |
| **SalesAmount** | int | Line revenue = UnitPrice × Quantity (VND). |
| **Unit Cost** | string (decimal `,`) | Unit cost (VND). |
| **Cost Amount** | int | Line cost = Unit Cost × Quantity (VND). |

## Return transactions

Transactions where **Quantity**, **SalesAmount**, and **Cost Amount** are all **negative** are customer return transactions. Handle this return data appropriately in your models.

Regardless of your approach, the final prediction for **Quantity** in the submission **must be non-negative**.

## `sample_submission.csv` format

The submission file is a CSV with the following structure:

- **Header:** `id, F1, F2, ..., F28`
- **Row structure:** Each `id` is `<ItemCode>` plus a suffix:
  - `_validation` — Public leaderboard. Columns `F1`–`F28` are dates **2025-09-06 → 2025-10-03**.
  - `_evaluation` — Private leaderboard. Columns `F1`–`F28` are dates **2025-10-04 → 2025-10-31**.
- **Example IDs:** `SKU-00001_validation`, `SKU-00001_evaluation`

You predict demand for **28 forecast days** (`F1`–`F28`) per row. There are **31,944** rows total (**15,972** SKUs × 2 time windows).

**Value constraints:**

- Forecast values must be **non-negative floats**.
- Any negative values in the submission are clipped to 0.
- The scoring metric uses floats directly with **no rounding**.

**Submission rules:**

- Submissions with duplicate `id` values will be rejected.
- The submission must contain the **exact same set of IDs** as `sample_submission.csv`. Missing any row results in a "Submission Scoring Error."
