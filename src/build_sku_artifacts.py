"""
Build WRMSSE profit-weight artifacts from train.csv.

Outputs (default: main/output/):

Fixed rank (pipeline profit parse):
  - top99_skus.csv, top653_skus.csv
  - sku_weights.parquet

Cumulative profit share (note.md / eda.py: cum < threshold):
  - top50pct_skus_eda.csv, top80pct_skus_eda.csv
  - sku_weights_eda.parquet

Same thresholds on pipeline parse:
  - top50pct_skus.csv, top80pct_skus.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = REPO_ROOT / "main" / "output"


def parse_vn_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(x, errors="coerce")


def load_transactions(train_path: Path) -> pd.DataFrame:
    """Pipeline / notebook: full VN number parse."""
    df = pd.read_csv(train_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Unit Cost": "UnitCost", "Cost Amount": "CostAmount"})
    for col in ["Quantity", "UnitPrice", "SalesAmount", "UnitCost", "CostAmount"]:
        df[col] = parse_vn_number(df[col])
    df["line_profit"] = df["SalesAmount"] - df["CostAmount"]
    return df


def load_transactions_eda(train_path: Path) -> pd.DataFrame:
    """EDA / note.md: strip commas on Cost Amount only (SalesAmount as int)."""
    df = pd.read_csv(train_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    cost = df["Cost Amount"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["line_profit"] = df["SalesAmount"] - cost
    return df[["Date", "ItemCode", "line_profit"]]


def compute_sku_weights(transactions: pd.DataFrame) -> pd.DataFrame:
    """Match pipeline notebook: profit, profit_pos (clip), weight."""
    w = (
        transactions.groupby("ItemCode")["line_profit"]
        .sum()
        .rename("profit")
        .reset_index()
    )
    w["profit_pos"] = w["profit"].clip(lower=0)
    total = w["profit_pos"].sum()
    w["weight"] = np.where(total > 0, w["profit_pos"] / total, 0.0)
    w = w.sort_values("profit_pos", ascending=False).reset_index(drop=True)
    w["profit_rank"] = np.arange(1, len(w) + 1)
    pos_total = w["profit_pos"].sum()
    w["cum_profit_pct"] = w["profit_pos"].cumsum() / pos_total if pos_total > 0 else 0.0
    return w


def count_skus_below_cumulative_threshold(weights: pd.DataFrame, threshold: float) -> int:
    """Same as eda.py: (cum_pct < threshold).sum() on positive-profit SKUs."""
    pos = weights.loc[weights["profit_pos"] > 0, "profit_pos"].sort_values(ascending=False)
    if pos.empty:
        return 0
    cum = pos.cumsum() / pos.sum()
    return int((cum < threshold).sum())


def build_top_sku_table(weights: pd.DataFrame, n: int) -> pd.DataFrame:
    top = weights.nlargest(n, "profit_pos").copy()
    top = top.sort_values("profit_pos", ascending=False).reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    pos_total = weights["profit_pos"].sum()
    top["cum_profit_pct"] = top["profit_pos"].cumsum() / pos_total if pos_total > 0 else 0.0
    return top[
        ["rank", "ItemCode", "profit", "profit_pos", "weight", "cum_profit_pct"]
    ]


def build_top_by_cumulative_threshold(
    weights: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Top N SKUs where N = count with cumulative profit share still < threshold (note.md)."""
    n = count_skus_below_cumulative_threshold(weights, threshold)
    if n == 0:
        n = 1
    return build_top_sku_table(weights, n)


def write_artifacts(
    weights: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
    thresholds: tuple[float, float] = (0.50, 0.80),
) -> None:
    """Write weights parquet + threshold CSVs. prefix '' or '_eda'."""
    suffix = prefix
    weights.to_parquet(output_dir / f"sku_weights{suffix}.parquet", index=False)

    for thr, label in zip(thresholds, ("50pct", "80pct")):
        table = build_top_by_cumulative_threshold(weights, thr)
        path = output_dir / f"top{label}_skus{suffix}.csv"
        table.to_csv(path, index=False)
        n = len(table)
        cum = table["cum_profit_pct"].iloc[-1] if n else 0.0
        print(f"  {path.name}: {n} SKUs, cum_profit_pct={cum:.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKU weight artifacts for WRMSSE.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Pipeline parse (notebook / metric-aligned):")
    weights = compute_sku_weights(load_transactions(args.train))
    top99 = build_top_sku_table(weights, 99)
    top653 = build_top_sku_table(weights, 653)
    top99.to_csv(args.output_dir / "top99_skus.csv", index=False)
    top653.to_csv(args.output_dir / "top653_skus.csv", index=False)
    write_artifacts(weights, args.output_dir, prefix="")

    print("\nEDA parse (note.md / eda.py — Cost Amount comma strip only):")
    weights_eda = compute_sku_weights(load_transactions_eda(args.train))
    write_artifacts(weights_eda, args.output_dir, prefix="_eda")

    n50_eda = count_skus_below_cumulative_threshold(weights_eda, 0.50)
    n80_eda = count_skus_below_cumulative_threshold(weights_eda, 0.80)
    n50_pipe = count_skus_below_cumulative_threshold(weights, 0.50)
    n80_pipe = count_skus_below_cumulative_threshold(weights, 0.80)

    print(f"\nTrain: {args.train}")
    print(f"Unique SKUs: {weights['ItemCode'].nunique():,}")
    print(f"SKUs with weight > 0: {(weights['weight'] > 0).sum():,}")
    print(f"Threshold (<50% cum): EDA={n50_eda} SKUs (~note.md top 99), pipeline={n50_pipe} SKUs")
    print(f"Threshold (<80% cum): EDA={n80_eda} SKUs (~note.md top 653), pipeline={n80_pipe} SKUs")
    print(f"Fixed top99 cum_profit_pct: {top99['cum_profit_pct'].iloc[-1]:.4%}")
    print(f"Fixed top653 cum_profit_pct: {top653['cum_profit_pct'].iloc[-1]:.4%}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
