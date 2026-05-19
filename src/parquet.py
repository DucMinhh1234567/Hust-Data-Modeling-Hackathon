"""Load SKU artifacts from main/output (run build_sku_artifacts.py first)."""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"

weights = pd.read_parquet(OUT / "sku_weights.parquet")
top50pct = pd.read_csv(OUT / "top50pct_skus.csv")
top80pct = pd.read_csv(OUT / "top80pct_skus.csv")

TOP_50PCT = top50pct["ItemCode"].tolist()
TOP_80PCT = top80pct["ItemCode"].tolist()
