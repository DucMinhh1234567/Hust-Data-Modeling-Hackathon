"""Load SKU artifacts from main/output (run build_sku_artifacts.py first)."""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"

# Fixed rank (pipeline profit)
top99 = pd.read_csv(OUT / "top99_skus.csv")
top653 = pd.read_csv(OUT / "top653_skus.csv")
weights = pd.read_parquet(OUT / "sku_weights.parquet")

# Cumulative ~50% / ~80% profit (note.md / eda.py)
top50pct_eda = pd.read_csv(OUT / "top50pct_skus_eda.csv")
top80pct_eda = pd.read_csv(OUT / "top80pct_skus_eda.csv")
weights_eda = pd.read_parquet(OUT / "sku_weights_eda.parquet")

# Same thresholds, pipeline parse (WRMSSE notebooks)
top50pct = pd.read_csv(OUT / "top50pct_skus.csv")
top80pct = pd.read_csv(OUT / "top80pct_skus.csv")

TOP_99 = top99["ItemCode"].tolist()
TOP_653 = top653["ItemCode"].tolist()
TOP_50PCT_EDA = top50pct_eda["ItemCode"].tolist()
TOP_80PCT_EDA = top80pct_eda["ItemCode"].tolist()
TOP_50PCT = top50pct["ItemCode"].tolist()
TOP_80PCT = top80pct["ItemCode"].tolist()
