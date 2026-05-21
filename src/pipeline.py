from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(x):
        print(x)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None
    HAS_LIGHTGBM = False

from .config import *

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

LAG_FEATURES = [1, 2, 3, 7, 14, 28, 56, 91, 182, 364, 365]
ROLL_WINDOWS = [7, 14, 28, 56, 90]

def parse_vn_number(s):
    """
    Chuyển cột số/tiền về numeric an toàn.
    Xử lý được:
    - số đã là int/float
    - chuỗi dùng dấu phẩy thập phân: '123559,1'
    - chuỗi có dấu cách
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = (
        s.astype(str)
         .str.strip()
         .str.replace(" ", "", regex=False)
         .str.replace(".", "", regex=False)   # phòng trường hợp có dấu ngăn cách nghìn
         .str.replace(",", ".", regex=False)  # đổi dấu phẩy thập phân
         .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(x, errors="coerce")

def aggregate_daily_y(transactions, value_col="demand_qty"):
    return (
        transactions.groupby(["ItemCode", "Date"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "y"})
    )

def compute_sku_weights(transactions):
    tmp = transactions.copy()
    if "line_profit" not in tmp.columns:
        tmp["line_profit"] = tmp["SalesAmount"] - tmp["CostAmount"]

    w = (
        tmp.groupby("ItemCode")["line_profit"]
           .sum()
           .rename("profit")
           .reset_index()
    )
    w["profit_pos"] = w["profit"].clip(lower=0)
    total = w["profit_pos"].sum()
    w["weight"] = np.where(total > 0, w["profit_pos"] / total, 0)
    return w[["ItemCode", "profit", "profit_pos", "weight"]]

def compute_rmsse_scale(train_daily_sparse, sku_list, train_dates):
    date_to_idx = pd.Series(np.arange(len(train_dates)), index=train_dates)
    grouped = dict(tuple(train_daily_sparse.groupby("ItemCode")))
    result = []

    for sku in sku_list:
        arr = np.zeros(len(train_dates), dtype=np.float32)
        if sku in grouped:
            g = grouped[sku]
            idx = date_to_idx.loc[g["Date"]].to_numpy()
            arr[idx] = g["y"].to_numpy(dtype=np.float32)

        if len(arr) <= 1:
            scale = EPS
        else:
            diffs = np.diff(arr)
            scale = float(np.mean(diffs ** 2))
            if scale <= 0:
                scale = EPS
        result.append((sku, scale))

    return pd.DataFrame(result, columns=["ItemCode", "scale"])

def make_actual_matrix_long(daily_sparse, sku_list, dates, value_col="y"):
    idx = pd.MultiIndex.from_product([sku_list, dates], names=["ItemCode", "Date"])
    s = daily_sparse.set_index(["ItemCode", "Date"])[value_col]
    out = s.reindex(idx, fill_value=0).reset_index()
    out = out.rename(columns={value_col: "actual"})
    return out

def wrmsse_score(actual_long, pred_long, scale_df, weight_df):
    eval_df = (
        actual_long.merge(pred_long, on=["ItemCode", "Date"], how="left")
                   .merge(scale_df, on="ItemCode", how="left")
                   .merge(weight_df[["ItemCode", "weight"]], on="ItemCode", how="left")
    )

    eval_df["pred"] = eval_df["pred"].fillna(0).clip(lower=0)
    eval_df["actual"] = eval_df["actual"].fillna(0)
    eval_df["scale"] = eval_df["scale"].fillna(EPS).clip(lower=EPS)
    eval_df["weight"] = eval_df["weight"].fillna(0)
    eval_df["se"] = (eval_df["actual"] - eval_df["pred"]) ** 2

    sku_eval = (
        eval_df.groupby("ItemCode")
        .agg(
            mse=("se", "mean"),
            scale=("scale", "first"),
            weight=("weight", "first"),
            actual_sum=("actual", "sum"),
            pred_sum=("pred", "sum")
        )
        .reset_index()
    )

    sku_eval["rmsse"] = np.sqrt(sku_eval["mse"] / sku_eval["scale"])
    sku_eval["contribution"] = sku_eval["weight"] * sku_eval["rmsse"]
    score = float(sku_eval["contribution"].sum())

    return score, sku_eval, eval_df

def build_baseline_forecast(
    transactions,
    forecast_dates,
    sku_list,
    windows=BASELINE_WINDOWS,
    window_weights=BASELINE_WEIGHTS,
    sunday_floor=SUNDAY_FACTOR_FLOOR,
    sparse_penalty=True,
    weight_df=None,
    zero_weight_to_zero=True,
    sparse_rules=None,
    recency_rules=None
):
    assert len(windows) == len(window_weights)

    transactions = transactions.copy()
    train_end = transactions["Date"].max()
    train_start = transactions["Date"].min()
    daily_sparse = aggregate_daily_y(transactions)

    all_sku = pd.DataFrame({"ItemCode": sku_list})

    # Positive-day stats: chỉ tính ngày y > 0
    pos_daily = daily_sparse[daily_sparse["y"] > 0].copy()
    sku_stats = (
        pos_daily.groupby("ItemCode")
        .agg(
            total_y_pos=("y", "sum"),
            positive_days=("Date", "nunique"),
            last_positive_date=("Date", "max")
        )
        .reset_index()
    )

    sku_stats = all_sku.merge(sku_stats, on="ItemCode", how="left")
    sku_stats["total_y_pos"] = sku_stats["total_y_pos"].fillna(0)
    sku_stats["positive_days"] = sku_stats["positive_days"].fillna(0)
    sku_stats["last_positive_date"] = pd.to_datetime(sku_stats["last_positive_date"])
    sku_stats["days_since_last_sale"] = (train_end - sku_stats["last_positive_date"]).dt.days
    sku_stats["days_since_last_sale"] = sku_stats["days_since_last_sale"].fillna(9999)

    # Rolling window averages per SKU, chia cho số ngày window để tính cả zero days
    base = all_sku.copy()
    for w in windows:
        start = train_end - pd.Timedelta(days=w-1)
        tmp = (
            daily_sparse[(daily_sparse["Date"] >= start) & (daily_sparse["Date"] <= train_end)]
            .groupby("ItemCode")["y"].sum()
            .rename(f"sum_{w}")
            .reset_index()
        )
        base = base.merge(tmp, on="ItemCode", how="left")
        base[f"sum_{w}"] = base[f"sum_{w}"].fillna(0)
        base[f"mean_{w}"] = base[f"sum_{w}"] / w

    base["base_pred"] = 0.0
    for w, ww in zip(windows, window_weights):
        base["base_pred"] += ww * base[f"mean_{w}"]

    base = base.merge(
        sku_stats[["ItemCode", "positive_days", "days_since_last_sale"]],
        on="ItemCode",
        how="left"
    )

    # Sparse rule
    if sparse_rules is None:
        sparse_rules = [
            (1, 0.02),
            (3, 0.10),
            (7, 0.25),
            (14, 0.55),
            (28, 0.75),
        ]

    # Recency rule: áp dụng từ ngưỡng nhỏ đến lớn để SKU quá lâu không bán không bị ghi đè sai.
    if recency_rules is None:
        recency_rules = [
            (60, 0.70),
            (90, 0.50),
            (180, 0.20),
            (365, 0.00),
        ]

    if sparse_penalty:
        base["sparse_factor"] = 1.0
        for max_days, factor in sparse_rules:
            base.loc[base["positive_days"] <= max_days, "sparse_factor"] = factor

        base["recency_factor"] = 1.0
        # Important: thứ tự phải là 60 -> 90 -> 180 -> 365.
        # Nếu để 365 trước, SKU không bán rất lâu sẽ bị ghi đè lại thành 0.70.
        for min_days, factor in recency_rules:
            base.loc[base["days_since_last_sale"] > min_days, "recency_factor"] = factor

        base["base_pred"] = base["base_pred"] * base["sparse_factor"] * base["recency_factor"]

    if weight_df is not None:
        base = base.merge(weight_df[["ItemCode", "weight"]], on="ItemCode", how="left")
        base["weight"] = base["weight"].fillna(0)
        if zero_weight_to_zero:
            base.loc[base["weight"] <= 0, "base_pred"] = 0.0

    # Global weekday factor
    total_daily = daily_sparse.groupby("Date", as_index=False)["y"].sum()
    all_train_dates = pd.DataFrame({"Date": pd.date_range(train_start, train_end, freq="D")})
    total_daily = all_train_dates.merge(total_daily, on="Date", how="left").fillna({"y": 0})
    total_daily["dow"] = total_daily["Date"].dt.dayofweek

    weekday_mean = total_daily.groupby("dow")["y"].mean()
    global_mean = total_daily["y"].mean()
    weekday_factor = (weekday_mean / max(global_mean, EPS)).clip(lower=0.0, upper=2.0).to_dict()

    # Chủ nhật: dùng floor rất thấp để tránh dự báo Chủ nhật quá cao
    weekday_factor[6] = max(float(weekday_factor.get(6, sunday_floor)), sunday_floor)

    # Cross join SKU x forecast_dates
    fdates = pd.DataFrame({"Date": forecast_dates})
    fdates["dow"] = fdates["Date"].dt.dayofweek
    fdates["weekday_factor"] = fdates["dow"].map(weekday_factor).fillna(1.0)

    pred = base[["ItemCode", "base_pred"]].merge(fdates, how="cross")
    pred["pred"] = pred["base_pred"] * pred["weekday_factor"]
    pred["pred"] = pred["pred"].clip(lower=0)

    return pred[["ItemCode", "Date", "pred"]], base, weekday_factor

def build_baseline_model_forecast(
    transactions,
    forecast_dates,
    sku_list,
    cfg=None,
    weight_df=None,
    zero_weight_to_zero=True,
    use_ensemble=USE_BASELINE_ENSEMBLE,
):
    """
    Wrapper cho baseline:
    - cfg != None: dùng đúng một cấu hình baseline, thường dùng khi grid-search.
    - use_ensemble=True: ensemble fast/stable/long baseline.
    - mặc định: dùng BASELINE_WINDOWS / BASELINE_WEIGHTS đang set ở CONFIG.
    """
    if cfg is not None:
        return build_baseline_forecast(
            transactions,
            forecast_dates,
            sku_list,
            windows=cfg["windows"],
            window_weights=cfg["weights"],
            sunday_floor=cfg.get("sunday_floor", SUNDAY_FACTOR_FLOOR),
            sparse_penalty=cfg.get("sparse_penalty", True),
            weight_df=weight_df,
            zero_weight_to_zero=cfg.get("zero_weight_to_zero", zero_weight_to_zero),
        )

    if not use_ensemble:
        return build_baseline_forecast(
            transactions,
            forecast_dates,
            sku_list,
            windows=BASELINE_WINDOWS,
            window_weights=BASELINE_WEIGHTS,
            sunday_floor=SUNDAY_FACTOR_FLOOR,
            sparse_penalty=True,
            weight_df=weight_df,
            zero_weight_to_zero=zero_weight_to_zero,
        )

    pred_frames = []
    base_tables = []
    weekday_factors = {}
    total_weight = sum(c["ensemble_weight"] for c in BASELINE_ENSEMBLE_CONFIGS)

    for cfg_i in BASELINE_ENSEMBLE_CONFIGS:
        pred_i, base_i, wf_i = build_baseline_forecast(
            transactions,
            forecast_dates,
            sku_list,
            windows=cfg_i["windows"],
            window_weights=cfg_i["weights"],
            sunday_floor=SUNDAY_FACTOR_FLOOR,
            sparse_penalty=True,
            weight_df=weight_df,
            zero_weight_to_zero=zero_weight_to_zero,
        )
        pred_i = pred_i.rename(columns={"pred": f"pred_{cfg_i['name']}"})
        pred_i[f"pred_{cfg_i['name']}"] *= cfg_i["ensemble_weight"] / total_weight
        pred_frames.append(pred_i)
        base_i["baseline_name"] = cfg_i["name"]
        base_tables.append(base_i)
        weekday_factors[cfg_i["name"]] = wf_i

    out = pred_frames[0]
    for p in pred_frames[1:]:
        out = out.merge(p, on=["ItemCode", "Date"], how="left")

    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    out["pred"] = out[pred_cols].sum(axis=1).clip(lower=0)
    base_table = pd.concat(base_tables, ignore_index=True)
    return out[["ItemCode", "Date", "pred"]], base_table, weekday_factors

def apply_special_sku_multipliers(pred_long, multipliers=None, enabled=USE_SPECIAL_SKU_MULTIPLIER):
    """Optional rule thủ công cho top contribution SKU. Mặc định tắt để tránh overfit."""
    if multipliers is None:
        multipliers = SPECIAL_SKU_MULTIPLIER
    out = pred_long.copy()
    if not enabled or len(multipliers) == 0:
        return out
    for sku, mult in multipliers.items():
        out.loc[out["ItemCode"] == sku, "pred"] *= float(mult)
    out["pred"] = out["pred"].clip(lower=0)
    return out

def make_single_validation_split(transactions, horizon=56):
    max_date = transactions["Date"].max()
    valid_start = max_date - pd.Timedelta(days=horizon - 1)
    valid_end = max_date
    train_part = transactions[transactions["Date"] < valid_start].copy()
    valid_part = transactions[(transactions["Date"] >= valid_start) & (transactions["Date"] <= valid_end)].copy()
    valid_dates = pd.date_range(valid_start, valid_end, freq="D")
    return train_part, valid_part, valid_dates, valid_start, valid_end

def build_top_contribution_table(sku_eval, top_n=100):
    out = sku_eval.copy()
    out = out.sort_values("contribution", ascending=False).reset_index(drop=True)
    out["rank_contribution"] = np.arange(1, len(out) + 1)
    out["cum_contribution"] = out["contribution"].cumsum()
    total = out["contribution"].sum()
    out["contribution_share"] = np.where(total > 0, out["contribution"] / total, 0)
    out["cum_contribution_share"] = out["contribution_share"].cumsum()
    return out.head(top_n)

def suggest_special_multipliers(sku_eval, top_n=30, min_actual_sum=1, lower=0.70, upper=1.30):
    """
    Gợi ý multiplier cho top contribution SKU dựa trên actual_sum / pred_sum.
    Chỉ dùng để tham khảo/EDA; không tự apply vào final nếu USE_SPECIAL_SKU_MULTIPLIER=False.
    """
    tmp = sku_eval.sort_values("contribution", ascending=False).head(top_n).copy()
    tmp["raw_ratio_actual_over_pred"] = tmp["actual_sum"] / tmp["pred_sum"].replace(0, np.nan)
    tmp["suggested_multiplier"] = tmp["raw_ratio_actual_over_pred"].replace([np.inf, -np.inf], np.nan)
    tmp["suggested_multiplier"] = tmp["suggested_multiplier"].fillna(1.0).clip(lower, upper)
    tmp.loc[tmp["actual_sum"] < min_actual_sum, "suggested_multiplier"] = 1.0
    return tmp[[
        "ItemCode", "weight", "rmsse", "contribution", "actual_sum", "pred_sum",
        "raw_ratio_actual_over_pred", "suggested_multiplier"
    ]]

def plot_validation_sku(item_code, eval_df, title_prefix="Validation"):
    tmp = eval_df[eval_df["ItemCode"] == item_code].copy().sort_values("Date")
    if tmp.empty:
        print("No data for", item_code)
        return

    info = (
        baseline_sku_eval[baseline_sku_eval["ItemCode"] == item_code]
        [["ItemCode", "weight", "rmsse", "contribution", "actual_sum", "pred_sum"]]
    )
    display(info)

    plt.figure(figsize=(13, 4))
    plt.plot(tmp["Date"], tmp["actual"], marker="o", label="actual", linewidth=2)
    plt.plot(tmp["Date"], tmp["pred"], marker="o", label="pred", linewidth=2)
    plt.title(f"{title_prefix}: {item_code}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def make_rolling_folds(transactions, horizon=VALIDATION_HORIZON, n_folds=N_ROLLING_FOLDS, step_days=ROLLING_STEP_DAYS):
    max_date = transactions["Date"].max()
    folds = []
    for i in range(n_folds):
        valid_end = max_date - pd.Timedelta(days=i * step_days)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train_end = valid_start - pd.Timedelta(days=1)
        folds.append({
            "fold": i + 1,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "valid_dates": pd.date_range(valid_start, valid_end, freq="D")
        })
    return folds

def evaluate_baseline_on_fold(transactions, fold, sku_list, cfg=None):
    train_fold = transactions[transactions["Date"] <= fold["train_end"]].copy()
    valid_fold = transactions[(transactions["Date"] >= fold["valid_start"]) & (transactions["Date"] <= fold["valid_end"])].copy()

    train_daily = aggregate_daily_y(train_fold)
    valid_daily = aggregate_daily_y(valid_fold)
    weight_fold = compute_sku_weights(train_fold)
    train_dates_fold = pd.date_range(train_fold["Date"].min(), train_fold["Date"].max(), freq="D")
    scale_fold = compute_rmsse_scale(train_daily, sku_list, train_dates_fold)
    actual_long = make_actual_matrix_long(valid_daily, sku_list, fold["valid_dates"])

    if cfg is None:
        pred, base_table, wf = build_baseline_model_forecast(
            train_fold,
            fold["valid_dates"],
            sku_list,
            weight_df=weight_fold,
            zero_weight_to_zero=True,
            use_ensemble=USE_BASELINE_ENSEMBLE,
        )
    else:
        pred, base_table, wf = build_baseline_model_forecast(
            train_fold,
            fold["valid_dates"],
            sku_list,
            cfg=cfg,
            weight_df=weight_fold,
            zero_weight_to_zero=cfg.get("zero_weight_to_zero", True),
        )

    pred = apply_special_sku_multipliers(pred)
    score, sku_eval, eval_df = wrmsse_score(actual_long, pred, scale_fold, weight_fold)
    return score, sku_eval, eval_df

def get_top_weight_skus(weight_df, top_n=TOP_N_SKUS):
    return (
        weight_df.sort_values("weight", ascending=False)
                 .head(top_n)["ItemCode"]
                 .tolist()
    )

def add_date_features(frame):
    frame = frame.copy()
    frame["dow"] = frame["Date"].dt.dayofweek
    frame["day"] = frame["Date"].dt.day
    frame["month"] = frame["Date"].dt.month
    frame["weekofyear"] = frame["Date"].dt.isocalendar().week.astype(int)
    frame["is_saturday"] = (frame["dow"] == 5).astype(int)
    frame["is_sunday"] = (frame["dow"] == 6).astype(int)
    frame["is_month_start"] = frame["Date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["Date"].dt.is_month_end.astype(int)
    return frame

def _weighted_baseline_from_roll_cols(frame, windows=BASELINE_WINDOWS, weights=BASELINE_WEIGHTS):
    """Baseline reference dùng trong LightGBM ratio training: weighted rolling mean, past-only."""
    out = np.zeros(len(frame), dtype=np.float64)
    total_w = 0.0
    for w, wt in zip(windows, weights):
        col = f"roll_mean_{w}"
        if col in frame.columns:
            out += float(wt) * frame[col].fillna(0).to_numpy(dtype=np.float64)
            total_w += float(wt)
    if total_w <= 0:
        return np.zeros(len(frame), dtype=np.float64)
    return out / total_w

def _weighted_baseline_from_history(arr, windows=BASELINE_WINDOWS, weights=BASELINE_WEIGHTS):
    """Baseline reference cho một SKU ở future date dựa trên history hiện tại."""
    vals = []
    total = 0.0
    denom = 0.0
    for w, wt in zip(windows, weights):
        recent = arr[-w:] if len(arr) >= w else arr
        mean_val = float(np.mean(recent)) if len(recent) > 0 else 0.0
        total += float(wt) * mean_val
        denom += float(wt)
    return total / denom if denom > 0 else 0.0

def compute_global_weekday_factor_from_daily(daily_sparse, train_start, train_end, sunday_floor=SUNDAY_FACTOR_FLOOR):
    """
    Tính weekday factor giống build_baseline_forecast.

    Dùng lại cho LightGBM ratio mode để baseline_ref ở train và inference
    đều cùng định nghĩa với baseline đã weekday-adjusted.
    """
    total_daily = daily_sparse.groupby("Date", as_index=False)["y"].sum()
    all_train_dates = pd.DataFrame({"Date": pd.date_range(train_start, train_end, freq="D")})
    total_daily = all_train_dates.merge(total_daily, on="Date", how="left").fillna({"y": 0})
    total_daily["dow"] = total_daily["Date"].dt.dayofweek

    weekday_mean = total_daily.groupby("dow")["y"].mean()
    global_mean = total_daily["y"].mean()
    weekday_factor = (weekday_mean / max(global_mean, EPS)).clip(lower=0.0, upper=2.0).to_dict()
    weekday_factor[6] = max(float(weekday_factor.get(6, sunday_floor)), sunday_floor)
    return weekday_factor

def _safe_divide_feature(numer, denom, eps=RATIO_EPS):
    """Tạo ratio feature an toàn, tránh inf/nan."""
    out = numer / (denom + eps)
    return out.replace([np.inf, -np.inf], 0).fillna(0)

def _ewma_last_from_history(arr, span):
    """EWMA feature cho future date: chỉ dùng history trước ngày cần dự đoán."""
    if len(arr) == 0:
        return 0.0
    return float(pd.Series(arr, dtype="float64").ewm(span=span, min_periods=1, adjust=False).mean().iloc[-1])

def make_lgbm_train_frame(
    transactions,
    cutoff_date,
    top_skus,
    weight_df,
    scale_df,
    min_history_days=120,
    target_mode=LGBM_TARGET_MODE,
):
    """
    Tạo training frame cho LightGBM top SKU.

    target_mode:
    - "quantity": model học y trực tiếp như bản cũ.
    - "ratio": model học correction ratio = (actual + eps) / (baseline_ref + eps), clip [0, RATIO_CLIP_MAX].

    Feature lag/rolling dùng shift(1) để không leakage.
    """
    cutoff_date = pd.to_datetime(cutoff_date)
    train_tx = transactions[transactions["Date"] <= cutoff_date].copy()
    daily_sparse = aggregate_daily_y(train_tx)

    train_dates = pd.date_range(train_tx["Date"].min(), cutoff_date, freq="D")
    idx = pd.MultiIndex.from_product([top_skus, train_dates], names=["ItemCode", "Date"])

    g = (
        daily_sparse.set_index(["ItemCode", "Date"])["y"]
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    g = g.sort_values(["ItemCode", "Date"]).reset_index(drop=True)
    g = add_date_features(g)

    # Map SKU -> item_id categorical/int
    item_map = {sku: i for i, sku in enumerate(top_skus)}
    g["item_id"] = g["ItemCode"].map(item_map).astype(int)

    # Lag features
    grp = g.groupby("ItemCode")["y"]
    for lag in LAG_FEATURES:
        g[f"lag_{lag}"] = grp.shift(lag)

    # Rolling features dùng past only: shift(1).rolling()
    shifted = grp.shift(1)
    for w in ROLL_WINDOWS:
        g[f"roll_mean_{w}"] = (
            shifted.groupby(g["ItemCode"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        g[f"roll_sum_{w}"] = (
            shifted.groupby(g["ItemCode"])
            .rolling(w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # EWMA + trend features: dùng shifted y nên không leakage.
    if USE_EWMA_TREND_PROFIT_FEATURES:
        for span in EWMA_SPANS:
            g[f"ewma_{span}"] = (
                shifted.groupby(g["ItemCode"])
                .transform(lambda x: x.ewm(span=span, min_periods=1, adjust=False).mean())
            )

        g["trend_7_28"] = _safe_divide_feature(g["roll_mean_7"] - g["roll_mean_28"], g["roll_mean_28"])
        g["trend_14_90"] = _safe_divide_feature(g["roll_mean_14"] - g["roll_mean_90"], g["roll_mean_90"])
        g["trend_28_90"] = _safe_divide_feature(g["roll_mean_28"] - g["roll_mean_90"], g["roll_mean_90"])

    # P1: baseline_ref phải cùng định nghĩa với baseline dùng lúc inference.
    # Trước đây target ratio học trên raw rolling baseline, nhưng inference lại nhân ratio
    # với baseline đã weekday-adjusted -> dễ double-count weekday effect.
    weekday_factor_dict = compute_global_weekday_factor_from_daily(
        daily_sparse, train_tx["Date"].min(), cutoff_date, sunday_floor=SUNDAY_FACTOR_FLOOR
    )
    g["weekday_factor_for_ratio"] = g["dow"].map(weekday_factor_dict).fillna(1.0)
    g["baseline_ref_raw"] = _weighted_baseline_from_roll_cols(g)
    g["baseline_ref"] = g["baseline_ref_raw"] * g["weekday_factor_for_ratio"]

    # Static SKU stats tính đến cutoff
    pos_daily = daily_sparse[daily_sparse["y"] > 0].copy()
    static = (
        pos_daily.groupby("ItemCode")
        .agg(
            total_y_pos=("y", "sum"),
            positive_days=("Date", "nunique"),
            last_positive_date=("Date", "max"),
            mean_y_when_sold=("y", "mean"),
            max_y_when_sold=("y", "max")
        )
        .reset_index()
    )

    all_top = pd.DataFrame({"ItemCode": top_skus})
    static = all_top.merge(static, on="ItemCode", how="left")
    static["total_y_pos"] = static["total_y_pos"].fillna(0)
    static["positive_days"] = static["positive_days"].fillna(0)
    static["mean_y_when_sold"] = static["mean_y_when_sold"].fillna(0)
    static["max_y_when_sold"] = static["max_y_when_sold"].fillna(0)
    static["last_positive_date"] = pd.to_datetime(static["last_positive_date"])
    static["days_since_last_sale"] = (cutoff_date - static["last_positive_date"]).dt.days.fillna(9999)
    static["avg_y_per_day"] = static["total_y_pos"] / max(len(train_dates), 1)

    static = static.merge(weight_df[["ItemCode", "weight"]], on="ItemCode", how="left")
    static = static.merge(scale_df[["ItemCode", "scale"]], on="ItemCode", how="left")
    static["weight"] = static["weight"].fillna(0)
    static["scale"] = static["scale"].fillna(EPS).clip(lower=EPS)

    # Profit rank static features: giúp model phân biệt nhóm SKU cực kỳ quan trọng trong WRMSSE.
    rank_df = weight_df[["ItemCode", "weight"]].copy()
    rank_df = rank_df.sort_values("weight", ascending=False).reset_index(drop=True)
    rank_df["profit_rank"] = np.arange(1, len(rank_df) + 1)
    rank_df["profit_rank_pct"] = rank_df["profit_rank"] / max(len(rank_df), 1)
    static = static.merge(rank_df[["ItemCode", "profit_rank", "profit_rank_pct"]], on="ItemCode", how="left")
    static["profit_rank"] = static["profit_rank"].fillna(len(rank_df) + 1)
    static["profit_rank_pct"] = static["profit_rank_pct"].fillna(1.0)

    g = g.merge(
        static[[
            "ItemCode", "total_y_pos", "positive_days", "mean_y_when_sold",
            "max_y_when_sold", "days_since_last_sale", "avg_y_per_day",
            "weight", "scale", "profit_rank", "profit_rank_pct"
        ]],
        on="ItemCode",
        how="left"
    )

    # Bỏ giai đoạn đầu chưa đủ lag
    min_date = train_dates.min() + pd.Timedelta(days=min_history_days)
    g = g[g["Date"] >= min_date].copy()

    # Fill lag missing còn lại
    feature_cols = (
        ["item_id", "dow", "day", "month", "weekofyear", "is_saturday", "is_sunday",
         "is_month_start", "is_month_end", "baseline_ref", "baseline_ref_raw", "weekday_factor_for_ratio"]
        + [f"lag_{l}" for l in LAG_FEATURES]
        + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
        + [f"roll_sum_{w}" for w in ROLL_WINDOWS]
        + ["total_y_pos", "positive_days", "mean_y_when_sold", "max_y_when_sold",
           "days_since_last_sale", "avg_y_per_day", "weight", "scale"]
    )

    if USE_EWMA_TREND_PROFIT_FEATURES:
        feature_cols += [f"ewma_{span}" for span in EWMA_SPANS]
        feature_cols += ["trend_7_28", "trend_14_90", "trend_28_90", "profit_rank", "profit_rank_pct"]

    for c in feature_cols:
        g[c] = g[c].fillna(0)

    # Target cho LightGBM
    if target_mode == "ratio":
        g["target_lgbm"] = ((g["y"] + RATIO_EPS) / (g["baseline_ref"].clip(lower=0) + RATIO_EPS)).clip(0, RATIO_CLIP_MAX)
    elif target_mode == "quantity":
        g["target_lgbm"] = g["y"].clip(lower=0)
    else:
        raise ValueError("target_mode must be 'quantity' or 'ratio'")

    # Sample weight gần với WRMSSE: weight / scale
    g["sample_weight"] = g["weight"] / g["scale"].clip(lower=EPS)
    mean_sw = g["sample_weight"].mean()
    if mean_sw > 0:
        g["sample_weight"] = g["sample_weight"] / mean_sw
    g["sample_weight"] = g["sample_weight"].clip(lower=0.01, upper=100)

    return g, feature_cols, item_map, static

def train_lgbm_model(train_frame, feature_cols, target_mode=LGBM_TARGET_MODE):
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM chưa được cài. Hãy chạy: pip install lightgbm")

    if target_mode == "ratio":
        # Ratio đã bị clip [0, RATIO_CLIP_MAX], regression thường ổn định hơn Tweedie cho correction factor.
        objective_params = {
            "objective": "regression",
            "metric": "rmse",
        }
    else:
        objective_params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.2,
            "metric": "rmse",
        }

    params = {
        **objective_params,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "force_col_wise": True,
    }

    print("Training LightGBM target_mode:", target_mode, "| objective:", params["objective"])

    X_full = train_frame[feature_cols]
    y_full = train_frame["target_lgbm"].clip(lower=0) if "target_lgbm" in train_frame.columns else train_frame["y"].clip(lower=0)
    sw_full = train_frame["sample_weight"]

    # Early stopping theo thời gian: dùng 15% ngày cuối để tìm best_iteration,
    # sau đó train lại trên full train_frame bằng best_iteration để không mất dữ liệu mới nhất.
    if USE_LGBM_EARLY_STOPPING and "Date" in train_frame.columns:
        unique_dates = np.sort(pd.to_datetime(train_frame["Date"].unique()))
        if len(unique_dates) >= 20:
            cutoff_idx = int(len(unique_dates) * (1.0 - LGBM_ES_VALID_FRAC))
            cutoff_idx = min(max(cutoff_idx, 1), len(unique_dates) - 1)
            cutoff_es = pd.to_datetime(unique_dates[cutoff_idx])
            tr_mask = pd.to_datetime(train_frame["Date"]) < cutoff_es
            va_mask = ~tr_mask

            print(
                "Early stopping split:",
                "cutoff =", cutoff_es.date(),
                "| train rows =", int(tr_mask.sum()),
                "| valid rows =", int(va_mask.sum())
            )

            dtrain_es = lgb.Dataset(
                train_frame.loc[tr_mask, feature_cols],
                label=y_full.loc[tr_mask],
                weight=sw_full.loc[tr_mask],
                categorical_feature=["item_id"],
                free_raw_data=False,
            )
            dvalid_es = lgb.Dataset(
                train_frame.loc[va_mask, feature_cols],
                label=y_full.loc[va_mask],
                weight=sw_full.loc[va_mask],
                categorical_feature=["item_id"],
                free_raw_data=False,
            )

            model_es = lgb.train(
                params,
                dtrain_es,
                num_boost_round=LGBM_NUM_BOOST_ROUND,
                valid_sets=[dvalid_es],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(200),
                ],
            )
            best_iter = int(model_es.best_iteration or LGBM_FIXED_NUM_BOOST_ROUND)
            print("Best iteration:", best_iter)

            if not RETRAIN_FULL_AFTER_EARLY_STOPPING:
                return model_es

            dtrain_full = lgb.Dataset(
                X_full,
                label=y_full,
                weight=sw_full,
                categorical_feature=["item_id"],
                free_raw_data=False,
            )
            model = lgb.train(
                params,
                dtrain_full,
                num_boost_round=best_iter,
            )
            return model
        else:
            print("Skip early stopping: not enough unique dates.")

    dtrain = lgb.Dataset(
        X_full,
        label=y_full,
        weight=sw_full,
        categorical_feature=["item_id"],
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_FIXED_NUM_BOOST_ROUND
    )

    return model

def build_history_dict(transactions, cutoff_date, top_skus):
    cutoff_date = pd.to_datetime(cutoff_date)
    train_tx = transactions[transactions["Date"] <= cutoff_date].copy()
    daily_sparse = aggregate_daily_y(train_tx)
    train_dates = pd.date_range(train_tx["Date"].min(), cutoff_date, freq="D")

    idx = pd.MultiIndex.from_product([top_skus, train_dates], names=["ItemCode", "Date"])
    hist_df = (
        daily_sparse.set_index(["ItemCode", "Date"])["y"]
        .reindex(idx, fill_value=0)
        .reset_index()
        .sort_values(["ItemCode", "Date"])
    )

    history = {}
    for sku, g in hist_df.groupby("ItemCode"):
        history[sku] = g["y"].to_numpy(dtype=np.float32).tolist()

    return history, train_dates

def get_days_since_last_positive(arr):
    for i, v in enumerate(reversed(arr), start=1):
        if v > 0:
            return i - 1
    return 9999

def make_future_feature_frame_for_date(date, top_skus, history, item_map, static_lookup, feature_cols, weekday_factor_dict=None):
    rows = []
    date = pd.to_datetime(date)
    if weekday_factor_dict is None:
        weekday_factor_for_date = 1.0
    else:
        weekday_factor_for_date = float(weekday_factor_dict.get(date.dayofweek, 1.0))

    for sku in top_skus:
        arr = history[sku]
        row = {
            "ItemCode": sku,
            "Date": date,
            "item_id": item_map[sku],
            "dow": date.dayofweek,
            "day": date.day,
            "month": date.month,
            "weekofyear": int(date.isocalendar().week),
            "is_saturday": int(date.dayofweek == 5),
            "is_sunday": int(date.dayofweek == 6),
            "is_month_start": int(date.is_month_start),
            "is_month_end": int(date.is_month_end),
        }

        for lag in LAG_FEATURES:
            row[f"lag_{lag}"] = arr[-lag] if len(arr) >= lag else 0.0

        for w in ROLL_WINDOWS:
            vals = arr[-w:] if len(arr) >= w else arr
            row[f"roll_mean_{w}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            row[f"roll_sum_{w}"] = float(np.sum(vals)) if len(vals) > 0 else 0.0

        if USE_EWMA_TREND_PROFIT_FEATURES:
            for span in EWMA_SPANS:
                row[f"ewma_{span}"] = _ewma_last_from_history(arr, span)
            row["trend_7_28"] = (row["roll_mean_7"] - row["roll_mean_28"]) / (row["roll_mean_28"] + RATIO_EPS)
            row["trend_14_90"] = (row["roll_mean_14"] - row["roll_mean_90"]) / (row["roll_mean_90"] + RATIO_EPS)
            row["trend_28_90"] = (row["roll_mean_28"] - row["roll_mean_90"]) / (row["roll_mean_90"] + RATIO_EPS)

        # P1: future baseline_ref feature cũng dùng weekday-adjusted definition
        # để khớp với baseline_ref trong training ratio mode.
        raw_baseline_ref = _weighted_baseline_from_history(arr)
        row["baseline_ref_raw"] = raw_baseline_ref
        row["weekday_factor_for_ratio"] = weekday_factor_for_date
        row["baseline_ref"] = raw_baseline_ref * weekday_factor_for_date

        static_row = static_lookup.get(sku, {})
        for c in ["total_y_pos", "positive_days", "mean_y_when_sold", "max_y_when_sold",
                  "avg_y_per_day", "weight", "scale", "profit_rank", "profit_rank_pct"]:
            row[c] = static_row.get(c, 0.0)

        # Với future, days_since_last_sale cần update theo history bao gồm prediction trước đó
        row["days_since_last_sale"] = get_days_since_last_positive(arr)

        rows.append(row)

    out = pd.DataFrame(rows)
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0)

    return out

def _make_baseline_lookup_for_lgbm(baseline_pred):
    """Tạo lookup baseline exact theo ItemCode-Date để convert ratio -> quantity."""
    if baseline_pred is None:
        return None
    tmp = baseline_pred[["ItemCode", "Date", "pred"]].copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    return tmp.set_index(["ItemCode", "Date"])["pred"].to_dict()

def predict_lgbm_recursive(
    model,
    transactions,
    cutoff_date,
    forecast_dates,
    top_skus,
    item_map,
    static_table,
    feature_cols,
    target_mode=LGBM_TARGET_MODE,
    baseline_pred=None,
):
    history, train_dates = build_history_dict(transactions, cutoff_date, top_skus)

    static_lookup = (
        static_table.set_index("ItemCode")
        .to_dict(orient="index")
    )
    baseline_lookup = _make_baseline_lookup_for_lgbm(baseline_pred)

    # P1: dùng cùng weekday factor với build_baseline_forecast cho feature baseline_ref tương lai.
    train_tx_for_wf = transactions[transactions["Date"] <= pd.to_datetime(cutoff_date)].copy()
    daily_sparse_for_wf = aggregate_daily_y(train_tx_for_wf)
    weekday_factor_dict = compute_global_weekday_factor_from_daily(
        daily_sparse_for_wf,
        train_tx_for_wf["Date"].min(),
        pd.to_datetime(cutoff_date),
        sunday_floor=SUNDAY_FACTOR_FLOOR,
    )

    preds = []

    for date in forecast_dates:
        date = pd.to_datetime(date)
        feat = make_future_feature_frame_for_date(
            date,
            top_skus,
            history,
            item_map,
            static_lookup,
            feature_cols,
            weekday_factor_dict=weekday_factor_dict
        )

        raw_pred = model.predict(feat[feature_cols])

        if target_mode == "ratio":
            ratio_pred = np.clip(raw_pred, 0, RATIO_CLIP_MAX)
            if baseline_lookup is not None:
                base_vals = np.array([
                    baseline_lookup.get((sku, date), np.nan)
                    for sku in top_skus
                ], dtype=np.float64)
                # fallback nếu thiếu baseline exact
                fallback = feat["baseline_ref"].to_numpy(dtype=np.float64)
                base_vals = np.where(np.isfinite(base_vals), base_vals, fallback)
            else:
                base_vals = feat["baseline_ref"].to_numpy(dtype=np.float64)
            yhat = np.clip(base_vals * ratio_pred, 0, None)
            day_pred = pd.DataFrame({
                "ItemCode": top_skus,
                "Date": date,
                "pred": yhat,
                "ratio_pred": ratio_pred,
                "baseline_ref_for_ratio": base_vals,
            })
        else:
            yhat = np.clip(raw_pred, 0, None)
            day_pred = pd.DataFrame({
                "ItemCode": top_skus,
                "Date": date,
                "pred": yhat
            })
        preds.append(day_pred)

        # Append quantity prediction vào history để forecast recursive.
        # Với ratio mode, append yhat sau khi đã quy về Quantity.
        for sku, pred_val in zip(top_skus, yhat):
            history[sku].append(float(pred_val))

    return pd.concat(preds, ignore_index=True)

def make_alpha_table(weight_df, top_n=TOP_N_SKUS):
    """Tạo alpha gốc theo profit rank."""
    rank_df = weight_df.sort_values("weight", ascending=False).reset_index(drop=True).copy()
    rank_df["profit_rank"] = np.arange(1, len(rank_df) + 1)

    rank_df["rank_alpha"] = RANK_ALPHA_CONFIG["tail"]
    rank_df.loc[rank_df["profit_rank"] <= 2000, "rank_alpha"] = RANK_ALPHA_CONFIG["rank_1404_2000"]
    rank_df.loc[rank_df["profit_rank"] <= 1403, "rank_alpha"] = RANK_ALPHA_CONFIG["rank_214_1403"]
    rank_df.loc[rank_df["profit_rank"] <= 213, "rank_alpha"] = RANK_ALPHA_CONFIG["top_213"]
    rank_df.loc[rank_df["profit_rank"] > top_n, "rank_alpha"] = 0.0

    return rank_df[["ItemCode", "profit_rank", "weight", "rank_alpha"]]

def horizon_alpha_factor(h):
    if h <= 7:
        return HORIZON_ALPHA_CONFIG["h_1_7"]
    elif h <= 14:
        return HORIZON_ALPHA_CONFIG["h_8_14"]
    elif h <= 28:
        return HORIZON_ALPHA_CONFIG["h_15_28"]
    else:
        return HORIZON_ALPHA_CONFIG["h_29_56"]

def blend_baseline_lgbm(
    baseline_pred,
    lgbm_pred_top,
    top_skus,
    alpha=0.10,
    weight_df=None,
    forecast_start_date=None,
    use_rank_horizon_alpha=USE_RANK_HORIZON_ALPHA,
):
    """
    Blend baseline + LGBM.

    Nếu use_rank_horizon_alpha=False:
        alpha là alpha cố định.
    Nếu use_rank_horizon_alpha=True:
        alpha là alpha_scale, alpha thực tế = scale × rank_alpha × horizon_factor.
    """
    out = baseline_pred.copy()
    key_cols = ["ItemCode", "Date"]

    lgb = lgbm_pred_top.copy().rename(columns={"pred": "lgbm_pred"})
    out = out.merge(lgb[key_cols + ["lgbm_pred"]], on=key_cols, how="left")

    if use_rank_horizon_alpha and weight_df is not None:
        alpha_table = make_alpha_table(weight_df, top_n=len(top_skus))
        out = out.merge(alpha_table[["ItemCode", "profit_rank", "rank_alpha"]], on="ItemCode", how="left")
        out["rank_alpha"] = out["rank_alpha"].fillna(0.0)

        if forecast_start_date is None:
            forecast_start_date = out["Date"].min()
        forecast_start_date = pd.to_datetime(forecast_start_date)
        out["horizon"] = (out["Date"] - forecast_start_date).dt.days + 1
        out["horizon_factor"] = out["horizon"].apply(horizon_alpha_factor)
        out["alpha_final"] = (float(alpha) * out["rank_alpha"] * out["horizon_factor"]).clip(0, 0.40)
    else:
        out["alpha_final"] = float(alpha)

    mask = out["ItemCode"].isin(top_skus) & out["lgbm_pred"].notna() & (out["alpha_final"] > 0)
    out.loc[mask, "pred"] = (
        out.loc[mask, "alpha_final"] * out.loc[mask, "lgbm_pred"] +
        (1 - out.loc[mask, "alpha_final"]) * out.loc[mask, "pred"]
    )
    out["pred"] = out["pred"].clip(lower=0)
    return out[key_cols + ["pred"]]

def switch_predictions_for_skus(base_pred, replacement_pred, replace_skus):
    """Replace prediction values for selected SKUs using another prediction DataFrame."""
    replace_skus = list(dict.fromkeys([str(x) for x in replace_skus]))
    if len(replace_skus) == 0:
        return base_pred.copy()

    out = base_pred.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    rep = replacement_pred[replacement_pred["ItemCode"].isin(replace_skus)][["ItemCode", "Date", "pred"]].copy()
    rep["Date"] = pd.to_datetime(rep["Date"])
    rep = rep.rename(columns={"pred": "pred_replacement"})

    out = out.merge(rep, on=["ItemCode", "Date"], how="left")
    mask = out["pred_replacement"].notna() & out["ItemCode"].isin(replace_skus)
    out.loc[mask, "pred"] = out.loc[mask, "pred_replacement"].clip(lower=0)
    out = out.drop(columns=["pred_replacement"])
    return out

def compare_sku_contribution(main_sku_eval, segment_sku_eval, candidate_skus, fold_id=None):
    """
    So sánh contribution_i = weight_i * RMSSE_i giữa Model A và Model B.
    improvement > 0 nghĩa là segment model tốt hơn.
    """
    cand = set(candidate_skus)
    a = main_sku_eval[main_sku_eval["ItemCode"].isin(cand)][[
        "ItemCode", "weight", "rmsse", "contribution", "actual_sum", "pred_sum"
    ]].rename(columns={
        "rmsse": "rmsse_main",
        "contribution": "contribution_main",
        "pred_sum": "pred_sum_main",
    })
    b = segment_sku_eval[segment_sku_eval["ItemCode"].isin(cand)][[
        "ItemCode", "rmsse", "contribution", "pred_sum"
    ]].rename(columns={
        "rmsse": "rmsse_segment",
        "contribution": "contribution_segment",
        "pred_sum": "pred_sum_segment",
    })
    comp = a.merge(b, on="ItemCode", how="left")
    comp["contribution_segment"] = comp["contribution_segment"].fillna(comp["contribution_main"])
    comp["rmsse_segment"] = comp["rmsse_segment"].fillna(comp["rmsse_main"])
    comp["pred_sum_segment"] = comp["pred_sum_segment"].fillna(comp["pred_sum_main"])
    comp["improvement"] = comp["contribution_main"] - comp["contribution_segment"]
    comp["is_better"] = comp["improvement"] > SEGMENT_SELECT_MIN_MEAN_IMPROVEMENT
    if fold_id is not None:
        comp["fold"] = fold_id
    return comp.sort_values("improvement", ascending=False)

def select_segment_skus(segment_compare_df):
    """Chọn SKU để replace ở final dựa trên rolling folds, tránh auto replace toàn bộ top50."""
    if segment_compare_df is None or len(segment_compare_df) == 0:
        return [], pd.DataFrame()

    summary = (
        segment_compare_df.groupby("ItemCode")
        .agg(
            weight=("weight", "mean"),
            mean_improvement=("improvement", "mean"),
            sum_improvement=("improvement", "sum"),
            better_folds=("is_better", "sum"),
            n_folds=("fold", "nunique"),
            mean_main_contribution=("contribution_main", "mean"),
            mean_segment_contribution=("contribution_segment", "mean"),
            mean_actual_sum=("actual_sum", "mean"),
            mean_pred_sum_main=("pred_sum_main", "mean"),
            mean_pred_sum_segment=("pred_sum_segment", "mean"),
        )
        .reset_index()
    )
    summary["better_folds"] = summary["better_folds"].astype(int)
    summary = summary.sort_values("mean_improvement", ascending=False)

    selected_df = summary[
        (summary["better_folds"] >= SEGMENT_SELECT_MIN_BETTER_FOLDS)
        & (summary["mean_improvement"] > SEGMENT_SELECT_MIN_MEAN_IMPROVEMENT)
    ].copy()
    selected_df = selected_df.head(SEGMENT_MAX_SELECTED_SKUS)
    selected = selected_df["ItemCode"].tolist()
    return selected, summary

def train_predict_lgbm_blend_for_skus(
    transactions,
    cutoff_date,
    forecast_dates,
    sku_subset,
    sku_list_for_metric,
    weight_df,
    scale_df,
    baseline_pred,
    alpha,
):
    """Train một LightGBM trên sku_subset rồi blend vào baseline."""
    train_frame, feature_cols, item_map, static_table = make_lgbm_train_frame(
        transactions,
        cutoff_date,
        sku_subset,
        weight_df,
        scale_df,
        min_history_days=120,
        target_mode=LGBM_TARGET_MODE,
    )
    model = train_lgbm_model(train_frame, feature_cols, target_mode=LGBM_TARGET_MODE)
    lgbm_pred_top = predict_lgbm_recursive(
        model,
        transactions,
        cutoff_date,
        forecast_dates,
        sku_subset,
        item_map,
        static_table,
        feature_cols,
        target_mode=LGBM_TARGET_MODE,
        baseline_pred=baseline_pred,
    )
    blend_pred = blend_baseline_lgbm(
        baseline_pred,
        lgbm_pred_top,
        sku_subset,
        alpha=alpha,
        weight_df=weight_df,
        forecast_start_date=pd.to_datetime(forecast_dates).min(),
        use_rank_horizon_alpha=USE_RANK_HORIZON_ALPHA,
    )
    blend_pred = apply_special_sku_multipliers(blend_pred)
    return blend_pred

def apply_sku_correction_factors(pred_long, factor_dict):
    """Nhân dự báo của từng SKU theo factor đã học từ rolling validation."""
    if factor_dict is None or len(factor_dict) == 0:
        return pred_long.copy()

    out = pred_long.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    factor_map = {str(k): float(v) for k, v in factor_dict.items()}
    out["_corr_factor"] = out["ItemCode"].map(factor_map).fillna(1.0)
    out["pred"] = (out["pred"] * out["_corr_factor"]).clip(lower=0)
    out = out.drop(columns=["_corr_factor"])
    return out

def tune_sku_correction_factors_on_fold(
    actual_long,
    pred_long,
    scale_df,
    weight_df,
    candidate_skus,
    factor_grid=CORR_FACTOR_GRID,
    fold_id=None,
):
    """
    Với mỗi SKU candidate, thử nhiều correction factor và chọn factor làm contribution nhỏ nhất.
    improvement > 0 nghĩa là factor tốt hơn dự báo main hiện tại.
    """
    candidate_skus = list(dict.fromkeys([str(x) for x in candidate_skus]))
    if len(candidate_skus) == 0:
        return pd.DataFrame()

    eval_df = (
        actual_long[actual_long["ItemCode"].isin(candidate_skus)]
        .merge(pred_long[["ItemCode", "Date", "pred"]], on=["ItemCode", "Date"], how="left")
        .merge(scale_df[["ItemCode", "scale"]], on="ItemCode", how="left")
        .merge(weight_df[["ItemCode", "weight"]], on="ItemCode", how="left")
    )
    eval_df["actual"] = eval_df["actual"].fillna(0)
    eval_df["pred"] = eval_df["pred"].fillna(0).clip(lower=0)
    eval_df["scale"] = eval_df["scale"].fillna(EPS).clip(lower=EPS)
    eval_df["weight"] = eval_df["weight"].fillna(0)

    rows = []
    for sku, g in eval_df.groupby("ItemCode", sort=False):
        actual = g["actual"].to_numpy(dtype=np.float64)
        pred = g["pred"].to_numpy(dtype=np.float64)
        scale = float(g["scale"].iloc[0])
        weight = float(g["weight"].iloc[0])
        actual_sum = float(actual.sum())
        pred_sum = float(pred.sum())

        if pred_sum < CORR_MIN_PRED_SUM or weight <= 0:
            continue
        if CORR_ONLY_OVERPRED and not (pred_sum > actual_sum):
            continue

        base_mse = float(np.mean((actual - pred) ** 2))
        base_rmsse = float(np.sqrt(base_mse / max(scale, EPS)))
        base_contribution = weight * base_rmsse

        best_factor = 1.0
        best_rmsse = base_rmsse
        best_contribution = base_contribution
        for factor in factor_grid:
            factor = float(factor)
            adj_pred = pred * factor
            mse = float(np.mean((actual - adj_pred) ** 2))
            rmsse = float(np.sqrt(mse / max(scale, EPS)))
            contribution = weight * rmsse
            if contribution < best_contribution:
                best_factor = factor
                best_rmsse = rmsse
                best_contribution = contribution

        rows.append({
            "ItemCode": sku,
            "weight": weight,
            "actual_sum": actual_sum,
            "pred_sum_main": pred_sum,
            "base_rmsse": base_rmsse,
            "base_contribution": base_contribution,
            "best_factor": best_factor,
            "best_rmsse": best_rmsse,
            "best_contribution": best_contribution,
            "improvement": base_contribution - best_contribution,
            "is_better": (base_contribution - best_contribution) > CORR_SELECT_MIN_MEAN_IMPROVEMENT,
            "fold": fold_id,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("improvement", ascending=False).reset_index(drop=True)

def select_sku_correction_factors(
    correction_df,
    min_better_folds=None,
    min_mean_improvement=None,
    max_selected_skus=None,
    factor_shrink=None,
    clip_low=None,
    clip_high=None,
    agg_method=None,
):
    """Tổng hợp best_factor qua rolling folds và chọn SKU/factor cho final submission.

    Bản sửa: không dùng median_factor trực tiếp nữa, vì median có thể chọn sai hướng
    khi 3 folds mâu thuẫn. Ví dụ factors [0.60, 1.20, 1.20] nhưng improvement
    lớn nhất nằm ở fold factor 0.60 thì median=1.20 sẽ sai. Vì vậy mặc định dùng
    improvement-weighted factor rồi shrink về 1.0 để giảm overfit.

    Có thể truyền tham số để tạo nhiều submission variants mà không cần train lại:
    - factor_shrink=0.60, max_selected_skus=10
    - factor_shrink=0.50, max_selected_skus=15, min_mean_improvement=0.00005
    """
    if correction_df is None or len(correction_df) == 0:
        return {}, pd.DataFrame()

    min_better_folds = CORR_SELECT_MIN_BETTER_FOLDS if min_better_folds is None else int(min_better_folds)
    min_mean_improvement = CORR_SELECT_MIN_MEAN_IMPROVEMENT if min_mean_improvement is None else float(min_mean_improvement)
    max_selected_skus = CORR_MAX_SELECTED_SKUS if max_selected_skus is None else int(max_selected_skus)
    factor_shrink = CORR_FACTOR_SHRINK if factor_shrink is None else float(factor_shrink)
    clip_low = CORR_FACTOR_CLIP_LOW if clip_low is None else float(clip_low)
    clip_high = CORR_FACTOR_CLIP_HIGH if clip_high is None else float(clip_high)
    agg_method = CORR_FACTOR_AGG_METHOD if agg_method is None else str(agg_method)

    def _weighted_factor(g):
        factors = g["best_factor"].astype(float).to_numpy()
        weights = g["improvement"].clip(lower=0).astype(float).to_numpy()
        if weights.sum() <= EPS:
            return float(np.median(factors))
        return float(np.average(factors, weights=weights))

    weighted_factor_df = (
        correction_df.groupby("ItemCode")
        .apply(_weighted_factor)
        .rename("weighted_factor")
        .reset_index()
    )

    summary = (
        correction_df.groupby("ItemCode")
        .agg(
            weight=("weight", "mean"),
            mean_improvement=("improvement", "mean"),
            sum_improvement=("improvement", "sum"),
            better_folds=("is_better", "sum"),
            n_folds=("fold", "nunique"),
            median_factor=("best_factor", "median"),
            mean_factor=("best_factor", "mean"),
            mean_actual_sum=("actual_sum", "mean"),
            mean_pred_sum_main=("pred_sum_main", "mean"),
            mean_base_contribution=("base_contribution", "mean"),
            mean_best_contribution=("best_contribution", "mean"),
        )
        .reset_index()
        .merge(weighted_factor_df, on="ItemCode", how="left")
    )

    summary["better_folds"] = summary["better_folds"].astype(int)
    summary["overpred_ratio"] = summary["mean_pred_sum_main"] / (summary["mean_actual_sum"] + EPS)

    if agg_method == "median":
        summary["raw_final_factor"] = summary["median_factor"].astype(float)
    elif agg_method == "mean":
        summary["raw_final_factor"] = summary["mean_factor"].astype(float)
    else:
        summary["raw_final_factor"] = summary["weighted_factor"].astype(float)

    # Shrink về 1.0 để tránh chọn factor quá cực đoan từ local validation.
    summary["final_factor"] = 1.0 + factor_shrink * (summary["raw_final_factor"] - 1.0)
    summary["final_factor"] = summary["final_factor"].clip(clip_low, clip_high)

    summary["variant_min_better_folds"] = min_better_folds
    summary["variant_min_mean_improvement"] = min_mean_improvement
    summary["variant_max_selected_skus"] = max_selected_skus
    summary["variant_factor_shrink"] = factor_shrink
    summary["variant_clip_low"] = clip_low
    summary["variant_clip_high"] = clip_high
    summary["variant_agg_method"] = agg_method

    summary = summary.sort_values("mean_improvement", ascending=False)

    selected_df = summary[
        (summary["better_folds"] >= min_better_folds)
        & (summary["mean_improvement"] > min_mean_improvement)
    ].copy()
    selected_df = selected_df.head(max_selected_skus)

    factor_dict = {
        row["ItemCode"]: float(row["final_factor"])
        for _, row in selected_df.iterrows()
        if abs(float(row["final_factor"]) - 1.0) > 1e-9
    }

    # Manual factors sẽ override factor học được nếu bạn muốn ép tay.
    if MANUAL_SKU_CORRECTION_FACTORS:
        factor_dict.update({str(k): float(v) for k, v in MANUAL_SKU_CORRECTION_FACTORS.items()})

    return factor_dict, summary

def evaluate_lgbm_blend_on_fold(transactions, fold, sku_list, alpha=0.5, top_n=TOP_N_SKUS):
    train_fold = transactions[transactions["Date"] <= fold["train_end"]].copy()
    valid_fold = transactions[(transactions["Date"] >= fold["valid_start"]) & (transactions["Date"] <= fold["valid_end"])].copy()

    train_daily = aggregate_daily_y(train_fold)
    valid_daily = aggregate_daily_y(valid_fold)
    weight_fold = compute_sku_weights(train_fold)
    train_dates_fold = pd.date_range(train_fold["Date"].min(), train_fold["Date"].max(), freq="D")
    scale_fold = compute_rmsse_scale(train_daily, sku_list, train_dates_fold)
    actual_long = make_actual_matrix_long(valid_daily, sku_list, fold["valid_dates"])

    baseline_pred, base_table, wf = build_baseline_model_forecast(
        train_fold,
        fold["valid_dates"],
        sku_list,
        weight_df=weight_fold,
        zero_weight_to_zero=True,
    )
    baseline_score, _, _ = wrmsse_score(actual_long, baseline_pred, scale_fold, weight_fold)

    # Model A: current model trên TOP_N_SKUS
    print("\n=== Fold", fold["fold"], "| Model A main top", top_n, "===")
    top_skus_main = get_top_weight_skus(weight_fold, top_n=top_n)
    main_blend_pred = train_predict_lgbm_blend_for_skus(
        train_fold,
        train_fold["Date"].max(),
        fold["valid_dates"],
        top_skus_main,
        sku_list,
        weight_fold,
        scale_fold,
        baseline_pred,
        alpha,
    )
    main_score, sku_eval_main, eval_df_main = wrmsse_score(actual_long, main_blend_pred, scale_fold, weight_fold)

    correction_table = pd.DataFrame()
    correction_score = np.nan
    correction_factor_dict_fold = {}
    sku_eval_correction = pd.DataFrame()
    eval_df_correction = pd.DataFrame()

    if USE_SKU_CORRECTION_FACTOR:
        print("\n=== Fold", fold["fold"], "| SKU correction factor tuning top", CORR_TOP_N_CANDIDATES, "===")
        correction_candidates = get_top_weight_skus(weight_fold, top_n=CORR_TOP_N_CANDIDATES)
        correction_table = tune_sku_correction_factors_on_fold(
            actual_long,
            main_blend_pred,
            scale_fold,
            weight_fold,
            candidate_skus=correction_candidates,
            factor_grid=CORR_FACTOR_GRID,
            fold_id=fold["fold"],
        )
        if len(correction_table) > 0:
            fold_selected = correction_table[
                correction_table["improvement"] > CORR_SELECT_MIN_MEAN_IMPROVEMENT
            ].head(CORR_MAX_SELECTED_SKUS)
            correction_factor_dict_fold = dict(zip(fold_selected["ItemCode"], fold_selected["best_factor"]))
            correction_pred = apply_sku_correction_factors(main_blend_pred, correction_factor_dict_fold)
            correction_score, sku_eval_correction, eval_df_correction = wrmsse_score(
                actual_long, correction_pred, scale_fold, weight_fold
            )
            print("Fold", fold["fold"], "correction_score=", round(correction_score, 6),
                  "| n_factors=", len(correction_factor_dict_fold))
            display(correction_table.head(CORR_SHOW_TOP_N))
        else:
            print("No valid correction candidates in this fold.")

    res = {
        "baseline_score": baseline_score,
        "blend_score": main_score,
        "sku_eval": sku_eval_main,
        "eval_df": eval_df_main,
        "segment_score": np.nan,
        "oracle_switch_score": np.nan,
        "segment_compare": pd.DataFrame(),
        "oracle_switch_skus": [],
        "correction_score": correction_score,
        "correction_factor_dict_fold": correction_factor_dict_fold,
        "sku_correction_table": correction_table,
        "correction_sku_eval": sku_eval_correction,
        "correction_eval_df": eval_df_correction,
    }

    if USE_SEGMENT_TOP_MODEL:
        # Model B: specialized model train trên top100/topN, chỉ xét replace top50/topN.
        print("\n=== Fold", fold["fold"], "| Model B segment train top", SEGMENT_TOP_N_TRAIN, "===")
        top_skus_segment_train = get_top_weight_skus(weight_fold, top_n=SEGMENT_TOP_N_TRAIN)
        top_skus_segment_replace = get_top_weight_skus(weight_fold, top_n=SEGMENT_TOP_N_REPLACE)

        segment_blend_pred = train_predict_lgbm_blend_for_skus(
            train_fold,
            train_fold["Date"].max(),
            fold["valid_dates"],
            top_skus_segment_train,
            sku_list,
            weight_fold,
            scale_fold,
            baseline_pred,
            alpha,
        )
        segment_score, sku_eval_segment, eval_df_segment = wrmsse_score(actual_long, segment_blend_pred, scale_fold, weight_fold)

        comp = compare_sku_contribution(
            sku_eval_main,
            sku_eval_segment,
            candidate_skus=top_skus_segment_replace,
            fold_id=fold["fold"],
        )
        oracle_replace_skus = comp.loc[comp["improvement"] > SEGMENT_SELECT_MIN_MEAN_IMPROVEMENT, "ItemCode"].tolist()
        oracle_switch_pred = switch_predictions_for_skus(main_blend_pred, segment_blend_pred, oracle_replace_skus)
        oracle_switch_score, sku_eval_switch, eval_df_switch = wrmsse_score(actual_long, oracle_switch_pred, scale_fold, weight_fold)

        print("Fold", fold["fold"], "scores:",
              "baseline=", round(baseline_score, 6),
              "main=", round(main_score, 6),
              "segment_all=", round(segment_score, 6),
              "oracle_switch=", round(oracle_switch_score, 6),
              "oracle_n_skus=", len(oracle_replace_skus))
        display(comp.head(SEGMENT_SHOW_TOP_N))

        res.update({
            "segment_score": segment_score,
            "segment_sku_eval": sku_eval_segment,
            "segment_eval_df": eval_df_segment,
            "oracle_switch_score": oracle_switch_score,
            "oracle_switch_sku_eval": sku_eval_switch,
            "oracle_switch_eval_df": eval_df_switch,
            "segment_compare": comp,
            "oracle_switch_skus": oracle_replace_skus,
        })

    return res

def make_submission_from_long_pred(sample_sub, pred_long):
    sub = sample_sub.copy()
    fcols = [c for c in sub.columns if c.startswith("F")]

    sub["window"] = sub["id"].str.extract(r"_(validation|evaluation)$")[0]
    sub["ItemCode"] = sub["id"].str.replace(r"_(validation|evaluation)$", "", regex=True)

    date_map = {
        "validation": pd.date_range("2025-09-06", periods=28, freq="D"),
        "evaluation": pd.date_range("2025-10-04", periods=28, freq="D")
    }

    pred_lookup = pred_long.set_index(["ItemCode", "Date"])["pred"]

    for idx, row in sub.iterrows():
        dates = date_map[row["window"]]
        item = row["ItemCode"]
        vals = [float(pred_lookup.get((item, d), 0.0)) for d in dates]
        sub.loc[idx, fcols] = vals

    sub = sub[sample_sub.columns]

    assert sub["id"].is_unique, "Duplicate id detected!"
    assert set(sub["id"]) == set(sample_sub["id"]), "Submission id set mismatch!"
    assert sub.shape == sample_sub.shape, "Submission shape mismatch!"
    assert np.isfinite(sub[fcols].to_numpy()).all(), "NaN/Inf predictions detected!"
    assert (sub[fcols].to_numpy() >= 0).all(), "Negative predictions detected!"

    return sub



def load_train_clean(train_path=TRAIN_PATH):
    """Load raw train.csv and create the clean demand target used by the final pipeline."""
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train data: {train_path}. Put train.csv in data/raw/.")

    df = pd.read_csv(train_path)
    required_cols = ["Date", "ItemCode", "Quantity", "SalesAmount", "Cost Amount"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["ItemCode"] = df["ItemCode"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["SalesAmount"] = pd.to_numeric(df["SalesAmount"], errors="coerce").fillna(0)
    df["Cost Amount"] = pd.to_numeric(df["Cost Amount"], errors="coerce").fillna(0)

    if "UnitPrice" in df.columns:
        df["UnitPrice_num"] = parse_vn_number(df["UnitPrice"])
    if "Unit Cost" in df.columns:
        df["UnitCost_num"] = parse_vn_number(df["Unit Cost"])

    df["line_profit"] = df["SalesAmount"] - df["Cost Amount"]
    # Target demand only uses positive outgoing sales; returns/free/internal movements are excluded from y.
    df["demand_qty"] = df["Quantity"].clip(lower=0)
    df["return_qty"] = (-df["Quantity"].clip(upper=0)).astype(float)
    return df.sort_values(["Date", "ItemCode"]).reset_index(drop=True)


def load_sample_submission(sample_path=SAMPLE_PATH):
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Cannot find sample submission: {sample_path}. Put sample_submission.csv in data/raw/.")
    return pd.read_csv(sample_path)


def save_preprocessing_artifacts(df, output_dir=PROCESSED_DIR):
    """Save lightweight processed artifacts used by the modeling notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "train_clean.csv", index=False)
    sku_list = pd.DataFrame({"ItemCode": sorted(df["ItemCode"].unique())})
    sku_list.to_csv(output_dir / "sku_list.csv", index=False)
    weight_df = compute_sku_weights(df)
    weight_df.to_csv(output_dir / "sku_weights.csv", index=False)
    daily = aggregate_daily_y(df)
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    scale_df = compute_rmsse_scale(daily, sku_list["ItemCode"].tolist(), all_dates)
    scale_df.to_csv(output_dir / "rmsse_scale.csv", index=False)
    return {"sku_list": sku_list, "weight_df": weight_df, "daily": daily, "scale_df": scale_df}


def load_preprocessed_or_raw(train_path=TRAIN_PATH, processed_dir=PROCESSED_DIR):
    """Use data/processed/train_clean.csv if present; otherwise clean data/raw/train.csv."""
    processed_dir = Path(processed_dir)
    clean_path = processed_dir / "train_clean.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path, parse_dates=["Date"])
        return df
    return load_train_clean(train_path)

