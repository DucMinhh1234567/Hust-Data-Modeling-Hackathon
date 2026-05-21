from pathlib import Path
import pandas as pd

from src.config import *
from src.pipeline import *


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_preprocessed_or_raw()
    sample_sub = load_sample_submission()
    sku_list = sorted(df["ItemCode"].unique())

    folds = make_rolling_folds(df, horizon=VALIDATION_HORIZON, n_folds=N_ROLLING_FOLDS, step_days=ROLLING_STEP_DAYS)
    rolling_rows, correction_tables = [], []
    for fold in folds:
        res = evaluate_lgbm_blend_on_fold(df, fold, sku_list, alpha=BEST_ALPHA, top_n=TOP_N_SKUS)
        rolling_rows.append({
            "fold": fold["fold"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "baseline_score": res["baseline_score"],
            "blend_score": res["blend_score"],
            "correction_score": res["correction_score"],
        })
        if len(res["sku_correction_table"]) > 0:
            correction_tables.append(res["sku_correction_table"])

    rolling_results = pd.DataFrame(rolling_rows)
    rolling_results.to_csv(OUTPUT_DIR / "rolling_validation_results.csv", index=False)
    all_corr = pd.concat(correction_tables, ignore_index=True) if correction_tables else pd.DataFrame()

    variant = CORR_VARIANT_CONFIGS[0]
    factors, corr_summary = select_sku_correction_factors(
        all_corr,
        min_better_folds=variant["min_better_folds"],
        min_mean_improvement=variant["min_mean_improvement"],
        max_selected_skus=variant["max_selected_skus"],
        factor_shrink=variant["factor_shrink"],
        clip_low=variant["clip_low"],
        clip_high=variant["clip_high"],
        agg_method=CORR_FACTOR_AGG_METHOD,
    )
    corr_summary.to_csv(OUTPUT_DIR / "sku_correction_summary.csv", index=False)

    forecast_dates_56 = pd.date_range(FORECAST_START, periods=FORECAST_DAYS, freq="D")
    full_weight_df = compute_sku_weights(df)
    final_baseline_pred, _, _ = build_baseline_model_forecast(
        df, forecast_dates_56, sku_list, weight_df=full_weight_df, zero_weight_to_zero=True, use_ensemble=USE_BASELINE_ENSEMBLE
    )
    full_daily = aggregate_daily_y(df)
    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_scale_df = compute_rmsse_scale(full_daily, sku_list, full_dates)
    top_skus_full = get_top_weight_skus(full_weight_df, top_n=TOP_N_SKUS)

    train_frame, feature_cols, item_map, static_table = make_lgbm_train_frame(
        df, df["Date"].max(), top_skus_full, full_weight_df, full_scale_df, min_history_days=120, target_mode=LGBM_TARGET_MODE
    )
    model = train_lgbm_model(train_frame, feature_cols, target_mode=LGBM_TARGET_MODE)
    lgbm_pred_top = predict_lgbm_recursive(
        model, df, df["Date"].max(), forecast_dates_56, top_skus_full, item_map, static_table,
        feature_cols, target_mode=LGBM_TARGET_MODE, baseline_pred=final_baseline_pred
    )
    final_pred = blend_baseline_lgbm(
        final_baseline_pred, lgbm_pred_top, top_skus_full, alpha=BEST_ALPHA,
        weight_df=full_weight_df, forecast_start_date=forecast_dates_56.min(),
        use_rank_horizon_alpha=USE_RANK_HORIZON_ALPHA,
    )
    final_pred = apply_sku_correction_factors(final_pred, factors)
    sub = make_submission_from_long_pred(sample_sub, final_pred)
    out_path = OUTPUT_DIR / FINAL_SUBMISSION_NAME
    sub.to_csv(out_path, index=False)
    print(f"Saved final submission: {out_path}")
    print("Correction factors:", factors)


if __name__ == "__main__":
    main()
