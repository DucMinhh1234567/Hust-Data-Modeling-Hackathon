from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

TRAIN_PATH = RAW_DIR / "train.csv"
SAMPLE_PATH = RAW_DIR / "sample_submission.csv"

# =========================
# Reproducibility
# =========================
RANDOM_STATE = 42
EPS = 1e-9

# =========================
# Model configuration used for the submitted solution
# =========================
RUN_LIGHTGBM = True
RUN_ROLLING_VALIDATION = True
TOP_N_SKUS = 2892

LGBM_TARGET_MODE = "ratio"
RATIO_EPS = 0.10
RATIO_CLIP_MAX = 2.5

USE_EWMA_TREND_PROFIT_FEATURES = True
EWMA_SPANS = [7, 14, 28]

USE_LGBM_EARLY_STOPPING = True
LGBM_NUM_BOOST_ROUND = 3000
LGBM_FIXED_NUM_BOOST_ROUND = 700
LGBM_EARLY_STOPPING_ROUNDS = 100
LGBM_ES_VALID_FRAC = 0.15
RETRAIN_FULL_AFTER_EARLY_STOPPING = True

BASELINE_WINDOWS = (14, 28, 56, 90)
BASELINE_WEIGHTS = (0.40, 0.30, 0.20, 0.10)
SUNDAY_FACTOR_FLOOR = 0.001

VALIDATION_HORIZON = 28
N_ROLLING_FOLDS = 3
ROLLING_STEP_DAYS = 28

USE_BASELINE_ENSEMBLE = False
BASELINE_ENSEMBLE_CONFIGS = [
    {"name": "fast", "windows": (7, 14, 28, 56), "weights": (0.30, 0.35, 0.25, 0.10), "ensemble_weight": 0.20},
    {"name": "stable", "windows": (14, 28, 56, 90), "weights": (0.40, 0.30, 0.20, 0.10), "ensemble_weight": 0.60},
    {"name": "long", "windows": (28, 56, 90, 180), "weights": (0.35, 0.30, 0.20, 0.15), "ensemble_weight": 0.20},
]

ALPHA_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
BEST_ALPHA = 1.0
BEST_BLEND_MODE = "constant"

USE_RANK_HORIZON_ALPHA = False
ALPHA_SCALE_GRID = [0.50, 0.75, 1.00, 1.25, 1.50]
RANK_ALPHA_CONFIG = {"top_213": 0.12, "rank_214_1403": 0.08, "rank_1404_2000": 0.04, "tail": 0.00}
HORIZON_ALPHA_CONFIG = {"h_1_7": 1.20, "h_8_14": 1.00, "h_15_28": 0.80, "h_29_56": 0.50}

USE_SPECIAL_SKU_MULTIPLIER = False
SPECIAL_SKU_MULTIPLIER = {}

# Segment model was tested and kept disabled in the final solution.
USE_SEGMENT_TOP_MODEL = False
SEGMENT_TOP_N_TRAIN = 100
SEGMENT_TOP_N_REPLACE = 50
SEGMENT_SELECT_MIN_BETTER_FOLDS = 2
SEGMENT_SELECT_MIN_MEAN_IMPROVEMENT = 0.0
SEGMENT_MAX_SELECTED_SKUS = 50
SEGMENT_SHOW_TOP_N = 30

# Final SKU-level correction-factor configuration.
USE_SKU_CORRECTION_FACTOR = True
CORR_TOP_N_CANDIDATES = 50
CORR_FACTOR_GRID = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
CORR_SELECT_MIN_BETTER_FOLDS = 1
CORR_SELECT_MIN_MEAN_IMPROVEMENT = 0.00005
CORR_MAX_SELECTED_SKUS = 15
CORR_SHOW_TOP_N = 30
CORR_ONLY_OVERPRED = False
CORR_MIN_PRED_SUM = 1.0
CORR_FACTOR_AGG_METHOD = "improvement_weighted"
CORR_FACTOR_SHRINK = 0.60
CORR_FACTOR_CLIP_LOW = 0.75
CORR_FACTOR_CLIP_HIGH = 1.15

CORR_VARIANT_CONFIGS = [
    {
        "name": "shrink060_sel15_min005",
        "factor_shrink": 0.60,
        "max_selected_skus": 15,
        "min_mean_improvement": 0.00005,
        "min_better_folds": 1,
        "clip_low": 0.75,
        "clip_high": 1.15,
    }
]

MANUAL_SKU_CORRECTION_FACTORS = {}

FORECAST_START = "2025-09-06"
FORECAST_DAYS = 56
FINAL_SUBMISSION_NAME = "submission_lgbm_ratio_sku_corr_shrink060_sel15.csv"
