"""
Configuration for Credit Risk Default Prediction.

Contains all constants, paths, and configuration settings.
"""

from pathlib import Path
from typing import List, Dict, Any
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Kaggle dataset info
KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_DATA_FILE = "accepted_2007_to_2018Q4.csv"

# Alternative: Direct download URL (if Kaggle auth fails)
# You can also manually download from Kaggle

# Target variable
TARGET_COLUMN = "loan_status"

# Target mapping (what counts as default)
DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
]

PAID_STATUSES = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]

# Statuses to exclude (loan still ongoing)
EXCLUDE_STATUSES = [
    "Current",
    "In Grace Period",
    "Issued",
]

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Columns to use from raw data
NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
]

DATE_FEATURES = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d",
]

# Features to drop (leakage or not useful)
DROP_FEATURES = [
    "id",
    "member_id",
    "url",
    "desc",
    "title",
    "zip_code",
    "emp_title",
    "policy_code",
    "pymnt_plan",  # Almost all 'n'
    "hardship_flag",
    "debt_settlement_flag",
]

# Features that leak future information (should not be used for prediction)
LEAKAGE_FEATURES = [
    "out_prncp",
    "out_prncp_inv", 
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

FEATURE_ENGINEERING_CONFIG = {
    # Credit utilization thresholds
    "high_util_threshold": 0.8,
    "very_high_util_threshold": 0.95,
    
    # Income thresholds
    "low_income_threshold": 30000,
    "high_income_threshold": 100000,
    
    # DTI thresholds
    "high_dti_threshold": 20,
    "very_high_dti_threshold": 35,
    
    # Delinquency thresholds
    "recent_delinq_months": 24,
    
    # Employment length encoding
    "emp_length_mapping": {
        "< 1 year": 0.5,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    },
    
    # Grade encoding (ordinal)
    "grade_mapping": {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
    },
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    # Train/validation/test split
    "test_size": 0.2,
    "val_size": 0.2,  # Of remaining after test split
    "random_state": 42,
    
    # Cross-validation
    "cv_folds": 5,
    "stratified": True,
    
    # Class imbalance handling
    "handle_imbalance": True,
    "imbalance_method": "smote",  # Options: smote, class_weight, undersample
    
    # LightGBM default parameters
    "lgbm_params": {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "random_state": 42,
    },
    
    # XGBoost default parameters
    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42,
    },
    
    # Logistic Regression parameters
    "lr_params": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    },
    
    # Optuna hyperparameter tuning
    "optuna_trials": 100,
    "optuna_timeout": 3600,  # 1 hour
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    # Threshold for classification
    "default_threshold": 0.5,
    
    # Threshold optimization
    "optimize_threshold": True,
    "optimization_metric": "f1",  # Options: f1, precision, recall, balanced_accuracy
    
    # Metrics to compute
    "metrics": [
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "brier_score",
    ],
    
    # Fairness evaluation
    "protected_attributes": ["addr_state"],  # Note: race/gender not in Lending Club data
    "fairness_metrics": [
        "demographic_parity_difference",
        "equalized_odds_difference",
    ],
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_CONFIG = {
    "title": "Credit Risk Prediction API",
    "description": "ML-powered API for predicting loan default risk",
    "version": "1.0.0",
    "model_path": MODELS_DIR / "best_model.joblib",
    "preprocessor_path": MODELS_DIR / "preprocessor.joblib",
    "feature_columns_path": MODELS_DIR / "feature_columns.joblib",
}

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

MONITORING_CONFIG = {
    # Data drift thresholds
    "drift_threshold": 0.1,  # PSI threshold
    "feature_drift_threshold": 0.15,
    
    # Performance thresholds
    "auc_threshold": 0.70,
    "precision_threshold": 0.50,
    
    # Alert settings
    "alert_email": "elian.jose12@gmail.com",
    "check_frequency": "daily",
}

# =============================================================================
# AWS CONFIGURATION
# =============================================================================

AWS_CONFIG = {
    "region": "ap-south-1",
    "s3_bucket": "credit-risk-models",
    "lambda_function": "credit-risk-api",
}
