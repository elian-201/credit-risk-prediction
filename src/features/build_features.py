"""
Feature Engineering Module for Credit Risk Prediction.

This module creates all features required for the credit risk model:
- Loan characteristics features
- Borrower profile features
- Credit history features
- Debt profile features
- Derived risk indicators

Author: Elian Jose
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.data.config import (
    FEATURE_ENGINEERING_CONFIG,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    MODELS_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CreditRiskFeatureEngineer:
    """
    Feature engineering pipeline for credit risk prediction.
    
    Creates interpretable and predictive features from raw loan data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FEATURE_ENGINEERING_CONFIG
        self.label_encoders = {}
        self.preprocessor = None
        self.feature_names = []
        
    # =========================================================================
    # LOAN CHARACTERISTICS FEATURES
    # =========================================================================
    
    def create_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from loan characteristics.
        
        Args:
            df: DataFrame with loan data
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating loan characteristic features...")
        
        df = df.copy()
        
        # Loan amount relative to income
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['loan_to_income'] = df['loan_to_income'].clip(0, 10)  # Cap outliers
        
        # Monthly payment burden
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['installment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
            df['installment_to_income'] = df['installment_to_income'].clip(0, 1)
        
        # Interest rate categories
        if 'int_rate' in df.columns:
            df['high_interest'] = (df['int_rate'] > 15).astype(int)
            df['very_high_interest'] = (df['int_rate'] > 20).astype(int)
            
            # Interest rate buckets
            df['int_rate_bucket'] = pd.cut(
                df['int_rate'],
                bins=[0, 8, 12, 16, 20, 40],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Term as numeric
        if 'term' in df.columns and 'term_months' not in df.columns:
            df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
        
        # Is long term loan
        if 'term_months' in df.columns:
            df['is_long_term'] = (df['term_months'] == 60).astype(int)
        
        # Grade encoding (ordinal)
        if 'grade' in df.columns:
            grade_map = self.config['grade_mapping']
            df['grade_numeric'] = df['grade'].map(grade_map)
            
            # Risk tier
            df['is_subprime'] = (df['grade'].isin(['D', 'E', 'F', 'G'])).astype(int)
            df['is_prime'] = (df['grade'].isin(['A', 'B'])).astype(int)
        
        # Sub-grade encoding
        if 'sub_grade' in df.columns:
            # Convert A1-G5 to numeric (A1=1, A5=5, B1=6, etc.)
            def encode_subgrade(sg):
                if pd.isna(sg):
                    return np.nan
                grade = sg[0]
                num = int(sg[1])
                grade_base = (ord(grade) - ord('A')) * 5
                return grade_base + num
            
            df['sub_grade_numeric'] = df['sub_grade'].apply(encode_subgrade)
        
        return df
    
    # =========================================================================
    # BORROWER PROFILE FEATURES
    # =========================================================================
    
    def create_borrower_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from borrower profile.
        
        Args:
            df: DataFrame with borrower data
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating borrower profile features...")
        
        df = df.copy()
        
        # Employment length encoding
        if 'emp_length' in df.columns:
            emp_map = self.config['emp_length_mapping']
            df['emp_length_numeric'] = df['emp_length'].map(emp_map)
            df['emp_length_numeric'] = df['emp_length_numeric'].fillna(0)
            
            # Employment stability indicators
            df['is_new_employee'] = (df['emp_length_numeric'] < 2).astype(int)
            df['is_stable_employee'] = (df['emp_length_numeric'] >= 5).astype(int)
        
        # Income features
        if 'annual_inc' in df.columns:
            # Log transform (reduces skew)
            df['log_annual_inc'] = np.log1p(df['annual_inc'])
            
            # Income categories
            df['low_income'] = (df['annual_inc'] < self.config['low_income_threshold']).astype(int)
            df['high_income'] = (df['annual_inc'] > self.config['high_income_threshold']).astype(int)
            
            # Income quintiles
            df['income_quintile'] = pd.qcut(
                df['annual_inc'].clip(lower=0),
                q=5,
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
        
        # Home ownership features
        if 'home_ownership' in df.columns:
            df['is_renter'] = (df['home_ownership'] == 'RENT').astype(int)
            df['is_homeowner'] = (df['home_ownership'].isin(['OWN', 'MORTGAGE'])).astype(int)
            df['has_mortgage'] = (df['home_ownership'] == 'MORTGAGE').astype(int)
        
        # Verification status
        if 'verification_status' in df.columns:
            df['is_verified'] = (df['verification_status'] != 'Not Verified').astype(int)
            df['is_source_verified'] = (df['verification_status'] == 'Source Verified').astype(int)
        
        return df
    
    # =========================================================================
    # CREDIT HISTORY FEATURES
    # =========================================================================
    
    def create_credit_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from credit history.
        
        Args:
            df: DataFrame with credit history data
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating credit history features...")
        
        df = df.copy()
        
        # Credit history length
        if 'credit_history_years' in df.columns:
            df['short_credit_history'] = (df['credit_history_years'] < 5).astype(int)
            df['long_credit_history'] = (df['credit_history_years'] > 15).astype(int)
        
        # Delinquency features
        if 'delinq_2yrs' in df.columns:
            df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
            df['multiple_delinquencies'] = (df['delinq_2yrs'] > 2).astype(int)
        
        # Public records
        if 'pub_rec' in df.columns:
            df['has_public_record'] = (df['pub_rec'] > 0).astype(int)
        
        # Bankruptcies
        if 'pub_rec_bankruptcies' in df.columns:
            df['has_bankruptcy'] = (df['pub_rec_bankruptcies'] > 0).astype(int)
        
        # Recent inquiries
        if 'inq_last_6mths' in df.columns:
            df['many_recent_inquiries'] = (df['inq_last_6mths'] > 3).astype(int)
            df['no_recent_inquiries'] = (df['inq_last_6mths'] == 0).astype(int)
        
        # Account statistics
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            df['account_utilization'] = df['open_acc'] / (df['total_acc'] + 1)
            df['closed_accounts'] = df['total_acc'] - df['open_acc']
        
        # Months since last delinquency (impute missing with large value)
        if 'mths_since_last_delinq' in df.columns:
            df['mths_since_last_delinq_filled'] = df['mths_since_last_delinq'].fillna(999)
            df['recent_delinquency'] = (
                df['mths_since_last_delinq_filled'] < self.config['recent_delinq_months']
            ).astype(int)
        
        return df
    
    # =========================================================================
    # DEBT PROFILE FEATURES
    # =========================================================================
    
    def create_debt_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from debt profile.
        
        Args:
            df: DataFrame with debt data
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating debt profile features...")
        
        df = df.copy()
        
        # DTI features
        if 'dti' in df.columns:
            df['high_dti'] = (df['dti'] > self.config['high_dti_threshold']).astype(int)
            df['very_high_dti'] = (df['dti'] > self.config['very_high_dti_threshold']).astype(int)
            df['low_dti'] = (df['dti'] < 10).astype(int)
        
        # Revolving utilization features
        if 'revol_util' in df.columns:
            df['revol_util_filled'] = df['revol_util'].fillna(df['revol_util'].median())
            df['high_revol_util'] = (
                df['revol_util_filled'] > self.config['high_util_threshold'] * 100
            ).astype(int)
            df['maxed_out'] = (
                df['revol_util_filled'] > self.config['very_high_util_threshold'] * 100
            ).astype(int)
            df['low_revol_util'] = (df['revol_util_filled'] < 30).astype(int)
        
        # Revolving balance features
        if 'revol_bal' in df.columns:
            df['log_revol_bal'] = np.log1p(df['revol_bal'])
            df['has_revol_bal'] = (df['revol_bal'] > 0).astype(int)
        
        # Total balance to limit ratio
        if 'tot_cur_bal' in df.columns and 'total_rev_hi_lim' in df.columns:
            df['total_util'] = df['tot_cur_bal'] / (df['total_rev_hi_lim'] + 1)
            df['total_util'] = df['total_util'].clip(0, 2)
        
        # Mortgage indicator
        if 'mort_acc' in df.columns:
            df['has_mortgage_account'] = (df['mort_acc'] > 0).astype(int)
        
        return df
    
    # =========================================================================
    # RISK INDICATOR FEATURES
    # =========================================================================
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk indicator features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with risk indicators
        """
        logger.info("Creating composite risk indicators...")
        
        df = df.copy()
        
        # Risk score (simple additive model)
        risk_factors = []
        
        if 'is_subprime' in df.columns:
            risk_factors.append(df['is_subprime'] * 2)
        if 'high_dti' in df.columns:
            risk_factors.append(df['high_dti'])
        if 'high_revol_util' in df.columns:
            risk_factors.append(df['high_revol_util'])
        if 'has_delinquency' in df.columns:
            risk_factors.append(df['has_delinquency'] * 2)
        if 'short_credit_history' in df.columns:
            risk_factors.append(df['short_credit_history'])
        if 'is_renter' in df.columns:
            risk_factors.append(df['is_renter'] * 0.5)
        if 'low_income' in df.columns:
            risk_factors.append(df['low_income'])
        if 'is_long_term' in df.columns:
            risk_factors.append(df['is_long_term'] * 0.5)
        
        if risk_factors:
            df['simple_risk_score'] = sum(risk_factors)
            df['high_risk_flag'] = (df['simple_risk_score'] > 5).astype(int)
        
        # Stability score (positive indicators)
        stability_factors = []
        
        if 'is_stable_employee' in df.columns:
            stability_factors.append(df['is_stable_employee'])
        if 'is_homeowner' in df.columns:
            stability_factors.append(df['is_homeowner'])
        if 'is_verified' in df.columns:
            stability_factors.append(df['is_verified'])
        if 'long_credit_history' in df.columns:
            stability_factors.append(df['long_credit_history'])
        if 'low_dti' in df.columns:
            stability_factors.append(df['low_dti'])
        if 'no_recent_inquiries' in df.columns:
            stability_factors.append(df['no_recent_inquiries'])
        
        if stability_factors:
            df['stability_score'] = sum(stability_factors)
        
        return df
    
    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Grade × DTI interaction
        if 'grade_numeric' in df.columns and 'dti' in df.columns:
            df['grade_dti_interaction'] = df['grade_numeric'] * df['dti']
        
        # Income × Loan amount interaction
        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['income_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
        
        # Subprime × High DTI
        if 'is_subprime' in df.columns and 'high_dti' in df.columns:
            df['subprime_high_dti'] = df['is_subprime'] * df['high_dti']
        
        # Renter × Low income
        if 'is_renter' in df.columns and 'low_income' in df.columns:
            df['renter_low_income'] = df['is_renter'] * df['low_income']
        
        return df
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Args:
            df: Processed loan DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 50)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 50)
        
        # Create all feature groups
        df = self.create_loan_features(df)
        df = self.create_borrower_features(df)
        df = self.create_credit_history_features(df)
        df = self.create_debt_features(df)
        df = self.create_risk_indicators(df)
        df = self.create_interaction_features(df)
        
        logger.info("=" * 50)
        logger.info(f"Feature engineering complete: {len(df.columns)} columns")
        logger.info("=" * 50)
        
        return df
    
    def get_feature_columns(
        self, 
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Get lists of numeric and categorical feature columns.
        
        Args:
            df: Feature DataFrame
            exclude_cols: Columns to exclude
            
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        if exclude_cols is None:
            exclude_cols = ['target', 'loan_status', 'issue_d', 'earliest_cr_line']
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Categorical columns (only those we want to encode)
        categorical_cols = [
            'grade', 'home_ownership', 'verification_status', 
            'purpose', 'application_type', 'initial_list_status',
            'int_rate_bucket', 'income_quintile'
        ]
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        return numeric_cols, categorical_cols
    
    def create_preprocessor(
        self, 
        df: pd.DataFrame,
    ) -> ColumnTransformer:
        """
        Create sklearn preprocessor for model training.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Fitted ColumnTransformer
        """
        numeric_cols, categorical_cols = self.get_feature_columns(df)
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        
        # Combined preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric_cols),
                ('categorical', categorical_pipeline, categorical_cols),
            ],
            remainder='drop',
        )
        
        self.preprocessor = preprocessor
        self.feature_names = numeric_cols + categorical_cols
        
        logger.info(f"Created preprocessor with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        
        return preprocessor
    
    def save_preprocessor(self, path: Optional[Path] = None) -> Path:
        """Save preprocessor to disk."""
        if self.preprocessor is None:
            raise ValueError("No preprocessor to save. Run create_preprocessor() first.")
        
        if path is None:
            path = MODELS_DIR / "preprocessor.joblib"
        
        joblib.dump(self.preprocessor, path)
        
        # Also save feature names
        feature_path = MODELS_DIR / "feature_columns.joblib"
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Saved preprocessor to {path}")
        
        return path


def build_feature_dataset(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build complete feature dataset.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        save: Whether to save feature datasets
        
    Returns:
        Tuple of (train_features, test_features)
    """
    engineer = CreditRiskFeatureEngineer()
    
    # Engineer features
    train_features = engineer.engineer_features(train_df)
    
    if test_df is not None:
        test_features = engineer.engineer_features(test_df)
    else:
        test_features = None
    
    # Save
    if save:
        timestamp = datetime.now().strftime("%Y%m%d")
        
        train_path = FEATURES_DIR / f"train_features_{timestamp}.parquet"
        train_features.to_parquet(train_path, index=False)
        logger.info(f"Saved train features to {train_path}")
        
        if test_features is not None:
            test_path = FEATURES_DIR / f"test_features_{timestamp}.parquet"
            test_features.to_parquet(test_path, index=False)
            logger.info(f"Saved test features to {test_path}")
    
    return train_features, test_features


if __name__ == "__main__":
    print("Credit Risk Feature Engineering Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from src.features.build_features import CreditRiskFeatureEngineer")
    print("  engineer = CreditRiskFeatureEngineer()")
    print("  df = engineer.engineer_features(processed_df)")
