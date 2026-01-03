"""
Data Ingestion Module for Credit Risk Prediction.

This module handles:
1. Loading Lending Club data
2. Initial preprocessing
3. Target variable creation
4. Train/test splitting

Author: Elian Jose
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import pandas as pd
import numpy as np

from src.data.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    DEFAULT_STATUSES,
    PAID_STATUSES,
    EXCLUDE_STATUSES,
    DROP_FEATURES,
    LEAKAGE_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_CONFIG,
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LendingClubDataLoader:
    """Loads and preprocesses Lending Club loan data."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to raw CSV file. If None, looks in RAW_DATA_DIR.
        """
        if data_path is None:
            data_path = RAW_DATA_DIR / RAW_DATA_FILE
        
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        
    def load_raw_data(
        self, 
        nrows: Optional[int] = None,
        sample_frac: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load raw Lending Club data.
        
        Args:
            nrows: Number of rows to load (for testing)
            sample_frac: Fraction of data to sample (for faster iteration)
            
        Returns:
            Raw DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please download from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
                f"And place the CSV file in {RAW_DATA_DIR}"
            )
        
        # Load with optimized dtypes
        dtype_dict = {
            'id': str,
            'member_id': str,
            'term': str,
            'grade': str,
            'sub_grade': str,
            'emp_title': str,
            'emp_length': str,
            'home_ownership': str,
            'verification_status': str,
            'loan_status': str,
            'purpose': str,
            'title': str,
            'zip_code': str,
            'addr_state': str,
            'application_type': str,
            'initial_list_status': str,
        }
        
        # Read CSV
        if nrows:
            self.raw_df = pd.read_csv(
                self.data_path, 
                nrows=nrows,
                dtype=dtype_dict,
                low_memory=False,
            )
        else:
            self.raw_df = pd.read_csv(
                self.data_path,
                dtype=dtype_dict,
                low_memory=False,
            )
        
        # Sample if requested
        if sample_frac and sample_frac < 1.0:
            original_len = len(self.raw_df)
            self.raw_df = self.raw_df.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled {len(self.raw_df)} rows from {original_len}")
        
        logger.info(f"Loaded {len(self.raw_df)} rows, {len(self.raw_df.columns)} columns")
        
        return self.raw_df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable from loan_status.
        
        Default = 1 (bad loan)
        Paid = 0 (good loan)
        
        Args:
            df: DataFrame with loan_status column
            
        Returns:
            DataFrame with target column added
        """
        logger.info("Creating target variable...")
        
        df = df.copy()
        
        # Log original distribution
        logger.info(f"Original loan_status distribution:\n{df['loan_status'].value_counts()}")
        
        # Filter to only completed loans (exclude current/ongoing)
        completed_mask = ~df['loan_status'].isin(EXCLUDE_STATUSES)
        df = df[completed_mask].copy()
        logger.info(f"Filtered to {len(df)} completed loans")
        
        # Create binary target
        df['target'] = df['loan_status'].apply(
            lambda x: 1 if x in DEFAULT_STATUSES else 0
        )
        
        # Log target distribution
        default_rate = df['target'].mean()
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        logger.info(f"Default rate: {default_rate:.2%}")
        
        return df
    
    def remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that leak future information.
        
        These are features that wouldn't be available at loan origination.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with leakage features removed
        """
        logger.info("Removing leakage features...")
        
        cols_to_drop = [col for col in LEAKAGE_FEATURES if col in df.columns]
        cols_to_drop.extend([col for col in DROP_FEATURES if col in df.columns])
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        logger.info(f"Removed {len(cols_to_drop)} features")
        
        return df
    
    def clean_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert numeric features.
        
        Args:
            df: DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning numeric features...")
        
        df = df.copy()
        
        # Convert interest rate (remove % sign)
        if 'int_rate' in df.columns and df['int_rate'].dtype == 'object':
            df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)
        
        # Convert revolving utilization (remove % sign)
        if 'revol_util' in df.columns and df['revol_util'].dtype == 'object':
            df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)
        
        # Convert term to numeric (e.g., "36 months" -> 36)
        if 'term' in df.columns and df['term'].dtype == 'object':
            df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
        
        return df
    
    def clean_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean categorical features.
        
        Args:
            df: DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning categorical features...")
        
        df = df.copy()
        
        # Consolidate home ownership categories
        if 'home_ownership' in df.columns:
            df['home_ownership'] = df['home_ownership'].replace({
                'ANY': 'OTHER',
                'NONE': 'OTHER',
            })
        
        # Clean employment length
        if 'emp_length' in df.columns:
            # Keep as is for now, will encode in feature engineering
            df['emp_length'] = df['emp_length'].fillna('Unknown')
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns and extract features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with date features
        """
        logger.info("Parsing date features...")
        
        df = df.copy()
        
        # Issue date
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            df['issue_year'] = df['issue_d'].dt.year
            df['issue_month'] = df['issue_d'].dt.month
        
        # Earliest credit line
        if 'earliest_cr_line' in df.columns:
            df['earliest_cr_line'] = pd.to_datetime(
                df['earliest_cr_line'], format='%b-%Y', errors='coerce'
            )
            # Calculate credit history length in years
            if 'issue_d' in df.columns:
                df['credit_history_years'] = (
                    (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
                )
        
        return df
    
    def preprocess(
        self, 
        nrows: Optional[int] = None,
        sample_frac: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Args:
            nrows: Number of rows to load
            sample_frac: Fraction to sample
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_raw_data(nrows=nrows, sample_frac=sample_frac)
        
        # Create target
        df = self.create_target_variable(df)
        
        # Remove leakage
        df = self.remove_leakage_features(df)
        
        # Clean features
        df = self.clean_numeric_features(df)
        df = self.clean_categorical_features(df)
        df = self.parse_dates(df)
        
        self.processed_df = df
        
        logger.info("=" * 50)
        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        logger.info("=" * 50)
        
        return df
    
    def save_processed_data(
        self, 
        df: Optional[pd.DataFrame] = None,
        filename: str = "processed_loans.parquet",
    ) -> Path:
        """
        Save processed data to parquet.
        
        Args:
            df: DataFrame to save (uses self.processed_df if None)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if df is None:
            df = self.processed_df
        
        if df is None:
            raise ValueError("No data to save. Run preprocess() first.")
        
        output_path = PROCESSED_DATA_DIR / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify: bool = True,
    time_based: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split.
    
    Args:
        df: Processed DataFrame
        test_size: Fraction for test set
        stratify: Whether to stratify by target
        time_based: Whether to split by time (more realistic)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Creating train/test split (test_size={test_size})")
    
    if time_based and 'issue_d' in df.columns:
        # Sort by issue date and split
        df_sorted = df.sort_values('issue_d')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"Time-based split:")
        logger.info(f"  Train: {train_df['issue_d'].min()} to {train_df['issue_d'].max()}")
        logger.info(f"  Test: {test_df['issue_d'].min()} to {test_df['issue_d'].max()}")
    else:
        # Random stratified split
        stratify_col = df['target'] if stratify else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=MODEL_CONFIG['random_state'],
        )
    
    logger.info(f"Train set: {len(train_df)} rows ({train_df['target'].mean():.2%} default rate)")
    logger.info(f"Test set: {len(test_df)} rows ({test_df['target'].mean():.2%} default rate)")
    
    return train_df, test_df


def run_ingestion_pipeline(
    nrows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete data ingestion pipeline.
    
    Args:
        nrows: Number of rows to load
        sample_frac: Fraction to sample
        save: Whether to save processed data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load and preprocess
    loader = LendingClubDataLoader()
    df = loader.preprocess(nrows=nrows, sample_frac=sample_frac)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # Save
    if save:
        loader.save_processed_data(df, "processed_loans.parquet")
        
        train_path = PROCESSED_DATA_DIR / "train.parquet"
        test_path = PROCESSED_DATA_DIR / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved test data to {test_path}")
    
    return train_df, test_df


if __name__ == "__main__":
    print("Credit Risk Data Ingestion")
    print("=" * 50)
    print("\nUsage:")
    print("  from src.data.ingestion import run_ingestion_pipeline")
    print("  train_df, test_df = run_ingestion_pipeline()")
    print("\nOr for testing with smaller data:")
    print("  train_df, test_df = run_ingestion_pipeline(nrows=10000)")
    print("\nMake sure to download the Lending Club dataset first:")
    print("  https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print(f"  Place the CSV file in: {RAW_DATA_DIR}")
