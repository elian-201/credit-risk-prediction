"""
Data Validation Module for Credit Risk Prediction.

This module provides:
1. Pandera schemas for data validation
2. Data quality checks
3. Validation reporting

Author: Elian Jose
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.errors import SchemaError

from src.data.config import PROCESSED_DATA_DIR, REPORTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PANDERA SCHEMAS
# =============================================================================

# Schema for raw loan data (minimal validation)
RawLoanSchema = DataFrameSchema(
    columns={
        "loan_amnt": Column(
            float,
            Check.greater_than(0, error="Loan amount must be positive"),
            nullable=True,
            coerce=True,
        ),
        "loan_status": Column(
            str,
            nullable=False,
            description="Loan status (target source)",
        ),
        "int_rate": Column(
            float,
            Check.in_range(0, 50, error="Interest rate should be 0-50%"),
            nullable=True,
            coerce=True,
        ),
        "annual_inc": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
    },
    strict=False,
    coerce=True,
)

# Schema for processed loan data
ProcessedLoanSchema = DataFrameSchema(
    columns={
        "loan_amnt": Column(
            float,
            Check.greater_than(0),
            nullable=False,
        ),
        "target": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description="Binary target: 1=default, 0=paid",
        ),
        "int_rate": Column(
            float,
            Check.in_range(0, 40),
            nullable=True,
        ),
        "annual_inc": Column(
            float,
            Check.greater_than(0),
            nullable=True,
        ),
        "dti": Column(
            float,
            Check.in_range(-10, 100, error="DTI should be reasonable"),
            nullable=True,
        ),
        "grade": Column(
            str,
            Check.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']),
            nullable=True,
        ),
        "home_ownership": Column(
            str,
            Check.isin(['RENT', 'OWN', 'MORTGAGE', 'OTHER']),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
)

# Schema for feature data (model input)
FeatureSchema = DataFrameSchema(
    columns={
        "target": Column(int, Check.isin([0, 1]), nullable=False),
        # Add more feature validations as needed
    },
    strict=False,
    coerce=True,
)

# Schema for API input
APIInputSchema = DataFrameSchema(
    columns={
        "loan_amnt": Column(float, Check.greater_than(0), nullable=False),
        "term_months": Column(int, Check.isin([36, 60]), nullable=False),
        "int_rate": Column(float, Check.in_range(1, 35), nullable=False),
        "annual_inc": Column(float, Check.greater_than(0), nullable=False),
        "dti": Column(float, Check.in_range(0, 100), nullable=False),
        "grade": Column(str, Check.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']), nullable=False),
        "home_ownership": Column(str, nullable=False),
        "emp_length": Column(str, nullable=True),
        "purpose": Column(str, nullable=False),
        "verification_status": Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate raw loan data.
    
    Args:
        df: Raw loan DataFrame
        
    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    report = {
        "input_rows": len(df),
        "validation_time": datetime.now().isoformat(),
        "schema": "RawLoanSchema",
        "errors": [],
        "warnings": [],
    }
    
    try:
        validated_df = RawLoanSchema.validate(df, lazy=True)
        report["status"] = "PASSED"
        report["output_rows"] = len(validated_df)
        logger.info(f"Raw data validation passed: {len(validated_df)} rows")
        
    except SchemaError as e:
        report["status"] = "FAILED"
        report["errors"] = str(e)
        logger.error(f"Raw data validation failed: {e}")
        validated_df = df
        report["output_rows"] = len(df)
    
    return validated_df, report


def validate_processed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate processed loan data.
    
    Args:
        df: Processed loan DataFrame
        
    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    report = {
        "input_rows": len(df),
        "validation_time": datetime.now().isoformat(),
        "schema": "ProcessedLoanSchema",
        "errors": [],
        "warnings": [],
    }
    
    try:
        validated_df = ProcessedLoanSchema.validate(df, lazy=True)
        report["status"] = "PASSED"
        report["output_rows"] = len(validated_df)
        logger.info(f"Processed data validation passed: {len(validated_df)} rows")
        
    except SchemaError as e:
        report["status"] = "FAILED"
        report["errors"] = str(e)
        logger.error(f"Processed data validation failed: {e}")
        validated_df = df
        report["output_rows"] = len(df)
    
    return validated_df, report


def validate_api_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate API input data.
    
    Args:
        data: Dictionary of input features
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Required fields
    required_fields = [
        'loan_amnt', 'term_months', 'int_rate', 'annual_inc', 
        'dti', 'grade', 'home_ownership', 'purpose', 'verification_status'
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Value validations
    if data['loan_amnt'] <= 0:
        errors.append("loan_amnt must be positive")
    
    if data['term_months'] not in [36, 60]:
        errors.append("term_months must be 36 or 60")
    
    if not 1 <= data['int_rate'] <= 35:
        errors.append("int_rate must be between 1 and 35")
    
    if data['annual_inc'] <= 0:
        errors.append("annual_inc must be positive")
    
    if not 0 <= data['dti'] <= 100:
        errors.append("dti must be between 0 and 100")
    
    if data['grade'] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        errors.append("grade must be A, B, C, D, E, F, or G")
    
    return len(errors) == 0, errors


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

class DataQualityChecker:
    """Comprehensive data quality checks for credit risk data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.warnings = []
        self.metrics = {}
        
    def check_target_distribution(self) -> None:
        """Check target variable distribution for imbalance."""
        if 'target' not in self.df.columns:
            self.warnings.append("Target column not found")
            return
        
        target_dist = self.df['target'].value_counts(normalize=True)
        self.metrics["target_distribution"] = target_dist.to_dict()
        
        minority_rate = target_dist.min()
        
        if minority_rate < 0.05:
            self.issues.append(f"Severe class imbalance: minority class at {minority_rate:.2%}")
        elif minority_rate < 0.20:
            self.warnings.append(f"Moderate class imbalance: minority class at {minority_rate:.2%}")
        
        logger.info(f"Target distribution: {target_dist.to_dict()}")
    
    def check_missing_values(self) -> None:
        """Check for missing values across features."""
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        high_missing = missing_pct[missing_pct > 50]
        
        self.metrics["missing_percentage"] = missing_pct.to_dict()
        self.metrics["high_missing_columns"] = high_missing.to_dict()
        
        for col, pct in high_missing.items():
            self.warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
        
        logger.info(f"Found {len(high_missing)} columns with >50% missing values")
    
    def check_duplicates(self) -> None:
        """Check for duplicate rows."""
        dup_count = self.df.duplicated().sum()
        self.metrics["duplicate_rows"] = dup_count
        
        if dup_count > 0:
            dup_pct = dup_count / len(self.df) * 100
            if dup_pct > 1:
                self.issues.append(f"Found {dup_count} duplicate rows ({dup_pct:.2f}%)")
            else:
                self.warnings.append(f"Found {dup_count} duplicate rows ({dup_pct:.2f}%)")
        
        logger.info(f"Duplicate rows: {dup_count}")
    
    def check_numeric_ranges(self) -> None:
        """Check for suspicious values in numeric columns."""
        checks = {
            'loan_amnt': (500, 50000),
            'annual_inc': (0, 10000000),
            'int_rate': (0, 40),
            'dti': (-10, 100),
            'open_acc': (0, 100),
            'pub_rec': (0, 50),
            'revol_util': (0, 200),
        }
        
        for col, (min_val, max_val) in checks.items():
            if col not in self.df.columns:
                continue
            
            out_of_range = (
                (self.df[col] < min_val) | (self.df[col] > max_val)
            ).sum()
            
            if out_of_range > 0:
                pct = out_of_range / len(self.df) * 100
                self.warnings.append(
                    f"Column '{col}': {out_of_range} values ({pct:.2f}%) outside expected range [{min_val}, {max_val}]"
                )
    
    def check_categorical_cardinality(self) -> None:
        """Check cardinality of categorical columns."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        cardinality = {}
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            cardinality[col] = n_unique
            
            if n_unique > 100:
                self.warnings.append(f"High cardinality in '{col}': {n_unique} unique values")
            elif n_unique == 1:
                self.issues.append(f"Constant column '{col}': only 1 unique value")
        
        self.metrics["categorical_cardinality"] = cardinality
    
    def check_correlation_with_target(self) -> None:
        """Check feature correlation with target."""
        if 'target' not in self.df.columns:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'target']
        
        correlations = {}
        for col in numeric_cols:
            try:
                corr = self.df[col].corr(self.df['target'])
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                pass
        
        self.metrics["target_correlations"] = correlations
        
        # Check for suspiciously high correlations (potential leakage)
        high_corr = {k: v for k, v in correlations.items() if abs(v) > 0.5}
        if high_corr:
            for col, corr in high_corr.items():
                self.warnings.append(
                    f"High correlation with target in '{col}': {corr:.3f} (check for leakage)"
                )
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks and return report."""
        logger.info("Running all data quality checks...")
        
        self.check_target_distribution()
        self.check_missing_values()
        self.check_duplicates()
        self.check_numeric_ranges()
        self.check_categorical_cardinality()
        self.check_correlation_with_target()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "issues": self.issues,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "status": "PASSED" if len(self.issues) == 0 else "ISSUES_FOUND",
        }
        
        logger.info(f"Quality check complete: {report['status']}")
        logger.info(f"  Issues: {len(self.issues)}, Warnings: {len(self.warnings)}")
        
        return report


def run_full_validation(
    df: pd.DataFrame,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Run complete validation pipeline.
    
    Args:
        df: DataFrame to validate
        save_report: Whether to save validation report
        
    Returns:
        Validation report dictionary
    """
    # Schema validation
    validated_df, schema_report = validate_processed_data(df)
    
    # Quality checks
    quality_report = DataQualityChecker(validated_df).run_all_checks()
    
    # Combined report
    report = {
        "schema_validation": schema_report,
        "quality_checks": quality_report,
        "overall_status": "PASSED" if (
            schema_report["status"] == "PASSED" and 
            quality_report["status"] == "PASSED"
        ) else "ISSUES_FOUND",
    }
    
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"validation_report_{timestamp}.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {report_path}")
    
    return report


if __name__ == "__main__":
    print("Data Validation Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from src.data.validation import run_full_validation")
    print("  report = run_full_validation(df)")
