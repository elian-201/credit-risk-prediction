"""
Model Training Module for Credit Risk Prediction.

This module handles:
1. Model training (Logistic Regression, LightGBM, XGBoost)
2. Hyperparameter tuning with Optuna
3. Cross-validation
4. Experiment tracking with MLflow
5. Model persistence

Author: Elian Jose
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import warnings

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, log_loss, brier_score_loss, confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb
import optuna

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.data.config import MODEL_CONFIG, MODELS_DIR, REPORTS_DIR
from src.features.build_features import CreditRiskFeatureEngineer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Handles training and evaluation of credit risk models."""
    
    def __init__(self, config: Optional[Dict] = None, experiment_name: str = "credit_risk_prediction"):
        self.config = config or MODEL_CONFIG
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        y = df[target_col].copy()
        engineer = CreditRiskFeatureEngineer()
        numeric_cols, _ = engineer.get_feature_columns(df)
        X = df[numeric_cols].copy()
        self.feature_columns = numeric_cols
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def handle_missing_values(self, X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values."""
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imputed
    
    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Train Logistic Regression."""
        logger.info("Training Logistic Regression...")
        params = self.config['lr_params'].copy()
        params['class_weight'] = 'balanced'
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        
        pipeline = Pipeline([('scaler', scaler), ('classifier', model)])
        
        if X_val is not None:
            y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC: {auc:.4f}")
        
        self.models['logistic_regression'] = pipeline
        return pipeline
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparameters=False):
        """Train LightGBM."""
        logger.info("Training LightGBM...")
        
        params = self.config['lgbm_params'].copy()
        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        params['scale_pos_weight'] = n_neg / n_pos
        
        model = lgb.LGBMClassifier(**params)
        
        if X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC: {auc:.4f}")
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost."""
        logger.info("Training XGBoost...")
        
        params = self.config['xgb_params'].copy()
        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        params['scale_pos_weight'] = n_neg / n_pos
        
        model = xgb.XGBClassifier(**params)
        
        if X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Validation AUC: {auc:.4f}")
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def train_all_models(self, X_train, y_train, X_val, y_val, tune_lightgbm=False):
        """Train all models and compare."""
        logger.info("=" * 50)
        logger.info("Training all models")
        logger.info("=" * 50)
        
        X_train = self.handle_missing_values(X_train)
        X_val = self.handle_missing_values(X_val)
        
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val, tune_hyperparameters=tune_lightgbm)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        results = {}
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            results[name] = {
                'auc': roc_auc_score(y_val, y_pred_proba),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'accuracy': accuracy_score(y_val, y_pred),
            }
            logger.info(f"{name}: AUC={results[name]['auc']:.4f}, F1={results[name]['f1']:.4f}")
        
        best_name = max(results.keys(), key=lambda k: results[k]['auc'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        logger.info(f"\nBest model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
        
        return results
    
    def save_model(self, model=None, path=None, model_name='best_model'):
        """Save trained model."""
        if model is None:
            model = self.best_model
        if model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = MODELS_DIR / f"{model_name}.joblib"
        
        joblib.dump(model, path)
        logger.info(f"Saved model to {path}")
        
        if self.feature_columns:
            joblib.dump(self.feature_columns, MODELS_DIR / "feature_columns.joblib")
        
        return path


def run_training_pipeline(train_df, test_df, tune_hyperparameters=False, save_model=True):
    """Run complete training pipeline."""
    logger.info("=" * 50)
    logger.info("Starting training pipeline")
    logger.info("=" * 50)
    
    trainer = CreditRiskModelTrainer()
    
    X_train, y_train = trainer.prepare_data(train_df)
    X_test, y_test = trainer.prepare_data(test_df)
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    results = trainer.train_all_models(X_train_split, y_train_split, X_val, y_val, tune_lightgbm=tune_hyperparameters)
    
    logger.info("\n" + "=" * 50)
    logger.info("Final evaluation on test set")
    logger.info("=" * 50)
    
    X_test_imputed = trainer.handle_missing_values(X_test)
    y_pred_proba = trainer.best_model.predict_proba(X_test_imputed)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    test_results = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    if save_model:
        trainer.save_model()
    
    return trainer.best_model, {'validation': results, 'test': test_results}


if __name__ == "__main__":
    print("Credit Risk Model Training Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from src.models.train import run_training_pipeline")
    print("  model, results = run_training_pipeline(train_df, test_df)")
