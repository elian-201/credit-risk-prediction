"""
FastAPI Application for Credit Risk Prediction.

This module provides REST API endpoints for:
1. Single loan prediction
2. Batch predictions
3. Model information
4. Health checks

Author: Elian Jose
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"

# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class LoanApplication(BaseModel):
    """Schema for loan application input."""
    
    loan_amnt: float = Field(..., gt=0, description="Loan amount requested")
    term_months: int = Field(..., description="Loan term in months (36 or 60)")
    int_rate: float = Field(..., ge=1, le=35, description="Interest rate (%)")
    installment: float = Field(..., gt=0, description="Monthly installment amount")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio")
    
    # Credit history
    delinq_2yrs: int = Field(0, ge=0, description="Delinquencies in past 2 years")
    inq_last_6mths: int = Field(0, ge=0, description="Inquiries in last 6 months")
    open_acc: int = Field(1, ge=0, description="Number of open accounts")
    pub_rec: int = Field(0, ge=0, description="Number of public records")
    revol_bal: float = Field(0, ge=0, description="Revolving balance")
    revol_util: Optional[float] = Field(None, ge=0, le=200, description="Revolving utilization (%)")
    total_acc: int = Field(1, ge=0, description="Total number of accounts")
    
    # Categorical features
    grade: str = Field(..., description="Loan grade (A-G)")
    home_ownership: str = Field(..., description="Home ownership status")
    verification_status: str = Field(..., description="Income verification status")
    purpose: str = Field(..., description="Purpose of loan")
    emp_length: Optional[str] = Field(None, description="Employment length")
    
    # Additional numeric features
    pub_rec_bankruptcies: int = Field(0, ge=0, description="Number of bankruptcies")
    mort_acc: int = Field(0, ge=0, description="Number of mortgage accounts")
    
    @validator('grade')
    def validate_grade(cls, v):
        if v.upper() not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            raise ValueError('Grade must be A, B, C, D, E, F, or G')
        return v.upper()
    
    @validator('term_months')
    def validate_term(cls, v):
        if v not in [36, 60]:
            raise ValueError('Term must be 36 or 60 months')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000,
                "term_months": 36,
                "int_rate": 12.5,
                "installment": 500,
                "annual_inc": 75000,
                "dti": 18.5,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 35.5,
                "total_acc": 15,
                "grade": "B",
                "home_ownership": "MORTGAGE",
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "emp_length": "5 years",
                "pub_rec_bankruptcies": 0,
                "mort_acc": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    probability_of_default: float = Field(..., description="Probability of loan default (0-1)")
    risk_category: str = Field(..., description="Risk category (LOW, MEDIUM, HIGH)")
    default_prediction: bool = Field(..., description="Binary prediction (True = will default)")
    confidence_score: float = Field(..., description="Model confidence (0-1)")
    
    # Additional information
    loan_amnt: float
    annual_inc: float
    grade: str
    prediction_timestamp: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    
    applications: List[LoanApplication]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse]
    total_applications: int
    high_risk_count: int
    average_default_probability: float


class ModelInfo(BaseModel):
    """Schema for model information."""
    
    model_name: str
    model_version: str
    features_count: int
    training_date: str
    performance_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str
    model_loaded: bool
    timestamp: str


# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="""
    Machine Learning API for predicting loan default risk.
    
    ## Features
    - Single loan prediction
    - Batch predictions
    - Risk categorization (LOW, MEDIUM, HIGH)
    - Probability scores with confidence
    
    ## Model
    The API uses a LightGBM classifier trained on Lending Club data
    to predict the probability of loan default.
    """,
    version="1.0.0",
    contact={
        "name": "Elian Jose",
        "email": "elian.jose12@gmail.com",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
feature_columns = None
model_version = "1.0.0"


# =============================================================================
# STARTUP/SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, feature_columns
    
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Loaded model from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model = None
        
        if FEATURE_COLUMNS_PATH.exists():
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        else:
            logger.warning("Feature columns file not found")
            feature_columns = None
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_features(application: LoanApplication) -> pd.DataFrame:
    """Convert application to feature DataFrame."""
    
    # Create base features from application
    data = {
        'loan_amnt': application.loan_amnt,
        'int_rate': application.int_rate,
        'installment': application.installment,
        'annual_inc': application.annual_inc,
        'dti': application.dti,
        'delinq_2yrs': application.delinq_2yrs,
        'inq_last_6mths': application.inq_last_6mths,
        'open_acc': application.open_acc,
        'pub_rec': application.pub_rec,
        'revol_bal': application.revol_bal,
        'revol_util': application.revol_util if application.revol_util else 0,
        'total_acc': application.total_acc,
        'pub_rec_bankruptcies': application.pub_rec_bankruptcies,
        'mort_acc': application.mort_acc,
    }
    
    # Create engineered features
    data['loan_to_income'] = data['loan_amnt'] / (data['annual_inc'] + 1)
    data['installment_to_income'] = (data['installment'] * 12) / (data['annual_inc'] + 1)
    data['log_annual_inc'] = np.log1p(data['annual_inc'])
    data['log_revol_bal'] = np.log1p(data['revol_bal'])
    
    # Grade encoding
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    data['grade_numeric'] = grade_map.get(application.grade, 4)
    
    # Binary features
    data['is_subprime'] = 1 if application.grade in ['D', 'E', 'F', 'G'] else 0
    data['is_prime'] = 1 if application.grade in ['A', 'B'] else 0
    data['high_dti'] = 1 if application.dti > 20 else 0
    data['very_high_dti'] = 1 if application.dti > 35 else 0
    data['high_revol_util'] = 1 if data['revol_util'] > 80 else 0
    data['has_delinquency'] = 1 if application.delinq_2yrs > 0 else 0
    data['has_public_record'] = 1 if application.pub_rec > 0 else 0
    data['has_bankruptcy'] = 1 if application.pub_rec_bankruptcies > 0 else 0
    data['many_recent_inquiries'] = 1 if application.inq_last_6mths > 3 else 0
    data['is_renter'] = 1 if application.home_ownership == 'RENT' else 0
    data['is_homeowner'] = 1 if application.home_ownership in ['OWN', 'MORTGAGE'] else 0
    data['has_mortgage'] = 1 if application.home_ownership == 'MORTGAGE' else 0
    data['is_verified'] = 1 if application.verification_status != 'Not Verified' else 0
    data['is_long_term'] = 1 if application.term_months == 60 else 0
    data['low_income'] = 1 if application.annual_inc < 30000 else 0
    data['high_income'] = 1 if application.annual_inc > 100000 else 0
    
    # Account features
    data['account_utilization'] = data['open_acc'] / (data['total_acc'] + 1)
    data['closed_accounts'] = data['total_acc'] - data['open_acc']
    data['has_mortgage_account'] = 1 if application.mort_acc > 0 else 0
    
    # Emp length encoding
    emp_map = {'< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    data['emp_length_numeric'] = emp_map.get(application.emp_length, 0) if application.emp_length else 0
    data['is_new_employee'] = 1 if data['emp_length_numeric'] < 2 else 0
    data['is_stable_employee'] = 1 if data['emp_length_numeric'] >= 5 else 0
    
    # Risk scores
    data['simple_risk_score'] = (
        data['is_subprime'] * 2 + 
        data['high_dti'] + 
        data['high_revol_util'] + 
        data['has_delinquency'] * 2 +
        data['is_renter'] * 0.5 +
        data['low_income'] +
        data['is_long_term'] * 0.5
    )
    data['high_risk_flag'] = 1 if data['simple_risk_score'] > 5 else 0
    
    data['stability_score'] = (
        data['is_stable_employee'] +
        data['is_homeowner'] +
        data['is_verified'] +
        (1 if application.inq_last_6mths == 0 else 0)
    )
    
    # Interaction features
    data['grade_dti_interaction'] = data['grade_numeric'] * data['dti']
    data['income_loan_ratio'] = data['annual_inc'] / (data['loan_amnt'] + 1)
    data['subprime_high_dti'] = data['is_subprime'] * data['high_dti']
    data['renter_low_income'] = data['is_renter'] * data['low_income']
    
    df = pd.DataFrame([data])
    
    # Align with feature columns if available
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    
    return df


def get_risk_category(probability: float) -> str:
    """Convert probability to risk category."""
    if probability < 0.15:
        return "LOW"
    elif probability < 0.35:
        return "MEDIUM"
    else:
        return "HIGH"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic information."""
    return {
        "name": "Credit Risk Prediction API",
        "version": model_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(application: LoanApplication):
    """
    Predict default risk for a single loan application.
    
    Returns probability of default, risk category, and additional details.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        X = prepare_features(application)
        
        # Get prediction
        probability = model.predict_proba(X)[0, 1]
        prediction = probability >= 0.5
        
        # Calculate confidence
        confidence = abs(probability - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        
        return PredictionResponse(
            probability_of_default=round(float(probability), 4),
            risk_category=get_risk_category(probability),
            default_prediction=bool(prediction),
            confidence_score=round(float(confidence), 4),
            loan_amnt=application.loan_amnt,
            annual_inc=application.annual_inc,
            grade=application.grade,
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_version,
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict default risk for multiple loan applications.
    
    Returns individual predictions plus summary statistics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.applications) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 applications per batch")
    
    predictions = []
    probabilities = []
    
    for app in request.applications:
        try:
            X = prepare_features(app)
            probability = model.predict_proba(X)[0, 1]
            probabilities.append(probability)
            
            predictions.append(PredictionResponse(
                probability_of_default=round(float(probability), 4),
                risk_category=get_risk_category(probability),
                default_prediction=bool(probability >= 0.5),
                confidence_score=round(float(abs(probability - 0.5) * 2), 4),
                loan_amnt=app.loan_amnt,
                annual_inc=app.annual_inc,
                grade=app.grade,
                prediction_timestamp=datetime.now().isoformat(),
                model_version=model_version,
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            continue
    
    high_risk_count = sum(1 for p in predictions if p.risk_category == "HIGH")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_applications=len(predictions),
        high_risk_count=high_risk_count,
        average_default_probability=round(float(np.mean(probabilities)), 4) if probabilities else 0,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the deployed model."""
    return ModelInfo(
        model_name="LightGBM Credit Risk Classifier",
        model_version=model_version,
        features_count=len(feature_columns) if feature_columns else 0,
        training_date="2024-12-30",  # Update as needed
        performance_metrics={
            "auc": 0.75,  # Update with actual metrics
            "precision": 0.65,
            "recall": 0.70,
            "f1": 0.67,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
