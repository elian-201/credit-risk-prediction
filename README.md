# Credit Risk Default Prediction Model

End-to-end ML pipeline for predicting loan default risk using the Lending Club dataset.

## Project Overview

This project demonstrates the complete ML model lifecycle for credit risk assessment:
1. **Data Engineering** - Ingestion, validation, feature store
2. **Model Development** - Feature engineering, training, experiment tracking
3. **Model Validation** - Cross-validation, backtesting, fairness analysis
4. **Deployment** - FastAPI service, Docker containerization, AWS deployment
5. **Monitoring** - Drift detection, performance tracking, alerting

## Problem Statement

Predict whether a loan applicant will **default on their loan** based on their credit history, financial profile, and loan characteristics. This is a binary classification problem critical to banking risk management.

**Target Variable:** `loan_status` - Whether the borrower defaulted (1) or fully paid (0)

## Business Context

Credit risk modeling is fundamental to banking operations:
- **Loan Approval Decisions**: Should we approve this application?
- **Interest Rate Pricing**: What rate appropriately compensates for risk?
- **Portfolio Management**: How much capital should we reserve for losses?
- **Regulatory Compliance**: Basel III and IFRS 9 require robust risk models

## Dataset

**Source:** Lending Club Loan Data (2007-2018)
- **Size:** 2+ million loans
- **Features:** 150+ variables
- **Download:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

### Key Features

| Category | Features |
|----------|----------|
| Loan Characteristics | loan_amount, term, interest_rate, installment, grade |
| Borrower Profile | annual_income, employment_length, home_ownership |
| Credit History | credit_score, open_accounts, delinquencies, bankruptcies |
| Debt Profile | debt_to_income, revolving_utilization, total_debt |

## Project Structure

```
credit-risk-prediction/
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/                    # Original Lending Club data
│   ├── processed/              # Cleaned, validated data
│   └── features/               # Feature store
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_fairness_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Data loading and preprocessing
│   │   ├── validation.py       # Pandera schemas, quality checks
│   │   └── config.py           # Configuration settings
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training with MLflow
│   │   ├── predict.py          # Inference logic
│   │   └── evaluate.py         # Model evaluation metrics
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Custom metrics
│   │   └── fairness.py         # Bias and fairness analysis
│   └── monitoring/
│       ├── __init__.py
│       └── drift_detection.py  # Evidently AI integration
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── models/                     # Serialized models
├── reports/                    # Evaluation and monitoring reports
└── tests/
    ├── test_data_validation.py
    ├── test_features.py
    ├── test_model.py
    └── test_api.py
```

## Models

1. **Baseline:** Logistic Regression (interpretable benchmark)
2. **Primary:** LightGBM Classifier (high performance)
3. **Comparison:** XGBoost Classifier
4. **Deep Learning:** Simple Neural Network (optional)

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| ROC-AUC | Overall discrimination ability | > 0.75 |
| Precision | Of predicted defaults, how many actually defaulted | > 0.60 |
| Recall | Of actual defaults, how many did we catch | > 0.70 |
| F1 Score | Balance of precision and recall | > 0.65 |
| KS Statistic | Maximum separation between classes | > 0.40 |

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd credit-risk-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data (see data/README.md for instructions)
# Place accepted_loans.csv in data/raw/

# Run preprocessing
python src/data/ingestion.py

# Train model
python src/models/train.py

# Run API locally
uvicorn app.main:app --reload

# Run with Docker
docker-compose up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single loan prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predict/explain` | POST | Prediction with SHAP explanation |
| `/model/info` | GET | Model metadata |

## Fairness Considerations

Credit risk models must be fair across protected classes. We evaluate:
- **Demographic Parity**: Equal approval rates across groups
- **Equal Opportunity**: Equal true positive rates
- **Predictive Parity**: Equal precision across groups

## Author

Elian Jose | Data Analyst | ITUS Capital Advisors

## License

MIT
