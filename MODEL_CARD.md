# Model Card: Credit Risk Default Prediction

## Model Details

### Basic Information
- **Model Name:** Credit Risk Default Classifier
- **Model Version:** 1.0.0
- **Model Type:** Binary Classification (LightGBM)
- **Task:** Predict loan default probability
- **Training Date:** December 2024
- **Developer:** Elian Jose

### Intended Use
- **Primary Use:** Assess credit risk for loan applications
- **Primary Users:** Risk analysts, loan officers, portfolio managers
- **Out-of-Scope:** Real-time trading decisions, regulatory capital calculations

## Training Data

### Dataset
- **Source:** Lending Club Loan Data (2007-2018)
- **Size:** ~2.2 million loans
- **Features:** 50+ features covering loan, borrower, and credit characteristics

### Data Split
- **Training:** 64% (2007-2017)
- **Validation:** 16% (held out from training)
- **Test:** 20% (2018, time-based split)

### Target Variable
- **Definition:** Binary indicator of loan default
- **Positive Class (1):** Charged Off, Default, Late (31-120 days)
- **Negative Class (0):** Fully Paid

### Class Distribution
- Default Rate: ~20% (imbalanced)
- Handled via: Class weighting (scale_pos_weight)

## Features

### Feature Categories
1. **Loan Characteristics** (8 features)
   - loan_amnt, term, int_rate, installment, grade, sub_grade
   
2. **Borrower Profile** (6 features)
   - annual_inc, emp_length, home_ownership, verification_status
   
3. **Credit History** (10 features)
   - delinq_2yrs, inq_last_6mths, open_acc, pub_rec, total_acc
   
4. **Debt Profile** (8 features)
   - dti, revol_bal, revol_util, mort_acc
   
5. **Engineered Features** (20+ features)
   - loan_to_income, grade_numeric, simple_risk_score, etc.

### Important Features (by SHAP importance)
1. int_rate (Interest Rate)
2. grade_numeric (Loan Grade)
3. dti (Debt-to-Income)
4. annual_inc (Annual Income)
5. revol_util (Revolving Utilization)

## Model Performance

### Validation Metrics
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.75-0.78 |
| Precision | 0.60-0.65 |
| Recall | 0.65-0.70 |
| F1 Score | 0.62-0.67 |

### Test Metrics (2018 data)
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.74-0.77 |
| Precision | 0.58-0.63 |
| Recall | 0.63-0.68 |
| F1 Score | 0.60-0.65 |

### Threshold Analysis
| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.30 | 0.35 | 0.85 | 0.50 |
| 0.50 | 0.60 | 0.65 | 0.62 |
| 0.70 | 0.75 | 0.40 | 0.52 |

## Limitations

### Known Limitations
1. **Temporal Drift:** Model trained on 2007-2017 data may not reflect current lending patterns
2. **Geographic Bias:** Lending Club operates primarily in the US
3. **Economic Conditions:** Model may underperform during economic crises (trained pre-COVID)
4. **Feature Availability:** Some features may not be available at loan application time

### What the Model Cannot Do
- Predict exact dollar amount of loss
- Account for macroeconomic shocks
- Replace human judgment in edge cases
- Guarantee regulatory compliance

## Ethical Considerations

### Fairness Analysis
- Model does NOT use protected attributes (race, gender) as features
- Geographic features (addr_state) may encode socioeconomic disparities
- Income-based features may correlate with protected classes

### Bias Mitigation
- Regular monitoring of prediction disparities across demographics
- Threshold calibration to ensure equitable false positive rates
- Human review for borderline cases

### Recommendations
- Conduct regular fairness audits
- Combine with human review for high-stakes decisions
- Monitor for performance drift across subgroups

## Caveats and Recommendations

### When to Use
✅ Screening large volumes of applications
✅ Prioritizing applications for manual review
✅ Portfolio risk assessment
✅ Stress testing and scenario analysis

### When NOT to Use
❌ As sole basis for credit decisions
❌ Without human oversight
❌ In jurisdictions with different lending regulations
❌ For loan types not in training data (e.g., mortgages)

## Monitoring

### Recommended Monitoring
1. **Data Drift:** Monitor feature distributions weekly
2. **Performance:** Track actual vs predicted default rates monthly
3. **Fairness:** Audit approval rates across subgroups quarterly

### Retraining Triggers
- ROC-AUC drops below 0.70
- Data drift score exceeds 0.15
- Significant change in macroeconomic conditions

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2024 | Initial release |

## Contact

For questions or issues:
- **Developer:** Elian Jose
- **Email:** elian.jose12@gmail.com
