# ❤️ Machine Learning-Based Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **⚠️ Medical Disclaimer:** This project is developed for educational and internship purposes.  
> All model predictions are based on statistical patterns in a population-level dataset and are  
> **NOT clinical diagnoses**. Never use this tool to make healthcare decisions.  
> Always consult a qualified healthcare professional.

---

## 📋 Project Overview

This project implements a complete, end-to-end **Machine Learning pipeline** for predicting
cardiovascular disease (CVD) risk from patient health and lifestyle attributes. The system
trains, compares, and tunes five classification algorithms, selecting the best performer
using rigorous cross-validation and held-out test evaluation.

---

## 🎯 Problem Statement

Cardiovascular diseases are the leading cause of death globally. Early identification of
at-risk individuals enables timely intervention. This project builds a screening-support tool
that flags potential CVD risk based on measurable health attributes, **not** as a diagnostic
device but as a decision-support aid to guide further clinical investigation.

---

## 🏆 Objectives

1. Load, inspect, and thoroughly clean the cardiovascular disease dataset.
2. Perform detailed Exploratory Data Analysis (EDA) with professional visualisations.
3. Compute Pearson and Spearman correlation analyses.
4. Train and cross-validate five ML algorithms under identical conditions.
5. Tune hyperparameters using GridSearchCV on training data only.
6. Evaluate all models on a held-out test set and create a ranked comparison table.
7. Select the best model, save the pipeline, and expose a reusable prediction function.
8. Document every decision to ensure reproducibility and academic integrity.

---

## 📊 Dataset Description

| Field | Description |
|-------|-------------|
| **Source** | Cardiovascular Disease Dataset (`cardio_train.csv`) |
| **Separator** | Semicolon (`;`) |
| **Raw size** | 70,000 rows × 13 columns |
| **Missing values** | None |
| **Target** | `cardio` (0 = no recorded CVD, 1 = CVD present) |
| **Class balance** | ~50/50 (near-perfectly balanced) |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Age in days (converted to years during preprocessing) |
| `gender` | int | 1 = female, 2 = male |
| `height` | int | Height in centimetres |
| `weight` | float | Weight in kilograms |
| `ap_hi` | int | Systolic blood pressure (mmHg) |
| `ap_lo` | int | Diastolic blood pressure (mmHg) |
| `cholesterol` | int | 1=normal, 2=above normal, 3=well above normal |
| `gluc` | int | 1=normal, 2=above normal, 3=well above normal |
| `smoke` | int | 0=non-smoker, 1=smoker |
| `alco` | int | 0=no alcohol, 1=alcohol consumer |
| `active` | int | 0=inactive, 1=physically active |
| `cardio` | int | **Target**: 0=No CVD, 1=CVD |

---

## 🔧 Data Preprocessing Steps

| Step | Action | Rows Affected |
|------|--------|---------------|
| 1 | Load (semicolon-separated CSV) | 70,000 rows |
| 2 | Drop `id` column (surrogate key) | — |
| 3 | Remove duplicate records | 24 removed |
| 4 | Convert age: days → years (÷ 365.25) | — |
| 5 | Remove invalid blood pressure (non-positive, extreme, inverted) | ~1,329 removed |
| 6 | Remove implausible height/weight | 33 removed |
| 7 | Calculate BMI = weight / (height_m)² | New column added |
| 8 | Remove implausible BMI (< 10 or > 70) | 11 removed |
| **Final** | **Cleaned dataset** | **68,603 rows** |

**Key design principles:**
- Raw DataFrame is never mutated; all cleaning on a working copy.
- Outliers removed using **domain-informed medical plausibility rules**, not blind statistical thresholds.
- StandardScaler fitted **only on training folds** (inside sklearn Pipelines) to prevent data leakage.

---

## 📈 EDA Findings

| Finding | Observation |
|---------|-------------|
| **Age** | CVD prevalence rises steeply with age (~25% at 30-35 years to ~65% at 60-65 years) |
| **Blood Pressure** | CVD patients consistently show higher systolic and diastolic BP |
| **Cholesterol** | Monotonic increase in CVD prevalence with cholesterol level |
| **Glucose** | Higher glucose associated with higher CVD prevalence |
| **BMI** | CVD patients have modestly higher median BMI |
| **Physical Activity** | Active individuals show lower CVD prevalence |
| **Smoking/Alcohol** | Weak associations in this dataset (possible self-reporting bias) |
| **Class Balance** | ~50/50 split — no class-weight adjustment required |

> **Note:** All findings describe **associations**, not causal relationships.

---

## 🤖 Algorithms Used

| Algorithm | Scaling | Cross-validation Note |
|-----------|---------|----------------------|
| Logistic Regression | ✅ StandardScaler | Full training set |
| K-Nearest Neighbors (KNN) | ✅ StandardScaler | Full training set |
| Decision Tree | ❌ (scale-invariant) | Full training set |
| Random Forest | ❌ (scale-invariant) | Full training set |
| SVM (RBF kernel) | ✅ StandardScaler | Stratified 20% subsample (documented) |

---

## 📊 Model Comparison Table

*(Generated from actual execution — see `outputs/model_comparison.csv`)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Random Forest** (Selected) | **0.7425** | **0.7624** | **0.6967** | **0.7281** | **0.8061** |
| **KNN** | 0.7358 | 0.7602 | 0.6808 | 0.7183 | 0.7994 |
| **Decision Tree** | 0.7354 | 0.7622 | 0.6761 | 0.7166 | 0.7981 |
| **Logistic Regression** | 0.7338 | 0.7600 | 0.6751 | 0.7150 | 0.7960 |
| **SVM (RBF Kernel)** | 0.7322 | 0.7477 | 0.6926 | 0.7191 | 0.7897 |

> *All metrics evaluated on the held-out test split (13,721 samples) after 5-fold Stratified CV hyperparameter tuning.*

---

## 🏅 Final Model

The model with the highest ROC-AUC and strong Recall is automatically selected.
The saved pipeline is stored at `models/heart_disease_model.joblib`.

**Selection rationale:** For cardiovascular disease *screening*, **recall (sensitivity)**
is prioritised because a false negative (missing a true CVD case) is far more harmful
than a false positive (unnecessary follow-up). ROC-AUC is used as primary ranking
because it reflects discriminative ability across all decision thresholds.

---

## 🚀 Installation & Execution

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### 1. Clone / set up project
```bash
cd D:\Project\heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline
```bash
set PYTHONIOENCODING=utf-8
python src/train_model.py
```

This will:
- Clean the data
- Train and tune all 5 models
- Save `models/heart_disease_model.joblib`
- Save `outputs/model_comparison.csv`
- Save all figures to `outputs/figures/`

### 4. Launch the Jupyter notebook
```bash
python generate_notebook.py     # regenerate notebook (if needed)
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

Or run the notebook non-interactively:
```bash
set PYTHONIOENCODING=utf-8
jupyter nbconvert --to notebook --execute notebooks/heart_disease_analysis.ipynb --output notebooks/heart_disease_analysis_executed.ipynb
```

### 4. Interactive Terminal Prediction (Console)
Prompt-based interactive console with real-time input validation:
```bash
python src/interactive_predict.py
```

### 5. Programmatic Python Prediction
```python
from src.predict import predict_cardiovascular_risk

result = predict_cardiovascular_risk(
    age_years=55, gender=2, height_cm=175, weight_kg=88,
    ap_hi=150, ap_lo=95, cholesterol=2, gluc=1,
    smoke=0, alco=0, active=1,
    model_path="models/heart_disease_model.joblib",
)
print(result["risk_label"])       # "High Risk" or "Low Risk"
print(result["cvd_probability"])  # e.g. 0.7234
print(result["disclaimer"])       # safety notice
```

---

## 📁 Project Structure

```
heart-disease-prediction/
├── data/
│   └── cardio_train.csv              # Raw dataset
├── notebooks/
│   └── heart_disease_analysis.ipynb  # Complete analysis notebook
├── src/
│   ├── preprocessing.py              # Data loading & cleaning pipeline
│   ├── train_model.py                # Training, tuning, evaluation & saving
│   └── predict.py                    # Reusable prediction function
├── models/
│   └── heart_disease_model.joblib    # Saved final pipeline (generated)
├── outputs/
│   ├── figures/                      # All EDA and model plots (generated)
│   └── model_comparison.csv          # Metric comparison table (generated)
├── generate_notebook.py              # Script to regenerate the .ipynb
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
1. **Dataset bias:** The dataset comes from a single population and may not generalise globally.
2. **Self-reported features:** Smoking and alcohol are self-reported and prone to under-reporting.
3. **No temporal data:** Snapshot measurements; disease progression not captured.
4. **Ordinal encoding:** Cholesterol and glucose are ordinal (1/2/3), losing granularity vs. actual values.
5. **Binary classification only:** No severity or probability calibration.

### Future Improvements
1. **Feature engineering:** Pulse pressure (ap_hi − ap_lo), age-BMI interaction features.
2. **Ensemble stacking:** Combine multiple model predictions.
3. **Probability calibration:** Platt scaling or isotonic regression for calibrated probabilities.
4. **SHAP explainability:** Per-prediction feature importance for clinical interpretability.
5. **External validation:** Test on an independent dataset from a different population.
6. **Deep learning:** Tabular DL models (TabNet, SAINT) for comparison.
7. **Clinical integration:** Threshold optimisation for specific recall targets.

---

## 📝 Internship Report Outline

### Abstract
Brief summary of the project goals, methodology, and key results.

### 1. Introduction
- Background on cardiovascular disease prevalence
- Motivation for ML-based screening tools
- Project scope and objectives

### 2. Literature Review
- ML applications in CVD risk prediction
- Existing datasets and benchmarks
- Evaluation metrics used in medical ML

### 3. Methodology
- Dataset description and preprocessing strategy
- Feature engineering (age conversion, BMI)
- Outlier handling rationale
- Model selection rationale
- Evaluation framework (stratified CV, held-out test)

### 4. Implementation
- Data pipeline (`preprocessing.py`)
- Model pipeline with `sklearn` Pipelines
- Hyperparameter tuning strategy
- Anti-leakage measures

### 5. Results & Discussion
- EDA findings and visualisations
- Cross-validation comparison table
- Test-set metrics and confusion matrices
- ROC curves and AUC interpretation
- False positive/negative trade-offs in CVD screening

### 6. Conclusion
- Summary of findings
- Best model and justification
- Clinical relevance (with disclaimer)

### 7. Future Scope
- As listed in the Limitations section above

### References
- Dataset source
- scikit-learn documentation
- Relevant cardiovascular ML papers

---

*Project developed for internship demonstration. Not intended for clinical use.*
