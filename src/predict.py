"""
predict.py
==========
Reusable prediction function for the Cardiovascular Disease Prediction model.

Usage:
    from src.predict import predict_cardiovascular_risk

    result = predict_cardiovascular_risk(
        age_years   = 55,
        gender      = 2,       # 1 = female, 2 = male (dataset encoding)
        height_cm   = 170,
        weight_kg   = 80,
        ap_hi       = 140,
        ap_lo       = 90,
        cholesterol = 2,       # 1=normal, 2=above normal, 3=well above normal
        gluc        = 1,
        smoke       = 0,
        alco        = 0,
        active      = 1,
        model_path  = "models/heart_disease_model.joblib",
    )
    print(result)
"""

import os
import numpy as np
import joblib


# ---------------------------------------------------------------------------
# Input validation ranges (same thresholds as preprocessing.py)
# ---------------------------------------------------------------------------
VALID_RANGES = {
    "age_years":   (29, 65,    "Age in years"),
    "gender":      (1, 2,      "Gender (1=female, 2=male)"),
    "height_cm":   (100, 220,  "Height in cm"),
    "weight_kg":   (30, 200,   "Weight in kg"),
    "ap_hi":       (60, 250,   "Systolic blood pressure (mmHg)"),
    "ap_lo":       (40, 180,   "Diastolic blood pressure (mmHg)"),
    "cholesterol": (1, 3,      "Cholesterol level (1/2/3)"),
    "gluc":        (1, 3,      "Glucose level (1/2/3)"),
    "smoke":       (0, 1,      "Smoking (0=no, 1=yes)"),
    "alco":        (0, 1,      "Alcohol (0=no, 1=yes)"),
    "active":      (0, 1,      "Physical activity (0=no, 1=yes)"),
}

# Feature order must exactly match the order used during training
FEATURE_ORDER = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "bmi",
]


def _validate_inputs(**kwargs) -> None:
    """
    Validate every input against known plausible ranges.
    Raises ValueError with a descriptive message on the first invalid input.
    Also checks that diastolic < systolic blood pressure.
    """
    # Map external kwarg names to validation-range keys
    key_map = {
        "age_years":   "age_years",
        "gender":      "gender",
        "height_cm":   "height_cm",
        "weight_kg":   "weight_kg",
        "ap_hi":       "ap_hi",
        "ap_lo":       "ap_lo",
        "cholesterol": "cholesterol",
        "gluc":        "gluc",
        "smoke":       "smoke",
        "alco":        "alco",
        "active":      "active",
    }
    for kwarg_name, range_key in key_map.items():
        value = kwargs[kwarg_name]
        lo, hi, label = VALID_RANGES[range_key]
        if not (lo <= value <= hi):
            raise ValueError(
                f"Invalid value for '{kwarg_name}' ({label}): "
                f"got {value}, expected between {lo} and {hi}."
            )

    # Physiological check
    if kwargs["ap_lo"] >= kwargs["ap_hi"]:
        raise ValueError(
            f"Diastolic pressure (ap_lo={kwargs['ap_lo']}) must be strictly "
            f"less than systolic pressure (ap_hi={kwargs['ap_hi']})."
        )


def predict_cardiovascular_risk(
    age_years:   float,
    gender:      int,
    height_cm:   float,
    weight_kg:   float,
    ap_hi:       int,
    ap_lo:       int,
    cholesterol: int,
    gluc:        int,
    smoke:       int,
    alco:        int,
    active:      int,
    model_path:  str = "models/heart_disease_model.joblib",
) -> dict:
    """
    Predict cardiovascular disease risk for a single individual.

    ⚠️  IMPORTANT DISCLAIMER
    ========================
    This function returns a prediction from a machine-learning model trained
    on a population-level dataset.  It is NOT a medical diagnosis.  It should
    NOT be used to make clinical decisions.  Always consult a qualified
    healthcare professional for any health concerns.

    Parameters
    ----------
    age_years   : float – Age in years (29–65).
    gender      : int   – 1 = female, 2 = male (dataset encoding).
    height_cm   : float – Height in centimetres (100–220).
    weight_kg   : float – Weight in kilograms (30–200).
    ap_hi       : int   – Systolic blood pressure in mmHg (60–250).
    ap_lo       : int   – Diastolic blood pressure in mmHg (40–180).
    cholesterol : int   – 1 = normal, 2 = above normal, 3 = well above normal.
    gluc        : int   – 1 = normal, 2 = above normal, 3 = well above normal.
    smoke       : int   – 0 = non-smoker, 1 = smoker.
    alco        : int   – 0 = no alcohol, 1 = alcohol consumer.
    active      : int   – 0 = not physically active, 1 = physically active.
    model_path  : str   – Path to the saved joblib pipeline.

    Returns
    -------
    dict with keys:
        predicted_class      : int  – 0 (no CVD) or 1 (CVD present).
        cvd_probability      : float – Estimated probability of CVD [0, 1].
        risk_label           : str   – "Low Risk" or "High Risk".
        disclaimer           : str   – Safety notice.
        input_summary        : dict  – Echo of derived BMI alongside inputs.
    """
    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    _validate_inputs(
        age_years=age_years, gender=gender, height_cm=height_cm,
        weight_kg=weight_kg, ap_hi=ap_hi, ap_lo=ap_lo,
        cholesterol=cholesterol, gluc=gluc, smoke=smoke,
        alco=alco, active=active,
    )

    # -----------------------------------------------------------------------
    # Derive BMI (same formula as preprocessing)
    # -----------------------------------------------------------------------
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)

    # -----------------------------------------------------------------------
    # Assemble feature vector in exact training order
    # -----------------------------------------------------------------------
    feature_vector = np.array([[
        round(age_years, 1),   # age (years, as cleaned)
        gender,
        height_cm,
        weight_kg,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        bmi,
    ]])

    # -----------------------------------------------------------------------
    # Load model and predict
    # -----------------------------------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Run train_model.py first to generate the model."
        )

    pipeline = joblib.load(model_path)
    predicted_class = int(pipeline.predict(feature_vector)[0])
    cvd_probability = float(pipeline.predict_proba(feature_vector)[0, 1])

    # -----------------------------------------------------------------------
    # Build result
    # -----------------------------------------------------------------------
    result = {
        "predicted_class":  predicted_class,
        "cvd_probability":  round(cvd_probability, 4),
        "risk_label":       "High Risk" if predicted_class == 1 else "Low Risk",
        "disclaimer": (
            "⚠️  This is a machine-learning prediction based on statistical patterns "
            "in a population dataset.  It is NOT a clinical diagnosis.  "
            "Please consult a qualified healthcare professional for medical advice."
        ),
        "input_summary": {
            "age_years":   age_years,
            "gender":      "Male" if gender == 2 else "Female",
            "height_cm":   height_cm,
            "weight_kg":   weight_kg,
            "bmi":         bmi,
            "ap_hi":       ap_hi,
            "ap_lo":       ap_lo,
            "cholesterol": cholesterol,
            "gluc":        gluc,
            "smoke":       smoke,
            "alco":        alco,
            "active":      active,
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")

    print("=== Prediction Function Demo ===\n")

    # Example 1 – healthy profile
    try:
        r1 = predict_cardiovascular_risk(
            age_years=40, gender=1, height_cm=165, weight_kg=60,
            ap_hi=110, ap_lo=70, cholesterol=1, gluc=1,
            smoke=0, alco=0, active=1,
            model_path=MODEL_PATH,
        )
        print("Profile 1 – Healthy 40-year-old:")
        print(f"  Risk label   : {r1['risk_label']}")
        print(f"  CVD Prob     : {r1['cvd_probability']:.4f}")
        print(f"  Predicted    : {r1['predicted_class']}")
        print(f"  BMI          : {r1['input_summary']['bmi']}")
        print(f"  Disclaimer   : {r1['disclaimer']}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Example 2 – high-risk profile
    try:
        r2 = predict_cardiovascular_risk(
            age_years=60, gender=2, height_cm=175, weight_kg=100,
            ap_hi=180, ap_lo=110, cholesterol=3, gluc=3,
            smoke=1, alco=1, active=0,
            model_path=MODEL_PATH,
        )
        print("Profile 2 – High-risk 60-year-old:")
        print(f"  Risk label   : {r2['risk_label']}")
        print(f"  CVD Prob     : {r2['cvd_probability']:.4f}")
        print(f"  Predicted    : {r2['predicted_class']}")
        print(f"  BMI          : {r2['input_summary']['bmi']}")
        print(f"  Disclaimer   : {r2['disclaimer']}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Example 3 – invalid input (should raise ValueError)
    print("Profile 3 – Invalid input test (ap_lo > ap_hi):")
    try:
        predict_cardiovascular_risk(
            age_years=50, gender=1, height_cm=160, weight_kg=70,
            ap_hi=80, ap_lo=120,   # diastolic > systolic → invalid
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            model_path=MODEL_PATH,
        )
    except ValueError as e:
        print(f"  Correctly caught ValueError: {e}\n")
