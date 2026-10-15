"""
interactive_predict.py
======================
Interactive command-line interface for Cardiovascular Disease Prediction.

Run directly in terminal:
    python src/interactive_predict.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_cardiovascular_risk, VALID_RANGES


def get_validated_input(prompt: str, cast_type, min_val, max_val, hint: str):
    """Prompt user until a valid value within [min_val, max_val] is provided."""
    while True:
        try:
            raw = input(f"{prompt} [{hint}]: ").strip()
            if not raw:
                print("   ⚠️  Input cannot be empty. Please enter a value.")
                continue
            val = cast_type(raw)
            if not (min_val <= val <= max_val):
                print(f"   ⚠️  Value must be between {min_val} and {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print(f"   ⚠️  Invalid format. Please enter a valid {cast_type.__name__}.")


def main():
    print("=" * 70)
    print("  ❤️  CARDIOVASCULAR DISEASE PREDICTION - USER INPUT CONSOLE")
    print("=" * 70)
    print("Enter patient details below to predict cardiovascular disease risk.\n")

    # 1. Age
    age = get_validated_input("1. Age in years", float, 29, 65, "29 - 65")

    # 2. Gender
    print("\n   [1 = Female, 2 = Male]")
    gender = get_validated_input("2. Gender", int, 1, 2, "1 or 2")

    # 3. Height
    height = get_validated_input("3. Height (cm)", float, 100, 220, "100 - 220 cm")

    # 4. Weight
    weight = get_validated_input("4. Weight (kg)", float, 30, 200, "30 - 200 kg")

    # 5. Systolic BP
    ap_hi = get_validated_input("5. Systolic Blood Pressure (ap_hi)", int, 60, 250, "60 - 250 mmHg")

    # 6. Diastolic BP
    while True:
        ap_lo = get_validated_input("6. Diastolic Blood Pressure (ap_lo)", int, 40, 180, "40 - 180 mmHg")
        if ap_lo >= ap_hi:
            print(f"   ⚠️  Diastolic pressure ({ap_lo}) must be strictly less than Systolic ({ap_hi})! Try again.")
        else:
            break

    # 7. Cholesterol
    print("\n   [1 = Normal, 2 = Above Normal, 3 = Well Above Normal]")
    chol = get_validated_input("7. Cholesterol Level", int, 1, 3, "1, 2, or 3")

    # 8. Glucose
    print("\n   [1 = Normal, 2 = Above Normal, 3 = Well Above Normal]")
    gluc = get_validated_input("8. Glucose Level", int, 1, 3, "1, 2, or 3")

    # 9. Smoke
    print("\n   [0 = Non-Smoker, 1 = Smoker]")
    smoke = get_validated_input("9. Smoking Status", int, 0, 1, "0 or 1")

    # 10. Alcohol
    print("\n   [0 = No Alcohol, 1 = Alcohol Consumer]")
    alco = get_validated_input("10. Alcohol Intake", int, 0, 1, "0 or 1")

    # 11. Physical Activity
    print("\n   [0 = Inactive, 1 = Physically Active]")
    active = get_validated_input("11. Physical Activity", int, 0, 1, "0 or 1")

    model_path = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")

    print("\n" + "-" * 70)
    print("⏳ Processing patient attributes with Machine Learning model...")
    print("-" * 70)

    try:
        res = predict_cardiovascular_risk(
            age_years=age,
            gender=gender,
            height_cm=height,
            weight_kg=weight,
            ap_hi=ap_hi,
            ap_lo=ap_lo,
            cholesterol=chol,
            gluc=gluc,
            smoke=smoke,
            alco=alco,
            active=active,
            model_path=model_path,
        )

        prob_pct = res["cvd_probability"] * 100
        bmi = res["input_summary"]["bmi"]

        print("\n" + "=" * 70)
        print("                  🩺 PREDICTION RESULTS")
        print("=" * 70)
        print(f"  • Calculated BMI           : {bmi:.2f} kg/m²")
        print(f"  • Predicted Risk Level     : {res['risk_label'].upper()}")
        print(f"  • Estimated CVD Probability: {prob_pct:.1f}%")
        print(f"  • Model Decision Class     : {'1 (Positive - CVD present)' if res['predicted_class'] == 1 else '0 (Negative - No CVD)'}")
        print("=" * 70)
        print(f"\n{res['disclaimer']}\n")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")


if __name__ == "__main__":
    main()
