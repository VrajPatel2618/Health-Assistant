"""
app.py
======
Interactive Streamlit Web Application for Cardiovascular Disease Prediction.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_cardiovascular_risk

st.set_page_config(
    page_title="Cardiovascular Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header
st.title("❤️ Machine Learning Cardiovascular Disease Prediction System")
st.markdown(
    "Predict the probability of cardiovascular disease using patient physiological "
    "and lifestyle attributes via our tuned **Random Forest ML Pipeline**."
)
st.write("---")

# Layout: 2 Columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographic & Body Metrics")
    age = st.slider("Age (years)", min_value=29, max_value=65, value=50, step=1)
    gender_label = st.radio("Gender", options=["Female", "Male"], horizontal=True)
    gender = 1 if gender_label == "Female" else 2

    height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=165.0, step=1.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

    # Derived live BMI
    bmi = weight / ((height / 100.0) ** 2)
    st.info(f"**Computed BMI:** `{bmi:.2f} kg/m²`")

with col2:
    st.subheader("🩺 Clinical & Laboratory Measurements")
    ap_hi = st.slider("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=125, step=1)
    ap_lo = st.slider("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=180, value=80, step=1)

    if ap_lo >= ap_hi:
        st.error("⚠️ Diastolic pressure must be strictly less than Systolic pressure!")

    chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    chol_label = st.selectbox("Cholesterol Level", options=list(chol_map.keys()))
    cholesterol = chol_map[chol_label]

    gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gluc_label = st.selectbox("Glucose Level", options=list(gluc_map.keys()))
    gluc = gluc_map[gluc_label]

st.subheader("🏃 Lifestyle Attributes")
c_life1, c_life2, c_life3 = st.columns(3)
with c_life1:
    smoke = 1 if st.checkbox("Smoker") else 0
with c_life2:
    alco = 1 if st.checkbox("Alcohol Consumer") else 0
with c_life3:
    active = 1 if st.checkbox("Physically Active", value=True) else 0

st.write("---")

# Predict button
if st.button("🔍 Predict Cardiovascular Risk", type="primary", use_container_width=True):
    if ap_lo >= ap_hi:
        st.error("Cannot predict: Diastolic pressure must be less than Systolic pressure.")
    else:
        model_path = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")
        try:
            res = predict_cardiovascular_risk(
                age_years=age,
                gender=gender,
                height_cm=height,
                weight_kg=weight,
                ap_hi=ap_hi,
                ap_lo=ap_lo,
                cholesterol=cholesterol,
                gluc=gluc,
                smoke=smoke,
                alco=alco,
                active=active,
                model_path=model_path,
            )

            prob = res["cvd_probability"]
            label = res["risk_label"]

            st.markdown("### 📊 Prediction Result")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if label == "High Risk":
                    st.error(f"## 🚨 {label}")
                else:
                    st.success(f"## 🟢 {label}")
                st.metric(label="Estimated Risk Probability", value=f"{prob * 100:.1f}%")

            with res_col2:
                st.progress(float(prob))
                st.write(
                    f"The machine learning model estimates a **{prob * 100:.1f}% probability** "
                    f"of cardiovascular disease presence based on the provided parameters."
                )
                st.caption(res["disclaimer"])

        except Exception as e:
            st.error(f"Error predicting risk: {e}")

st.write("---")
st.markdown(
    "<small>⚠️ **Disclaimer**: This application is a machine learning academic demonstration. "
    "It does not provide medical diagnoses or replace clinical examinations by medical practitioners.</small>",
    unsafe_allow_html=True,
)
