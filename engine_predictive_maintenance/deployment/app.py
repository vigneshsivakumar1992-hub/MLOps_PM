import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ------------------------------
# Load model (cached)
# ------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Vignesh-vigu/Engine-Predictive-Maintenance",
        filename="engine_predictive_maintenance_model.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔧 Engine Predictive Maintenance App")

st.write("""
This application predicts whether an **engine requires maintenance**
based on real-time sensor parameters.
""")

# ------------------------------
# Input fields (DATA-DRIVEN FROM CSV STATS)
# ------------------------------
engine_rpm = st.number_input(
    "Engine RPM",
    min_value=61.0,
    max_value=2239.0,
    value=800.0,
    step=10.0
)

lub_oil_pressure = st.number_input(
    "Lubricating Oil Pressure (bar)",
    min_value=0.0,
    max_value=7.3,
    value=3.3,
    step=0.1
)

fuel_pressure = st.number_input(
    "Fuel Pressure (bar)",
    min_value=0.0,
    max_value=21.2,
    value=6.6,
    step=0.1
)

coolant_pressure = st.number_input(
    "Coolant Pressure (bar)",
    min_value=0.0,
    max_value=7.5,
    value=2.3,
    step=0.1
)

lub_oil_temp = st.number_input(
    "Lubricating Oil Temperature (°C)",
    min_value=71.0,
    max_value=90.0,
    value=77.0,
    step=0.5
)

coolant_temp = st.number_input(
    "Coolant Temperature (°C)",
    min_value=61.0,
    max_value=196.0,
    value=78.0,
    step=0.5
)

# ------------------------------
# Prepare input (NORMALIZED)
# ------------------------------
input_df = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Maintenance Requirement"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.45)

    st.subheader("Prediction Result")
    st.write(f"Failure Probability: **{proba:.2f}**")

    if prediction == 1:
        st.error("⚠️ Maintenance Required")
    else:
        st.success("✅ Engine Operating Normally")


