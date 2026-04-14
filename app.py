import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="DOMUS AI", layout="centered")

st.title("DOMUS AI")
st.divider()

# Load dataset
data = pd.read_csv("clean_data.csv")

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# User Inputs
area = st.number_input("Enter Area (sq ft)", min_value=0, value=1000)
bhk = st.number_input("Enter BHK", min_value=0, value=2)
bathroom = st.number_input("Enter Bathrooms", min_value=0, value=2)
age = st.number_input("Enter Property Age", min_value=0, value=5)

status = st.selectbox("Select Status", data["status"].unique())
location = st.selectbox("Select Location", sorted(data["location"].unique()))
builder = st.selectbox("Select Builder", sorted(data["builder"].unique()))

st.divider()

# ✅ SINGLE BUTTON ONLY
if st.button("Predict Price", key="predict_main"):

    # Input Data
    input_data = pd.DataFrame({
        "area": [area],
        "status": [status],
        "bhk": [bhk],
        "bathroom": [bathroom],
        "age": [age],
        "location": [location],
        "builder": [builder]
    })

    # Convert categorical
    input_data = pd.get_dummies(input_data)

    # Match training features
    input_data = input_data.reindex(columns=features, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Show Price
    st.success(f"💰 Predicted House Price: ₹ {prediction:.2f} Lakhs")

# ---------------- GOOGLE MAP ----------------
st.subheader("📍 Property Location")

search_query = f"{builder}, {location}"
map_url = f"https://www.google.com/maps?q={search_query}&output=embed"

st.components.v1.iframe(map_url, height=400)

st.divider()