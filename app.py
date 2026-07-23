
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page title
st.set_page_config(page_title='CA House Price Predictor')

# Load the saved scaler and model
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('mlp_model.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Feature names matching the training set
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']

st.title("California House Price Predictor")
st.write("Enter the following details to estimate the median house value:")

# Create a layout with two columns for input fields
col1, col2 = st.columns(2)
input_data = []

for i, feat in enumerate(features):
    with col1 if i % 2 == 0 else col2:
        val = st.number_input(f"{feat.replace('_', ' ').title()}", value=0.0)
        input_data.append(val)

if st.button("Predict Price"):
    # Prepare data for prediction
    X_input = np.array([input_data])
    X_scaled = scaler.transform(X_input)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    
    # Display result
    st.success(f"### Estimated House Value: ${prediction[0]:,.2f}")
