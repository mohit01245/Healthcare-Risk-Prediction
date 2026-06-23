import streamlit as st
import pandas as pd
import joblib

model = joblib.load('risk_model.joblib')

st.title("Healthcare Risk Prediction")

age = st.number_input("Enter Age", min_value=0)
length_of_stay = st.number_input("Enter Length of Stay (days)", min_value=0)
treatment_cost = st.number_input("Enter Treatment Cost", min_value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[age, length_of_stay, treatment_cost]],
        columns=[
            'age',
            'length_of_stay',
            'treatment_cost'
        ]
    )

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.success(
        f"Predicted Risk: {'High Risk' if prediction[0] == 1 else 'Low Risk'}"
    )

    st.write(
        f"Probability of High Risk: {round(probability * 100, 2)}%"
    )