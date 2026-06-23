import streamlit as st
import pandas as pd
import joblib

model = joblib.load('risk_model.joblib')

st.title("healthcare risk prediction")

age = st.number_input("Enter age", min_value=0)
length_of_stay = st.number_input("Enter length of stay (days)", min_value=0)
treatment_cost = st.number_input("Enter treatment cost", min_value=0.0)

if st.button("predict"):
    input_data = pd.DataFrame([[age, length_of_stay, treatment_cost]], columns=['age', 'length_of_stay', 'treatment_cost'])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Assuming binary classification

    st.write(f"Predicted risk: {'High risk' if prediction == 1 else 'Low risk'}")
    st.write(f"Probability of high risk: {round(probability , 2)}")
