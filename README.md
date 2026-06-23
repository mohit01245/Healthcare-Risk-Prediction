# Healthcare-Risk-Prediction
# Healthcare Risk Prediction

## Project Overview

This project predicts patient healthcare risk using Machine Learning. The model analyzes patient information such as age, length of hospital stay, and treatment cost to classify patients into risk categories.

## Features

* Patient Data Analysis
* Data Preprocessing using Pandas
* Feature Engineering
* Logistic Regression Model
* Risk Prediction (High Risk / Low Risk)
* Interactive Streamlit Web Application

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

## Dataset

The project uses healthcare-related datasets containing:

* Patient Information
* Diagnosis Data
* Outcomes Data
* Laboratory Data

## Machine Learning Workflow

1. Load and merge datasets
2. Data preprocessing
3. Create length_of_stay feature
4. Encode outcome categories
5. Train Logistic Regression model
6. Evaluate model performance
7. Save model using Joblib
8. Deploy using Streamlit

## Input Features

* Age
* Length of Stay
* Treatment Cost

## Output

* High Risk
* Low Risk
* Risk Probability

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py
├── risk_model.joblib
├── Patients_200_rows.csv
├── Diagnosis_200_rows.csv
├── Outcomes_200_rows.csv
├── Labs_200_rows.csv
├── requirements.txt
└── README.md
```

## Author

mohit Prajapati
