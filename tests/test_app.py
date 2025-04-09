import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import preprocessing, predict  # Now this will work

def test_preprocessing_and_prediction():
    sample_input = pd.DataFrame({
        "Gender": ["Male"],
        "Married": ["Yes"],
        "Dependents": ["1"],
        "Education": ["Graduate"],
        "Self_Employed": ["No"],
        "ApplicantIncome": [5000],
        "CoapplicantIncome": [1500],
        "LoanAmount": [120],
        "Loan_Amount_Term": [360],
        "Credit_History": [1.0],
        "Property_Area": ["Urban"]
    })

    df_processed = preprocessing.preprocess(sample_input)
    prediction = predict.predict(df_processed)

    assert prediction[0] in [0, 1]  # Assuming the model outputs binary predictions (0 or 1)
    assert df_processed.shape[0] == sample_input.shape[0]  # Ensure the shape remains the same after preprocessing