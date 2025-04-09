import joblib
import pandas as pd

def predict(input_data: pd.DataFrame):
    model = joblib.load("models/loan_model.pkl")
    prediction = model.predict(input_data)
    return prediction