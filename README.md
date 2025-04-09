# Loan Eligibility Predictor

This project is a **machine learning web application** built with **Streamlit**. It predicts whether a loan applicant is eligible for a loan based on personal and financial information using a trained logistic regression model.
[Click here to try the app](https://junjinch-loan-eligibility-app-app-2o3xae.streamlit.app/)
---

## Features

- Interactive web interface (Streamlit)
- Logistic regression model trained on sample loan data
- Preprocessing pipeline for real-time inputs
- Feature coefficient visualization (via bar chart)
- Logging system that records user input, prediction, and errors
- Modular and testable code structure

---

## Model Info

- **Algorithm**: `LogisticRegression` (from scikit-learn)
- **Target**: `Loan_Status` (1 = Approved, 0 = Not Approved)
- **Features**:
  - Gender, Married, Dependents, Education, Self_Employed
  - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
  - Credit_History, Property_Area
- **Trained Model Path**: `models/loan_model.pkl`

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Launch the Streamlit App

streamlit run app.py

## Folder Structure
```
Loan_Eligibility_App/
├── app.py                         # Main Streamlit app
├── train_and_save_model.py       # Model training script
├── data/
│   └── credit.csv                 # dataset
├── logs/
│   └── app_report.txt             # Auto-generated log file
├── models/
│   └── loan_model.pkl             # Trained model
├── notebooks/
│   └── Loan_Eligibility_Model_Solution.ipynb  # Jupyter exploration
├── src/                          # Modular Python code
│   ├── data_loader.py
│   ├── logger.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── visualization.py
├── tests/
│   └── test_app.py                # Unit test
├── requirements.txt
└── README.md
```
## Logging
All predictions, inputs, and runtime errors are saved to:

logs/app_report.txt

## Testing
pytest tests/test_app.py

##  Author
- Project for `CST2216 Machine Learning`
- Student: Junjin Chen
- Year: 2025 Winter
