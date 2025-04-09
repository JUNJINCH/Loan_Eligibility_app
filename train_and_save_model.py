import pandas as pd
from src import preprocessing, model

# Load the data
df = pd.read_csv("data/credit.csv")

# Show basic column info
print("Original columns:", df.columns.tolist())

# Prepare target and features
target = df["Loan_Approved"].map({"Y": 1, "N": 0})
features = df.drop(columns=["Loan_ID", "Loan_Approved"])

# Preprocess features
features_processed = preprocessing.preprocess(features)

# Reattach target
df_processed = features_processed.copy()
df_processed["Loan_Approved"] = target

# Train model
model.train_model(df_processed, target_col="Loan_Approved")

print("Model trained and saved to models/loan_model.pkl")