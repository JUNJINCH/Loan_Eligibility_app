from sklearn.preprocessing import LabelEncoder
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Fill missing numerical values with median
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode and encode them
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = LabelEncoder().fit_transform(df[col])

    return df