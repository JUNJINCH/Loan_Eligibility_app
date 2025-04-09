import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Error loading file: {e}")