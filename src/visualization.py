import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_feature_importance_plot(model_path: str, feature_df: pd.DataFrame):
    model = joblib.load(model_path)
    
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        features = feature_df.columns

        importance_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": coefs
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=importance_df, x="Coefficient", y="Feature", ax=ax)
        ax.set_title("Feature Importance")
        ax.grid(True)
        return fig
    else:
        raise ValueError("Model does not support coefficient-based feature importance.")