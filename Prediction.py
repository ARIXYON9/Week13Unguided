import joblib
import pandas as pd

model = joblib.load("randomforest.sav")
feature_cols = joblib.load("feature_columns.pkl")

def prepare_input(row):
    df = pd.DataFrame([row])
    df = pd.get_dummies(df)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    df = df[feature_cols]
    return df

def predict(df):
    return model.predict(df)
