import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from preprocessing import load_and_engineer_features

def clean_columns(df):
    """Ensure flat, clean column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df.columns = df.columns.astype(str).str.strip().str.replace(" ", "", regex=False)
    return df

def fetch_latest_data(symbol):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=60)

    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data fetched for symbol {symbol}")
    
    df = clean_columns(df)

    df.columns = [col.split("_")[0] for col in df.columns]
    
    return df

def prepare_features(df):
    processed_df = load_and_engineer_features(df)
    return clean_columns(processed_df)

def predict_tomorrow(symbol):
    model = joblib.load("Stock_predictor.pkl")

    raw_df = fetch_latest_data(symbol)
    processed_df = prepare_features(raw_df)

    latest_features = processed_df.iloc[-1:].drop(columns=["Date", "Target"], errors="ignore")

    
    latest_features = latest_features.reindex(model.feature_names_in_, axis=1, fill_value=0)

    prediction = model.predict(latest_features)[0]
    
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(latest_features)[0][prediction]
    else:
        proba = None

    movement = "UP" if prediction == 1 else "DOWN"
    print(f"Predicted price movement for {symbol} tomorrow: {movement}")
    
    if proba is not None:
        print(f"Prediction confidence: {proba * 100:.2f}%")

if __name__ == "__main__":
    predict_tomorrow("AAPL")
