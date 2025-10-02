import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from preprocessing import load_and_engineer_features

def fetch_latest_data(symbol):

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=60)

    df = yf.download(symbol, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
    # flatten multiindex and remove ticker suffix
    # df.columns = [col[0] if isinstance(col, tuple)else col for col in df.columns]
    df.columns = df.columns.astype(str).str.strip()
    # df.columns = [col.split()[0].strip() for col in df.columns]

    if df.empty:
        raise ValueError(f"No data fetched for symbol {symbol}")
    
    return df

def prepare_features(df):

    return load_and_engineer_features(df)

def predict_tomorrow(symbol = "AAPL"):

    # Load the trained model
    model = joblib.load("Stock_predictor.pkl")

    # Fetch new data
    raw_df = fetch_latest_data(symbol)

    # Apply feature engineering
    processed_df = prepare_features(raw_df)
    if isinstance(processed_df.columns, pd.MultiIndex):
        processed_df.columns = [col[0] if isinstance(col, tuple) else col for col in processed_df.columns]

    processed_df.columns = processed_df.columns.astype(str).str.strip()
    # Latest row of features
    latest_features = processed_df.iloc[-1:].drop(columns=["Date", "Target"], errors="ignore")

    latest_features.columns = [col.split()[0].strip() for col in latest_features.columns]

    latest_features = latest_features.reindex(model.feature_names_in_, axis = 1, fill_value = 0)

    # Make predictions
    prediction = model.predict(latest_features)[0]
    proba = model.predict_proba(latest_features)[0] if hasattr(model, "predict_prob")else None

    # Interpret result
    movement = "UP" if prediction == 1 else "DOWN"
    print(f"Predicted price movement for {symbol} tomorrow: {movement}")
   


    if proba is not None:
        print(f"Prediction confidence: {proba[prediction]: 2f}")

if __name__ == "__main__":
    predict_tomorrow("AAPL")


    