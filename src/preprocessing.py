import pandas as pd
import os

def load_and_engineer_features(df):

    # df = pd.read_csv(TSLA_data)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    # if "Date" in df.columns:
    #     df["Date"] = pd.to_datetime(df["Date"])
    #     df.set_index("Date", inplace=True)

    df["Return"] = df["Close"].pct_change()

    df["SMA10"] = df["Close"].rolling(window=10).mean()
    # df["MA20"] = df["Close"].rolling(window=20).mean()

    df["Volatility"] = df["Return"].rolling(window=10).std()

    df["High_Low_spread"] = df["High"] - df["Low"]
    df["Close_Open_change"] = df["Close"] - df["Open"]


    df["Return_1d"] = df["Close"].pct_change(1)
    

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    ##base_dir = os.path.dirname(os.path.dirname(__file__))   # go up from src/ → STOCK_PREDICTOR/
    csv_file = os.path.join("data", "TSLA_data.csv")
    processed_df = load_and_engineer_features(csv_file)
    print(processed_df.head())
    