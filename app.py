import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import sys
sys.path.append('src')  # Ensure src/ is discoverable

from src.preprocessing import load_and_engineer_features

model_path = "Stock_predictor.pkl"
model = joblib.load(model_path)

st.title("Stock price movement predictor")
st.write("Predicts whether the stock price will go UP or DOWN tomorrow based on recent data")

symbol = st.text_input("Enter Stock Symbol:", value = "TSLA").upper()
days = st.slider("Number of past days to fetch:", min_value=30, max_value=120, value = 60)

if st.button("Predict"):
    try:
        df=yf.download(symbol, period=f"{days}d")

        if df.empty:
            st.error(f"Failed to fetch data for {symbol}. Try another symbol.")

        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            df.columns = [col.split('_')[0] for col in df.columns]

            df = df.reset_index()

            processed_df = load_and_engineer_features(df)

            latest_features = processed_df.drop(columns=["Date", "Target"], errors="ignore")
            latest_row = latest_features.iloc[-1:].copy()
            # latest_row.columns = [col.split()[0].strip() for col in latest_row.columns]
            # clean_col = []
            # for col in latest_row.columns:
            #     if isinstance(col, tuple):
            #         clean_col.append(col[0].strip())
            #     else:
            #         clean_col.append(str(col).strip())

            expected_features = [
                        "Close", "High", "Low", "Open",
                        "Return", "SMA10", "Volatility",
                        "High_Low_spread", "Close_Open_change",
                        "Return_1d", "EMA_12", "EMA_26", "MACD"
                        ]
            
            latest_row.columns = latest_row.columns.str.strip()

            latest_row = latest_row[[col for col in latest_features.columns if col in expected_features]]   

            for col in expected_features:
                if col not in latest_row.columns:
                    latest_row[col] = 0

            latest_row = latest_row[expected_features]
            # latest_row.columns = clean_col


            prediction = model.predict(latest_row)[0]
            proba =  model.predict_proba(latest_row)[0][prediction]

            label = "UP" if prediction == 1 else "DOWN"
            st.subheader(f"Prediction for {symbol}: **{label}**")
            st.write(f"**Confidence:** {proba * 100:.2f}%")


            st.subheader(f"Last {days} Days Closing Prices")
            fig, ax = plt.subplots()
            ax.plot(df["Date"], df["Close"], linewidth=2)

            ax.grid(alpha = 0.3)

            last_date = df["Date"].iloc[-1]
            last_price = df["Close"].iloc[-1]
            ax.scatter(last_date, last_price, s=80, edgecolors="black")

            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.set_title(f"{symbol} Price Trend")

            plt.xticks(rotation = 30)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occured: {e}")
