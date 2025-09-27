import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, start_date, end_date, save_csv=True):

    data = yf.download(symbol, start=start_date, end=end_date)

    if save_csv:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = os.path.join(data_dir, f"{symbol}_data.csv")
        data.to_csv(filename)
        print(f"Saved data to {filename}")

    return data