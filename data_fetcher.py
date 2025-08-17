# file: data_fetcher.py
import yfinance as yf
import pandas as pd
import os


def fetch_yfinance_data(ticker: str, period: str = "1mo", interval: str = "1m"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    print(f"[INFO] Fetching {ticker} data from yfinance...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}. Check symbol or internet connection.")

    data.reset_index(inplace=True)
    return data


def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save DataFrame into the data/processed folder.
    """
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    file_path = os.path.join(processed_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"[INFO] Saved data to {file_path}")


if __name__ == "__main__":
    # Example: Fetch Apple 1-min data for the last 5 days
    ticker_symbol = "AAPL"
    df = fetch_yfinance_data(ticker_symbol, period="5d", interval="1m")
    save_processed_data(df, f"{ticker_symbol}_1min_processed.csv")
