"""
Load raw CSVs and build processed dataset with technical indicators.
"""
import pandas as pd
import numpy as np
import ta
from pathlib import Path
from .config import DATA_RAW, DATA_PROCESSED, FUTURE_PERIOD
from .utils import save_processed, load_csv

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # expects df with Close, Open, High, Low, Volume
    data = df.copy()
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    # Bollinger
    bb = ta.volatility.BollingerBands(data['Close'], window=20)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    # EMA
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
    # Drop rows with NaN
    data = data.dropna()
    return data

def create_labels(df: pd.DataFrame, future_period: int = FUTURE_PERIOD):
    data = df.copy()
    data['Future_Close'] = data['Close'].shift(-future_period)
    # threshold 0.2% up/down
    up = data['Future_Close'] > data['Close'] * 1.002
    down = data['Future_Close'] < data['Close'] * 0.998
    data['Target'] = 0
    data.loc[up, 'Target'] = 1
    data.loc[down, 'Target'] = -1
    data = data.dropna()
    return data

def process_all_raw():
    for f in Path(DATA_RAW).glob("*.csv"):
        print("Processing", f)
        df = load_csv(f)
        df = compute_technical_indicators(df)
        df = create_labels(df)
        out = save_processed(df, name=f.stem + "_processed.csv")
        print("Saved:", out)

if __name__ == "__main__":
    process_all_raw()
