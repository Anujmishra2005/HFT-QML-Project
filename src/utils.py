import pandas as pd
from pathlib import Path
from .config import DATA_RAW, DATA_PROCESSED

def load_csv(filepath):
    return pd.read_csv(filepath, parse_dates=True, index_col=0)

def save_processed(df, name="processed.csv"):
    path = DATA_PROCESSED / name
    df.to_csv(path)
    return path
