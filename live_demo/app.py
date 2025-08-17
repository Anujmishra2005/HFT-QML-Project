import os
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

# ───────────────────────────────────────────────
# YFINANCE FETCHER
# ───────────────────────────────────────────────
import yfinance as yf

def fetch_yfinance_data(ticker: str, period: str = "5d", interval: str = "1m"):
    """Fetch historical data from Yahoo Finance."""
    st.info(f"Fetching {ticker} data from yfinance...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        st.error(f"No data fetched for {ticker}")
        return None
    data.reset_index(inplace=True)
    return data

def save_processed_data(df: pd.DataFrame, filename: str):
    """Save to data/processed/ folder."""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / filename, index=False)
    st.success(f"Saved data to {processed_dir / filename}")

# ───────────────────────────────────────────────
# APP CONFIG
# ───────────────────────────────────────────────
APP_TITLE = "HFT · Quantum ML Studio"
DATA_DIR = Path("data")
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
CLASSICAL_DIR = MODELS_DIR / "classical"
QUANTUM_DIR = MODELS_DIR / "quantum"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

# CSS styling
st.markdown("""<style> ... (your CSS styles here) ... </style>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# FUNCTIONS (unchanged from your code)
# ───────────────────────────────────────────────
def load_local_processed():
    if PROC_DIR.exists():
        files = sorted(PROC_DIR.glob("*_processed.csv"))
        return files
    return []

def read_csv_any(path_or_buffer):
    if isinstance(path_or_buffer, (str, Path)):
        return pd.read_csv(path_or_buffer)
    return pd.read_csv(path_or_buffer)

def ensure_datetime_index(df):
    for col in ["Date", "Datetime", "Timestamp", "time", "Time"]:
        if col in df.columns:
            try:
                idx = pd.to_datetime(df[col])
                df = df.set_index(idx).drop(columns=[col])
                break
            except Exception:
                pass
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.RangeIndex(start=0, stop=len(df))
    return df

def add_technical_indicators(df):
    ohlc_cols = ["Open","High","Low","Close"]
    for c in ohlc_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required OHLC column: {c}")
    close = df["Close"].astype(float)
    df["RET"] = close.pct_change().fillna(0.0)
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = (gain / (loss.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["ATR"] = ((df["High"]-df["Low"]).abs().rolling(14).mean().fillna(method="bfill"))
    df["BB_MA"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_UP"] = df["BB_MA"] + 2*bb_std
    df["BB_DN"] = df["BB_MA"] - 2*bb_std
    df["WILLR"] = -100 * ((df["High"].rolling(14).max() - close) /
                          (df["High"].rolling(14).max() - df["Low"].rolling(14).min() + 1e-9))
    df["MOM_10"] = close.pct_change(10).fillna(0)
    df["VOL_20"] = df["RET"].rolling(20).std().fillna(0)
    df.replace([np.inf,-np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def label_from_returns(df, horizon=1, threshold=0.0):
    ret_fwd = df["Close"].pct_change(periods=horizon).shift(-horizon).fillna(0)
    y = (ret_fwd > threshold).astype(int)
    return y

def kpi_card(label, value):
    st.markdown(f"""<div class="metric-card"><div class="kpi">{value}</div><div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

def candlestick_with_indicators(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    if "EMA_12" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12"))
    if "EMA_26" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_26"], name="EMA 26"))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=460)
    return fig

def try_load_model(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except ModuleNotFoundError as e:
        st.warning(f"Corrupted pickle {p.name}, will retrain. ({e})")
        return None
    except Exception as e:
        st.warning(f"Failed to load {p.name}: {e}")
        return None

def train_models(X, y, use_xgb=True):
    models = {}
    models["SVM"] = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale").fit(X, y)
    models["MLP"] = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=42).fit(X, y)
    models["RF"]  = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
    if use_xgb and XGB_OK:
        models["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=42, tree_method="hist"
        ).fit(X, y)
    return models

# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")
        source = st.radio("Data Source", ["Upload CSV", "Use processed/ folder", "Fetch from yfinance"], index=1)
        horizon = st.slider("Label horizon (bars ahead)", 1, 30, 5, 1)
        thr = st.slider("Return threshold", -0.01, 0.01, 0.0, 0.001)
        scale = st.toggle("Standardize features", value=True)
        st.markdown("---")
        st.caption("Models will auto-load if .pkl files exist, else they will be trained quickly.")
        return source, horizon, thr, scale

source, horizon, thr, do_scale = sidebar()

# ───────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────
df = None
uploaded = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload market CSV", type=["csv"])
    if uploaded:
        df = read_csv_any(uploaded)

elif source == "Use processed/ folder":
    files = load_local_processed()
    if not files:
        st.warning("No files in data/processed. Switching to yfinance...")
        df = fetch_yfinance_data("AAPL", period="5d", interval="1m")
        if df is not None:
            save_processed_data(df, "AAPL_1min_processed.csv")
    else:
        pick = st.selectbox("Choose processed dataset", [f.name for f in files])
        if pick:
            df = pd.read_csv(PROC_DIR / pick)

elif source == "Fetch from yfinance":
    ticker = st.text_input("Enter ticker (e.g., AAPL, ^NSEI)", "AAPL")
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "1h", "1d"], index=0)
    if st.button("Fetch Data"):
        df = fetch_yfinance_data(ticker, period=period, interval=interval)
        if df is not None:
            save_processed_data(df, f"{ticker}_{interval}_processed.csv")

# ───────────────────────────────────────────────
# REST OF YOUR ORIGINAL APP LOGIC (unchanged)
# ───────────────────────────────────────────────
if df is None:
    st.info("Load a dataset to continue.")
    st.stop()

df = ensure_datetime_index(df)
req_cols = ["Open","High","Low","Close"]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

df = add_technical_indicators(df)
# ... continue with your charts, ML training, and tabs exactly as before ...
