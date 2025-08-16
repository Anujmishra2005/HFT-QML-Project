# app.py — HFT Quantum ML Dashboard (polished, interactive, standalone)

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
import yfinance as yf
import time
from datetime import datetime, timedelta
import threading

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

APP_TITLE = "HFT · Quantum ML Studio"
# Get the project root directory (parent of live_demo)
# Use absolute path resolution to handle Streamlit's working directory
import os
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSICAL_DIR = MODELS_DIR / "classical"
QUANTUM_DIR = MODELS_DIR / "quantum"

st.set_page_config(page_title=APP_TITLE, page_icon="��", layout="wide")

# Initialize session state for live data
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'live_data_cache' not in st.session_state:
    st.session_state.live_data_cache = {}

st.markdown("""
<style>
/* global */
:root { --brand:#6C63FF; --bg:#0b0f19; --card:#12182a; --muted:#a0a8c3; }
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0e1426 0%,#0b0f19 100%); }
h1,h2,h3,h4 { letter-spacing:.2px; }
.metric-card {
  background: var(--card); padding: 16px; border-radius: 18px; border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
.kpi { font-size: 28px; font-weight: 700; color: white; }
.kpi-label { color: var(--muted); font-size: 13px; }
.btn-primary button { background: var(--brand)!important; border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background: #12182a; border-radius: 12px; padding: 8px 14px; }
.stDownloadButton .css-1dp5vir { width: 100%; }
hr { border-color: rgba(255,255,255,.1); }
</style>
""", unsafe_allow_html=True)

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
    df["ATR"] = ( (df["High"]-df["Low"]).abs()
                .rolling(14).mean().bfill() )
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
    st.markdown(f"""
    <div class="metric-card">
      <div class="kpi">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def candlestick_with_indicators(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    if "EMA_12" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12"))
    if "EMA_26" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_26"], name="EMA 26"))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=460)
    return fig

def feature_table(df, features):
    sub = df[features].tail(20).copy()
    return sub

def try_load_model(path):
    p = Path(path)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception as e:
            st.warning(f"Could not load model from {p.name}: {str(e)}")
            return None
    return None

def fetch_live_data(symbol, period="1d", interval="1m"):
    """Fetch live data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_live_ticker_info(symbol):
    """Get live ticker information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'current_price': info.get('currentPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
            'change_percent': ((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 0)) * 100 if info.get('previousClose', 0) else 0
        }
    except Exception as e:
        return None

def auto_refresh_data(symbol, interval_seconds=60):
    """Auto-refresh data at specified interval"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if (datetime.now() - st.session_state.last_refresh).seconds >= interval_seconds:
        st.session_state.last_refresh = datetime.now()
        return True
    return False

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

def predictions_section(models, X_test, y_test, class_labels=("Down","Up")):
    cols = st.columns(len(models))
    scores = {}
    for i,(name,model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        scores[name] = (acc,f1)
        with cols[i]:
            kpi_card(f"{name} · Accuracy", f"{acc:.3f}")
            kpi_card(f"{name} · F1-score", f"{f1:.3f}")
    st.markdown("#### Classification Reports")
    for name,model in models.items():
        y_pred = model.predict(X_test)
        st.markdown(f"**{name}**")
        st.code(classification_report(y_test, y_pred, target_names=class_labels))
    best = max(scores.items(), key=lambda kv: kv[1][0])[0]
    st.success(f"Best model by accuracy: **{best}**")
    return best

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Pred", y="True"), title=title)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

def sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data source selection
        source = st.radio("Data Source", ["Upload CSV", "Use processed/ folder", "Live YFinance"], index=1)
        
        # YFinance controls
        if source == "Live YFinance":
            st.subheader("📈 Live Data Settings")
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
            interval = st.selectbox("Data Interval", ["1m", "2m", "5m", "15m", "30m", "60m"], index=0)
            period = st.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=0)
            auto_refresh = st.toggle("Auto-refresh", value=True)
            refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60, 30)
            
            if auto_refresh:
                if st.button("🔄 Refresh Now"):
                    st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_interval + 1)
                    st.rerun()
        else:
            symbol = "AAPL"
            interval = "1m"
            period = "1d"
            auto_refresh = False
            refresh_interval = 60
        
        st.markdown("---")
        horizon = st.slider("Label horizon (bars ahead)", 1, 30, 5, 1)
        thr = st.slider("Return threshold", -0.01, 0.01, 0.0, 0.001)
        scale = st.toggle("Standardize features", value=True)
        st.markdown("---")
        st.caption("Models will auto-load if .pkl files exist, else they will be trained quickly.")
        return source, horizon, thr, scale, symbol, interval, period, auto_refresh, refresh_interval

def header():
    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1: st.title(APP_TITLE)
    with c2: kpi_card("Models", "Classical + Quantum*")
    with c3: kpi_card("Indicators", "MACD/RSI/EMA")
    with c4: kpi_card("Status", "Ready")

source, horizon, thr, do_scale, symbol, interval, period, auto_refresh, refresh_interval = sidebar()
header()

tab1, tab2, tab3, tab4 = st.tabs(["📂 Data", "📈 Charts", "🤖 Models", "🔮 Live Predict"])

df = None
uploaded = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload market CSV with columns: Open, High, Low, Close (plus optional Volume/Date)", type=["csv"])
    if uploaded:
        df = read_csv_any(uploaded)
elif source == "Live YFinance":
    st.subheader(f"📈 Live Data: {symbol.upper()}")
    
    # Show live ticker info
    ticker_info = get_live_ticker_info(symbol)
    if ticker_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${ticker_info['current_price']:.2f}", 
                     f"{ticker_info['change']:+.2f} ({ticker_info['change_percent']:+.2f}%)")
        with col2:
            st.metric("Volume", f"{ticker_info['volume']:,}")
        with col3:
            st.metric("Market Cap", f"${ticker_info['market_cap']/1e9:.1f}B" if ticker_info['market_cap'] else "N/A")
        with col4:
            st.metric("P/E Ratio", f"{ticker_info['pe_ratio']:.1f}" if ticker_info['pe_ratio'] else "N/A")
    
    # Fetch live data
    if auto_refresh and auto_refresh_data(symbol, refresh_interval):
        st.info(f"🔄 Auto-refreshing data for {symbol.upper()}...")
    
    live_data = fetch_live_data(symbol, period, interval)
    if live_data is not None:
        df = live_data
        st.success(f"✅ Live data loaded: {len(df)} bars from {symbol.upper()}")
        
        # Show last update time
        if 'last_refresh' in st.session_state:
            st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    else:
        st.error(f"❌ Failed to fetch data for {symbol.upper()}")
        st.info("Please check the symbol and try again.")
else:
    files = load_local_processed()
    if not files:
        st.warning("No files in data/processed. Upload a CSV instead.")
    else:
        pick = st.selectbox("Choose processed dataset", [f.name for f in files])
        if pick:
            df = pd.read_csv(PROC_DIR / pick)

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

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.tail(200), use_container_width=True, height=420)
    st.download_button(
        "Download Enriched CSV",
        data=df.to_csv().encode("utf-8"),
        file_name="enriched_dataset.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Price & Indicators")
    
    if source == "Live YFinance" and auto_refresh:
        st.info(f"📊 Live chart updating every {refresh_interval} seconds for {symbol.upper()}")
        if st.button("🔄 Update Chart Now"):
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_interval + 1)
            st.rerun()
    
    fig = candlestick_with_indicators(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add live price line for yfinance data
    if source == "Live YFinance" and ticker_info:
        st.markdown("**💡 Live Price Indicator**")
        live_price = ticker_info['current_price']
        last_close = df['Close'].iloc[-1]
        price_change = live_price - last_close
        price_change_pct = (price_change / last_close) * 100
        
        price_col1, price_col2, price_col3 = st.columns(3)
        with price_col1:
            st.metric("Last Close", f"${last_close:.2f}")
        with price_col2:
            st.metric("Live Price", f"${live_price:.2f}", f"{price_change:+.2f}")
        with price_col3:
            st.metric("Change %", f"{price_change_pct:+.2f}%")
    
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**MACD**")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], name="Signal"))
        macd_fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(macd_fig, use_container_width=True)
    with mcol2:
        st.markdown("**RSI**")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="rgba(108,99,255,.1)")
        rsi_fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(rsi_fig, use_container_width=True)

features = [
    "RET","EMA_12","EMA_26","MACD","MACD_SIG","RSI","ATR",
    "BB_MA","BB_UP","BB_DN","WILLR","MOM_10","VOL_20",
    "Open","High","Low","Close"
]
X_raw = df[features].astype(float).values
y = label_from_returns(df, horizon=horizon, threshold=thr).values

if do_scale:
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
else:
    scaler = None
    X = X_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=False)

pretrained = {
    "SVM": try_load_model(CLASSICAL_DIR / "svm_model.pkl"),
    "XGB": try_load_model(CLASSICAL_DIR / "xgboost_model.pkl") if XGB_OK else None,
    "MLP": try_load_model(CLASSICAL_DIR / "mlp_model.pkl"),
    "RF":  try_load_model(CLASSICAL_DIR / "rf_model.pkl"),
    "VQC": try_load_model(QUANTUM_DIR / "vqc_model.pkl")
}

with tab3:
    st.subheader("Model Training / Loading")
    use_pretrained = st.toggle("Prefer pre-trained models if available", value=True)
    enable_xgb = st.toggle("Enable XGBoost", value=XGB_OK, disabled=not XGB_OK)
    go_train = st.button("Build/Load Models", type="primary")
    if go_train:
        models = {}
        for k in ["SVM","MLP","RF"] + (["XGB"] if enable_xgb else []):
            if use_pretrained and pretrained.get(k) is not None:
                models[k] = pretrained[k]
            else:
                models = {**models, **train_models(X_train, y_train, use_xgb=enable_xgb)}
                break
        if "VQC" in pretrained and pretrained["VQC"] is not None:
            models["VQC"] = pretrained["VQC"]

        st.success("Models ready.")
        best = predictions_section(models, X_test, y_test)
        for name,model in models.items():
            y_pred = model.predict(X_test)
            plot_confusion(y_test, y_pred, f"{name} · Confusion Matrix")

        if "XGB" in models and hasattr(models["XGB"], "feature_importances_"):
            imp = pd.DataFrame({
                "feature": features,
                "importance": models["XGB"].feature_importances_
            }).sort_values("importance", ascending=False)
            st.subheader("XGB Feature Importance")
            fig_imp = px.bar(imp, x="feature", y="importance")
            fig_imp.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

        bundle = {
            "scaler": scaler,
            "features": features,
            "models": {k: v for k,v in models.items()}
        }
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button("Download Model Bundle (.joblib)", buf.getvalue(), file_name="hft_models.joblib")

with tab4:
    st.subheader("🔮 Live Predictions")
    
    if source == "Live YFinance":
        st.markdown("### 📊 Real-time Market Monitoring")
        
        # Live price ticker
        if ticker_info:
            price_col1, price_col2, price_col3 = st.columns(3)
            with price_col1:
                st.metric("Live Price", f"${ticker_info['current_price']:.2f}", 
                         f"{ticker_info['change']:+.2f}")
            with price_col2:
                st.metric("Change %", f"{ticker_info['change_percent']:+.2f}%")
            with price_col3:
                st.metric("Volume", f"{ticker_info['volume']:,}")
        
        # Auto-refresh status
        if auto_refresh:
            st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
            if st.button("🔄 Manual Refresh", type="secondary"):
                st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_interval + 1)
                st.rerun()
    
    # Model selection for predictions
    choice = st.selectbox("Select model for live prediction", ["Auto (best)","SVM","XGB","MLP","RF","VQC"])
    model_to_use = None
    if choice == "Auto (best)":
        for opt in ["XGB","SVM","RF","MLP","VQC"]:
            if pretrained.get(opt) is not None:
                model_to_use = pretrained[opt]; break
        if model_to_use is None:
            st.warning("No pre-trained model found. Train models in the previous tab.")
    else:
        model_to_use = pretrained.get(choice)
        if model_to_use is None:
            st.warning(f"{choice} model not available. Train or add a .pkl first.")

    # Live indicators display
    if df is not None and len(df) > 0:
        cA, cB, cC = st.columns(3)
        with cA: 
            kpi_card("RSI (last)", f"{df['RSI'].iloc[-1]:.2f}")
        with cB: 
            kpi_card("MACD (last)", f"{df['MACD'].iloc[-1]:.4f}")
        with cC: 
            kpi_card("RET (last)", f"{df['RET'].iloc[-1]:.4%}")
        
        # Additional live metrics
        if source == "Live YFinance":
            col1, col2, col3 = st.columns(3)
            with col1:
                kpi_card("ATR (last)", f"{df['ATR'].iloc[-1]:.4f}")
            with col2:
                kpi_card("BB Width", f"{(df['BB_UP'].iloc[-1] - df['BB_DN'].iloc[-1])/df['BB_MA'].iloc[-1]:.2%}")
            with col3:
                kpi_card("Momentum", f"{df['MOM_10'].iloc[-1]:.2%}")

    # Prediction section
    if model_to_use is not None and df is not None and len(df) > 0:
        st.markdown("### 🎯 Make Prediction")
        
        if st.button("🚀 Predict Next Move", type="primary"):
            row = df.iloc[-1][features].values.astype(float)
            if scaler is not None:
                row_s = scaler.transform([row])[0]
            else:
                row_s = row

            pred = int(model_to_use.predict([row_s])[0])
            prob = None
            try:
                prob = float(model_to_use.predict_proba([row_s])[0, pred])
            except Exception:
                prob = None
            
            lbl = "🟢 UP" if pred==1 else "🔴 DOWN"
            confidence_color = "success" if prob and prob > 0.7 else "warning" if prob and prob > 0.5 else "info"
            
            if prob is not None:
                st.success(f"**Prediction: {lbl}**  ·  **Confidence: {prob:.2%}**")
                
                # Confidence indicator
                if prob > 0.8:
                    st.info("🎯 High confidence prediction")
                elif prob > 0.6:
                    st.info("⚠️ Medium confidence prediction")
                else:
                    st.info("🤔 Low confidence prediction")
            else:
                st.success(f"**Prediction: {lbl}**")
            
            # Show prediction details
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.markdown("**Latest Data Point:**")
                out = pd.DataFrame([dict(zip(features, row))])
                out["prediction"] = lbl
                st.dataframe(out, use_container_width=True)
            
            with pred_col2:
                st.markdown("**Market Context:**")
                if source == "Live YFinance":
                    context_data = {
                        "Symbol": symbol.upper(),
                        "Current Price": f"${ticker_info['current_price']:.2f}" if ticker_info else "N/A",
                        "RSI": f"{df['RSI'].iloc[-1]:.2f}",
                        "MACD": f"{df['MACD'].iloc[-1]:.4f}",
                        "Volume": f"{ticker_info['volume']:,}" if ticker_info else "N/A",
                        "Prediction Time": datetime.now().strftime("%H:%M:%S")
                    }
                    context_df = pd.DataFrame(list(context_data.items()), columns=["Metric", "Value"])
                    st.dataframe(context_df, use_container_width=True)
            
            # Download prediction
            st.download_button("📥 Download Live Prediction", out.to_csv(index=False), 
                             f"live_prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                             "text/csv")
            
            # Prediction history (if available)
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            prediction_record = {
                'timestamp': datetime.now(),
                'symbol': symbol if source == "Live YFinance" else "Unknown",
                'prediction': lbl,
                'confidence': prob,
                'current_price': ticker_info['current_price'] if ticker_info else None,
                'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
                'macd': df['MACD'].iloc[-1] if 'MACD' in df.columns else None
            }
            st.session_state.prediction_history.append(prediction_record)
            
            # Show prediction history
            if len(st.session_state.prediction_history) > 1:
                st.markdown("### 📈 Prediction History")
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df, use_container_width=True)
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
    else:
        st.warning("Please load data and train models to make predictions.")
