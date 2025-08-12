import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from src.utils import load_csv
from src.classical_models import DEFAULT_FEATURES
from src.predict import load_models, predict_sample

st.set_page_config(layout="wide")
st.title("HFT-ML Project - 7th Sem Demo")

DATA_PROCESSED = Path("data/processed")

# Sidebar
st.sidebar.header("Controls")
files = list(DATA_PROCESSED.glob("*_processed.csv"))
selected = st.sidebar.selectbox("Select dataset", [f.name for f in files]) if files else None

if selected:
    df = load_csv(DATA_PROCESSED / selected)
    st.subheader("Price chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model prediction on last row")
    last = df.iloc[-1]
    features = [float(last[f]) for f in DEFAULT_FEATURES]
    if st.button("Load models and predict"):
        try:
            models = load_models()
            preds = predict_sample(models, features)
            st.write("Predictions:", preds)
        except Exception as e:
            st.error("Error loading models: " + str(e))
else:
    st.info("No processed dataset found. Run data preprocessing first.")
