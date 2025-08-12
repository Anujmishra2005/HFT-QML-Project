# Methodology

1. Data collection: CSVs in data/raw
2. Feature engineering: Technical indicators (RSI, MACD, Bollinger, EMA)
3. Labeling: future period threshold (FUTURE_PERIOD)
4. Training: SVM, XGBoost, MLP
5. Evaluation: classification report + backtesting (future)
6. Quantum: angle encoding + VQC prototype (PennyLane)
