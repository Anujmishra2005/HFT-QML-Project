"""
Train and save SVM, XGBoost, MLP baseline models with scaling.
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from .config import CLASSICAL_DIR, TRAIN_TEST_SPLIT, DATA_PROCESSED
from .utils import load_csv

DEFAULT_FEATURES = ['RSI', 'MACD', 'MACD_Signal', 'BB_high', 'BB_low', 'EMA_12', 'EMA_26']

def load_processed(name=None):
    if name:
        return pd.read_csv(Path(DATA_PROCESSED)/name, index_col=0, parse_dates=True)
    files = list(Path(DATA_PROCESSED).glob("*_processed.csv"))
    if not files:
        raise FileNotFoundError("No processed data found in data/processed")
    return pd.read_csv(files[0], index_col=0, parse_dates=True)

def train_and_save(models_out=True):
    df = load_processed()
    X = df[DEFAULT_FEATURES].values
    y = df['Target'].values

    split = int(len(X) * (1 - TRAIN_TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}

    # Standardize (only for SVM, MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, CLASSICAL_DIR / "scaler.pkl")

    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    results['svm'] = classification_report(y_test, svm_pred, output_dict=True)
    joblib.dump(svm, CLASSICAL_DIR / "svm_model.pkl")

    # XGBoost (raw features, tree-based so no scaling)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results['xgb'] = classification_report(y_test, xgb_pred, output_dict=True)
    joblib.dump(xgb, CLASSICAL_DIR / "xgboost_model.pkl")

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    results['mlp'] = classification_report(y_test, mlp_pred, output_dict=True)
    joblib.dump(mlp, CLASSICAL_DIR / "mlp_model.pkl")

    if models_out:
        print("Training results (macro precision summary):")
        for k, v in results.items():
            print(f"{k.upper()}: {v['macro avg']['precision']:.3f}")
    return results

if __name__ == "__main__":
    train_and_save()