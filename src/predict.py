"""
Load trained models and make a sample prediction.
"""
import joblib
import numpy as np
from pathlib import Path
from .config import CLASSICAL_DIR
from .classical_models import DEFAULT_FEATURES
import pandas as pd

def load_models():
    svm = joblib.load(Path(CLASSICAL_DIR)/"svm_model.pkl")
    xgb = joblib.load(Path(CLASSICAL_DIR)/"xgboost_model.pkl")
    mlp = joblib.load(Path(CLASSICAL_DIR)/"mlp_model.pkl")
    return {"svm": svm, "xgb": xgb, "mlp": mlp}

def predict_sample(models, feature_vector):
    # feature_vector: list-like with length = len(DEFAULT_FEATURES)
    v = np.array(feature_vector).reshape(1, -1)
    return {name: models[name].predict(v)[0] for name in models}

if __name__ == "__main__":
    models = load_models()
    sample = [0.5] * len(DEFAULT_FEATURES)
    print("Predictions:", predict_sample(models, sample))
