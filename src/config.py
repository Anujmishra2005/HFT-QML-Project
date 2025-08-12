from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CLASSICAL_DIR = MODELS_DIR / "classical"
QUANTUM_DIR = MODELS_DIR / "quantum"
RESULTS_DIR = ROOT / "results"

# Ensure dirs exist
for p in [DATA_PROCESSED, CLASSICAL_DIR, QUANTUM_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Model hyperparams
TRAIN_TEST_SPLIT = 0.2
FUTURE_PERIOD = 5  # minutes forward for labels
