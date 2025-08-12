import json
from pathlib import Path
from .config import RESULTS_DIR

def save_metrics(metrics: dict, name="metrics.json"):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(RESULTS_DIR)/name, "w") as f:
        json.dump(metrics, f, indent=2)
