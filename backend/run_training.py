"""
Step 2: train the solar and wind models, score them on the held-out year,
save the models and write metrics and figures.

    cd backend
    python run_training.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from ml.plots import backtest_plot, mae_bar_chart
from ml.train import train_and_evaluate
from utils.config import DOCS_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("run_training")
DATASET_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def main() -> int:
    if not DATASET_PATH.exists():
        logger.error(f"{DATASET_PATH} not found. Run: python run_initialization.py")
        return 1
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["time"])
    logger.info(f"Loaded {len(dataset)} rows")

    metrics, predictors, forecasts = train_and_evaluate(dataset)

    for predictor in predictors.values():
        predictor.save()
    forecasts.to_csv(MODELS_DIR / "test_forecasts.csv", index=False)
    for path in (MODELS_DIR / "metrics.json", DOCS_DIR / "metrics.json"):
        path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved models and metrics to {MODELS_DIR}")

    DOCS_DIR.mkdir(exist_ok=True)
    backtest_plot(forecasts, "de_aar", "2026-01-10", 10, DOCS_DIR / "backtest_de_aar.png")
    mae_bar_chart(metrics, DOCS_DIR / "mae_by_site.png")
    logger.info(f"Wrote figures to {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
