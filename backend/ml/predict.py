"""
Load trained models and make forecasts
"""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import MODELS_DIR, PLANT
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)

CAPACITY_MW = {"solar": PLANT["pv_ac_capacity_mw"], "wind": PLANT["wind_capacity_mw"]}


def postprocess(target: str, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Clip to plant limits; solar is zero when the sun is below the horizon"""
    preds = np.clip(np.asarray(predictions, dtype=float), 0, CAPACITY_MW[target])
    if target == "solar":
        preds = np.where(features["solar_elevation"].to_numpy() <= 0, 0.0, preds)
    return preds


class EnergyPredictor:
    """Wraps one trained XGBoost model (solar or wind)"""

    def __init__(self, target: str, model: xgb.XGBRegressor, feature_columns: list):
        self.target = target
        self.model = model
        self.feature_columns = feature_columns

    @classmethod
    def load(cls, target: str, models_dir: Optional[Path] = None) -> "EnergyPredictor":
        model_dir = (models_dir or MODELS_DIR) / target
        model_path = model_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train first: python run_training.py"
            )
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        with open(model_dir / "features.json") as f:
            feature_columns = json.load(f)
        logger.info(f"Loaded {target} model with {len(feature_columns)} features")
        return cls(target, model, feature_columns)

    def save(self, models_dir: Optional[Path] = None):
        model_dir = (models_dir or MODELS_DIR) / self.target
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_dir / "model.json")
        with open(model_dir / "features.json", "w") as f:
            json.dump(self.feature_columns, f, indent=2)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        start_time = time.time()
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        raw = self.model.predict(features[self.feature_columns])
        preds = postprocess(self.target, raw, features)
        log_performance(logger, f"{self.target} prediction", (time.time() - start_time) * 1000,
                        metadata={"n_samples": len(preds)})
        return preds
