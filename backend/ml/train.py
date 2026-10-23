"""
Train and evaluate the day-ahead solar and wind models.

Split by time, never at random: train on the oldest data, stop early on a
later validation window, then score once on a held-out year that the model
never saw. The model is compared with two naive baselines and with the
physical power model run on the same forecast weather.
"""
import time
from typing import Dict, Tuple

import pandas as pd
import xgboost as xgb

from data_processing.dataset import TARGETS
from data_processing.feature_engineering import FEATURE_COLUMNS
from ml.evaluate import evaluate
from ml.predict import EnergyPredictor, postprocess
from utils.config import TEST_START, VALIDATION_START, XGBOOST_PARAMS
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)

PHYSICAL_COLUMN = {"solar": "pv_physical_mw", "wind": "wind_physical_mw"}


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train < VALIDATION_START <= validation < TEST_START <= test"""
    train = df[df["time"] < VALIDATION_START]
    val = df[(df["time"] >= VALIDATION_START) & (df["time"] < TEST_START)]
    test = df[df["time"] >= TEST_START]
    return train, val, test


def train_model(target: str, train: pd.DataFrame, val: pd.DataFrame) -> EnergyPredictor:
    y_col = TARGETS[target]
    start_time = time.time()
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        train[FEATURE_COLUMNS], train[y_col],
        eval_set=[(val[FEATURE_COLUMNS], val[y_col])],
        verbose=False,
    )
    log_performance(logger, f"{target} training", (time.time() - start_time) * 1000,
                    metadata={"best_iteration": model.best_iteration})
    return EnergyPredictor(target, model, list(FEATURE_COLUMNS))


def forecast_table(target: str, predictor: EnergyPredictor, test: pd.DataFrame) -> pd.DataFrame:
    """Actual, model and baseline forecasts for every test hour"""
    y_col = TARGETS[target]
    return pd.DataFrame({
        "time": test["time"].to_numpy(),
        "location": test["location"].to_numpy(),
        "solar_elevation": test["solar_elevation"].to_numpy(),
        "actual": test[y_col].to_numpy(),
        "model": predictor.predict(test),
        "persistence": test[f"{y_col}_persistence"].to_numpy(),
        "seasonal_naive": test[f"{y_col}_seasonal_naive"].to_numpy(),
        "physical_nwp": postprocess(target, test[PHYSICAL_COLUMN[target]], test),
    })


def train_and_evaluate(dataset: pd.DataFrame) -> Tuple[Dict, Dict[str, EnergyPredictor], pd.DataFrame]:
    """Returns (metrics, predictors, test forecasts)"""
    # Baselines need the previous day; drop the few hours where it is missing
    baseline_cols = [f"{c}_{b}" for c in TARGETS.values() for b in ("persistence", "seasonal_naive")]
    dataset = dataset.dropna(subset=baseline_cols).sort_values(["location", "time"])
    train, val, test = time_split(dataset)
    logger.info(f"Split - train: {len(train)}, validation: {len(val)}, test: {len(test)} rows")

    metrics = {
        "split": {
            "train": [str(train["time"].min()), str(train["time"].max()), len(train)],
            "validation": [str(val["time"].min()), str(val["time"].max()), len(val)],
            "test": [str(test["time"].min()), str(test["time"].max()), len(test)],
        },
        "targets": {},
    }
    predictors, tables = {}, []
    for target in TARGETS:
        predictor = train_model(target, train, val)
        predictors[target] = predictor
        table = forecast_table(target, predictor, test)
        tables.append(table.assign(target=target))

        # Solar is scored on daylight hours only; night hours are zero for
        # every method and would flatter the numbers
        scored = table[table["solar_elevation"] > 0] if target == "solar" else table
        result = evaluate(scored)
        result["hours_scored"] = "daylight only" if target == "solar" else "all hours"
        result["best_iteration"] = int(predictor.model.best_iteration)
        metrics["targets"][target] = result

        o = result["overall"]
        logger.info(
            f"{target}: model MAE {o['model']['mae']:.2f} MW, "
            f"persistence {o['persistence']['mae']:.2f}, "
            f"seasonal naive {o['seasonal_naive']['mae']:.2f}, "
            f"physical on forecast {o['physical_nwp']['mae']:.2f}"
        )
    return metrics, predictors, pd.concat(tables, ignore_index=True)
