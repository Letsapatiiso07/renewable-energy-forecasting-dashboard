import os
from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(DATA_DIR / "models")))
DOCS_DIR = BASE_DIR / "docs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Open-Meteo APIs (free, no API key). Data licence: CC BY 4.0.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Africa/Johannesburg"

# Pin the weather models. Open-Meteo's default "best_match" switched to a
# newer ECMWF model in late 2025, which changed the forecast bias between
# the training and test periods. One fixed model for training and serving
# avoids that.
REANALYSIS_MODEL = "era5"
FORECAST_MODEL = "ecmwf_ifs025"

# South African sites. Each one gets a hypothetical 100 MW PV plant and a
# 100 MW wind farm whose output is modelled from weather data.
LOCATIONS: Dict[str, Dict] = {
    "pretoria": {"lat": -25.7479, "lon": 28.2293, "name": "Pretoria, Gauteng"},
    "de_aar": {"lat": -30.65, "lon": 24.01, "name": "De Aar, Northern Cape"},
    "jeffreys_bay": {"lat": -34.03, "lon": 24.91, "name": "Jeffreys Bay, Eastern Cape"},
}

# Hourly weather variables. The same names are used for the reanalysis
# ("actual" weather), the archived day-ahead forecasts and the live forecast.
WEATHER_PARAMS: List[str] = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
]

# Day-ahead forecasts of 100 m wind are archived from about April 2024
HISTORY_START_DATE = "2024-04-01"
HISTORY_END_DATE = "2026-09-30"

# Time-based split: train, then a validation window for early stopping,
# then a full held-out year for testing
VALIDATION_START = "2025-07-01"
TEST_START = "2025-10-01"

# Plant assumptions used by the physical power models
PLANT = {
    "pv_ac_capacity_mw": 100.0,
    "pv_dc_ac_ratio": 1.2,
    "pv_gamma_pdc": -0.004,  # power change per degree C of cell temperature
    "pv_system_losses": 0.14,  # PVWatts default losses
    "pv_inverter_efficiency": 0.96,
    "wind_capacity_mw": 100.0,
    "wind_cut_in_ms": 3.0,
    "wind_rated_ms": 12.0,
    "wind_cut_out_ms": 25.0,
    "wind_farm_losses": 0.10,  # wake and availability losses
}

RANDOM_STATE = 42

# XGBoost parameters (small enough to train in seconds on 2 cores)
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": int(os.getenv("N_JOBS", "2")),
}

# Illustrative marginal costs in USD/MWh for the dispatch demo.
# These are placeholders, not market data.
ENERGY_COSTS = {
    "solar": 5.0,
    "wind": 5.0,
    "hydro": 40.0,
    "coal": 60.0,
    "gas": 150.0,
}
CAPACITIES_MW = {
    "hydro": 200.0,
    "coal": 500.0,
    "gas": 300.0,
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# HTTP
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT = 120
