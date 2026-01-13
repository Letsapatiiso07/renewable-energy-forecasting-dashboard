import os
from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{DATA_DIR / 'energy_forecast.db'}"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Open-Meteo API Configuration (Free, no API key required)
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
OPEN_METEO_ARCHIVE_URL = f"{OPEN_METEO_BASE_URL}/archive"
OPEN_METEO_FORECAST_URL = f"{OPEN_METEO_BASE_URL}/forecast"

# South African Locations (Lat, Lon)
LOCATIONS: Dict[str, Dict[str, float]] = {
    "pretoria": {"lat": -25.7479, "lon": 28.2293, "name": "Pretoria"},
    "cape_town": {"lat": -33.9249, "lon": 18.4241, "name": "Cape Town"},
    "johannesburg": {"lat": -26.2041, "lon": 28.0473, "name": "Johannesburg"},
    "durban": {"lat": -29.8587, "lon": 31.0218, "name": "Durban"},
    "port_elizabeth": {"lat": -33.9608, "lon": 25.6022, "name": "Port Elizabeth"}
}

# Weather parameters for Open-Meteo
WEATHER_PARAMS: List[str] = [
    "temperature_2m",
    "relativehumidity_2m",
    "windspeed_10m",
    "winddirection_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloudcover"
]

# Historical data range
HISTORICAL_START_DATE = "2019-01-01"
HISTORICAL_END_DATE = "2024-12-31"

# Forecast configuration
FORECAST_HOURS = 168  # 7 days
UPDATE_INTERVAL_MINUTES = 30

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# XGBoost parameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# LSTM parameters
LSTM_PARAMS = {
    "sequence_length": 24,  # 24 hours lookback
    "units": 128,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32
}

# Feature engineering
FEATURE_LAG_HOURS = [1, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOW_HOURS = [6, 12, 24]

# Cost optimization ($/kWh)
ENERGY_COSTS = {
    "solar": 0.05,
    "wind": 0.05,
    "coal": 0.15,
    "gas": 0.12,
    "hydro": 0.08
}

# Performance targets
TARGET_ACCURACY = 0.88
TARGET_LATENCY_MS = 300
TARGET_UPTIME = 0.995
TARGET_COST_REDUCTION = 0.35

# Monitoring and logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Airflow configuration
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", str(BASE_DIR / "backend" / "airflow"))
AIRFLOW_DAGS_FOLDER = os.path.join(AIRFLOW_HOME, "dags")

# Error handling
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT = 30