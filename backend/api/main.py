"""
FastAPI app: live day-ahead forecasts, held-out test metrics and a dispatch demo.

Run from the backend folder:
    uvicorn api.main:app --port 8000
"""
import json
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_processing.feature_engineering import build_features
from data_processing.ingestion import WeatherDataIngestion
from ml.predict import EnergyPredictor
from optimization.cost_optimizer import GridCostOptimizer
from utils.config import FORECAST_MODEL, LOCATIONS, MODELS_DIR, PLANT
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)

weather_ingestion = WeatherDataIngestion()
optimizer = GridCostOptimizer()

# Loaded at startup; tests may replace these
state = {"predictors": {}, "metrics": None}
started_at = datetime.now()
forecast_latencies_ms: List[float] = []


def load_artifacts():
    for target in ("solar", "wind"):
        try:
            state["predictors"][target] = EnergyPredictor.load(target)
        except FileNotFoundError as e:
            logger.warning(f"{e}. /forecast will return 503 until models are trained.")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        state["metrics"] = json.loads(metrics_path.read_text())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Renewable Energy Forecasting API",
    description="Day-ahead solar and wind forecasts for hypothetical 100 MW plants "
                "at South African sites, with a least-cost dispatch demo",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    location: str
    days: int = Field(7, ge=1, le=7)


class ForecastPoint(BaseModel):
    timestamp: str
    solar_mw: float
    wind_mw: float


class ForecastResponse(BaseModel):
    location: str
    issued_at: str
    weather_source: str
    points: List[ForecastPoint]


class OptimizationRequest(BaseModel):
    demand_mw: float = Field(..., gt=0)
    solar_available: float = Field(0, ge=0)
    wind_available: float = Field(0, ge=0)


@app.get("/")
async def root():
    return {
        "message": "Renewable Energy Forecasting API",
        "endpoints": ["/locations", "/forecast", "/optimize", "/metrics", "/health", "/docs"],
    }


@app.get("/health")
async def health_check():
    loaded = {t: t in state["predictors"] for t in ("solar", "wind")}
    return {
        "status": "ok" if all(loaded.values()) else "models_missing",
        "models": loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/locations")
async def get_locations():
    return {"locations": list(LOCATIONS), "details": LOCATIONS, "plant": PLANT}


@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(request: ForecastRequest):
    """Hourly solar and wind forecast from the live Open-Meteo forecast"""
    start_time = time.time()
    if request.location not in LOCATIONS:
        raise HTTPException(400, f"Invalid location: {request.location}. "
                                 f"Valid locations: {list(LOCATIONS)}")
    if len(state["predictors"]) < 2:
        raise HTTPException(503, "Models not loaded. Train them first: python run_training.py")

    weather = weather_ingestion.fetch_forecast_weather(request.location, request.days)
    if weather is None or weather.empty:
        raise HTTPException(502, "Could not fetch the weather forecast from Open-Meteo")
    weather = weather.dropna()

    features = build_features(weather, request.location)
    solar = state["predictors"]["solar"].predict(features)
    wind = state["predictors"]["wind"].predict(features)

    points = [
        ForecastPoint(timestamp=t.isoformat(), solar_mw=round(float(s), 2), wind_mw=round(float(w), 2))
        for t, s, w in zip(features["time"], solar, wind)
    ]
    duration_ms = (time.time() - start_time) * 1000
    forecast_latencies_ms.append(duration_ms)
    log_performance(logger, "forecast request", duration_ms)
    return ForecastResponse(
        location=request.location,
        issued_at=datetime.now().isoformat(timespec="seconds"),
        weather_source=f"Open-Meteo Forecast API ({FORECAST_MODEL})",
        points=points,
    )


@app.post("/optimize")
def optimize_energy(request: OptimizationRequest):
    """Least-cost dispatch for one hour under illustrative costs"""
    return optimizer.optimize_allocation(
        demand_mw=request.demand_mw,
        solar_available=request.solar_available,
        wind_available=request.wind_available,
    )


@app.get("/metrics")
async def get_metrics():
    """Held-out test scores from the last training run, plus request timings
    measured by this API process since it started"""
    latencies = forecast_latencies_ms
    return {
        "evaluation": state["metrics"],
        "api": {
            "since": started_at.isoformat(timespec="seconds"),
            "forecast_requests": len(latencies),
            "median_forecast_ms": statistics.median(latencies) if latencies else None,
            "last_forecast_ms": latencies[-1] if latencies else None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
