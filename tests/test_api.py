"""
Tests for FastAPI endpoints and the dispatch optimiser
"""
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from api import main
from conftest import make_weather
from data_processing.feature_engineering import FEATURE_COLUMNS, build_features
from ml.predict import EnergyPredictor
from optimization.cost_optimizer import GridCostOptimizer


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def tiny_models(monkeypatch):
    """Small models trained on synthetic data, plus a fake forecast fetch"""
    feats = build_features(make_weather("2025-01-01", 24 * 5), "pretoria")
    predictors = {}
    for target, col in (("solar", "pv_physical_mw"), ("wind", "wind_physical_mw")):
        model = xgb.XGBRegressor(n_estimators=10, max_depth=2, n_jobs=1)
        model.fit(feats[FEATURE_COLUMNS], feats[col])
        predictors[target] = EnergyPredictor(target, model, list(FEATURE_COLUMNS))
    monkeypatch.setitem(main.state, "predictors", predictors)
    monkeypatch.setattr(main.weather_ingestion, "fetch_forecast_weather",
                        lambda loc, days: make_weather("2026-10-24", 24 * days))


def test_root_and_locations(client):
    assert client.get("/").status_code == 200
    data = client.get("/locations").json()
    assert "pretoria" in data["locations"]
    assert data["plant"]["pv_ac_capacity_mw"] == 100


def test_health_reports_missing_models(client, monkeypatch):
    monkeypatch.setitem(main.state, "predictors", {})
    data = client.get("/health").json()
    assert data["status"] == "models_missing"
    assert client.post("/forecast", json={"location": "pretoria"}).status_code == 503


def test_forecast(client, tiny_models):
    response = client.post("/forecast", json={"location": "pretoria", "days": 2})
    assert response.status_code == 200
    points = response.json()["points"]
    assert len(points) == 48
    assert all(0 <= p["solar_mw"] <= 100 and 0 <= p["wind_mw"] <= 100 for p in points)
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_rejects_bad_input(client, tiny_models):
    assert client.post("/forecast", json={"location": "atlantis"}).status_code == 400
    assert client.post("/forecast", json={"location": "pretoria", "days": 30}).status_code == 422


def test_metrics_are_measured_not_fixed(client, tiny_models):
    before = client.get("/metrics").json()["api"]["forecast_requests"]
    client.post("/forecast", json={"location": "pretoria", "days": 1})
    api = client.get("/metrics").json()["api"]
    assert api["forecast_requests"] == before + 1
    assert api["last_forecast_ms"] > 0


def test_optimize_endpoint(client):
    data = client.post("/optimize", json={"demand_mw": 400, "solar_available": 80,
                                          "wind_available": 60}).json()
    assert data["status"] == "optimal"
    assert sum(data["allocation"].values()) == pytest.approx(400)
    assert client.post("/optimize", json={"demand_mw": 0}).status_code == 422


def test_optimizer_merit_order():
    result = GridCostOptimizer().optimize_allocation(400, solar_available=80, wind_available=60)
    alloc = result["allocation"]
    # cheapest sources first: all solar and wind, then hydro, then coal
    assert alloc["solar_mw"] == pytest.approx(80)
    assert alloc["wind_mw"] == pytest.approx(60)
    assert alloc["hydro_mw"] == pytest.approx(200)
    assert alloc["coal_mw"] == pytest.approx(60)
    assert alloc["gas_mw"] == pytest.approx(0)
    assert result["baseline_cost_usd"] > result["cost_usd"]


def test_optimizer_infeasible():
    result = GridCostOptimizer().optimize_allocation(5000, solar_available=0, wind_available=0)
    assert result["status"] == "infeasible"
