"""
Tests for data download helpers, physical power models and baselines
"""
import numpy as np
import pandas as pd

from data_processing.ingestion import WeatherDataIngestion, month_chunks
from data_processing.power_model import pv_output_mw, wind_output_mw
from ml.baselines import persistence, seasonal_naive
from utils.config import PLANT


def test_month_chunks_cover_range_without_overlap():
    chunks = list(month_chunks("2024-04-01", "2025-02-10", 3))
    assert chunks[0][0] == "2024-04-01"
    assert chunks[-1][1] == "2025-02-10"
    for (_, end), (start, _) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(start) - pd.Timestamp(end) == pd.Timedelta(days=1)


def test_invalid_location_returns_none():
    assert WeatherDataIngestion().fetch_forecast_weather("atlantis") is None


def test_pv_output_zero_at_night_and_within_capacity(weather):
    out = pv_output_mw(weather, -30.65, 24.01)
    night = weather["shortwave_radiation"] == 0
    assert np.all(out[night.to_numpy()] == 0)
    assert out.max() > 0
    assert out.max() <= PLANT["pv_ac_capacity_mw"]


def test_pv_output_drops_under_cloud(weather):
    noon = weather[weather["time"].dt.hour == 12].head(1)
    clear = noon.assign(shortwave_radiation=1000.0, direct_normal_irradiance=900.0, diffuse_radiation=100.0)
    cloudy = noon.assign(shortwave_radiation=300.0, direct_normal_irradiance=0.0, diffuse_radiation=300.0)
    assert pv_output_mw(cloudy, -30.65, 24.01)[0] < pv_output_mw(clear, -30.65, 24.01)[0]


def test_wind_power_curve():
    rated = PLANT["wind_capacity_mw"] * (1 - PLANT["wind_farm_losses"])
    out = wind_output_mw([2.0, 3.0, 8.0, 12.0, 20.0, 26.0])
    assert out[0] == 0 and out[1] == 0
    assert 0 < out[2] < rated
    assert out[3] == rated and out[4] == rated
    assert out[5] == 0
    speeds = np.linspace(3, 12, 50)
    assert np.all(np.diff(wind_output_mw(speeds)) >= 0)


def test_baselines_use_only_the_past():
    time = pd.date_range("2025-01-01", periods=72, freq="h")
    actual = pd.DataFrame({"time": time, "y": np.arange(72, dtype=float)})

    naive = seasonal_naive(actual, "y")
    assert naive.iloc[:24].isna().all()
    assert (naive.iloc[24:].to_numpy() == np.arange(48)).all()

    pers = persistence(actual, "y")
    assert pers.iloc[:24].isna().all()
    # every hour of day 2 gets 23:00 of day 1; day 3 gets 23:00 of day 2
    assert (pers.iloc[24:48] == 23).all()
    assert (pers.iloc[48:72] == 47).all()
