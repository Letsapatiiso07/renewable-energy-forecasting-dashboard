"""
Features for day-ahead solar and wind forecasting.

Every feature comes from forecast weather, the clock or the site position,
so the same code runs on archived day-ahead forecasts (training and testing)
and on the live forecast (dashboard). No feature uses plant output, which
is what made the old version score 100%.
"""
import numpy as np
import pandas as pd

from data_processing.power_model import pv_output_mw, solar_position, wind_output_mw
from utils.config import LOCATIONS, WEATHER_PARAMS

FEATURE_COLUMNS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin",
    "wind_dir_cos",
    "shortwave_radiation_3h",
    "wind_speed_100m_3h",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "solar_elevation",
    "pv_physical_mw",
    "wind_physical_mw",
    "latitude",
    "longitude",
]


def build_features(weather: pd.DataFrame, location_key: str) -> pd.DataFrame:
    """Turn one site's hourly forecast weather into model features.

    `weather` must hold `time` plus WEATHER_PARAMS for a single site.
    Returns `time`, `location` and FEATURE_COLUMNS.
    """
    site = LOCATIONS[location_key]
    df = weather.sort_values("time").reset_index(drop=True)
    out = df[["time"] + WEATHER_PARAMS].copy()
    out["location"] = location_key

    direction = np.radians(df["wind_direction_100m"])
    out["wind_dir_sin"] = np.sin(direction)
    out["wind_dir_cos"] = np.cos(direction)

    # Centred 3-hour means of the forecast soften timing errors. All three
    # hours come from the same forecast, so nothing is taken from the future.
    out["shortwave_radiation_3h"] = df["shortwave_radiation"].rolling(3, center=True, min_periods=1).mean()
    out["wind_speed_100m_3h"] = df["wind_speed_100m"].rolling(3, center=True, min_periods=1).mean()

    hour = df["time"].dt.hour
    doy = df["time"].dt.day_of_year
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["solar_elevation"] = solar_position(df["time"], site["lat"], site["lon"])["apparent_elevation"].to_numpy()

    # The physical models applied to forecast weather
    out["pv_physical_mw"] = pv_output_mw(df, site["lat"], site["lon"])
    out["wind_physical_mw"] = wind_output_mw(df["wind_speed_100m"])

    out["latitude"] = site["lat"]
    out["longitude"] = site["lon"]
    return out[["time", "location"] + FEATURE_COLUMNS]
