"""
Build the modelling table: one row per site and hour with
- features from the archived day-ahead forecast,
- modelled "actual" solar and wind output from reanalysis weather,
- the two naive baselines, computed from past actuals only.
"""
import pandas as pd

from data_processing.feature_engineering import build_features
from data_processing.power_model import pv_output_mw, wind_output_mw
from ml.baselines import persistence, seasonal_naive
from utils.config import LOCATIONS, WEATHER_PARAMS

TARGETS = {"solar": "solar_mw", "wind": "wind_mw"}


def modelled_actuals(reanalysis: pd.DataFrame, location_key: str) -> pd.DataFrame:
    site = LOCATIONS[location_key]
    df = reanalysis.sort_values("time").reset_index(drop=True)
    out = pd.DataFrame({"time": df["time"], "location": location_key})
    out["solar_mw"] = pv_output_mw(df, site["lat"], site["lon"])
    out["wind_mw"] = wind_output_mw(df["wind_speed_100m"])
    for col in TARGETS.values():
        out[f"{col}_persistence"] = persistence(out, col).to_numpy()
        out[f"{col}_seasonal_naive"] = seasonal_naive(out, col).to_numpy()
    return out


def build_site_dataset(reanalysis: pd.DataFrame, day_ahead: pd.DataFrame,
                       location_key: str) -> pd.DataFrame:
    actuals = modelled_actuals(reanalysis, location_key)
    features = build_features(day_ahead, location_key)
    # Drop hours where the archived forecast has gaps
    complete_times = day_ahead.loc[day_ahead[WEATHER_PARAMS].notna().all(axis=1), "time"]
    features = features[features["time"].isin(complete_times)]
    return features.merge(actuals, on=["time", "location"], how="inner")
