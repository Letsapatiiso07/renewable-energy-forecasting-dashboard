"""
Pytest configuration and fixtures. Tests use synthetic weather and never
call the network, so they run from a clean checkout.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Keep test models away from data/models; must be set before config is imported
os.environ["MODELS_DIR"] = tempfile.mkdtemp(prefix="ref_models_")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def make_weather(start: str, hours: int, seed: int = 0) -> pd.DataFrame:
    """Plausible hourly weather for one site"""
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=hours, freq="h")
    hour = time.hour.to_numpy()
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    cloud = rng.uniform(0, 100, hours)
    ghi = 1000 * sun * (1 - 0.7 * cloud / 100)
    wind100 = np.clip(8 + 4 * np.sin(np.arange(hours) / 9) + rng.normal(0, 2, hours), 0, None)
    return pd.DataFrame({
        "time": time,
        "shortwave_radiation": ghi,
        "direct_normal_irradiance": ghi * 0.9,
        "diffuse_radiation": ghi * 0.2,
        "temperature_2m": 20 + 6 * sun + rng.normal(0, 1, hours),
        "cloud_cover": cloud,
        "wind_speed_10m": wind100 * 0.7,
        "wind_speed_100m": wind100,
        "wind_direction_100m": rng.uniform(0, 360, hours),
    })


@pytest.fixture
def weather():
    return make_weather("2025-01-01", 24 * 10)
