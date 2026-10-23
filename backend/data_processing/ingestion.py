"""
Fetch hourly weather from Open-Meteo (free, no API key, CC BY 4.0).

Three sources, all with the same variable names:
- reanalysis: Historical Weather API, ERA5. Stands in for the weather that
  actually happened and drives the modelled "actual" output.
- day-ahead: Previous Runs API, ECMWF IFS 0.25 "_previous_day1" values, i.e.
  the forecast made 24 hours before each hour. These are the model inputs for
  training and testing, so the model only sees what a forecaster would have had.
- live: Forecast API, same ECMWF IFS 0.25 model, next 1 to 7 days.
"""
import time
from datetime import date, timedelta
from typing import Iterator, Optional, Tuple

import pandas as pd
import requests

from utils.config import (
    OPEN_METEO_ARCHIVE_URL, OPEN_METEO_PREVIOUS_RUNS_URL, OPEN_METEO_FORECAST_URL,
    WEATHER_PARAMS, LOCATIONS, RAW_DATA_DIR, TIMEZONE, REANALYSIS_MODEL, FORECAST_MODEL,
    MAX_RETRIES, RETRY_DELAY_SECONDS, REQUEST_TIMEOUT,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

DAY_AHEAD_SUFFIX = "_previous_day1"


def month_chunks(start: str, end: str, months: int) -> Iterator[Tuple[str, str]]:
    """Split [start, end] into ranges of a few months; long requests time out"""
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while current <= last:
        month = current.month - 1 + months
        next_start = date(current.year + month // 12, month % 12 + 1, 1)
        chunk_end = min(next_start - timedelta(days=1), last)
        yield current.isoformat(), chunk_end.isoformat()
        current = next_start


class WeatherDataIngestion:
    """Fetch weather data from Open-Meteo"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "renewable-energy-forecasting-dashboard"})

    def _make_request(self, url: str, params: dict) -> Optional[dict]:
        """GET with retries; returns parsed JSON or None"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        logger.error(f"All retry attempts failed for {url}")
        return None

    def _hourly_frame(self, url: str, location_key: str, suffix: str = "",
                      **extra) -> Optional[pd.DataFrame]:
        location = LOCATIONS[location_key]
        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "hourly": ",".join(p + suffix for p in WEATHER_PARAMS),
            "wind_speed_unit": "ms",
            "timezone": TIMEZONE,
            **extra,
        }
        data = self._make_request(url, params)
        if not data or "hourly" not in data:
            return None
        df = pd.DataFrame(data["hourly"])
        df = df.rename(columns={p + suffix: p for p in WEATHER_PARAMS})
        df["time"] = pd.to_datetime(df["time"])
        df["location"] = location_key
        return df

    def _fetch_range(self, url: str, location_key: str, start: str, end: str,
                     months: int, suffix: str = "", **extra) -> Optional[pd.DataFrame]:
        parts = []
        for chunk_start, chunk_end in month_chunks(start, end, months):
            df = self._hourly_frame(url, location_key, suffix,
                                    start_date=chunk_start, end_date=chunk_end, **extra)
            if df is None:
                logger.error(f"Failed to fetch {location_key} {chunk_start}..{chunk_end}")
                return None
            parts.append(df)
            time.sleep(1)  # be polite to the free API
        return pd.concat(parts, ignore_index=True)

    def fetch_reanalysis(self, location_key: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Hourly ERA5 reanalysis weather (Historical Weather API)"""
        logger.info(f"Fetching reanalysis weather for {location_key} {start}..{end}")
        return self._fetch_range(OPEN_METEO_ARCHIVE_URL, location_key, start, end, months=6,
                                 models=REANALYSIS_MODEL)

    def fetch_day_ahead(self, location_key: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Hourly forecasts issued 24 h before each hour (Previous Runs API)"""
        logger.info(f"Fetching archived day-ahead forecasts for {location_key} {start}..{end}")
        return self._fetch_range(OPEN_METEO_PREVIOUS_RUNS_URL, location_key, start, end,
                                 months=3, suffix=DAY_AHEAD_SUFFIX, models=FORECAST_MODEL)

    def fetch_forecast_weather(self, location_key: str, forecast_days: int = 7) -> Optional[pd.DataFrame]:
        """Live hourly forecast for the next `forecast_days` days"""
        if location_key not in LOCATIONS:
            logger.error(f"Invalid location: {location_key}")
            return None
        return self._hourly_frame(OPEN_METEO_FORECAST_URL, location_key,
                                  forecast_days=forecast_days, models=FORECAST_MODEL)

    def fetch_history(self, location_key: str, start: str, end: str,
                      refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reanalysis and day-ahead frames for one site, cached as CSV in data/raw"""
        frames = []
        sources = (("reanalysis", REANALYSIS_MODEL, self.fetch_reanalysis),
                   ("dayahead", FORECAST_MODEL, self.fetch_day_ahead))
        for kind, model, fetch in sources:
            path = RAW_DATA_DIR / f"{location_key}_{kind}_{model}_{start}_{end}.csv"
            if path.exists() and not refresh:
                df = pd.read_csv(path, parse_dates=["time"])
                logger.info(f"Loaded cached {path.name} ({len(df)} rows)")
            else:
                df = fetch(location_key, start, end)
                if df is None:
                    raise RuntimeError(f"Could not fetch {kind} data for {location_key}")
                df.to_csv(path, index=False)
                logger.info(f"Saved {path.name} ({len(df)} rows)")
            frames.append(df)
        return frames[0], frames[1]
