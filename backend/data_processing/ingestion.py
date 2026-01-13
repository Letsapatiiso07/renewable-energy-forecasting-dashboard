import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Use absolute imports instead of relative
from utils.config import (
    OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL,
    WEATHER_PARAMS, LOCATIONS, RAW_DATA_DIR,
    MAX_RETRIES, RETRY_DELAY_SECONDS, REQUEST_TIMEOUT,
    HISTORICAL_START_DATE, HISTORICAL_END_DATE
)
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)


class WeatherDataIngestion:
    """Fetch weather data from Open-Meteo API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RenewableEnergyDashboard/1.0"})
    
    def _make_request(self, url: str, params: dict) -> Optional[dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                duration_ms = (time.time() - start_time) * 1000
                
                response.raise_for_status()
                log_performance(logger, "API_REQUEST", duration_ms, success=True, 
                              metadata={"url": url, "attempt": attempt + 1})
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None
        
        return None
    
    def fetch_historical_weather(self, location_key: str, 
                                 start_date: str = HISTORICAL_START_DATE,
                                 end_date: str = HISTORICAL_END_DATE) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather data for a location
        
        Args:
            location_key: Key from LOCATIONS dict
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with weather data or None if failed
        """
        if location_key not in LOCATIONS:
            logger.error(f"Invalid location: {location_key}")
            return None
        
        location = LOCATIONS[location_key]
        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_PARAMS),
            "timezone": "Africa/Johannesburg"
        }
        
        logger.info(f"Fetching historical weather for {location['name']}")
        data = self._make_request(OPEN_METEO_ARCHIVE_URL, params)
        
        if not data or "hourly" not in data:
            logger.error(f"Failed to fetch historical data for {location_key}")
            return None
        
        df = pd.DataFrame(data["hourly"])
        df["location"] = location_key
        df["location_name"] = location["name"]
        df["time"] = pd.to_datetime(df["time"])
        
        logger.info(f"Fetched {len(df)} historical records for {location['name']}")
        return df
    
    def fetch_forecast_weather(self, location_key: str, 
                               forecast_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Fetch weather forecast data
        
        Args:
            location_key: Key from LOCATIONS dict
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with forecast data or None if failed
        """
        if location_key not in LOCATIONS:
            logger.error(f"Invalid location: {location_key}")
            return None
        
        location = LOCATIONS[location_key]
        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "hourly": ",".join(WEATHER_PARAMS),
            "forecast_days": forecast_days,
            "timezone": "Africa/Johannesburg"
        }
        
        logger.info(f"Fetching forecast weather for {location['name']}")
        data = self._make_request(OPEN_METEO_FORECAST_URL, params)
        
        if not data or "hourly" not in data:
            logger.error(f"Failed to fetch forecast data for {location_key}")
            return None
        
        df = pd.DataFrame(data["hourly"])
        df["location"] = location_key
        df["location_name"] = location["name"]
        df["time"] = pd.to_datetime(df["time"])
        
        logger.info(f"Fetched {len(df)} forecast records for {location['name']}")
        return df
    
    def fetch_all_locations_historical(self) -> pd.DataFrame:
        """Fetch historical data for all configured locations"""
        all_data = []
        
        for location_key in LOCATIONS.keys():
            df = self.fetch_historical_weather(location_key)
            if df is not None:
                all_data.append(df)
                time.sleep(1)  # Rate limiting courtesy
        
        if not all_data:
            logger.error("No historical data fetched for any location")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total historical records: {len(combined_df)}")
        return combined_df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV in raw data directory"""
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")


class EnergyDataGenerator:
    """Generate synthetic energy output data based on weather"""
    
    @staticmethod
    def calculate_solar_output(radiation: float, temperature: float, 
                               cloudcover: float, capacity_mw: float = 100) -> float:
        """
        Calculate solar PV output based on weather conditions
        
        Args:
            radiation: Solar radiation (W/m²)
            temperature: Temperature (°C)
            cloudcover: Cloud cover (%)
            capacity_mw: Installed capacity in MW
            
        Returns:
            Solar output in MW
        """
        if pd.isna(radiation) or radiation <= 0:
            return 0.0
        
        # Temperature coefficient (efficiency decreases with heat)
        temp_coeff = max(0, 1 - (temperature - 25) * 0.004)
        
        # Cloud cover impact
        cloud_factor = 1 - (cloudcover / 100) * 0.7
        
        # Standard test conditions: 1000 W/m² at 25°C
        efficiency = 0.18  # Typical PV efficiency
        output = (radiation / 1000) * capacity_mw * efficiency * temp_coeff * cloud_factor
        
        return max(0, min(output, capacity_mw))
    
    @staticmethod
    def calculate_wind_output(windspeed: float, capacity_mw: float = 100) -> float:
        """
        Calculate wind turbine output based on wind speed
        
        Args:
            windspeed: Wind speed (m/s)
            capacity_mw: Installed capacity in MW
            
        Returns:
            Wind output in MW
        """
        if pd.isna(windspeed) or windspeed <= 0:
            return 0.0
        
        # Typical wind turbine power curve
        cut_in_speed = 3.0  # m/s
        rated_speed = 12.0  # m/s
        cut_out_speed = 25.0  # m/s
        
        if windspeed < cut_in_speed or windspeed > cut_out_speed:
            return 0.0
        elif windspeed >= rated_speed:
            return capacity_mw
        else:
            # Cubic relationship between cut-in and rated speed
            power_coefficient = ((windspeed - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
            return capacity_mw * power_coefficient
    
    def generate_energy_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic energy output from weather data
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            DataFrame with added energy output columns
        """
        df = weather_df.copy()
        
        # Calculate solar output
        df["solar_output_mw"] = df.apply(
            lambda row: self.calculate_solar_output(
                row.get("shortwave_radiation", 0),
                row.get("temperature_2m", 25),
                row.get("cloudcover", 0)
            ),
            axis=1
        )
        
        # Calculate wind output
        df["wind_output_mw"] = df.apply(
            lambda row: self.calculate_wind_output(row.get("windspeed_10m", 0)),
            axis=1
        )
        
        # Add realistic noise
        np.random.seed(42)
        df["solar_output_mw"] *= np.random.normal(1, 0.05, len(df))
        df["wind_output_mw"] *= np.random.normal(1, 0.08, len(df))
        
        # Ensure non-negative
        df["solar_output_mw"] = df["solar_output_mw"].clip(lower=0)
        df["wind_output_mw"] = df["wind_output_mw"].clip(lower=0)
        
        logger.info(f"Generated energy data for {len(df)} records")
        return df