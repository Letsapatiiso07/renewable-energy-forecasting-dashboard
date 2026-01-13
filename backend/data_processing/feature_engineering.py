"""
Feature engineering for renewable energy forecasting
FIXED: Handle both training data (with targets) and forecast data (without targets)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

from utils.config import FEATURE_LAG_HOURS, ROLLING_WINDOW_HOURS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EnergyFeatureEngineer:
    """Engineer features for renewable energy prediction"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["day_of_year"] = df["time"].dt.dayofyear
        df["month"] = df["time"].dt.month
        df["quarter"] = df["time"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Cyclical encoding for day of year
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        
        logger.info("Created temporal features")
        return df
    
    def create_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create solar-specific features"""
        df = df.copy()
        
        # Solar angle and daylight features
        df["solar_elevation"] = self._calculate_solar_elevation(
            df["time"], df.get("location", pd.Series(["unknown"] * len(df)))
        )
        
        # Radiation features
        if "shortwave_radiation" in df.columns:
            df["radiation_potential"] = df["shortwave_radiation"] * (1 - df.get("cloudcover", 0) / 100)
            
            # FIXED: Only calculate efficiency if we have actual output data
            if "solar_output_mw" in df.columns:
                df["radiation_efficiency"] = np.where(
                    df["shortwave_radiation"] > 0,
                    df["solar_output_mw"] / (df["shortwave_radiation"] + 1),
                    0
                )
        
        # Direct vs diffuse ratio
        if "direct_radiation" in df.columns and "diffuse_radiation" in df.columns:
            df["direct_diffuse_ratio"] = np.where(
                df["diffuse_radiation"] > 0,
                df["direct_radiation"] / (df["diffuse_radiation"] + 1),
                0
            )
        
        # Temperature impact on efficiency
        if "temperature_2m" in df.columns:
            df["temp_deviation"] = df["temperature_2m"] - 25  # Optimal temp
            df["temp_efficiency_factor"] = 1 - (df["temp_deviation"] * 0.004)
        
        logger.info("Created solar features")
        return df
    
    def create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create wind-specific features"""
        df = df.copy()
        
        # Wind power potential (cubic relationship)
        if "windspeed_10m" in df.columns:
            df["wind_power_potential"] = df["windspeed_10m"] ** 3
            
            # Wind speed categories
            df["wind_speed_category"] = pd.cut(
                df["windspeed_10m"],
                bins=[0, 3, 8, 15, 25, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # Wind direction features (cyclical encoding)
        if "winddirection_10m" in df.columns:
            df["wind_dir_sin"] = np.sin(np.radians(df["winddirection_10m"]))
            df["wind_dir_cos"] = np.cos(np.radians(df["winddirection_10m"]))
        
        logger.info("Created wind features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lag_hours: List[int] = FEATURE_LAG_HOURS) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = df.copy()
        df = df.sort_values(["location", "time"])
        
        for col in columns:
            if col in df.columns:
                for lag in lag_hours:
                    lag_col_name = f"{col}_lag_{lag}h"
                    df[lag_col_name] = df.groupby("location")[col].shift(lag)
                    self.feature_names.append(lag_col_name)
        
        logger.info(f"Created lag features for {len(columns)} columns")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                                columns: List[str],
                                windows: List[int] = ROLLING_WINDOW_HOURS) -> pd.DataFrame:
        """Create rolling window statistics"""
        df = df.copy()
        df = df.sort_values(["location", "time"])
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    roll_mean_name = f"{col}_roll_mean_{window}h"
                    df[roll_mean_name] = df.groupby("location")[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    self.feature_names.append(roll_mean_name)
                    
                    # Rolling std
                    roll_std_name = f"{col}_roll_std_{window}h"
                    df[roll_std_name] = df.groupby("location")[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    self.feature_names.append(roll_std_name)
        
        logger.info(f"Created rolling features for {len(columns)} columns")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        df = df.copy()
        
        # Solar-specific interactions
        if all(col in df.columns for col in ["temperature_2m", "cloudcover"]):
            df["temp_cloud_interaction"] = df["temperature_2m"] * (100 - df["cloudcover"])
        
        if all(col in df.columns for col in ["shortwave_radiation", "relativehumidity_2m"]):
            df["radiation_humidity"] = df["shortwave_radiation"] * (100 - df["relativehumidity_2m"])
        
        # Wind-specific interactions
        if all(col in df.columns for col in ["windspeed_10m", "temperature_2m"]):
            df["wind_temp_interaction"] = df["windspeed_10m"] * df["temperature_2m"]
        
        logger.info("Created interaction features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, 
                             is_training: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (has target columns)
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting feature engineering on {len(df)} records")
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Solar features
        df = self.create_solar_features(df)
        
        # Wind features
        df = self.create_wind_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Lag and rolling features (only for training or if we have sufficient history)
        if is_training:
            # For training data with actual targets
            weather_cols = ["temperature_2m", "windspeed_10m", "shortwave_radiation"]
            df = self.create_lag_features(df, weather_cols)
            df = self.create_rolling_features(df, weather_cols)
            
            # Target lags (if targets exist)
            if "solar_output_mw" in df.columns:
                df = self.create_lag_features(df, ["solar_output_mw"])
                df = self.create_rolling_features(df, ["solar_output_mw"])
            
            if "wind_output_mw" in df.columns:
                df = self.create_lag_features(df, ["wind_output_mw"])
                df = self.create_rolling_features(df, ["wind_output_mw"])
        else:
            # For forecast data, only use weather-based features
            # Set lag/rolling features to 0 (they'll be filled during prediction)
            weather_cols = ["temperature_2m", "windspeed_10m", "shortwave_radiation"]
            
            # Create lag features but fill with mean values
            for col in weather_cols:
                if col in df.columns:
                    col_mean = df[col].mean()
                    for lag in FEATURE_LAG_HOURS:
                        df[f"{col}_lag_{lag}h"] = col_mean
                    for window in ROLLING_WINDOW_HOURS:
                        df[f"{col}_roll_mean_{window}h"] = col_mean
                        df[f"{col}_roll_std_{window}h"] = df[col].std()
            
            # For target lags (these don't exist in forecast mode), fill with zeros
            for target in ["solar_output_mw", "wind_output_mw"]:
                for lag in FEATURE_LAG_HOURS:
                    df[f"{target}_lag_{lag}h"] = 0
                for window in ROLLING_WINDOW_HOURS:
                    df[f"{target}_roll_mean_{window}h"] = 0
                    df[f"{target}_roll_std_{window}h"] = 0
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    @staticmethod
    def _calculate_solar_elevation(time: pd.Series, location: pd.Series) -> pd.Series:
        """Calculate solar elevation angle (simplified)"""
        hour = time.dt.hour
        day_of_year = time.dt.dayofyear
        
        # Peak at solar noon (around 12:00)
        hour_angle = (hour - 12) * 15  # degrees
        
        # Seasonal variation (simplified)
        declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
        
        # Elevation (simplified, assumes mid-latitude)
        elevation = 90 - abs(hour_angle / 15) * 7.5 + declination / 2
        
        return elevation.clip(lower=0)