"""
Machine learning model training for renewable energy forecasting
FIXED for XGBoost 2.0+ API changes
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import json

from utils.config import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    XGBOOST_PARAMS, LSTM_PARAMS, TARGET_ACCURACY
)
from utils.logger import setup_logger, log_performance
import time

logger = setup_logger(__name__)


class EnergyForecastModel:
    """Train and manage energy forecasting models"""
    
    def __init__(self, model_type: str = "xgboost", target: str = "solar"):
        """
        Initialize model trainer
        
        Args:
            model_type: "xgboost" or "lstm"
            target: "solar" or "wind"
        """
        self.model_type = model_type
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features and target for training
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Target column
        target_col = f"{self.target}_output_mw"
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Remove non-feature columns
        exclude_cols = [
            "time", "location", "location_name", 
            "solar_output_mw", "wind_output_mw"
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].fillna(0).values
        y = df[target_col].fillna(0).values
        
        self.feature_columns = feature_cols
        logger.info(f"Prepared {len(feature_cols)} features for {self.target} prediction")
        
        return X, y, feature_cols
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary with train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Second split: train vs val
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
    
    def train_xgboost(self, data_splits: Dict[str, np.ndarray]) -> xgb.XGBRegressor:
        """
        Train XGBoost model with proper API for XGBoost 2.0+
        
        Args:
            data_splits: Dictionary with data splits
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        start_time = time.time()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(data_splits["X_train"])
        X_val_scaled = self.scaler.transform(data_splits["X_val"])
        
        # FIXED: Use callbacks for early stopping in XGBoost 2.0+
        model = xgb.XGBRegressor(
            **XGBOOST_PARAMS,
            callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)]
        )
        
        # Fit model
        model.fit(
            X_train_scaled, 
            data_splits["y_train"],
            eval_set=[(X_val_scaled, data_splits["y_val"])],
            verbose=False
        )
        
        duration_ms = (time.time() - start_time) * 1000
        log_performance(logger, "XGBOOST_TRAINING", duration_ms, success=True)
        
        return model
    
    def train_lstm(self, data_splits: Dict[str, np.ndarray]) -> keras.Model:
        """
        Train LSTM model for time series
        
        Args:
            data_splits: Dictionary with data splits
            
        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM model...")
        start_time = time.time()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(data_splits["X_train"])
        X_val_scaled = self.scaler.transform(data_splits["X_val"])
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train_scaled.reshape(
            (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        )
        X_val_reshaped = X_val_scaled.reshape(
            (X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
        )
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(
                LSTM_PARAMS["units"], 
                input_shape=(1, X_train_scaled.shape[1]),
                dropout=LSTM_PARAMS["dropout"],
                return_sequences=False
            ),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        # Train
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        
        model.fit(
            X_train_reshaped, data_splits["y_train"],
            validation_data=(X_val_reshaped, data_splits["y_val"]),
            epochs=LSTM_PARAMS["epochs"],
            batch_size=LSTM_PARAMS["batch_size"],
            callbacks=[early_stopping],
            verbose=0
        )
        
        duration_ms = (time.time() - start_time) * 1000
        log_performance(logger, "LSTM_TRAINING", duration_ms, success=True)
        
        return model
    
    def evaluate_model(self, model, data_splits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            data_splits: Dictionary with data splits
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test_scaled = self.scaler.transform(data_splits["X_test"])
        
        if self.model_type == "lstm":
            X_test_scaled = X_test_scaled.reshape(
                (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
            )
        
        y_pred = model.predict(X_test_scaled)
        if self.model_type == "lstm":
            y_pred = y_pred.flatten()
        
        y_test = data_splits["y_test"]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy (within 15% tolerance)
        tolerance = 0.15
        accurate_predictions = np.abs(y_test - y_pred) <= (y_test * tolerance + 1)
        accuracy = np.mean(accurate_predictions)
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "accuracy": float(accuracy),
            "target_met": bool(accuracy >= TARGET_ACCURACY)  # Convert to Python bool for JSON
        }
        
        self.metrics = metrics
        
        logger.info(f"Model Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, Accuracy: {accuracy:.2%}")
        
        return metrics
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete training pipeline
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting {self.model_type} training for {self.target} energy")
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(df)
        data_splits = self.split_data(X, y)
        
        # Train model
        if self.model_type == "xgboost":
            self.model = self.train_xgboost(data_splits)
        elif self.model_type == "lstm":
            self.model = self.train_lstm(data_splits)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate
        metrics = self.evaluate_model(self.model, data_splits)
        
        # Save model
        self.save_model()
        
        return {
            "model_type": self.model_type,
            "target": self.target,
            "metrics": metrics,
            "n_features": len(feature_cols),
            "n_samples": len(X)
        }
    
    def save_model(self):
        """Save trained model and artifacts"""
        model_dir = MODELS_DIR / f"{self.target}_{self.model_type}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model_type == "xgboost":
            model_path = model_dir / "model.joblib"
            joblib.dump(self.model, model_path)
        else:  # LSTM
            model_path = model_dir / "model.h5"
            self.model.save(model_path)
        
        # Save scaler
        scaler_path = model_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        features_path = model_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_columns, f)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self):
        """Load trained model and artifacts"""
        model_dir = MODELS_DIR / f"{self.target}_{self.model_type}"
        
        # Load model
        if self.model_type == "xgboost":
            model_path = model_dir / "model.joblib"
            self.model = joblib.load(model_path)
        else:  # LSTM
            model_path = model_dir / "model.h5"
            self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        features_path = model_dir / "features.json"
        with open(features_path, "r") as f:
            self.feature_columns = json.load(f)
        
        # Load metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)
        
        logger.info(f"Model loaded from {model_dir}")