"""
Prediction module for energy forecasting
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time
from typing import Dict, List, Optional
from tensorflow import keras

from utils.config import MODELS_DIR
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)


class EnergyPredictor:
    """Make predictions using trained models"""
    
    def __init__(self, target: str = "solar", model_type: str = "xgboost"):
        self.target = target
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.load_model()
    
    def load_model(self):
        """Load trained model artifacts"""
        model_dir = MODELS_DIR / f"{self.target}_{self.model_type}"
        
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Please train models first using: python run_training.py"
            )
        
        # Load model
        if self.model_type == "xgboost":
            model_path = model_dir / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
        else:
            model_path = model_dir / "model.h5"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        import json
        features_path = model_dir / "features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        with open(features_path, "r") as f:
            self.feature_columns = json.load(f)
        
        logger.info(f"✓ Loaded {self.target} {self.model_type} model with {len(self.feature_columns)} features")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Array of predictions
        """
        start_time = time.time()
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with zeros.")
            for feat in missing_features:
                df[feat] = 0
        
        # Select and order features
        X = df[self.feature_columns].fillna(0).values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if self.model_type == "lstm":
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        duration_ms = (time.time() - start_time) * 1000
        log_performance(
            logger,
            f"{self.target.upper()}_PREDICTION",
            duration_ms,
            success=True,
            metadata={"n_samples": len(predictions)}
        )
        
        return predictions
    
    def predict_single(self, features: Dict) -> float:
        """
        Make single prediction
        
        Args:
            features: Dictionary of features
            
        Returns:
            Single prediction value
        """
        df = pd.DataFrame([features])
        prediction = self.predict(df)[0]
        return max(0, float(prediction))