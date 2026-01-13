"""
Tests for ML models
"""
import pytest
import numpy as np
import pandas as pd
from backend.ml.train import EnergyForecastModel

@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'hour': np.random.randint(0, 24, n_samples),
        'temperature_2m': np.random.uniform(10, 35, n_samples),
        'windspeed_10m': np.random.uniform(0, 20, n_samples),
        'shortwave_radiation': np.random.uniform(0, 1000, n_samples),
        'cloudcover': np.random.uniform(0, 100, n_samples),
        'solar_output_mw': np.random.uniform(0, 100, n_samples),
        'wind_output_mw': np.random.uniform(0, 100, n_samples),
    }
    
    return pd.DataFrame(data)

def test_model_initialization():
    """Test model initialization"""
    model = EnergyForecastModel(model_type="xgboost", target="solar")
    assert model.model_type == "xgboost"
    assert model.target == "solar"
    assert model.model is None  # Not trained yet

def test_data_preparation(sample_data):
    """Test data preparation"""
    model = EnergyForecastModel(model_type="xgboost", target="solar")
    X, y, feature_cols = model.prepare_data(sample_data)
    
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)
    assert len(feature_cols) > 0

def test_data_split(sample_data):
    """Test train/val/test split"""
    model = EnergyForecastModel(model_type="xgboost", target="solar")
    X, y, _ = model.prepare_data(sample_data)
    splits = model.split_data(X, y)
    
    assert "X_train" in splits
    assert "X_val" in splits
    assert "X_test" in splits
    assert len(splits["X_train"]) > len(splits["X_val"])
    assert len(splits["X_train"]) > len(splits["X_test"])