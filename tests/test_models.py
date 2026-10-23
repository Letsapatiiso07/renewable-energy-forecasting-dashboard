"""
Tests for features, the time split and the training pipeline
"""
import pandas as pd
import pytest

from conftest import make_weather
from data_processing.dataset import TARGETS, build_site_dataset
from data_processing.feature_engineering import FEATURE_COLUMNS, build_features
from ml import train
from ml.predict import EnergyPredictor
from utils.config import TEST_START, VALIDATION_START


def test_features_never_use_plant_output():
    banned = set(TARGETS.values())
    for col in FEATURE_COLUMNS:
        assert not any(t in col for t in banned), col
        assert "persistence" not in col and "seasonal_naive" not in col


def test_features_ignore_output_columns(weather):
    plain = build_features(weather, "de_aar")
    with_output = build_features(weather.assign(solar_mw=99.0, wind_mw=99.0), "de_aar")
    pd.testing.assert_frame_equal(plain, with_output)


@pytest.fixture(scope="module")
def dataset():
    # Synthetic history that spans the train / validation / test boundaries
    start, hours = "2025-05-01", 24 * 214
    reanalysis = make_weather(start, hours, seed=1)
    day_ahead = make_weather(start, hours, seed=1)
    day_ahead["wind_speed_100m"] *= 0.9  # forecasts are not perfect
    return build_site_dataset(reanalysis, day_ahead, "de_aar")


def test_time_split_is_ordered(dataset):
    tr, val, te = train.time_split(dataset)
    assert len(tr) and len(val) and len(te)
    assert tr["time"].max() < pd.Timestamp(VALIDATION_START) <= val["time"].min()
    assert val["time"].max() < pd.Timestamp(TEST_START) <= te["time"].min()


def test_train_and_evaluate(dataset, monkeypatch, tmp_path):
    monkeypatch.setitem(train.XGBOOST_PARAMS, "n_estimators", 40)
    metrics, predictors, forecasts = train.train_and_evaluate(dataset)

    assert forecasts["time"].min() >= pd.Timestamp(TEST_START)
    for target in TARGETS:
        overall = metrics["targets"][target]["overall"]
        for method in ("model", "persistence", "seasonal_naive", "physical_nwp"):
            assert overall[method]["mae"] >= 0
            assert overall[method]["rmse"] >= overall[method]["mae"]
        assert "seasonal_naive" in overall["skill_vs"]

    # save and reload gives the same predictions
    predictors["wind"].save(tmp_path)
    reloaded = EnergyPredictor.load("wind", tmp_path)
    test_rows = dataset[dataset["time"] >= TEST_START].head(48)
    assert (reloaded.predict(test_rows) == predictors["wind"].predict(test_rows)).all()
