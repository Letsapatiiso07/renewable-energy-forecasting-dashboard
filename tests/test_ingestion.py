"""
Tests for data ingestion module
"""
import pytest
import pandas as pd
from backend.data_processing.ingestion import WeatherDataIngestion, EnergyDataGenerator

def test_weather_ingestion_initialization():
    """Test WeatherDataIngestion initialization"""
    ingestion = WeatherDataIngestion()
    assert ingestion.session is not None

def test_energy_data_generator():
    """Test EnergyDataGenerator calculations"""
    generator = EnergyDataGenerator()
    
    # Test solar output calculation
    solar_output = generator.calculate_solar_output(
        radiation=800,  # W/m²
        temperature=25,  # °C
        cloudcover=20,  # %
        capacity_mw=100
    )
    assert 0 <= solar_output <= 100
    assert solar_output > 0  # Should produce some output
    
    # Test wind output calculation
    wind_output = generator.calculate_wind_output(
        windspeed=10,  # m/s
        capacity_mw=100
    )
    assert 0 <= wind_output <= 100
    assert wind_output > 0  # Should produce output at 10 m/s

def test_solar_output_edge_cases():
    """Test solar output edge cases"""
    generator = EnergyDataGenerator()
    
    # No radiation
    assert generator.calculate_solar_output(0, 25, 0) == 0
    
    # High cloud cover
    output_cloudy = generator.calculate_solar_output(1000, 25, 90)
    output_clear = generator.calculate_solar_output(1000, 25, 10)
    assert output_cloudy < output_clear

def test_wind_output_power_curve():
    """Test wind turbine power curve"""
    generator = EnergyDataGenerator()
    
    # Below cut-in speed
    assert generator.calculate_wind_output(2, 100) == 0
    
    # At rated speed
    assert generator.calculate_wind_output(12, 100) == 100
    
    # Above cut-out speed
    assert generator.calculate_wind_output(30, 100) == 0