"""
Standalone initialization script - Run this from the backend directory
This script doesn't rely on complex imports and will initialize your data
"""
import os
import sys

# Set the working directory to backend if not already there
if os.path.basename(os.getcwd()) != 'backend':
    if os.path.exists('backend'):
        os.chdir('backend')
    else:
        print("Error: Please run this from the project root or backend directory")
        sys.exit(1)

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("Renewable Energy Dashboard - Initialization")
print("=" * 60)
print()

# Now import after path is set
try:
    from data_processing.ingestion import WeatherDataIngestion, EnergyDataGenerator
    from data_processing.feature_engineering import EnergyFeatureEngineer
    from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from utils.logger import setup_logger
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print()
    print("Make sure you have:")
    print("1. Created all the required files in the correct locations")
    print("2. Installed all dependencies: pip install -r requirements.txt")
    print("3. Created __init__.py files in all package directories")
    sys.exit(1)

from datetime import datetime, timedelta
import pandas as pd

logger = setup_logger(__name__)

def main():
    """Initialize system with sample data"""
    print()
    print("Starting data initialization...")
    print()
    
    try:
        # Initialize components
        print("Initializing components...")
        ingestion = WeatherDataIngestion()
        generator = EnergyDataGenerator()
        engineer = EnergyFeatureEngineer()
        print("✓ Components initialized")
        
        # FIXED: Calculate proper historical date range
        # Open-Meteo archive API has data up to ~5 days ago, not including today
        # So we'll fetch data from 60 days ago to 10 days ago to be safe
        end_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        
        print()
        print(f"Fetching historical weather data:")
        print(f"  From: {start_date}")
        print(f"  To:   {end_date}")
        print()
        print("This may take 1-2 minutes...")
        print()
        
        # Fetch weather data
        df = ingestion.fetch_historical_weather("pretoria", start_date, end_date)
        
        if df is None or len(df) == 0:
            print()
            print("❌ Failed to fetch weather data")
            print()
            print("The Open-Meteo API might be:")
            print("  - Temporarily unavailable")
            print("  - Rate-limiting requests")
            print("  - Having issues with the specified date range")
            print()
            print("Solutions:")
            print("  1. Wait a few minutes and try again")
            print("  2. Check https://open-meteo.com to verify API status")
            print("  3. Try with a different date range")
            print()
            print("Alternative: Use the sample data generator below")
            print()
            
            # Offer to create sample data instead
            response = input("Create synthetic sample data instead? (y/n): ").lower()
            if response == 'y':
                print()
                print("Creating synthetic sample data...")
                df = create_synthetic_data(start_date, end_date)
                print(f"✓ Created {len(df)} synthetic weather records")
            else:
                return 1
        else:
            print(f"✓ Fetched {len(df)} weather records from Open-Meteo API")
        
        # Save raw weather data
        ingestion.save_to_csv(df, "sample_weather.csv")
        print(f"✓ Saved to {RAW_DATA_DIR / 'sample_weather.csv'}")
        
        # Generate energy data
        print()
        print("Generating synthetic energy output data...")
        energy_df = generator.generate_energy_data(df)
        energy_df.to_csv(RAW_DATA_DIR / "sample_energy.csv", index=False)
        print(f"✓ Generated {len(energy_df)} energy records")
        print(f"✓ Saved to {RAW_DATA_DIR / 'sample_energy.csv'}")
        
        # Engineer features
        print()
        print("Engineering features (this may take a minute)...")
        featured_df = engineer.engineer_all_features(energy_df, is_training=True)
        
        # Remove rows with NaN
        initial_count = len(featured_df)
        featured_df = featured_df.dropna()
        removed_count = initial_count - len(featured_df)
        
        if removed_count > 0:
            print(f"  Removed {removed_count} rows with missing values (from lag features)")
        
        featured_df.to_csv(PROCESSED_DATA_DIR / "sample_featured.csv", index=False)
        print(f"✓ Engineered features for {len(featured_df)} records")
        print(f"✓ Saved to {PROCESSED_DATA_DIR / 'sample_featured.csv'}")
        
        print()
        print("=" * 60)
        print("✓ Initialization complete!")
        print("=" * 60)
        print()
        print("Data summary:")
        print(f"  - Weather records: {len(df)}")
        print(f"  - Energy records: {len(energy_df)}")
        print(f"  - Featured records: {len(featured_df)}")
        print(f"  - Date range: {start_date} to {end_date}")
        print()
        print("Next steps:")
        print("1. Train models:")
        print("   python run_training.py")
        print()
        print("2. Start API:")
        print("   uvicorn api.main:app --reload")
        print()
        print("3. Start frontend:")
        print("   cd ../frontend && npm run dev")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Initialization failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("- Check your internet connection")
        print("- Verify dependencies: pip install -r requirements.txt")
        print("- Check that all files are in the correct locations")
        print("- View logs in data/logs/app.log")
        return 1


def create_synthetic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create synthetic weather data when API is unavailable
    This generates realistic weather patterns for Pretoria
    """
    import numpy as np
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate hourly timestamps
    timestamps = pd.date_range(start=start, end=end, freq='H')
    
    # Create synthetic weather data with realistic patterns
    np.random.seed(42)
    n = len(timestamps)
    
    # Extract hour and day of year for seasonal patterns
    hours = timestamps.hour
    days = timestamps.dayofyear
    
    # Temperature: 15-35°C with daily and seasonal variation
    base_temp = 25 + 8 * np.sin(2 * np.pi * days / 365)  # Seasonal
    daily_temp = -5 * np.cos(2 * np.pi * hours / 24)  # Daily cycle
    temp_noise = np.random.normal(0, 2, n)
    temperature = base_temp + daily_temp + temp_noise
    
    # Solar radiation: 0-1000 W/m² following sun pattern
    solar_base = 800 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_seasonal = 1 + 0.2 * np.sin(2 * np.pi * days / 365)
    solar_noise = np.random.normal(1, 0.15, n)
    shortwave_radiation = solar_base * solar_seasonal * solar_noise
    shortwave_radiation = np.clip(shortwave_radiation, 0, 1000)
    
    # Direct and diffuse radiation
    cloud_factor = np.random.uniform(0.3, 1.0, n)
    direct_radiation = shortwave_radiation * cloud_factor
    diffuse_radiation = shortwave_radiation * (1 - cloud_factor * 0.7)
    
    # Wind speed: 0-20 m/s with variation
    wind_base = 6 + 4 * np.sin(2 * np.pi * hours / 24)
    wind_noise = np.random.lognormal(0, 0.5, n)
    windspeed = wind_base * wind_noise
    windspeed = np.clip(windspeed, 0, 25)
    
    # Wind direction: 0-360 degrees
    winddirection = np.random.uniform(0, 360, n)
    
    # Relative humidity: 30-90%
    humidity_base = 60 - 20 * np.sin(2 * np.pi * hours / 24)
    humidity_noise = np.random.normal(0, 10, n)
    humidity = np.clip(humidity_base + humidity_noise, 20, 95)
    
    # Cloud cover: 0-100%
    cloudcover = 100 * (1 - cloud_factor)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'temperature_2m': temperature,
        'relativehumidity_2m': humidity,
        'windspeed_10m': windspeed,
        'winddirection_10m': winddirection,
        'shortwave_radiation': shortwave_radiation,
        'direct_radiation': direct_radiation,
        'diffuse_radiation': diffuse_radiation,
        'cloudcover': cloudcover,
        'location': 'pretoria',
        'location_name': 'Pretoria'
    })
    
    return df


if __name__ == "__main__":
    sys.exit(main())