import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path - CRITICAL for imports to work
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

# Change to backend directory
os.chdir(backend_path)

# Now import with absolute imports
from data_processing.ingestion import WeatherDataIngestion, EnergyDataGenerator
from data_processing.feature_engineering import EnergyFeatureEngineer
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Initialize system with sample data"""
    logger.info("=" * 60)
    logger.info("Starting data initialization...")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        ingestion = WeatherDataIngestion()
        generator = EnergyDataGenerator()
        engineer = EnergyFeatureEngineer()
        
        # Calculate date range (last 30 days for quick setup)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching weather data from {start_date} to {end_date}")
        logger.info("This may take a few minutes...")
        
        # Fetch weather data for Pretoria (sample location)
        df = ingestion.fetch_historical_weather("pretoria", start_date, end_date)
        
        if df is None or len(df) == 0:
            logger.error("Failed to fetch weather data. Please check your internet connection.")
            logger.info("You can try running this script again later.")
            return 1
        
        logger.info(f"✓ Fetched {len(df)} weather records")
        
        # Save raw weather data
        ingestion.save_to_csv(df, "sample_weather.csv")
        logger.info(f"✓ Saved to {RAW_DATA_DIR / 'sample_weather.csv'}")
        
        # Generate energy data
        logger.info("Generating synthetic energy output data...")
        energy_df = generator.generate_energy_data(df)
        energy_df.to_csv(RAW_DATA_DIR / "sample_energy.csv", index=False)
        logger.info(f"✓ Generated {len(energy_df)} energy records")
        logger.info(f"✓ Saved to {RAW_DATA_DIR / 'sample_energy.csv'}")
        
        # Engineer features
        logger.info("Engineering features (this may take a minute)...")
        featured_df = engineer.engineer_all_features(energy_df, is_training=True)
        
        # Remove rows with NaN (from lag features)
        featured_df = featured_df.dropna()
        
        featured_df.to_csv(PROCESSED_DATA_DIR / "sample_featured.csv", index=False)
        logger.info(f"✓ Engineered features for {len(featured_df)} records")
        logger.info(f"✓ Saved to {PROCESSED_DATA_DIR / 'sample_featured.csv'}")
        
        logger.info("=" * 60)
        logger.info("✓ Initialization complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Train models: python scripts/train_models.py")
        logger.info("2. Start API: uvicorn api.main:app --reload")
        logger.info("3. Start frontend: cd ../frontend && npm run dev")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("- Check your internet connection")
        logger.info("- Verify all dependencies are installed: pip install -r requirements.txt")
        logger.info("- Check logs in data/logs/app.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())