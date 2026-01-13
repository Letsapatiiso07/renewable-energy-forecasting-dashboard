"""
Airflow DAG for data ingestion
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))

from data_processing.ingestion import WeatherDataIngestion, EnergyDataGenerator
from data_processing.feature_engineering import EnergyFeatureEngineer
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'energy_data_ingestion',
    default_args=default_args,
    description='Ingest weather and energy data',
    schedule_interval='@daily',
    catchup=False,
)

def fetch_historical_data():
    """Fetch historical weather data"""
    logger.info("Fetching historical weather data")
    ingestion = WeatherDataIngestion()
    df = ingestion.fetch_all_locations_historical()
    ingestion.save_to_csv(df, "historical_weather.csv")
    logger.info(f"Saved {len(df)} historical records")

def generate_energy_data():
    """Generate energy output from weather data"""
    logger.info("Generating energy data")
    import pandas as pd
    
    weather_df = pd.read_csv(RAW_DATA_DIR / "historical_weather.csv")
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    
    generator = EnergyDataGenerator()
    energy_df = generator.generate_energy_data(weather_df)
    
    energy_df.to_csv(RAW_DATA_DIR / "historical_energy.csv", index=False)
    logger.info(f"Generated {len(energy_df)} energy records")

def engineer_features():
    """Engineer features from raw data"""
    logger.info("Engineering features")
    import pandas as pd
    
    energy_df = pd.read_csv(RAW_DATA_DIR / "historical_energy.csv")
    energy_df["time"] = pd.to_datetime(energy_df["time"])
    
    engineer = EnergyFeatureEngineer()
    featured_df = engineer.engineer_all_features(energy_df, is_training=True)
    
    featured_df.to_csv(PROCESSED_DATA_DIR / "featured_data.csv", index=False)
    logger.info(f"Engineered features for {len(featured_df)} records")

# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_historical_data',
    python_callable=fetch_historical_data,
    dag=dag,
)

generate_task = PythonOperator(
    task_id='generate_energy_data',
    python_callable=generate_energy_data,
    dag=dag,
)

feature_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

# Set dependencies
fetch_task >> generate_task >> feature_task