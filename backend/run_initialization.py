"""
Step 1: download weather history from Open-Meteo and build the modelling table.

    cd backend
    python run_initialization.py            # uses cached CSVs in data/raw if present
    python run_initialization.py --refresh  # download again
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data_processing.dataset import build_site_dataset
from data_processing.ingestion import WeatherDataIngestion
from utils.config import HISTORY_END_DATE, HISTORY_START_DATE, LOCATIONS, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("run_initialization")
DATASET_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download weather data and build the dataset")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    args = parser.parse_args()

    ingestion = WeatherDataIngestion()
    parts = []
    for key in LOCATIONS:
        reanalysis, day_ahead = ingestion.fetch_history(
            key, HISTORY_START_DATE, HISTORY_END_DATE, refresh=args.refresh
        )
        site_df = build_site_dataset(reanalysis, day_ahead, key)
        logger.info(f"{key}: {len(site_df)} usable hours "
                    f"({len(day_ahead) - len(site_df)} dropped for forecast gaps)")
        parts.append(site_df)

    dataset = pd.concat(parts, ignore_index=True)
    dataset.to_csv(DATASET_PATH, index=False)
    logger.info(f"Wrote {len(dataset)} rows to {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
