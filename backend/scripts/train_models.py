import sys
from pathlib import Path
import pandas as pd
import argparse

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from ml.train import EnergyForecastModel
from utils.config import PROCESSED_DATA_DIR, MODELS_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_models(data_path: Path, targets=['solar', 'wind'], model_type='xgboost'):
    """
    Train models for specified targets
    
    Args:
        data_path: Path to featured dataset
        targets: List of targets to train ('solar', 'wind')
        model_type: Type of model ('xgboost' or 'lstm')
    """
    logger.info("=" * 60)
    logger.info("Starting model training...")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    results = []
    
    # Train models for each target
    for target in targets:
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"Training {model_type.upper()} model for {target} energy")
        logger.info("-" * 60)
        
        try:
            # Initialize model
            model = EnergyForecastModel(model_type=model_type, target=target)
            
            # Train
            result = model.train(df)
            
            # Log results
            metrics = result['metrics']
            logger.info(f"✓ Training complete for {target}")
            logger.info(f"  - MAE: {metrics['mae']:.2f} MW")
            logger.info(f"  - RMSE: {metrics['rmse']:.2f} MW")
            logger.info(f"  - R² Score: {metrics['r2']:.3f}")
            logger.info(f"  - Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"  - Target Met: {'YES' if metrics['target_met'] else 'NO'}")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to train {target} model: {e}", exc_info=True)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    for result in results:
        target = result['target']
        metrics = result['metrics']
        logger.info(f"{target.upper()}: Accuracy={metrics['accuracy']:.2%}, "
                   f"R²={metrics['r2']:.3f}, MAE={metrics['mae']:.2f}")
    
    logger.info("")
    logger.info("✓ All models trained and saved!")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start API: uvicorn api.main:app --reload")
    logger.info("2. Start frontend: cd ../frontend && npm run dev")
    logger.info("")
    
    return results


def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(description='Train energy forecasting models')
    parser.add_argument(
        '--data',
        type=str,
        default=str(PROCESSED_DATA_DIR / 'sample_featured.csv'),
        help='Path to featured dataset'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['solar', 'wind'],
        choices=['solar', 'wind'],
        help='Targets to train (solar, wind, or both)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lstm'],
        help='Model type to train'
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run: python scripts/initialize.py first")
        return 1
    
    # Train models
    try:
        train_models(data_path, args.targets, args.model)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())