"""
Standalone model training script - Run this from the backend directory
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
print("Renewable Energy Dashboard - Model Training")
print("=" * 60)
print()

# Import modules
try:
    from ml.train import EnergyForecastModel
    from utils.config import PROCESSED_DATA_DIR, MODELS_DIR
    from utils.logger import setup_logger
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print()
    print("Make sure you have:")
    print("1. Installed all dependencies: pip install -r requirements.txt")
    print("2. Run initialization first: python run_initialization.py")
    sys.exit(1)

import pandas as pd
from pathlib import Path

logger = setup_logger(__name__)


def train_models(data_path: Path, targets=['solar', 'wind'], model_type='xgboost'):
    """
    Train models for specified targets
    
    Args:
        data_path: Path to featured dataset
        targets: List of targets to train ('solar', 'wind')
        model_type: Type of model ('xgboost' or 'lstm')
    """
    print()
    print("Starting model training...")
    print(f"Model type: {model_type.upper()}")
    print(f"Targets: {', '.join(targets)}")
    print()
    
    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print()
        print("Please run initialization first:")
        print("  python run_initialization.py")
        return 1
    
    # Load data
    print(f"Loading data from {data_path.name}...")
    try:
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    results = []
    
    # Train models for each target
    for idx, target in enumerate(targets, 1):
        print()
        print("-" * 60)
        print(f"Training {idx}/{len(targets)}: {model_type.upper()} model for {target.upper()} energy")
        print("-" * 60)
        
        try:
            # Initialize model
            model = EnergyForecastModel(model_type=model_type, target=target)
            
            # Train
            print("Training in progress (this may take 2-5 minutes)...")
            result = model.train(df)
            
            # Log results
            metrics = result['metrics']
            print()
            print("✓ Training complete!")
            print(f"  Results for {target.upper()}:")
            print(f"    • MAE (Mean Absolute Error):  {metrics['mae']:.2f} MW")
            print(f"    • RMSE (Root Mean Squared):   {metrics['rmse']:.2f} MW")
            print(f"    • R² Score:                   {metrics['r2']:.3f}")
            print(f"    • Accuracy:                   {metrics['accuracy']:.2%}")
            print(f"    • Target Met (88%+):          {'✓ YES' if metrics['target_met'] else '✗ NO'}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to train {target} model: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print()
        print("=" * 60)
        print("Training Summary")
        print("=" * 60)
        
        for result in results:
            target = result['target']
            metrics = result['metrics']
            status = '✓' if metrics['target_met'] else '✗'
            print(f"{status} {target.upper():6} | Accuracy: {metrics['accuracy']:.2%} | "
                  f"R²: {metrics['r2']:.3f} | MAE: {metrics['mae']:.2f} MW")
        
        print()
        print(f"✓ All models trained and saved to: {MODELS_DIR}")
        print()
        print("=" * 60)
        print("Next Steps")
        print("=" * 60)
        print()
        print("1. Start the API server:")
        print("   uvicorn api.main:app --reload --port 8000")
        print()
        print("2. In a new terminal, start the frontend:")
        print("   cd ../frontend")
        print("   npm run dev")
        print()
        print("3. Open your browser:")
        print("   http://localhost:5173")
        print()
        print("4. Test the API:")
        print("   http://localhost:8000/docs")
        print()
        
        return 0
    else:
        print()
        print("❌ No models were trained successfully")
        return 1


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train energy forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                    # Train solar and wind with XGBoost
  python run_training.py --model lstm       # Train with LSTM
  python run_training.py --targets solar    # Train only solar model
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(PROCESSED_DATA_DIR / 'sample_featured.csv'),
        help='Path to featured dataset (default: data/processed/sample_featured.csv)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['solar', 'wind'],
        choices=['solar', 'wind'],
        help='Targets to train: solar, wind, or both (default: both)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lstm'],
        help='Model type: xgboost or lstm (default: xgboost)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path
    data_path = Path(args.data)
    
    # Train models
    try:
        return train_models(data_path, args.targets, args.model)
    except KeyboardInterrupt:
        print()
        print()
        print("Training interrupted by user")
        return 1
    except Exception as e:
        print()
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())