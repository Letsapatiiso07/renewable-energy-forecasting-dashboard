set -e

echo " Renewable Energy Forecasting Dashboard - Setup"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}Error: Python 3.9+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check Node.js
echo "Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version | cut -d'v' -f2)
echo -e "${GREEN}✓ Node.js $NODE_VERSION detected${NC}"

echo ""
echo "Setting up backend..."
echo "--------------------"

# Create Python virtual environment
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p ../data/raw
mkdir -p ../data/processed
mkdir -p ../data/models
mkdir -p ../data/logs
mkdir -p airflow/dags

# Initialize Airflow (if not already done)
if [ ! -f "airflow/airflow.db" ]; then
    echo "Initializing Airflow..."
    export AIRFLOW_HOME=$(pwd)/airflow
    airflow db init
    
    # Create default admin user
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || true
fi

cd ..

echo ""
echo "Setting up frontend..."
echo "--------------------"

cd frontend

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

# Create environment file
if [ ! -f ".env" ]; then
    echo "Creating frontend .env file..."
    cat > .env << EOF
VITE_API_URL=http://localhost:8000
EOF
fi

cd ..

echo ""
echo "Fetching initial data..."
echo "----------------------"

# Create initialization script
cat > backend/scripts/initialize.py << 'EOF'
"""
Initialize the system with data
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from data_processing.ingestion import WeatherDataIngestion, EnergyDataGenerator
from data_processing.feature_engineering import EnergyFeatureEngineer
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Starting data initialization...")
    
    # Fetch sample weather data (last 30 days)
    ingestion = WeatherDataIngestion()
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    df = ingestion.fetch_historical_weather("pretoria", start_date, end_date)
    
    if df is not None:
        ingestion.save_to_csv(df, "sample_weather.csv")
        
        # Generate energy data
        logger.info("Generating energy output data...")
        generator = EnergyDataGenerator()
        energy_df = generator.generate_energy_data(df)
        energy_df.to_csv(RAW_DATA_DIR / "sample_energy.csv", index=False)
        
        # Engineer features
        logger.info("Engineering features...")
        engineer = EnergyFeatureEngineer()
        featured_df = engineer.engineer_all_features(energy_df, is_training=True)
        featured_df.to_csv(PROCESSED_DATA_DIR / "sample_featured.csv", index=False)
        
        logger.info(f"Initialization complete! Processed {len(featured_df)} records")
    else:
        logger.error("Failed to fetch initial data")

if __name__ == "__main__":
    main()
EOF

# Run initialization
cd backend
mkdir -p scripts
python scripts/initialize.py

cd ..

echo ""
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "To start the application:"
echo ""
echo -e "${YELLOW}Terminal 1 - Backend API:${NC}"
echo "  cd backend"
echo "  source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "  uvicorn api.main:app --reload --port 8000"
echo ""
echo -e "${YELLOW}Terminal 2 - Frontend:${NC}"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo -e "${YELLOW}Terminal 3 - Airflow (Optional):${NC}"
echo "  cd backend"
echo "  export AIRFLOW_HOME=\$(pwd)/airflow"
echo "  airflow webserver --port 8080 &"
echo "  airflow scheduler &"
echo ""
echo "Access the dashboard at: http://localhost:5173"
echo "Access Airflow at: http://localhost:8080 (admin/admin)"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo -e "${GREEN}Happy forecasting! ${NC}"