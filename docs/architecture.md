"""
# System Architecture

## Overview
The Renewable Energy Forecasting Dashboard is built on a microservices architecture with clear separation of concerns:

1. **Data Layer**: Ingestion, storage, and preprocessing
2. **ML Layer**: Model training, inference, and evaluation
3. **Optimization Layer**: Cost optimization using linear programming
4. **API Layer**: RESTful API with FastAPI
5. **Presentation Layer**: React dashboard

## Component Details

### Data Ingestion Pipeline
- **Source**: Open-Meteo API (weather data)
- **Frequency**: Every 30 minutes for forecasts, daily for historical
- **Processing**: 
  - Data validation
  - Feature engineering (15+ features)
  - Synthetic energy generation based on weather
- **Storage**: SQLite for metadata, CSV for bulk data

### Machine Learning Pipeline
- **Models**: 
  - XGBoost for structured prediction (primary)
  - LSTM for time-series patterns (secondary)
- **Features**: 
  - Temporal: Hour, day, season encoding
  - Weather: Temperature, radiation, wind
  - Derived: Lag features, rolling statistics
- **Training**: Weekly retraining on new data
- **Evaluation**: MAE, RMSE, R², custom accuracy metric

### Optimization Engine
- **Algorithm**: Linear Programming (PuLP)
- **Objective**: Minimize total grid operational cost
- **Constraints**: 
  - Meet demand requirements
  - Respect capacity limits
  - Prioritize renewable sources
- **Output**: Optimal allocation across 5 energy sources

### API Architecture
- **Framework**: FastAPI with async support
- **Authentication**: None (demo), JWT recommended for production
- **Rate Limiting**: 100 requests/minute per IP
- **Caching**: Redis recommended for production
- **Documentation**: Auto-generated OpenAPI/Swagger

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **State Management**: React hooks (useState, useEffect)
- **Styling**: TailwindCSS for utility-first styling
- **Charts**: Recharts for data visualization
- **Build**: Vite for fast development and optimized builds

## Data Flow

```
Weather API → Ingestion → Feature Engineering → Model Prediction
                ↓                                     ↓
            Storage ←─────────── Optimization ←──────┘
                ↓                     ↓
            FastAPI ←────────────────┘
                ↓
         React Dashboard
```

## Scalability Considerations

### Current (Single Machine)
- Handles 10,000+ predictions/day
- 5 locations in South Africa
- 7-day forecast horizon

### Production Scaling
- Horizontal: Multiple API instances with load balancer
- Database: PostgreSQL with replication
- Cache: Redis for frequent queries
- Queue: RabbitMQ for async processing
- Monitoring: Prometheus + Grafana

## Security
- Input validation on all endpoints
- SQL injection prevention (parameterized queries)
- CORS configured for frontend domain
- Rate limiting to prevent abuse
- Sensitive data encryption at rest

## Monitoring & Logging
- Application logs: Structured JSON logs
- Performance metrics: Request latency, throughput
- Model metrics: Prediction accuracy, drift detection
- System metrics: CPU, memory, disk usage
- Alerting: Email/Slack for critical issues
"""