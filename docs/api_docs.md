"""
# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (demo mode).
For production, implement JWT token-based authentication.

## Endpoints

### GET /
Root endpoint with API information.

**Response:**
```json
{
  "message": "Renewable Energy Forecasting API",
  "version": "1.0.0",
  "endpoints": ["/forecast", "/optimize", "/metrics", "/locations"]
}
```

### GET /locations
Get available forecast locations.

**Response:**
```json
{
  "locations": ["pretoria", "cape_town", "johannesburg", "durban", "port_elizabeth"],
  "details": {
    "pretoria": {
      "lat": -25.7479,
      "lon": 28.2293,
      "name": "Pretoria"
    }
  }
}
```

### POST /forecast
Get energy forecast for a specific location.

**Request Body:**
```json
{
  "location": "pretoria",
  "days": 7
}
```

**Parameters:**
- `location` (string, required): Location key from /locations
- `days` (integer, optional): Number of days to forecast (default: 7, max: 14)

**Response:**
```json
[
  {
    "location": "pretoria",
    "timestamp": "2025-01-10T12:00:00",
    "solar_forecast_mw": 82.5,
    "wind_forecast_mw": 45.2
  }
]
```

**Status Codes:**
- 200: Success
- 400: Invalid location or parameters
- 500: Server error (check logs)

### POST /optimize
Optimize energy allocation for cost minimization.

**Request Body:**
```json
{
  "demand_mw": 400,
  "solar_available": 80,
  "wind_available": 60
}
```

**Parameters:**
- `demand_mw` (float, required): Total grid demand in MW
- `solar_available` (float, required): Available solar capacity in MW
- `wind_available` (float, required): Available wind capacity in MW

**Response:**
```json
{
  "status": "optimal",
  "allocation": {
    "solar_mw": 80,
    "wind_mw": 60,
    "coal_mw": 60,
    "gas_mw": 0,
    "hydro_mw": 200
  },
  "optimized_cost": 36500,
  "baseline_cost": 60000,
  "cost_savings": 23500,
  "cost_reduction_pct": 39.2,
  "renewable_percentage": 85,
  "demand_mw": 400
}
```

### GET /metrics
Get system performance metrics.

**Response:**
```json
{
  "total_predictions": 10000,
  "avg_latency_ms": 250.0,
  "uptime_pct": 99.5,
  "cost_reduction_pct": 35.0
}
```

### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-10T12:00:00",
  "models_loaded": true
}
```

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message description"
}
```

Common error codes:
- 400: Bad Request (invalid parameters)
- 404: Not Found (invalid endpoint)
- 500: Internal Server Error
- 503: Service Unavailable (models not loaded)

## Rate Limiting
- Current: No rate limiting (demo)
- Recommended: 100 requests/minute per IP
- Headers: X-RateLimit-Remaining, X-RateLimit-Reset

## Best Practices
1. Cache forecast results (data updates every 30 min)
2. Handle errors gracefully with retries
3. Use batch optimization for multiple time periods
4. Monitor response times (<300ms target)
5. Log all API calls for debugging
"""