import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export interface ForecastData {
  location: string;
  timestamp: string;
  solar_forecast_mw: number;
  wind_forecast_mw: number;
}

export interface OptimizationResult {
  status: string;
  allocation: {
    solar_mw: number;
    wind_mw: number;
    coal_mw: number;
    gas_mw: number;
    hydro_mw: number;
  };
  optimized_cost: number;
  baseline_cost: number;
  cost_savings: number;
  cost_reduction_pct: number;
  renewable_percentage: number;
}

export interface Metrics {
  total_predictions: number;
  avg_latency_ms: number;
  uptime_pct: number;
  cost_reduction_pct: number;
}

export const fetchForecast = async (location: string, days: number = 7): Promise<ForecastData[]> => {
  const response = await api.post<ForecastData[]>('/forecast', { location, days });
  return response.data;
};

export const optimizeEnergy = async (
  demand_mw: number,
  solar_available: number,
  wind_available: number
): Promise<OptimizationResult> => {
  const response = await api.post<OptimizationResult>('/optimize', {
    demand_mw,
    solar_available,
    wind_available,
  });
  return response.data;
};

export const fetchMetrics = async (): Promise<Metrics> => {
  const response = await api.get<Metrics>('/metrics');
  return response.data;
};

export const fetchLocations = async () => {
  const response = await api.get('/locations');
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};