import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export interface ForecastPoint {
  timestamp: string;
  solar_mw: number;
  wind_mw: number;
}

export interface ForecastResponse {
  location: string;
  issued_at: string;
  weather_source: string;
  points: ForecastPoint[];
}

export interface OptimizationResult {
  status: string;
  message?: string;
  demand_mw: number;
  allocation: Record<string, number>;
  cost_usd: number;
  baseline_cost_usd: number;
  saving_usd: number;
  saving_pct: number;
  solar_wind_share_pct: number;
  assumed_costs_usd_per_mwh: Record<string, number>;
}

export interface Score {
  mae: number;
  rmse: number;
}

export interface SiteScores {
  model: Score;
  persistence: Score;
  seasonal_naive: Score;
  physical_nwp: Score;
  skill_vs: Record<string, { mae: number | null; rmse: number | null }>;
  n_hours: number;
}

export interface TargetEvaluation {
  overall: SiteScores;
  by_site: Record<string, SiteScores>;
  hours_scored: string;
}

export interface Metrics {
  evaluation: {
    split: Record<string, [string, string, number]>;
    targets: Record<'solar' | 'wind', TargetEvaluation>;
  } | null;
  api: {
    since: string;
    forecast_requests: number;
    median_forecast_ms: number | null;
    last_forecast_ms: number | null;
  };
}

export interface Location {
  lat: number;
  lon: number;
  name: string;
}

export const fetchForecast = async (location: string, days = 7): Promise<ForecastResponse> => {
  const response = await api.post<ForecastResponse>('/forecast', { location, days });
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

export const fetchLocations = async (): Promise<Record<string, Location>> => {
  const response = await api.get('/locations');
  return response.data.details;
};
