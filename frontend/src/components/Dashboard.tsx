import React, { useCallback, useEffect, useState } from 'react';
import { Sun, Wind, MapPin, RefreshCw } from 'lucide-react';
import {
  fetchForecast,
  fetchMetrics,
  fetchLocations,
  optimizeEnergy,
  ForecastPoint,
  ForecastResponse,
  Location,
  Metrics,
  OptimizationResult,
} from '../services/api';
import ForecastChart from './ForecastChart';
import MetricsCard from './MetricsCard';
import OptimizationPanel from './OptimizationPanel';

const pct = (x: number | null | undefined) => (x == null ? 'n/a' : `${(x * 100).toFixed(1)}%`);
const mw = (x: number) => `${x.toFixed(1)} MW`;
const day = (s: string) => s.slice(0, 10);

const Dashboard: React.FC = () => {
  const [locations, setLocations] = useState<Record<string, Location>>({});
  const [location, setLocation] = useState('de_aar');
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setOptimizationResult(null);
    try {
      setForecast(await fetchForecast(location, 7));
    } catch (e: any) {
      setForecast(null);
      setError(e?.response?.data?.detail ?? 'Could not reach the API.');
    }
    try {
      setMetrics(await fetchMetrics());
    } catch {
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  }, [location]);

  useEffect(() => {
    fetchLocations().then(setLocations).catch(() => setError('Could not reach the API.'));
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleOptimize = async (demand: number, point: ForecastPoint) => {
    try {
      setOptimizationResult(await optimizeEnergy(demand, point.solar_mw, point.wind_mw));
    } catch (e) {
      console.error('Optimization failed:', e);
    }
  };

  const evaluation = metrics?.evaluation;
  const solar = evaluation?.targets.solar.by_site[location];
  const wind = evaluation?.targets.wind.by_site[location];
  const test = evaluation?.split.test;
  const apiStats = metrics?.api;

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Renewable Energy Forecasting Dashboard</h1>
          <p className="text-gray-400 max-w-3xl">
            Day-ahead solar and wind forecasts for hypothetical 100 MW plants at South African sites.
            Plant output is modelled from weather data with physical formulas, not measured at a real plant.
          </p>
        </div>

        <div className="mb-6 flex flex-wrap items-center gap-3">
          <MapPin className="text-blue-400" />
          <select
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="bg-gray-800 text-white px-4 py-2 rounded-lg"
          >
            {Object.entries(locations).map(([key, loc]) => (
              <option key={key} value={key}>
                {loc.name}
              </option>
            ))}
          </select>
          <button
            onClick={loadData}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg flex items-center gap-2"
          >
            <RefreshCw className={loading ? 'animate-spin' : ''} size={18} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="mb-6 bg-red-900/40 border border-red-700 rounded-lg p-4 text-sm">{error}</div>
        )}

        {solar && wind && test && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-3">
              <MetricsCard
                title="Solar MAE, held-out year"
                value={mw(solar.model.mae)}
                detail={`Seasonal naive ${mw(solar.seasonal_naive.mae)}, physical model ${mw(solar.physical_nwp.mae)}. Daylight hours.`}
                icon={<Sun size={16} />}
              />
              <MetricsCard
                title="Solar skill vs seasonal naive"
                value={pct(solar.skill_vs.seasonal_naive.mae)}
                detail={`vs physical model on the same forecast: ${pct(solar.skill_vs.physical_nwp.mae)}`}
                icon={<Sun size={16} />}
              />
              <MetricsCard
                title="Wind MAE, held-out year"
                value={mw(wind.model.mae)}
                detail={`Seasonal naive ${mw(wind.seasonal_naive.mae)}, physical model ${mw(wind.physical_nwp.mae)}. All hours.`}
                icon={<Wind size={16} />}
              />
              <MetricsCard
                title="Wind skill vs seasonal naive"
                value={pct(wind.skill_vs.seasonal_naive.mae)}
                detail={`vs physical model on the same forecast: ${pct(wind.skill_vs.physical_nwp.mae)}`}
                icon={<Wind size={16} />}
              />
            </div>
            <p className="text-gray-500 text-xs mb-8">
              Scores from the last training run on held-out data, {day(test[0])} to {day(test[1])}. Skill is
              1 minus model MAE over baseline MAE.
              {apiStats?.last_forecast_ms != null &&
                ` The last forecast request took ${apiStats.last_forecast_ms.toFixed(0)} ms (median ${apiStats.median_forecast_ms?.toFixed(0)} ms over ${apiStats.forecast_requests} requests since the API started), including the call to Open-Meteo.`}
            </p>
          </>
        )}

        <div className="mb-8">
          <ForecastChart points={forecast?.points ?? []} loading={loading} source={forecast?.weather_source} />
        </div>

        <OptimizationPanel points={forecast?.points ?? []} onOptimize={handleOptimize} result={optimizationResult} />

        <footer className="mt-8 text-gray-500 text-xs">
          Weather data by <a className="underline" href="https://open-meteo.com/">Open-Meteo.com</a> (
          <a className="underline" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>): ERA5
          reanalysis from Copernicus C3S and ECMWF IFS forecasts.
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
