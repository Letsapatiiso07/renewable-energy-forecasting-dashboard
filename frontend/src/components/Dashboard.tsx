import React, { useState, useEffect } from 'react';
import { Sun, Wind, DollarSign, TrendingDown, MapPin, RefreshCw } from 'lucide-react';
import { fetchForecast, fetchMetrics, fetchLocations, optimizeEnergy } from '../services/api';
import ForecastChart from './ForecastChart';
import MetricsCard from './MetricsCard';
import OptimizationPanel from './OptimizationPanel';

const Dashboard: React.FC = () => {
  const [location, setLocation] = useState('pretoria');
  const [locations, setLocations] = useState<string[]>([]);
  const [forecastData, setForecastData] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState<any>(null);

  useEffect(() => {
    loadLocations();
    loadData();
  }, []);

  const loadLocations = async () => {
    try {
      const data = await fetchLocations();
      setLocations(data.locations);
    } catch (error) {
      console.error('Failed to load locations:', error);
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const [forecast, metricsData] = await Promise.all([
        fetchForecast(location, 7),
        fetchMetrics(),
      ]);
      setForecastData(forecast);
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async (demand: number) => {
    if (forecastData.length === 0) return;
    
    const latestForecast = forecastData[0];
    try {
      const result = await optimizeEnergy(
        demand,
        latestForecast.solar_forecast_mw,
        latestForecast.wind_forecast_mw
      );
      setOptimizationResult(result);
    } catch (error) {
      console.error('Optimization failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Renewable Energy Forecasting Dashboard</h1>
          <p className="text-gray-400">Real-time predictions for cost-optimized grids in South Africa</p>
        </div>

        {/* Location Selector */}
        <div className="mb-6 flex items-center gap-4">
          <MapPin className="text-blue-400" />
          <select
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="bg-gray-800 text-white px-4 py-2 rounded-lg"
          >
            {locations.map((loc) => (
              <option key={loc} value={loc}>
                {loc.charAt(0).toUpperCase() + loc.slice(1).replace('_', ' ')}
              </option>
            ))}
          </select>
          <button
            onClick={loadData}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg flex items-center gap-2"
          >
            <RefreshCw className={loading ? 'animate-spin' : ''} size={20} />
            Refresh
          </button>
        </div>

        {/* Metrics Cards */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <MetricsCard
              title="Total Predictions"
              value={metrics.total_predictions.toLocaleString()}
              icon={<TrendingDown />}
              color="blue"
            />
            <MetricsCard
              title="Avg Latency"
              value={`${metrics.avg_latency_ms}ms`}
              icon={<Sun />}
              color="green"
            />
            <MetricsCard
              title="System Uptime"
              value={`${metrics.uptime_pct}%`}
              icon={<Wind />}
              color="purple"
            />
            <MetricsCard
              title="Cost Reduction"
              value={`${metrics.cost_reduction_pct}%`}
              icon={<DollarSign />}
              color="yellow"
            />
          </div>
        )}

        {/* Forecast Chart */}
        <div className="mb-8">
          <ForecastChart data={forecastData} loading={loading} />
        </div>

        {/* Optimization Panel */}
        <OptimizationPanel
          onOptimize={handleOptimize}
          result={optimizationResult}
        />
      </div>
    </div>
  );
};

export default Dashboard;