import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ForecastChartProps {
  data: any[];
  loading: boolean;
}

const ForecastChart: React.FC<ForecastChartProps> = ({ data, loading }) => {
  const chartData = data.map(item => ({
    time: new Date(item.timestamp).toLocaleDateString('en-ZA', { month: 'short', day: 'numeric' }),
    solar: parseFloat(item.solar_forecast_mw.toFixed(1)),
    wind: parseFloat(item.wind_forecast_mw.toFixed(1)),
  }));

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-96 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading forecast data...</p>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-96 flex items-center justify-center">
        <p className="text-gray-400">No forecast data available. Select a location and click Refresh.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4">7-Day Energy Forecast</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="time" 
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="#9CA3AF"
            label={{ value: 'MW', angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF' } }}
            style={{ fontSize: '12px' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#F3F4F6' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="solar" 
            stroke="#F59E0B" 
            strokeWidth={3}
            name="Solar (MW)"
            dot={{ fill: '#F59E0B', r: 4 }}
          />
          <Line 
            type="monotone" 
            dataKey="wind" 
            stroke="#3B82F6" 
            strokeWidth={3}
            name="Wind (MW)"
            dot={{ fill: '#3B82F6', r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;