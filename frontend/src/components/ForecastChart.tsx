import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ForecastPoint } from '../services/api';

interface ForecastChartProps {
  points: ForecastPoint[];
  loading: boolean;
  source?: string;
}

const SOLAR = '#d95926';
const WIND = '#3987e5';

const formatDay = (ts: string) =>
  new Date(ts).toLocaleDateString('en-ZA', { weekday: 'short', day: 'numeric', month: 'short' });

const formatHour = (ts: string) =>
  new Date(ts).toLocaleString('en-ZA', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

const ForecastChart: React.FC<ForecastChartProps> = ({ points, loading, source }) => {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-96 flex items-center justify-center">
        <p className="text-gray-400">Loading forecast...</p>
      </div>
    );
  }

  if (points.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-96 flex items-center justify-center">
        <p className="text-gray-400">No forecast loaded. Check that the API is running and the models are trained.</p>
      </div>
    );
  }

  // One tick per day, at midnight
  const ticks = points.filter((p) => new Date(p.timestamp).getHours() === 0).map((p) => p.timestamp);

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-1">7-day hourly forecast</h2>
      <p className="text-gray-400 text-sm mb-4">
        XGBoost models applied to the live {source ?? 'weather'} forecast. Hypothetical 100 MW plants.
      </p>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={points} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid stroke="#374151" vertical={false} />
          <XAxis dataKey="timestamp" ticks={ticks} tickFormatter={formatDay} stroke="#9CA3AF" style={{ fontSize: '12px' }} />
          <YAxis
            domain={[0, 100]}
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
            label={{ value: 'MW', angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF' } }}
          />
          <Tooltip
            labelFormatter={(ts) => formatHour(String(ts))}
            formatter={(v: number) => `${v.toFixed(1)} MW`}
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#F3F4F6' }}
          />
          <Legend />
          <Line type="monotone" dataKey="solar_mw" stroke={SOLAR} strokeWidth={2} dot={false} name="Solar" />
          <Line type="monotone" dataKey="wind_mw" stroke={WIND} strokeWidth={2} dot={false} name="Wind" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;
