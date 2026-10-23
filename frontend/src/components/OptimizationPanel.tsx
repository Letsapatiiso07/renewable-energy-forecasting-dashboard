import React, { useEffect, useState } from 'react';
import { DollarSign, Zap } from 'lucide-react';
import { ForecastPoint, OptimizationResult } from '../services/api';

interface OptimizationPanelProps {
  points: ForecastPoint[];
  onOptimize: (demand: number, point: ForecastPoint) => void;
  result: OptimizationResult | null;
}

const SOURCES = ['solar', 'wind', 'hydro', 'coal', 'gas'];

const formatHour = (ts: string) =>
  new Date(ts).toLocaleString('en-ZA', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

const OptimizationPanel: React.FC<OptimizationPanelProps> = ({ points, onOptimize, result }) => {
  const [demand, setDemand] = useState(600);
  const [index, setIndex] = useState(0);

  // Default to the first forecast hour at noon
  useEffect(() => {
    const noon = points.findIndex((p) => new Date(p.timestamp).getHours() === 12);
    setIndex(noon >= 0 ? noon : 0);
  }, [points]);

  const point = points[index];

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-1 flex items-center gap-2">
        <DollarSign className="text-green-400" size={20} />
        Least-cost dispatch for one hour
      </h2>
      <p className="text-gray-400 text-sm mb-4">
        A linear program meets demand with forecast solar and wind first, then hydro, coal and gas.
        Costs are illustrative placeholders, not market prices.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-gray-300 text-sm mb-1">Forecast hour</label>
          <select
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg mb-1"
          >
            {points.map((p, i) => (
              <option key={p.timestamp} value={i}>
                {formatHour(p.timestamp)}
              </option>
            ))}
          </select>
          {point && (
            <p className="text-gray-400 text-xs mb-4">
              Forecast: solar {point.solar_mw.toFixed(1)} MW, wind {point.wind_mw.toFixed(1)} MW
            </p>
          )}

          <label className="block text-gray-300 text-sm mb-1">Grid demand (MW)</label>
          <input
            type="number"
            value={demand}
            onChange={(e) => setDemand(Number(e.target.value))}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg mb-4"
            min="10"
            max="1200"
            step="10"
          />

          <button
            onClick={() => point && onOptimize(demand, point)}
            disabled={!point}
            className="w-full bg-green-700 hover:bg-green-800 disabled:opacity-50 text-white font-semibold py-2 px-6 rounded-lg flex items-center justify-center gap-2"
          >
            <Zap size={18} />
            Run dispatch
          </button>
        </div>

        <div>
          {result?.status === 'optimal' ? (
            <>
              <div className="space-y-2 mb-4">
                {SOURCES.map((s) => {
                  const mw = result.allocation[`${s}_mw`] ?? 0;
                  return (
                    <div key={s} className="text-sm">
                      <div className="flex justify-between text-gray-300">
                        <span className="capitalize">{s}</span>
                        <span>
                          {mw.toFixed(1)} MW at ${result.assumed_costs_usd_per_mwh[s]}/MWh
                        </span>
                      </div>
                      <div className="h-1.5 bg-gray-700 rounded">
                        <div className="h-1.5 bg-green-500 rounded" style={{ width: `${(mw / result.demand_mw) * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <p className="text-gray-400">Cost this hour</p>
                  <p className="text-white text-lg font-semibold">${result.cost_usd.toFixed(0)}</p>
                  <p className="text-gray-400 text-xs">Without solar and wind: ${result.baseline_cost_usd.toFixed(0)}</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <p className="text-gray-400">Saving from solar and wind</p>
                  <p className="text-white text-lg font-semibold">{result.saving_pct.toFixed(1)}%</p>
                  <p className="text-gray-400 text-xs">Solar and wind share: {result.solar_wind_share_pct.toFixed(1)}%</p>
                </div>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
              <p>{result?.message ?? 'Pick an hour and run the dispatch.'}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptimizationPanel;
