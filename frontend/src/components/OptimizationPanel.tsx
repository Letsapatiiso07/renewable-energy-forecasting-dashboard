import React, { useState } from 'react';
import { DollarSign, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface OptimizationPanelProps {
  onOptimize: (demand: number) => void;
  result: any;
}

const OptimizationPanel: React.FC<OptimizationPanelProps> = ({ onOptimize, result }) => {
  const [demand, setDemand] = useState(400);

  const handleOptimize = () => {
    onOptimize(demand);
  };

  const COLORS = ['#F59E0B', '#3B82F6', '#EF4444', '#8B5CF6', '#10B981'];

  const chartData = result?.allocation ? [
    { name: 'Solar', value: result.allocation.solar_mw },
    { name: 'Wind', value: result.allocation.wind_mw },
    { name: 'Coal', value: result.allocation.coal_mw },
    { name: 'Gas', value: result.allocation.gas_mw },
    { name: 'Hydro', value: result.allocation.hydro_mw },
  ].filter(item => item.value > 0) : [];

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <DollarSign className="text-green-400" />
        Cost Optimization
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Input Section */}
        <div>
          <label className="block text-gray-300 mb-2">Grid Demand (MW)</label>
          <input
            type="number"
            value={demand}
            onChange={(e) => setDemand(Number(e.target.value))}
            className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg mb-4"
            min="100"
            max="1000"
            step="10"
          />
          
          <button
            onClick={handleOptimize}
            className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center gap-2 transition-all"
          >
            <Zap size={20} />
            Optimize Allocation
          </button>

          {result && result.status === 'optimal' && (
            <div className="mt-6 space-y-3">
              <div className="bg-gray-700 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Cost Reduction</p>
                <p className="text-green-400 text-2xl font-bold">
                  {result.cost_reduction_pct.toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-gray-700 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Renewable Percentage</p>
                <p className="text-blue-400 text-2xl font-bold">
                  {result.renewable_percentage.toFixed(1)}%
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Total Cost</p>
                <p className="text-white text-xl font-bold">
                  ${result.optimized_cost.toFixed(2)}
                </p>
                <p className="text-gray-500 text-xs">
                  Baseline: ${result.baseline_cost.toFixed(2)}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div>
          {result && result.status === 'optimal' ? (
            <>
              <h3 className="text-lg font-semibold mb-4">Energy Source Allocation</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value.toFixed(1)} MW`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                </PieChart>
              </ResponsiveContainer>

              <div className="mt-4 space-y-2">
                {Object.entries(result.allocation).map(([source, value]: [string, any], idx) => (
                  value > 0 && (
                    <div key={source} className="flex justify-between items-center bg-gray-700 rounded p-3">
                      <span className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                        />
                        <span className="capitalize">{source.replace('_mw', '')}</span>
                      </span>
                      <span className="font-semibold">{value.toFixed(1)} MW</span>
                    </div>
                  )
                ))}
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400">
              <p>Run optimization to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptimizationPanel;