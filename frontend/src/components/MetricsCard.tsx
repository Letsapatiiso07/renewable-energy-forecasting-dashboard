import React from 'react';

interface MetricsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'yellow' | 'purple';
}

const MetricsCard: React.FC<MetricsCardProps> = ({ title, value, icon, color }) => {
  const colorClasses = {
    blue: 'from-blue-600 to-blue-800',
    green: 'from-green-600 to-green-800',
    yellow: 'from-yellow-600 to-yellow-800',
    purple: 'from-purple-600 to-purple-800',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-lg p-6 shadow-lg`}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-white opacity-80">{icon}</div>
      </div>
      <h3 className="text-gray-200 text-sm font-medium mb-1">{title}</h3>
      <p className="text-white text-3xl font-bold">{value}</p>
    </div>
  );
};

export default MetricsCard;