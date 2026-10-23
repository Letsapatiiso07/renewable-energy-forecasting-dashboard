import React from 'react';

interface MetricsCardProps {
  title: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
}

const MetricsCard: React.FC<MetricsCardProps> = ({ title, value, detail, icon }) => (
  <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
    <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
      {icon}
      <h3>{title}</h3>
    </div>
    <p className="text-white text-3xl font-semibold mb-1">{value}</p>
    <p className="text-gray-400 text-xs leading-relaxed">{detail}</p>
  </div>
);

export default MetricsCard;
