'use client';

import { Users, Wifi, CreditCard, Clock, Flame } from 'lucide-react';
import { DashboardStats } from '../../lib/sis-types';

interface StatsBarProps {
  stats: DashboardStats;
}

export function StatsBar({ stats }: StatsBarProps) {
  const items = [
    { icon: Users, label: 'Total', value: stats.total, color: 'text-blue-400' },
    { icon: Wifi, label: 'Online', value: stats.online, color: 'text-green-400' },
    { icon: CreditCard, label: 'Subscribed', value: stats.subscribed, color: 'text-emerald-400' },
    { icon: Clock, label: 'Trial', value: stats.trial, color: 'text-yellow-400' },
    { icon: Flame, label: 'Streaks', value: stats.activeStreaks, color: 'text-orange-400' },
  ];

  return (
    <div className="flex flex-wrap gap-2 p-4 bg-gray-800/50 rounded-xl border border-gray-700/50">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex items-center gap-2 px-4 py-2 bg-gray-900/50 rounded-lg"
        >
          <item.icon className={`w-4 h-4 ${item.color}`} />
          <span className="text-white font-semibold">{item.value.toLocaleString()}</span>
          <span className="text-gray-400 text-sm">{item.label}</span>
        </div>
      ))}
    </div>
  );
}
