'use client';

import { Learner } from '../../lib/sis-types';
import { formatTimeAgo, getSubscriptionBadge } from '../../lib/sis-utils';
import { Star, AlertTriangle, MoreVertical } from 'lucide-react';

interface LearnersTableProps {
  learners: Learner[];
  selectedIds: Set<string>;
  onSelect: (id: string) => void;
  onSelectAll: () => void;
  onRowClick: (learner: Learner) => void;
}

export function LearnersTable({ learners, selectedIds, onSelect, onSelectAll, onRowClick }: LearnersTableProps) {
  const allSelected = learners.length > 0 && learners.every(l => selectedIds.has(l.id));

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-700/50">
            <th className="p-4 text-left">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={onSelectAll}
                aria-label="Select all learners"
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
              />
            </th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Learner</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Location</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">IP Address</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Status</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Streak</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Lessons</th>
            <th className="p-4 text-left text-sm font-medium text-gray-400">Last Seen</th>
            <th className="p-4 w-10"></th>
          </tr>
        </thead>
        <tbody>
          {learners.map((learner) => {
            const badge = getSubscriptionBadge(learner.subscriptionStatus, learner.subscriptionPlan);
            return (
              <tr
                key={learner.id}
                onClick={() => onRowClick(learner)}
                className="border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer transition-colors"
              >
                <td className="p-4" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selectedIds.has(learner.id)}
                    onChange={() => onSelect(learner.id)}
                    aria-label={`Select ${learner.name || 'learner'}`}
                    className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
                  />
                </td>
                <td className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      {learner.avatar ? (
                        <img src={learner.avatar} alt="" className="w-10 h-10 rounded-full bg-gray-700" />
                      ) : (
                        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center text-gray-400 text-sm">?</div>
                      )}
                      {learner.isOnline && (
                        <span className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-gray-900"></span>
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={learner.name ? 'text-white font-medium' : 'text-gray-500 italic'}>
                          {learner.name || 'Anonymous'}
                        </span>
                        {learner.isVIP && <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />}
                        {learner.needsAttention && <AlertTriangle className="w-4 h-4 text-orange-400" />}
                      </div>
                      <div className="text-sm text-gray-500">{learner.email || learner.id}</div>
                    </div>
                  </div>
                </td>
                <td className="p-4">
                  <div>
                    <span className="mr-2">{learner.countryFlag}</span>
                    <span className="text-white">{learner.city || 'Unknown'}</span>
                  </div>
                  <div className="text-sm text-gray-500">{learner.country || '—'}</div>
                </td>
                <td className="p-4">
                  <code className="text-sm text-gray-400 font-mono">{learner.ip || '—'}</code>
                </td>
                <td className="p-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${badge.color}`}>
                    {badge.label}
                  </span>
                </td>
                <td className="p-4">
                  {learner.currentStreak > 0 ? (
                    <span className="text-orange-400">🔥 {learner.currentStreak}</span>
                  ) : (
                    <span className="text-gray-500">—</span>
                  )}
                </td>
                <td className="p-4">
                  <div className="text-white">{learner.lessonsCompleted}</div>
                  <div className="text-sm text-gray-500">Day {learner.currentDay}</div>
                </td>
                <td className="p-4 text-gray-400">
                  {formatTimeAgo(learner.lastSeenAt)}
                </td>
                <td className="p-4">
                  <button className="p-1 hover:bg-gray-700 rounded" aria-label="More options">
                    <MoreVertical className="w-4 h-4 text-gray-400" />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      
      {learners.length === 0 && (
        <div className="p-12 text-center">
          <p className="text-gray-400">No learners found</p>
        </div>
      )}
    </div>
  );
}
