/**
 * Pipeline Status Dashboard Component
 * 
 * Displays real-time Kelly video pipeline status including:
 * - Provider availability
 * - Queue statistics
 * - Recent alerts
 * - Daily progress
 * 
 * @component
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================
// TYPES
// ============================================

export interface ProviderStatus {
  name: string;
  displayName: string;
  status: 'available' | 'degraded' | 'unavailable';
  lastCheck: string;
  successRate: number;
  avgProcessingTime?: number;
}

export interface QueueStats {
  queued: number;
  submitted: number;
  processing: number;
  completed_today: number;
  failed_today: number;
  blocked: number;
}

export interface PipelineAlert {
  id: string;
  type: 'eval_failure' | 'job_failure' | 'pipeline_error' | 'provider_down';
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  job_id?: string;
  day_of_year?: number;
  phase?: string;
  timestamp: string;
}

export interface PipelineStatusData {
  timestamp: string;
  providers: ProviderStatus[];
  queue: QueueStats;
  alerts: PipelineAlert[];
  health: 'healthy' | 'degraded' | 'critical';
  daily_progress: {
    target_day: number;
    phases_complete: number;
    phases_total: number;
    percent: number;
  };
}

export interface PipelineStatusProps {
  /** API endpoint URL (defaults to /api/pipeline/status) */
  apiUrl?: string;
  /** Refresh interval in milliseconds (default: 30000) */
  refreshInterval?: number;
  /** Show alerts section */
  showAlerts?: boolean;
  /** Compact mode for embedding */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

// ============================================
// STATUS INDICATOR COMPONENT
// ============================================

const StatusDot: React.FC<{ status: 'available' | 'degraded' | 'unavailable' | 'healthy' | 'critical' }> = ({ status }) => {
  const colors = {
    available: 'bg-green-500',
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    unavailable: 'bg-red-500',
    critical: 'bg-red-500',
  };
  
  return (
    <span 
      className={`inline-block w-3 h-3 rounded-full ${colors[status]} animate-pulse`}
      aria-label={status}
    />
  );
};

// ============================================
// PROGRESS BAR COMPONENT
// ============================================

const ProgressBar: React.FC<{ percent: number; label?: string }> = ({ percent, label }) => {
  const clampedPercent = Math.min(100, Math.max(0, percent));
  
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
          <span>{label}</span>
          <span>{percent}%</span>
        </div>
      )}
      <div 
        className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5"
        role="progressbar"
        aria-label={label || 'Progress'}
        aria-valuenow={clampedPercent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div 
          className={`bg-blue-600 h-2.5 rounded-full transition-all duration-500`}
          style={{ width: `${clampedPercent}%` }}
        />
      </div>
    </div>
  );
};

// ============================================
// ALERT ITEM COMPONENT
// ============================================

const AlertItem: React.FC<{ alert: PipelineAlert; onDismiss?: (id: string) => void }> = ({ alert, onDismiss }) => {
  const severityColors = {
    info: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
    warning: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
    error: 'border-red-500 bg-red-50 dark:bg-red-900/20',
    critical: 'border-red-700 bg-red-100 dark:bg-red-900/40',
  };
  
  const severityIcons = {
    info: 'ℹ️',
    warning: '⚠️',
    error: '❌',
    critical: '🚨',
  };
  
  const timeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };
  
  return (
    <div 
      className={`border-l-4 p-3 rounded-r ${severityColors[alert.severity]} mb-2`}
      role="alert"
    >
      <div className="flex justify-between items-start">
        <div className="flex items-start gap-2">
          <span aria-hidden="true">{severityIcons[alert.severity]}</span>
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {alert.day_of_year && `Day ${alert.day_of_year}`}
              {alert.phase && ` / ${alert.phase}`}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {alert.message.length > 100 ? `${alert.message.slice(0, 100)}...` : alert.message}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
              {timeAgo(alert.timestamp)}
            </p>
          </div>
        </div>
        {onDismiss && (
          <button
            onClick={() => onDismiss(alert.id)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            aria-label="Dismiss alert"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

// ============================================
// MAIN COMPONENT
// ============================================

export const PipelineStatus: React.FC<PipelineStatusProps> = ({
  apiUrl = '/api/pipeline/status',
  refreshInterval = 30000,
  showAlerts = true,
  compact = false,
  className = '',
}) => {
  const [data, setData] = useState<PipelineStatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(apiUrl);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const result = await response.json();
      setData(result);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);
  
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchStatus, refreshInterval]);
  
  const handleDismissAlert = (id: string) => {
    setDismissedAlerts(prev => new Set([...prev, id]));
  };
  
  if (loading && !data) {
    return (
      <div className={`p-4 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
          <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }
  
  if (error && !data) {
    return (
      <div className={`p-4 ${className}`}>
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-200">Failed to load pipeline status: {error}</p>
          <button
            onClick={fetchStatus}
            className="mt-2 text-sm text-red-600 dark:text-red-400 underline"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }
  
  if (!data) return null;
  
  const visibleAlerts = data.alerts.filter(a => !dismissedAlerts.has(a.id));
  
  return (
    <div className={`${className}`}>
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Pipeline Status
          </h2>
          <StatusDot status={data.health} />
          <span className="text-sm text-gray-500 dark:text-gray-400 capitalize">
            {data.health}
          </span>
        </div>
        {lastUpdated && (
          <span className="text-xs text-gray-400">
            Updated {lastUpdated.toLocaleTimeString()}
          </span>
        )}
      </div>
      
      {/* Providers Grid */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Video Providers
        </h3>
        <div className={`grid ${compact ? 'grid-cols-3' : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6'} gap-2`}>
          {data.providers.map(provider => (
            <div
              key={provider.name}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3"
            >
              <div className="flex items-center gap-2 mb-1">
                <StatusDot status={provider.status} />
                <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {provider.displayName}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                {provider.status}
              </p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Queue Stats */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Queue Status
        </h3>
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
          <div className={`grid ${compact ? 'grid-cols-3' : 'grid-cols-2 md:grid-cols-6'} gap-4`}>
            <div>
              <p className="text-2xl font-bold text-blue-600">{data.queue.queued}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Queued</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-600">{data.queue.processing}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Processing</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-green-600">{data.queue.completed_today}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Completed Today</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-red-600">{data.queue.failed_today}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Failed Today</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-orange-600">{data.queue.blocked}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Blocked</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-purple-600">{data.queue.submitted}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Submitted</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Daily Progress */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Day {data.daily_progress.target_day} Progress
        </h3>
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
          <ProgressBar 
            percent={data.daily_progress.percent} 
            label={`${data.daily_progress.phases_complete} / ${data.daily_progress.phases_total} phases`}
          />
        </div>
      </div>
      
      {/* Alerts */}
      {showAlerts && visibleAlerts.length > 0 && (
        <div>
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Recent Alerts ({visibleAlerts.length})
            </h3>
            {visibleAlerts.length > 0 && (
              <button
                onClick={() => setDismissedAlerts(new Set(data.alerts.map(a => a.id)))}
                className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              >
                Dismiss all
              </button>
            )}
          </div>
          <div className="max-h-64 overflow-y-auto">
            {visibleAlerts.map(alert => (
              <AlertItem 
                key={alert.id} 
                alert={alert} 
                onDismiss={handleDismissAlert}
              />
            ))}
          </div>
        </div>
      )}
      
      {/* No Alerts State */}
      {showAlerts && visibleAlerts.length === 0 && (
        <div className="text-center py-4 text-gray-500 dark:text-gray-400">
          <p className="text-sm">No recent alerts</p>
        </div>
      )}
    </div>
  );
};

export default PipelineStatus;
