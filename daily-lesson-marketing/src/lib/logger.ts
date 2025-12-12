type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const debugEnabled = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('debug');

function log(level: LogLevel, message: string, meta?: Record<string, unknown>) {
  if (level === 'debug' && !debugEnabled) {
    return;
  }
  const payload = { message, meta, level, timestamp: new Date().toISOString() };
  // eslint-disable-next-line no-console
  console[level === 'debug' ? 'log' : level](`[${level.toUpperCase()}] ${message}`, meta ?? '');
  window.dispatchEvent(new CustomEvent('logger:event', { detail: payload }));
}

export const logger = {
  debug: (message: string, meta?: Record<string, unknown>) => log('debug', message, meta),
  info: (message: string, meta?: Record<string, unknown>) => log('info', message, meta),
  warn: (message: string, meta?: Record<string, unknown>) => log('warn', message, meta),
  error: (message: string, meta?: Record<string, unknown>) => log('error', message, meta)
};












