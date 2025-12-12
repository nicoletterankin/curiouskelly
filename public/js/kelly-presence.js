/**
 * KellyPresence — shared presence/session/offline helpers (browser).
 *
 * Depends on `window.KellyTime` (load kelly-time.js first).
 */
(() => {
  function requireKellyTime() {
    if (!window.KellyTime) {
      throw new Error('KellyPresence requires window.KellyTime. Load /js/kelly-time.js first.');
    }
    return window.KellyTime;
  }

  function isOnline() {
    return typeof navigator !== 'undefined' ? !!navigator.onLine : true;
  }

  function attachOnlineOfflineHandlers({ onChange } = {}) {
    const handler = () => {
      try { onChange?.(isOnline()); } catch (_) {}
    };
    window.addEventListener('online', handler);
    window.addEventListener('offline', handler);
    return () => {
      window.removeEventListener('online', handler);
      window.removeEventListener('offline', handler);
    };
  }

  function startSession(storageKey = 'kellySessionStart') {
    try {
      localStorage.setItem(storageKey, String(Date.now()));
    } catch (_) {}
  }

  function endSession(storageKey = 'kellySessionStart') {
    try {
      localStorage.removeItem(storageKey);
    } catch (_) {}
  }

  function getSessionStart(storageKey = 'kellySessionStart') {
    try {
      const raw = localStorage.getItem(storageKey);
      const n = raw ? Number(raw) : NaN;
      return Number.isFinite(n) ? n : null;
    } catch (_) {
      return null;
    }
  }

  function getNowContext() {
    const KellyTime = requireKellyTime();
    const timeZone = KellyTime.getUserTimeZone();
    const utcMillis = Date.now();
    const { year, monthIndex, day } = KellyTime.getLocalDateParts(utcMillis, timeZone);
    return { utcMillis, timeZone, year, monthIndex, day };
  }

  const KellyPresence = {
    isOnline,
    attachOnlineOfflineHandlers,
    startSession,
    endSession,
    getSessionStart,
    getNowContext,
  };

  window.KellyPresence = KellyPresence;
})();

