/**
 * KellyTime — the only allowed place for date/time operations in Kelly OS (browser).
 *
 * KELLY TIME AUTHORITY: "When Kelly says 9:00:00 AM, class starts."
 * 
 * This module provides:
 * - Server-authoritative time synchronization (NTP-like)
 * - Timezone handling and conversion
 * - Live clock display components
 * - Live class countdown
 * - Calendar date/day-number conversions
 *
 * Rules:
 * - NEVER show "Day X" to users. Always show real dates like "December 12, 2025".
 * - Canonical lesson identity is `dayNumber` in range 1–365, keyed to calendar month/day.
 * - Leap day (Feb 29) does NOT create a 366th topic; it reuses the March 1 topic (dayNumber 60).
 * - Server time is TRUTH. Client clocks can be wrong.
 * - UTC is the backbone. Local times calculated client-side.
 *
 * Notes:
 * - This is a browser-native implementation (no build step required).
 * - Prefer providing an IANA timezone (e.g. "America/Los_Angeles") when mapping from UTC.
 */
(() => {
  const DEFAULT_TIME_ZONE = 'UTC';
  const DISPLAY_LOCALE = 'en-US';
  const LAUNCH_DATE_ISO = '2025-12-17';
  const TIME_SYNC_ENDPOINT = '/api/time';
  const SYNC_INTERVAL_MS = 60000; // Sync every 60 seconds

  const DAYS_IN_MONTH_COMMON = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const DAYS_IN_MONTH_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

  function isLeapYear(year) {
    if (year % 4 !== 0) return false;
    if (year % 100 !== 0) return true;
    return year % 400 === 0;
  }

  function getDayOfYear(year, monthIndex, dayOfMonth) {
    const daysInMonths = isLeapYear(year) ? DAYS_IN_MONTH_LEAP : DAYS_IN_MONTH_COMMON;
    let doy = dayOfMonth;
    for (let m = 0; m < monthIndex; m++) doy += daysInMonths[m];
    return doy; // 1-based
  }

  function dateFromDayOfYear(year, dayOfYear) {
    const leap = isLeapYear(year);
    const maxDay = leap ? 366 : 365;
    if (dayOfYear < 1 || dayOfYear > maxDay) {
      throw new Error(`Invalid dayOfYear ${dayOfYear} for year ${year}`);
    }

    const daysInMonths = leap ? DAYS_IN_MONTH_LEAP : DAYS_IN_MONTH_COMMON;
    let remaining = dayOfYear;
    let month = 0;
    while (month < 12 && remaining > daysInMonths[month]) {
      remaining -= daysInMonths[month];
      month++;
    }
    return new Date(year, month, remaining);
  }

  /**
   * Convert a local calendar date (Date in the current environment timezone) to canonical dayNumber (1–365).
   * Leap years compress after Feb 29; Feb 29 maps to dayNumber 60 (same as March 1).
   */
  function dateToDayNumber(date) {
    const year = date.getFullYear();
    const month = date.getMonth(); // 0-indexed
    const day = date.getDate();

    const leap = isLeapYear(year);
    const doy = getDayOfYear(year, month, day); // 1–365/366

    if (!leap) return doy;

    // Leap year compression rule:
    // - Feb 29 reuses March 1 topic (day 60)
    if (month === 1 && day === 29) return 60;

    // - Jan 1 – Feb 28 maps directly
    if (doy <= 59) return doy;

    // - Mar 1 – Dec 31 shift back by one (so Mar 1 is always 60)
    return doy - 1;
  }

  /**
   * Convert canonical dayNumber (1–365) to a Date in a specific year.
   * In leap years, dayNumbers >= 60 map to day-of-year + 1 (to skip Feb 29 slot).
   */
  function dayNumberToDate(dayNumber, year) {
    if (!Number.isFinite(dayNumber) || dayNumber < 1 || dayNumber > 365) {
      throw new Error(`Invalid dayNumber: ${dayNumber}. Must be 1–365.`);
    }
    const y = year ?? new Date().getFullYear();
    const leap = isLeapYear(y);
    const dayOfYear = leap && dayNumber >= 60 ? dayNumber + 1 : dayNumber;
    return dateFromDayOfYear(y, dayOfYear);
  }

  function getUserTimeZone() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_TIME_ZONE;
    } catch (_) {
      return DEFAULT_TIME_ZONE;
    }
  }

  /**
   * Get local date parts (year, monthIndex, day) for an instant in a specific timezone.
   * Uses Intl rather than trusting the device clock math.
   */
  function getLocalDateParts(utcMillis = Date.now(), timeZone = getUserTimeZone()) {
    const instant = new Date(utcMillis);
    const formatter = new Intl.DateTimeFormat('en-CA', {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    });
    const parts = formatter.formatToParts(instant);
    const lookup = {};
    for (const part of parts) {
      if (part.type !== 'literal') lookup[part.type] = part.value;
    }
    const year = Number(lookup.year);
    const monthIndex = Number(lookup.month) - 1;
    const day = Number(lookup.day);
    return { year, monthIndex, day };
  }

  function getLessonDayForTimeZone(utcMillis = Date.now(), timeZone = getUserTimeZone()) {
    const { year, monthIndex, day } = getLocalDateParts(utcMillis, timeZone);
    const localDate = new Date(year, monthIndex, day);
    return dateToDayNumber(localDate);
  }

  function formatDate(date, options = {}) {
    const { includeYear = false, includeWeekday = false, shortMonth = false } = options;
    const fmt = {
      month: shortMonth ? 'short' : 'long',
      day: 'numeric',
    };
    if (includeYear) fmt.year = 'numeric';
    if (includeWeekday) fmt.weekday = 'long';
    return date.toLocaleDateString(DISPLAY_LOCALE, fmt);
  }

  function formatDayNumber(dayNumber, options = {}) {
    const { utcMillis = Date.now(), timeZone = getUserTimeZone(), includeYear = true, includeWeekday = false } = options;
    const { year } = getLocalDateParts(utcMillis, timeZone);
    const date = dayNumberToDate(dayNumber, year);
    return formatDate(date, { includeYear, includeWeekday });
  }

  function parseISODate(iso) {
    // ISO `YYYY-MM-DD` -> Date in local environment timezone
    const [y, m, d] = String(iso).split('-').map(n => Number(n));
    return new Date(y, (m || 1) - 1, d || 1);
  }

  // Convenience methods for replacing new Date() calls
  function now() {
    return new Date();
  }

  function year() {
    return getLocalDateParts().year;
  }

  function month() {
    return getLocalDateParts().monthIndex + 1; // 1-indexed (1-12)
  }

  function day() {
    return getLocalDateParts().day;
  }

  function fullDate(date) {
    const d = date || now();
    return d.toLocaleDateString(DISPLAY_LOCALE, { month: 'long', day: 'numeric', year: 'numeric' });
  }

  const KellyTime = {
    LAUNCH_DATE_ISO,
    getLaunchDate: () => parseISODate(LAUNCH_DATE_ISO),

    isLeapYear,
    dateToDayNumber,
    dayNumberToDate,

    getUserTimeZone,
    getLocalDateParts,
    getLessonDayForTimeZone,

    formatDate,
    formatDayNumber,

    // Convenience methods
    now,
    year,
    month,
    day,
    fullDate,
  };

  // ============================================
  // KELLY TIME SYNC - Server-Authoritative Time
  // ============================================
  
  /**
   * KellyTimeSync - Synchronizes client time with Kelly's server
   * Uses NTP-like algorithm to account for network latency
   */
  const KellyTimeSync = {
    serverOffset: 0,        // Difference between server and client time (ms)
    lastSync: null,         // When we last synced
    syncInterval: SYNC_INTERVAL_MS,
    _intervalId: null,
    _initialized: false,
    
    /**
     * Initialize time sync with server
     */
    async init() {
      if (this._initialized) return;
      this._initialized = true;
      
      await this.sync();
      this._intervalId = setInterval(() => this.sync(), this.syncInterval);
      
      console.log('[KellyTimeSync] Initialized - syncing every', this.syncInterval / 1000, 'seconds');
    },
    
    /**
     * Sync with server time using NTP-like algorithm
     */
    async sync() {
      const t1 = Date.now(); // Client time before request
      
      try {
      const response = await fetch(TIME_SYNC_ENDPOINT);
      if (!response.ok) throw new Error('API not available');
        
        const data = await response.json();
        const t4 = Date.now(); // Client time after response
        const serverTime = data.utc;
        
        // Round-trip time and estimated one-way latency
        const rtt = t4 - t1;
        const latency = rtt / 2;
        
        // Server time at the moment we received it
        const adjustedServerTime = serverTime + latency;
        
        // Offset = how far ahead/behind client is from server
        this.serverOffset = adjustedServerTime - t4;
        this.lastSync = t4;
        
        console.log(`[KellyTimeSync] Synced. Offset: ${this.serverOffset}ms, RTT: ${rtt}ms`);
        
        return true;
      } catch (error) {
      console.log('Kelly Time: Using client time (server sync unavailable)');
      this.serverOffset = 0;
        return false;
      }
    },
    
    /**
     * Get the TRUE current time (server-authoritative)
     * @returns {number} Milliseconds since epoch (server time)
     */
    now() {
      return Date.now() + this.serverOffset;
    },
    
    /**
     * Get current server time as Date object
     * @returns {Date}
     */
    date() {
      return new Date(this.now());
    },
    
    /**
     * Check if time is synced (last sync within 2x interval)
     */
    isSynced() {
      if (!this.lastSync) return false;
      return (Date.now() - this.lastSync) < (this.syncInterval * 2);
    },
    
    /**
     * Stop syncing (for cleanup)
     */
    stop() {
      if (this._intervalId) {
        clearInterval(this._intervalId);
        this._intervalId = null;
      }
    }
  };

  // ============================================
  // KELLY TIMEZONE - Timezone Handling
  // ============================================
  
  const KellyTimezone = {
    /**
     * Get user's IANA timezone (e.g., "America/New_York")
     */
    get() {
      return getUserTimeZone();
    },
    
    /**
     * Get timezone abbreviation (e.g., "EST", "PST")
     */
    abbreviation(date = new Date()) {
      const tz = this.get();
      try {
        const parts = new Intl.DateTimeFormat('en-US', {
          timeZone: tz,
          timeZoneName: 'short'
        }).formatToParts(date);
        return parts.find(p => p.type === 'timeZoneName')?.value || '';
      } catch (e) {
        return '';
      }
    },
    
    /**
     * Get UTC offset in hours (e.g., -5 for EST)
     */
    offsetHours(date = new Date()) {
      return -date.getTimezoneOffset() / 60;
    },
    
    /**
     * Get UTC offset string (e.g., "-05:00")
     */
    offsetString(date = new Date()) {
      const offset = this.offsetHours(date);
      const sign = offset >= 0 ? '+' : '-';
      const hours = Math.abs(Math.floor(offset)).toString().padStart(2, '0');
      const minutes = Math.abs((offset % 1) * 60).toString().padStart(2, '0');
      return `${sign}${hours}:${minutes}`;
    },
    
    /**
     * Check if currently in Daylight Saving Time
     */
    isDST(date = new Date()) {
      const jan = new Date(date.getFullYear(), 0, 1);
      const jul = new Date(date.getFullYear(), 6, 1);
      const stdOffset = Math.max(jan.getTimezoneOffset(), jul.getTimezoneOffset());
      return date.getTimezoneOffset() < stdOffset;
    },
    
    /**
     * Get friendly timezone name (e.g., "New York (EST)")
     */
    friendlyName() {
      const tz = this.get();
      const city = tz.split('/').pop().replace(/_/g, ' ');
      const abbr = this.abbreviation();
      return abbr ? `${city} (${abbr})` : city;
    },
    
    /**
     * Format time for a specific timezone
     */
    formatInTimezone(date, timezone, options = {}) {
      const defaults = {
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
        timeZone: timezone
      };
      return new Intl.DateTimeFormat('en-US', { ...defaults, ...options }).format(date);
    },
    
    /**
     * Get world clock data for multiple timezones
     */
    worldClock(date = new Date()) {
      const zones = [
        { tz: 'America/Los_Angeles', label: 'Los Angeles' },
        { tz: 'America/New_York', label: 'New York' },
        { tz: 'Europe/London', label: 'London' },
        { tz: 'Europe/Paris', label: 'Paris' },
        { tz: 'Asia/Tokyo', label: 'Tokyo' },
        { tz: 'Asia/Shanghai', label: 'Shanghai' },
        { tz: 'Australia/Sydney', label: 'Sydney' }
      ];
      
      return zones.map(z => ({
        ...z,
        time: this.formatInTimezone(date, z.tz),
        date: new Intl.DateTimeFormat('en-US', {
          month: 'short',
          day: 'numeric',
          timeZone: z.tz
        }).format(date)
      }));
    }
  };

  // ============================================
  // KELLY LIVE SCHEDULE - Class Times
  // ============================================
  
  const KellyLiveSchedule = {
    // Live class times (local hours)
    classHours: [6, 9, 12, 18, 21],
    classDuration: 15, // minutes
    
    classLabels: {
      6: 'Early Birds',
      9: 'Morning',
      12: 'Lunch',
      18: 'Evening',
      21: 'Night Owls'
    },
    
    /**
     * Get next live class time
     */
    getNextClass() {
      const now = KellyTimeSync.date();
      const hour = now.getHours();
      const minute = now.getMinutes();
      
      // Find next class hour
      for (const h of this.classHours) {
        if (h > hour || (h === hour && minute < this.classDuration)) {
          const next = new Date(now);
          next.setHours(h, 0, 0, 0);
          return next;
        }
      }
      
      // Tomorrow's first class
      const tomorrow = new Date(now);
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(this.classHours[0], 0, 0, 0);
      return tomorrow;
    },
    
    /**
     * Check if class is currently live
     */
    isLive() {
      const now = KellyTimeSync.date();
      const hour = now.getHours();
      const minute = now.getMinutes();
      
      return this.classHours.includes(hour) && minute < this.classDuration;
    },
    
    /**
     * Get countdown to next class
     */
    getCountdown() {
      const now = KellyTimeSync.now();
      const next = this.getNextClass().getTime();
      const diff = next - now;
      
      if (diff <= 0 || this.isLive()) {
        return { live: true, hours: 0, minutes: 0, seconds: 0, display: 'LIVE NOW' };
      }
      
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);
      
      let display;
      if (hours > 0) {
        display = `${hours}h ${minutes}m ${seconds}s`;
      } else if (minutes > 0) {
        display = `${minutes}m ${seconds}s`;
      } else {
        display = `${seconds}s`;
      }
      
      return { live: false, hours, minutes, seconds, display };
    },
    
    /**
     * Get class label for an hour
     */
    getClassLabel(hour) {
      return this.classLabels[hour] || 'Class';
    }
  };

  // ============================================
  // KELLY CLOCK - Real-time Clock Display
  // ============================================
  
  /**
   * KellyClock - Real-time display with server-synced time
   * Updates every second, shows to the second
   */
  class KellyClock {
    constructor(elementId, options = {}) {
      this.element = typeof elementId === 'string' 
        ? document.getElementById(elementId) 
        : elementId;
      this.options = {
        showSeconds: true,
        showDate: true,
        showTimezone: true,
        showWorldClock: false,
        format24h: false,
        compact: false,
        ...options
      };
      this.intervalId = null;
    }
    
    start() {
      if (!this.element) {
        console.warn('[KellyClock] Element not found');
        return this;
      }
      this.render();
      this.intervalId = setInterval(() => this.render(), 1000);
      return this;
    }
    
    stop() {
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
      return this;
    }
    
    render() {
      if (!this.element) return;
      
      const now = KellyTimeSync.date();
      
      // Time components
      const hours = now.getHours();
      const minutes = now.getMinutes().toString().padStart(2, '0');
      const seconds = now.getSeconds().toString().padStart(2, '0');
      
      // Format time
      let timeStr;
      if (this.options.format24h) {
        timeStr = `${hours.toString().padStart(2, '0')}:${minutes}`;
        if (this.options.showSeconds) {
          timeStr += `:${seconds}`;
        }
      } else {
        const h = hours % 12 || 12;
        const ampm = hours < 12 ? 'AM' : 'PM';
        timeStr = `${h}:${minutes}`;
        if (this.options.showSeconds) {
          timeStr += `:${seconds}`;
        }
        timeStr += ` ${ampm}`;
      }
      
      // Format date
      const dateStr = now.toLocaleDateString(DISPLAY_LOCALE, {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
        year: 'numeric'
      });
      
      // Build HTML
      const compactClass = this.options.compact ? ' compact' : '';
      let html = `<div class="kelly-clock-time">${timeStr}</div>`;
      
      if (this.options.showDate) {
        html += `<div class="kelly-clock-date">${dateStr}</div>`;
      }
      
      if (this.options.showTimezone) {
        html += `<div class="kelly-clock-tz">${KellyTimezone.friendlyName()}</div>`;
      }
      
      this.element.innerHTML = html;
      this.element.className = `kelly-clock${compactClass}`;
    }
  }

  // ============================================
  // KELLY COUNTDOWN - Live Class Countdown
  // ============================================
  
  class KellyCountdown {
    constructor(elementId, options = {}) {
      this.element = typeof elementId === 'string'
        ? document.getElementById(elementId)
        : elementId;
      this.options = {
        showLabel: true,
        showTime: true,
        showCTA: true,
        ...options
      };
      this.intervalId = null;
    }
    
    start() {
      if (!this.element) {
        console.warn('[KellyCountdown] Element not found');
        return this;
      }
      this.render();
      this.intervalId = setInterval(() => this.render(), 1000);
      return this;
    }
    
    stop() {
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
      return this;
    }
    
    render() {
      if (!this.element) return;
      
      const countdown = KellyLiveSchedule.getCountdown();
      const next = KellyLiveSchedule.getNextClass();
      const label = KellyLiveSchedule.getClassLabel(next.getHours());
      
      if (countdown.live) {
        this.element.innerHTML = `
          <div class="kelly-countdown live">
            <span class="countdown-badge">🔴 LIVE</span>
            ${this.options.showLabel ? `<span class="countdown-label">${label} class in session</span>` : ''}
            ${this.options.showCTA ? `<a href="/live.html" class="countdown-cta">Join Now →</a>` : ''}
          </div>
        `;
      } else {
        const timeStr = next.toLocaleTimeString('en-US', {
          hour: 'numeric',
          minute: '2-digit',
          hour12: true
        });
        
        this.element.innerHTML = `
          <div class="kelly-countdown">
            ${this.options.showLabel ? `<span class="countdown-label">Next: ${label}</span>` : ''}
            <span class="countdown-timer">${countdown.display}</span>
            ${this.options.showTime ? `<span class="countdown-time">${timeStr}</span>` : ''}
          </div>
        `;
      }
    }
  }

  // ============================================
  // AUTO-INITIALIZATION
  // ============================================
  
  /**
   * Auto-start clocks and countdowns on elements with data attributes
   */
  function initTimeAuthority() {
    // Initialize time sync
    KellyTimeSync.init();
    
    // Auto-start clocks
    document.querySelectorAll('[data-kelly-clock]').forEach(el => {
      const clock = new KellyClock(el, {
        showSeconds: el.dataset.seconds !== 'false',
        showDate: el.dataset.date !== 'false',
        showTimezone: el.dataset.timezone !== 'false',
        format24h: el.dataset.format24h === 'true',
        compact: el.classList.contains('compact')
      });
      clock.start();
    });
    
    // Auto-start countdowns
    document.querySelectorAll('[data-kelly-countdown]').forEach(el => {
      const countdown = new KellyCountdown(el, {
        showLabel: el.dataset.label !== 'false',
        showTime: el.dataset.time !== 'false',
        showCTA: el.dataset.cta !== 'false'
      });
      countdown.start();
    });
  }
  
  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTimeAuthority);
  } else {
    // Small delay to ensure other scripts are loaded
    setTimeout(initTimeAuthority, 0);
  }

  // Global attach for non-module pages.
  window.KellyTime = KellyTime;
  window.KellyTimeSync = KellyTimeSync;
  window.KellyTimezone = KellyTimezone;
  window.KellyLiveSchedule = KellyLiveSchedule;
  window.KellyClock = KellyClock;
  window.KellyCountdown = KellyCountdown;
})();

