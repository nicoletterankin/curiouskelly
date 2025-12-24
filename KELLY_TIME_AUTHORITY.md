# KELLY TIME AUTHORITY
## The Trusted Clock for Global Learning

---

## THE VISION

When Kelly says 9:00:00 AM, class starts.
Everywhere. For everyone. To the second.

Kelly becomes the **internet authority on educational time** — the clock you trust, the calendar you sync to, the schedule that unites learners globally.

---

## CORE PRINCIPLES

### 1. Server Time is Truth
Client clocks can be wrong. Kelly's server clock is synced to NTP (time.google.com) and is the single source of truth.

### 2. UTC is the Backbone
All internal timestamps use UTC. Local times are calculated client-side from user timezone.

### 3. Every Second Counts
Live classes start at :00 seconds. Countdowns are precise. No "about 5 minutes" — it's "4:47" and ticking.

### 4. Sync to Their Calendar
One-click add to Google Calendar, Apple Calendar, Outlook. ICS files for everything.

---

## PART 1: SERVER-AUTHORITATIVE TIME

### The Problem
```javascript
// Client time can be WRONG
const clientTime = new Date(); // User's computer might be 3 minutes fast
```

### The Solution
```javascript
// Kelly's server provides authoritative time
// Client calculates offset and adjusts

const KellyTimeSync = {
  serverOffset: 0,      // Difference between server and client time (ms)
  lastSync: null,       // When we last synced
  syncInterval: 60000,  // Sync every 60 seconds
  
  /**
   * Initialize time sync with server
   */
  async init() {
    await this.sync();
    setInterval(() => this.sync(), this.syncInterval);
  },
  
  /**
   * Sync with server time
   * Uses NTP-like algorithm to account for network latency
   */
  async sync() {
    const t1 = Date.now(); // Client time before request
    
    try {
      const response = await fetch('/api/time');
      const data = await response.json();
      
      const t4 = Date.now(); // Client time after response
      const serverTime = data.utc; // Server's UTC timestamp
      
      // Round-trip time
      const rtt = t4 - t1;
      
      // Estimate one-way latency (assume symmetric)
      const latency = rtt / 2;
      
      // Server time at the moment we received it
      const adjustedServerTime = serverTime + latency;
      
      // Offset = how far ahead/behind client is
      this.serverOffset = adjustedServerTime - t4;
      this.lastSync = t4;
      
      console.log(`Kelly Time synced. Offset: ${this.serverOffset}ms`);
      
    } catch (error) {
      console.warn('Kelly Time sync failed, using client time');
    }
  },
  
  /**
   * Get the TRUE current time (server-authoritative)
   */
  now() {
    return Date.now() + this.serverOffset;
  },
  
  /**
   * Get current time as Date object
   */
  date() {
    return new Date(this.now());
  }
};
```

### Server Endpoint (Vercel/Cloudflare)

```javascript
// /api/time.js (Vercel Edge Function)
export const config = { runtime: 'edge' };

export default function handler(req) {
  const now = Date.now();
  
  return new Response(JSON.stringify({
    utc: now,
    iso: new Date(now).toISOString(),
    unix: Math.floor(now / 1000)
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store, max-age=0',
      'Access-Control-Allow-Origin': '*'
    }
  });
}
```

---

## PART 2: TIMEZONE HANDLING

### Detect User Timezone

```javascript
const KellyTimezone = {
  /**
   * Get user's IANA timezone (e.g., "America/New_York")
   */
  get() {
    return Intl.DateTimeFormat().resolvedOptions().timeZone;
  },
  
  /**
   * Get timezone abbreviation (e.g., "EST", "PST")
   */
  abbreviation(date = new Date()) {
    const tz = this.get();
    return new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      timeZoneName: 'short'
    }).formatToParts(date).find(p => p.type === 'timeZoneName')?.value || '';
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
   * Get friendly timezone name
   */
  friendlyName() {
    const tz = this.get();
    // Convert "America/New_York" to "New York"
    const city = tz.split('/').pop().replace(/_/g, ' ');
    return `${city} (${this.abbreviation()})`;
  }
};
```

### Convert Between Timezones

```javascript
const KellyTimeConvert = {
  /**
   * Convert UTC timestamp to user's local time
   */
  utcToLocal(utcMs) {
    return new Date(utcMs);
  },
  
  /**
   * Convert local time to UTC timestamp
   */
  localToUtc(localDate) {
    return localDate.getTime();
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
   * Get time in multiple timezones (for world clock display)
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
```

---

## PART 3: LIVE CLOCK DISPLAY

### The Kelly Clock Component

```javascript
/**
 * KELLY CLOCK
 * Real-time display with server-synced time
 * Updates every second, shows to the second
 */
class KellyClock {
  constructor(elementId, options = {}) {
    this.element = document.getElementById(elementId);
    this.options = {
      showSeconds: true,
      showDate: true,
      showTimezone: true,
      showWorldClock: false,
      format24h: false,
      ...options
    };
    this.intervalId = null;
  }
  
  start() {
    this.render();
    this.intervalId = setInterval(() => this.render(), 1000);
  }
  
  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
  
  render() {
    const now = KellyTimeSync.date();
    
    // Time components
    const hours = now.getHours();
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    
    // Format time
    let timeStr;
    if (this.options.format24h) {
      timeStr = `${hours.toString().padStart(2, '0')}:${minutes}`;
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
    const dateStr = now.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    });
    
    // Build HTML
    let html = `
      <div class="kelly-clock-time">${timeStr}</div>
    `;
    
    if (this.options.showDate) {
      html += `<div class="kelly-clock-date">${dateStr}</div>`;
    }
    
    if (this.options.showTimezone) {
      html += `<div class="kelly-clock-tz">${KellyTimezone.friendlyName()}</div>`;
    }
    
    this.element.innerHTML = html;
  }
}

// Auto-start on elements with data-kelly-clock
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-kelly-clock]').forEach(el => {
    const clock = new KellyClock(el.id, {
      showSeconds: el.dataset.seconds !== 'false',
      showDate: el.dataset.date !== 'false',
      showTimezone: el.dataset.timezone !== 'false'
    });
    clock.start();
  });
});
```

### Clock CSS

```css
.kelly-clock {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
  font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
}

.kelly-clock-time {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.02em;
  font-variant-numeric: tabular-nums;
}

.kelly-clock-date {
  font-size: 1rem;
  color: var(--text-secondary);
  font-family: var(--font-primary);
}

.kelly-clock-tz {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Compact version for headers */
.kelly-clock.compact .kelly-clock-time {
  font-size: 1.25rem;
}

.kelly-clock.compact .kelly-clock-date {
  font-size: 0.875rem;
}
```

### Clock HTML

```html
<!-- Full clock in hero -->
<div id="kelly-clock" class="kelly-clock" 
     data-kelly-clock
     data-seconds="true"
     data-date="true"
     data-timezone="true">
</div>

<!-- Compact clock in header -->
<div id="header-clock" class="kelly-clock compact"
     data-kelly-clock
     data-seconds="true"
     data-date="false"
     data-timezone="false">
</div>
```

---

## PART 4: LIVE CLASS COUNTDOWN

### Countdown to Next Class

```javascript
const KellyLiveSchedule = {
  // Live class times (local hours)
  classHours: [6, 9, 12, 18, 21],
  classDuration: 15, // minutes
  
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
    
    if (diff <= 0) {
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
    const labels = {
      6: 'Early Birds',
      9: 'Morning',
      12: 'Lunch',
      18: 'Evening',
      21: 'Night Owls'
    };
    return labels[hour] || 'Class';
  }
};

/**
 * Live countdown component
 */
class KellyCountdown {
  constructor(elementId) {
    this.element = document.getElementById(elementId);
    this.intervalId = null;
  }
  
  start() {
    this.render();
    this.intervalId = setInterval(() => this.render(), 1000);
  }
  
  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
  
  render() {
    const countdown = KellyLiveSchedule.getCountdown();
    const next = KellyLiveSchedule.getNextClass();
    const label = KellyLiveSchedule.getClassLabel(next.getHours());
    
    if (countdown.live) {
      this.element.innerHTML = `
        <div class="kelly-countdown live">
          <span class="countdown-badge">🔴 LIVE</span>
          <span class="countdown-label">${label} class in session</span>
          <a href="/live.html" class="countdown-cta">Join Now →</a>
        </div>
      `;
    } else {
      this.element.innerHTML = `
        <div class="kelly-countdown">
          <span class="countdown-label">Next: ${label}</span>
          <span class="countdown-timer">${countdown.display}</span>
          <span class="countdown-time">${next.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
          })}</span>
        </div>
      `;
    }
  }
}
```

### Countdown CSS

```css
.kelly-countdown {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1.5rem;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-full);
}

.kelly-countdown.live {
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
  border-color: rgba(239, 68, 68, 0.3);
}

.countdown-badge {
  animation: pulse 2s infinite;
  font-weight: 700;
}

.countdown-timer {
  font-family: 'SF Mono', monospace;
  font-size: 1.5rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--accent-primary);
}

.countdown-label {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.countdown-time {
  color: var(--text-muted);
  font-size: 0.875rem;
}

.countdown-cta {
  background: var(--accent-primary);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: var(--radius-md);
  font-weight: 600;
  text-decoration: none;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
```

---

## PART 5: CALENDAR SYNC

### Generate ICS Files

```javascript
const KellyCalendarExport = {
  /**
   * Generate ICS file content for an event
   */
  generateICS(event) {
    const {
      title,
      description,
      startDate,
      endDate,
      location = 'https://curiouskelly.com/live.html',
      recurring = false,
      rrule = null
    } = event;
    
    // Format dates for ICS (UTC)
    const formatDate = (date) => {
      return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
    };
    
    const uid = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}@curiouskelly.com`;
    
    let ics = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Curious Kelly//Live Classes//EN',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH',
      'BEGIN:VEVENT',
      `UID:${uid}`,
      `DTSTAMP:${formatDate(new Date())}`,
      `DTSTART:${formatDate(startDate)}`,
      `DTEND:${formatDate(endDate)}`,
      `SUMMARY:${title}`,
      `DESCRIPTION:${description.replace(/\n/g, '\\n')}`,
      `LOCATION:${location}`,
      `URL:${location}`
    ];
    
    if (recurring && rrule) {
      ics.push(`RRULE:${rrule}`);
    }
    
    ics.push('END:VEVENT', 'END:VCALENDAR');
    
    return ics.join('\r\n');
  },
  
  /**
   * Generate ICS for today's lesson
   */
  todayLesson(topic, time = '09:00') {
    const [hours, minutes] = time.split(':').map(Number);
    const start = new Date();
    start.setHours(hours, minutes, 0, 0);
    
    const end = new Date(start);
    end.setMinutes(end.getMinutes() + 5); // 5 min lesson
    
    return this.generateICS({
      title: `Kelly: ${topic}`,
      description: `Today's 5-minute lesson from Curious Kelly.\n\nTopic: ${topic}\n\nOpen: https://curiouskelly.com/learn.html`,
      startDate: start,
      endDate: end
    });
  },
  
  /**
   * Generate ICS for daily Kelly reminders (recurring)
   */
  dailyReminder(time = '09:00') {
    const [hours, minutes] = time.split(':').map(Number);
    const start = new Date();
    start.setHours(hours, minutes, 0, 0);
    
    const end = new Date(start);
    end.setMinutes(end.getMinutes() + 5);
    
    return this.generateICS({
      title: 'Kelly: Daily Lesson',
      description: 'Your daily 5-minute lesson from Curious Kelly.\n\nOpen: https://curiouskelly.com/learn.html',
      startDate: start,
      endDate: end,
      recurring: true,
      rrule: 'FREQ=DAILY'
    });
  },
  
  /**
   * Generate ICS for live class schedule
   */
  liveClassSchedule() {
    // Create a multi-event ICS with all live class times
    const events = [];
    const classHours = [6, 9, 12, 18, 21];
    const labels = ['Early Birds', 'Morning', 'Lunch', 'Evening', 'Night Owls'];
    
    classHours.forEach((hour, i) => {
      const start = new Date();
      start.setHours(hour, 0, 0, 0);
      
      const end = new Date(start);
      end.setMinutes(15);
      
      events.push({
        title: `Kelly LIVE: ${labels[i]}`,
        description: `Join the ${labels[i]} live class with Kelly and thousands of learners.\n\nJoin: https://curiouskelly.com/live.html`,
        startDate: start,
        endDate: end,
        recurring: true,
        rrule: 'FREQ=DAILY'
      });
    });
    
    // Build multi-event ICS
    let ics = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Curious Kelly//Live Classes//EN',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH'
    ];
    
    events.forEach(event => {
      const formatDate = (date) => date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
      const uid = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}@curiouskelly.com`;
      
      ics.push(
        'BEGIN:VEVENT',
        `UID:${uid}`,
        `DTSTAMP:${formatDate(new Date())}`,
        `DTSTART:${formatDate(event.startDate)}`,
        `DTEND:${formatDate(event.endDate)}`,
        `SUMMARY:${event.title}`,
        `DESCRIPTION:${event.description.replace(/\n/g, '\\n')}`,
        `URL:https://curiouskelly.com/live.html`,
        `RRULE:${event.rrule}`,
        'END:VEVENT'
      );
    });
    
    ics.push('END:VCALENDAR');
    return ics.join('\r\n');
  },
  
  /**
   * Download ICS file
   */
  download(icsContent, filename = 'kelly-lesson.ics') {
    const blob = new Blob([icsContent], { type: 'text/calendar;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
};
```

### Add to Calendar Buttons

```javascript
const KellyAddToCalendar = {
  /**
   * Generate Google Calendar link
   */
  googleCalendar(event) {
    const { title, description, startDate, endDate, location } = event;
    
    const formatDate = (date) => date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z/, 'Z');
    
    const params = new URLSearchParams({
      action: 'TEMPLATE',
      text: title,
      details: description,
      location: location || '',
      dates: `${formatDate(startDate)}/${formatDate(endDate)}`
    });
    
    return `https://calendar.google.com/calendar/render?${params}`;
  },
  
  /**
   * Generate Outlook Web link
   */
  outlookWeb(event) {
    const { title, description, startDate, endDate, location } = event;
    
    const params = new URLSearchParams({
      path: '/calendar/action/compose',
      rru: 'addevent',
      subject: title,
      body: description,
      location: location || '',
      startdt: startDate.toISOString(),
      enddt: endDate.toISOString()
    });
    
    return `https://outlook.live.com/calendar/0/deeplink/compose?${params}`;
  },
  
  /**
   * Generate Yahoo Calendar link
   */
  yahooCalendar(event) {
    const { title, description, startDate, endDate } = event;
    
    const formatDate = (date) => date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z/, 'Z');
    
    const params = new URLSearchParams({
      v: '60',
      title: title,
      desc: description,
      st: formatDate(startDate),
      et: formatDate(endDate)
    });
    
    return `https://calendar.yahoo.com/?${params}`;
  },
  
  /**
   * Render add-to-calendar dropdown
   */
  render(containerId, event) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="add-to-calendar">
        <button class="add-to-calendar-btn" onclick="this.parentElement.classList.toggle('open')">
          📅 Add to Calendar
        </button>
        <div class="add-to-calendar-dropdown">
          <a href="${this.googleCalendar(event)}" target="_blank">
            <img src="/icons/google-calendar.svg" alt=""> Google Calendar
          </a>
          <a href="${this.outlookWeb(event)}" target="_blank">
            <img src="/icons/outlook.svg" alt=""> Outlook
          </a>
          <a href="${this.yahooCalendar(event)}" target="_blank">
            <img src="/icons/yahoo.svg" alt=""> Yahoo
          </a>
          <button onclick="KellyCalendarExport.download(KellyCalendarExport.todayLesson('${event.title}'))">
            <img src="/icons/apple-calendar.svg" alt=""> Apple Calendar (.ics)
          </button>
        </div>
      </div>
    `;
  }
};
```

### Calendar Button CSS

```css
.add-to-calendar {
  position: relative;
  display: inline-block;
}

.add-to-calendar-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  color: var(--text-primary);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition-fast);
}

.add-to-calendar-btn:hover {
  background: var(--bg-hover);
}

.add-to-calendar-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  margin-top: 0.5rem;
  min-width: 200px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  opacity: 0;
  visibility: hidden;
  transform: translateY(-10px);
  transition: var(--transition-fast);
  z-index: 100;
}

.add-to-calendar.open .add-to-calendar-dropdown {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.add-to-calendar-dropdown a,
.add-to-calendar-dropdown button {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  padding: 0.75rem 1rem;
  color: var(--text-primary);
  text-decoration: none;
  background: none;
  border: none;
  cursor: pointer;
  text-align: left;
  font-size: 0.875rem;
}

.add-to-calendar-dropdown a:hover,
.add-to-calendar-dropdown button:hover {
  background: var(--bg-hover);
}

.add-to-calendar-dropdown img {
  width: 20px;
  height: 20px;
}
```

---

## PART 6: SUBSCRIBABLE CALENDAR FEED

### ICS Feed Endpoint

```javascript
// /api/calendar/feed.ics (Vercel function)
export default async function handler(req, res) {
  const { SUPABASE_URL, SUPABASE_KEY } = process.env;
  
  // Fetch upcoming lessons from Supabase
  const response = await fetch(
    `${SUPABASE_URL}/rest/v1/core_lessons?select=topic,calendar_month,calendar_day&order=calendar_month,calendar_day`,
    {
      headers: {
        'apikey': SUPABASE_KEY,
        'Authorization': `Bearer ${SUPABASE_KEY}`
      }
    }
  );
  
  const lessons = await response.json();
  
  // Build ICS feed
  const now = new Date();
  const year = now.getFullYear();
  
  let ics = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//Curious Kelly//Daily Lessons//EN',
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    'X-WR-CALNAME:Curious Kelly',
    'X-WR-CALDESC:Daily 5-minute lessons from Kelly'
  ];
  
  lessons.forEach(lesson => {
    const date = new Date(year, lesson.calendar_month - 1, lesson.calendar_day, 9, 0, 0);
    const endDate = new Date(date);
    endDate.setMinutes(5);
    
    const formatDate = (d) => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
    const uid = `lesson-${lesson.calendar_month}-${lesson.calendar_day}@curiouskelly.com`;
    
    ics.push(
      'BEGIN:VEVENT',
      `UID:${uid}`,
      `DTSTAMP:${formatDate(new Date())}`,
      `DTSTART:${formatDate(date)}`,
      `DTEND:${formatDate(endDate)}`,
      `SUMMARY:Kelly: ${lesson.topic}`,
      `DESCRIPTION:Today's lesson from Curious Kelly.\\n\\nOpen: https://curiouskelly.com/learn.html`,
      'URL:https://curiouskelly.com/learn.html',
      'RRULE:FREQ=YEARLY',
      'END:VEVENT'
    );
  });
  
  ics.push('END:VCALENDAR');
  
  res.setHeader('Content-Type', 'text/calendar; charset=utf-8');
  res.setHeader('Content-Disposition', 'attachment; filename="kelly-lessons.ics"');
  res.status(200).send(ics.join('\r\n'));
}
```

### Subscribe Instructions

```html
<div class="calendar-subscribe">
  <h3>Subscribe to Kelly's Calendar</h3>
  <p>Get every lesson automatically added to your calendar.</p>
  
  <div class="subscribe-options">
    <a href="https://calendar.google.com/calendar/r?cid=webcal://curiouskelly.com/api/calendar/feed.ics" 
       target="_blank" class="subscribe-btn google">
      Subscribe in Google Calendar
    </a>
    
    <a href="webcal://curiouskelly.com/api/calendar/feed.ics" 
       class="subscribe-btn apple">
      Subscribe in Apple Calendar
    </a>
    
    <button onclick="navigator.clipboard.writeText('https://curiouskelly.com/api/calendar/feed.ics')" 
            class="subscribe-btn copy">
      Copy Feed URL
    </button>
  </div>
</div>
```

---

## PART 7: IMPLEMENTATION CHECKLIST

### For Agent 1

```
Phase 1: Server Time Sync
[ ] Create /api/time endpoint
[ ] Add KellyTimeSync module to kelly-time.js
[ ] Initialize sync on page load
[ ] Test offset calculation

Phase 2: Clock Display
[ ] Create KellyClock component
[ ] Add to both index.html and learn.html headers
[ ] Show seconds
[ ] Show timezone
[ ] Update every second

Phase 3: Live Countdown
[ ] Create KellyCountdown component
[ ] Show next class time
[ ] Countdown to :00:00
[ ] "LIVE NOW" when class is active

Phase 4: Calendar Export
[ ] Create KellyCalendarExport module
[ ] Add-to-calendar buttons
[ ] ICS download for single events
[ ] ICS feed for all lessons

Phase 5: Subscribe
[ ] Create /api/calendar/feed.ics endpoint
[ ] Subscribe buttons on index.html
[ ] Test in Google Calendar
[ ] Test in Apple Calendar
```

### For Agent 2 Verification

```bash
# Verify time sync
curl https://curiouskelly.com/api/time
# Should return { utc: ..., iso: "2025-12-12T...", unix: ... }

# Verify clock updates every second
# Open browser, inspect element, confirm time changes

# Verify countdown accuracy
# Wait until :00:00 of a class hour, confirm "LIVE NOW" appears

# Verify ICS generation
curl https://curiouskelly.com/api/calendar/feed.ics
# Should return valid ICS content

# Verify timezone display
# Check multiple browsers/devices with different timezones
```

---

## THE RESULT

When Kelly is the time authority:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                    9:00:47 AM EST                            │
│            Friday, December 12, 2025                         │
│               New York (Eastern Time)                        │
│                                                              │
│     ┌─────────────────────────────────────────────────┐     │
│     │  🔴 LIVE CLASS STARTS IN                        │     │
│     │                                                  │     │
│     │              12 seconds                          │     │
│     │                                                  │     │
│     │    Morning class • 9:00 AM                      │     │
│     │    1,847 learners waiting                       │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
│     📅 Add to Calendar    🔔 Subscribe to All Classes       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

When the clock hits 9:00:00 AM — Kelly goes live.
Everyone sees it at the same moment.
The calendar is the interface.
Kelly is the time.

---

*Kelly Time Authority*
*December 12, 2025*
*"When Kelly says 9:00:00, class starts."*
