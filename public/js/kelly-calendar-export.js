/**
 * KellyCalendarExport - Calendar Export and "Add to Calendar" functionality
 * 
 * Part of Kelly Time Authority
 * 
 * Features:
 * - Generate ICS files for single events
 * - Generate ICS for recurring daily lessons
 * - Add-to-calendar links for Google, Outlook, Yahoo
 * - Subscribe to Kelly's full calendar feed
 * 
 * "Sync to their calendar. One-click add to Google Calendar, Apple Calendar, Outlook."
 */
(() => {
  'use strict';

  const KELLY_URL = 'https://curiouskelly.com';
  const CALENDAR_FEED_URL = `${KELLY_URL}/api/calendar/feed`;
  
  // ============================================
  // ICS GENERATION
  // ============================================
  
  const KellyCalendarExport = {
    /**
     * Format date for ICS format (YYYYMMDDTHHMMSSZ)
     */
    _formatICSDate(date) {
      return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
    },
    
    /**
     * Escape text for ICS format
     */
    _escapeICS(text) {
      return String(text)
        .replace(/\\/g, '\\\\')
        .replace(/,/g, '\\,')
        .replace(/;/g, '\\;')
        .replace(/\n/g, '\\n');
    },
    
    /**
     * Generate unique ID for ICS events
     */
    _generateUID(prefix = 'event') {
      return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}@curiouskelly.com`;
    },
    
    /**
     * Generate ICS file content for a single event
     */
    generateICS(event) {
      const {
        title,
        description = '',
        startDate,
        endDate,
        location = `${KELLY_URL}/learn.html`,
        recurring = false,
        rrule = null
      } = event;
      
      const uid = this._generateUID('lesson');
      const now = new Date();
      
      const lines = [
        'BEGIN:VCALENDAR',
        'VERSION:2.0',
        'PRODID:-//Curious Kelly//Kelly Time Authority//EN',
        'CALSCALE:GREGORIAN',
        'METHOD:PUBLISH',
        'BEGIN:VEVENT',
        `UID:${uid}`,
        `DTSTAMP:${this._formatICSDate(now)}`,
        `DTSTART:${this._formatICSDate(startDate)}`,
        `DTEND:${this._formatICSDate(endDate)}`,
        `SUMMARY:${this._escapeICS(title)}`,
        `DESCRIPTION:${this._escapeICS(description)}`,
        `LOCATION:${this._escapeICS(location)}`,
        `URL:${location}`
      ];
      
      if (recurring && rrule) {
        lines.push(`RRULE:${rrule}`);
      }
      
      lines.push('END:VEVENT', 'END:VCALENDAR');
      
      return lines.join('\r\n');
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
        description: `Today's 5-minute lesson from Curious Kelly.\n\nTopic: ${topic}\n\nOpen: ${KELLY_URL}/learn.html`,
        startDate: start,
        endDate: end,
        location: `${KELLY_URL}/learn.html`
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
        description: `Your daily 5-minute lesson from Curious Kelly.\n\nOpen: ${KELLY_URL}/learn.html`,
        startDate: start,
        endDate: end,
        recurring: true,
        rrule: 'FREQ=DAILY'
      });
    },
    
    /**
     * Generate ICS for live class schedule (all 5 daily classes)
     */
    liveClassSchedule() {
      const classHours = [6, 9, 12, 18, 21];
      const labels = ['Early Birds', 'Morning', 'Lunch', 'Evening', 'Night Owls'];
      const now = new Date();
      
      const lines = [
        'BEGIN:VCALENDAR',
        'VERSION:2.0',
        'PRODID:-//Curious Kelly//Live Classes//EN',
        'CALSCALE:GREGORIAN',
        'METHOD:PUBLISH',
        'X-WR-CALNAME:Kelly Live Classes',
        'X-WR-CALDESC:Live classes with Kelly and learners worldwide'
      ];
      
      classHours.forEach((hour, i) => {
        const start = new Date(now);
        start.setHours(hour, 0, 0, 0);
        
        const end = new Date(start);
        end.setMinutes(15); // 15 min class
        
        const uid = this._generateUID(`live-${hour}`);
        
        lines.push(
          'BEGIN:VEVENT',
          `UID:${uid}`,
          `DTSTAMP:${this._formatICSDate(now)}`,
          `DTSTART:${this._formatICSDate(start)}`,
          `DTEND:${this._formatICSDate(end)}`,
          `SUMMARY:Kelly LIVE: ${labels[i]}`,
          `DESCRIPTION:${this._escapeICS(`Join the ${labels[i]} live class with Kelly and learners worldwide.\n\nJoin: ${KELLY_URL}/live.html`)}`,
          `URL:${KELLY_URL}/live.html`,
          'RRULE:FREQ=DAILY',
          'END:VEVENT'
        );
      });
      
      lines.push('END:VCALENDAR');
      return lines.join('\r\n');
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
    },
    
    /**
     * Download today's lesson
     */
    downloadTodayLesson(topic) {
      const ics = this.todayLesson(topic || 'Daily Lesson');
      this.download(ics, 'kelly-today.ics');
    },
    
    /**
     * Download daily reminder (recurring)
     */
    downloadDailyReminder(time = '09:00') {
      const ics = this.dailyReminder(time);
      this.download(ics, 'kelly-daily-reminder.ics');
    },
    
    /**
     * Download live class schedule
     */
    downloadLiveClasses() {
      const ics = this.liveClassSchedule();
      this.download(ics, 'kelly-live-classes.ics');
    }
  };

  // ============================================
  // ADD TO CALENDAR LINKS
  // ============================================
  
  const KellyAddToCalendar = {
    /**
     * Format date for Google Calendar URL
     */
    _formatGoogleDate(date) {
      return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z/, 'Z');
    },
    
    /**
     * Generate Google Calendar link
     */
    googleCalendar(event) {
      const { title, description = '', startDate, endDate, location = '' } = event;
      
      const params = new URLSearchParams({
        action: 'TEMPLATE',
        text: title,
        details: description,
        location: location,
        dates: `${this._formatGoogleDate(startDate)}/${this._formatGoogleDate(endDate)}`
      });
      
      return `https://calendar.google.com/calendar/render?${params}`;
    },
    
    /**
     * Generate Outlook Web link
     */
    outlookWeb(event) {
      const { title, description = '', startDate, endDate, location = '' } = event;
      
      const params = new URLSearchParams({
        path: '/calendar/action/compose',
        rru: 'addevent',
        subject: title,
        body: description,
        location: location,
        startdt: startDate.toISOString(),
        enddt: endDate.toISOString()
      });
      
      return `https://outlook.live.com/calendar/0/deeplink/compose?${params}`;
    },
    
    /**
     * Generate Yahoo Calendar link
     */
    yahooCalendar(event) {
      const { title, description = '', startDate, endDate } = event;
      
      const params = new URLSearchParams({
        v: '60',
        title: title,
        desc: description,
        st: this._formatGoogleDate(startDate),
        et: this._formatGoogleDate(endDate)
      });
      
      return `https://calendar.yahoo.com/?${params}`;
    },
    
    /**
     * Generate all calendar links for an event
     */
    getAllLinks(event) {
      return {
        google: this.googleCalendar(event),
        outlook: this.outlookWeb(event),
        yahoo: this.yahooCalendar(event)
      };
    },
    
    /**
     * Create "Add to Calendar" dropdown HTML
     */
    createDropdown(containerId, event, options = {}) {
      const container = document.getElementById(containerId);
      if (!container) {
        console.warn('[KellyAddToCalendar] Container not found:', containerId);
        return;
      }
      
      const { buttonText = '📅 Add to Calendar', showICS = true } = options;
      const links = this.getAllLinks(event);
      const dropdownId = `atc-dropdown-${Date.now()}`;
      
      container.innerHTML = `
        <div class="kelly-add-to-calendar">
          <button class="atc-button" onclick="document.getElementById('${dropdownId}').classList.toggle('open')">
            ${buttonText}
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <path d="M2 4l4 4 4-4"/>
            </svg>
          </button>
          <div id="${dropdownId}" class="atc-dropdown">
            <a href="${links.google}" target="_blank" rel="noopener">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.5 3.5h-15A2 2 0 002.5 5.5v13a2 2 0 002 2h15a2 2 0 002-2v-13a2 2 0 00-2-2zM7 11h2v2H7v-2zm0 4h2v2H7v-2zm4-4h2v2h-2v-2zm0 4h2v2h-2v-2zm4-4h2v2h-2v-2zm0 4h2v2h-2v-2zM6.5 5.5h11v2h-11v-2z"/>
              </svg>
              Google Calendar
            </a>
            <a href="${links.outlook}" target="_blank" rel="noopener">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M21.5 3h-19A.5.5 0 002 3.5v17a.5.5 0 00.5.5h19a.5.5 0 00.5-.5v-17a.5.5 0 00-.5-.5zM12 16.5c-2.5 0-4.5-2-4.5-4.5s2-4.5 4.5-4.5 4.5 2 4.5 4.5-2 4.5-4.5 4.5z"/>
              </svg>
              Outlook
            </a>
            <a href="${links.yahoo}" target="_blank" rel="noopener">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15v-4H8l5-7v4h3l-5 7z"/>
              </svg>
              Yahoo
            </a>
            ${showICS ? `
              <button onclick="KellyCalendarExport.download(KellyCalendarExport.generateICS(${JSON.stringify(event).replace(/"/g, '&quot;')}), 'kelly-event.ics')">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 16l-6-6h4V4h4v6h4l-6 6zm-8 2h16v2H4v-2z"/>
                </svg>
                Download .ics
              </button>
            ` : ''}
          </div>
        </div>
      `;
      
      // Close dropdown when clicking outside
      document.addEventListener('click', (e) => {
        const dropdown = document.getElementById(dropdownId);
        if (dropdown && !container.contains(e.target)) {
          dropdown.classList.remove('open');
        }
      });
    }
  };

  // ============================================
  // CALENDAR SUBSCRIBE
  // ============================================
  
  const KellyCalendarSubscribe = {
    feedUrl: CALENDAR_FEED_URL,
    liveFeedUrl: `${CALENDAR_FEED_URL}?type=live`,
    
    /**
     * Get webcal URL for subscribing
     */
    getWebcalUrl(type = 'lessons') {
      const url = type === 'live' ? this.liveFeedUrl : this.feedUrl;
      return url.replace('https://', 'webcal://');
    },
    
    /**
     * Get Google Calendar subscribe URL
     */
    getGoogleSubscribeUrl(type = 'lessons') {
      const url = type === 'live' ? this.liveFeedUrl : this.feedUrl;
      return `https://calendar.google.com/calendar/r?cid=${encodeURIComponent(url.replace('https://', 'webcal://'))}`;
    },
    
    /**
     * Copy feed URL to clipboard
     */
    async copyFeedUrl(type = 'lessons') {
      const url = type === 'live' ? this.liveFeedUrl : this.feedUrl;
      try {
        await navigator.clipboard.writeText(url);
        return true;
      } catch (e) {
        console.error('Failed to copy:', e);
        return false;
      }
    },
    
    /**
     * Create subscribe buttons HTML
     */
    createSubscribeButtons(containerId, options = {}) {
      const container = document.getElementById(containerId);
      if (!container) return;
      
      const { type = 'lessons', title = "Subscribe to Kelly's Calendar" } = options;
      
      container.innerHTML = `
        <div class="kelly-calendar-subscribe">
          <h3 class="subscribe-title">${title}</h3>
          <p class="subscribe-desc">Get every lesson automatically added to your calendar.</p>
          <div class="subscribe-buttons">
            <a href="${this.getGoogleSubscribeUrl(type)}" target="_blank" class="subscribe-btn google">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.5 3.5h-15A2 2 0 002.5 5.5v13a2 2 0 002 2h15a2 2 0 002-2v-13a2 2 0 00-2-2z"/>
              </svg>
              Google Calendar
            </a>
            <a href="${this.getWebcalUrl(type)}" class="subscribe-btn apple">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83"/>
              </svg>
              Apple Calendar
            </a>
            <button class="subscribe-btn copy" onclick="KellyCalendarSubscribe.copyFeedUrl('${type}').then(ok => alert(ok ? 'Copied!' : 'Failed to copy'))">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
              </svg>
              Copy Feed URL
            </button>
          </div>
        </div>
      `;
    }
  };

  // ============================================
  // GLOBAL EXPORTS
  // ============================================
  
  window.KellyCalendarExport = KellyCalendarExport;
  window.KellyAddToCalendar = KellyAddToCalendar;
  window.KellyCalendarSubscribe = KellyCalendarSubscribe;
  
})();







