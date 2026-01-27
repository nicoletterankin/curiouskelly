# Kelly Time Authority

## The Trusted Clock for Global Learning

> "When Kelly says 9:00:00 AM, class starts. Everywhere. For everyone. To the second."

Kelly is the **internet authority on educational time** — the clock you trust, the calendar you sync to, the schedule that unites learners globally.

---

## Overview

The Kelly Time Authority system provides:

1. **Server-Authoritative Time Sync** - Client clocks can be wrong; Kelly's server is the truth
2. **Timezone Handling** - UTC backbone with client-side local time conversion
3. **Live Clock Display** - Real-time clock updating every second
4. **Live Class Countdown** - Precise countdowns to the next class
5. **Calendar Export** - ICS files and Add-to-Calendar buttons
6. **Calendar Feed** - Subscribable calendar for all platforms

---

## Architecture

### Core Principles

1. **Server Time is Truth** - The `/api/time` endpoint is synced to NTP and is the single source of truth
2. **UTC is the Backbone** - All internal timestamps use UTC; local times calculated client-side
3. **Every Second Counts** - Live classes start at :00 seconds; countdowns are precise
4. **Sync to Their Calendar** - One-click add to Google Calendar, Apple Calendar, Outlook

---

## Files Created/Modified

### API Endpoints

| File | Purpose |
|------|---------|
| `/api/time.ts` | Server time endpoint returning UTC timestamp, ISO format, and Unix timestamp |
| `/api/calendar/feed.ts` | Subscribable ICS calendar feed for lessons and live classes |

### Client-Side JavaScript

| File | Purpose |
|------|---------|
| `/public/js/kelly-time.js` | Core time module with sync, timezone, clock, and countdown classes |
| `/public/js/kelly-calendar-export.js` | ICS generation and Add-to-Calendar functionality |

### CSS

| File | Purpose |
|------|---------|
| `/public/css/kelly-time-authority.css` | Styles for clock, countdown, and calendar components |

### Icons

| File | Purpose |
|------|---------|
| `/public/images/icons/google-calendar.svg` | Google Calendar icon |
| `/public/images/icons/apple-calendar.svg` | Apple Calendar icon |
| `/public/images/icons/outlook.svg` | Outlook icon |
| `/public/images/icons/yahoo.svg` | Yahoo Calendar icon |
| `/public/images/icons/ics-download.svg` | ICS download icon |

---

## Usage

### Time Sync (Automatic)

The `KellyTimeSync` module automatically initializes and syncs with the server every 60 seconds:

```javascript
// Get server-authoritative current time
const now = KellyTimeSync.now();  // Milliseconds since epoch
const date = KellyTimeSync.date(); // Date object

// Check sync status
if (KellyTimeSync.isSynced()) {
  console.log('Time is synced with server');
}
```

### Timezone Handling

```javascript
// Get user's timezone
const tz = KellyTimezone.get(); // e.g., "America/New_York"
const abbr = KellyTimezone.abbreviation(); // e.g., "EST"
const friendly = KellyTimezone.friendlyName(); // e.g., "New York (EST)"

// Get UTC offset
const offset = KellyTimezone.offsetHours(); // e.g., -5
const offsetStr = KellyTimezone.offsetString(); // e.g., "-05:00"

// Check DST
const isDST = KellyTimezone.isDST();

// World clock
const worldClock = KellyTimezone.worldClock();
// Returns array of { tz, label, time, date } for major cities
```

### Live Clock Display

**HTML (auto-initialized):**
```html
<!-- Full clock with all options -->
<div id="my-clock" 
     data-kelly-clock
     data-seconds="true"
     data-date="true"
     data-timezone="true">
</div>

<!-- Compact clock for header -->
<div id="header-clock" 
     class="compact"
     data-kelly-clock
     data-seconds="true"
     data-date="false">
</div>
```

**JavaScript (manual):**
```javascript
// Create and start a clock
const clock = new KellyClock('my-clock-element', {
  showSeconds: true,
  showDate: true,
  showTimezone: true,
  format24h: false,
  compact: false
});
clock.start();

// Stop when done
clock.stop();
```

### Live Class Countdown

**HTML (auto-initialized):**
```html
<div id="countdown" 
     data-kelly-countdown
     data-label="true"
     data-time="true"
     data-cta="true">
</div>
```

**JavaScript (manual):**
```javascript
const countdown = new KellyCountdown('countdown-element', {
  showLabel: true,
  showTime: true,
  showCTA: true
});
countdown.start();
```

### Live Schedule API

```javascript
// Get next class time
const nextClass = KellyLiveSchedule.getNextClass(); // Date object

// Check if class is live now
if (KellyLiveSchedule.isLive()) {
  console.log('Class is in session!');
}

// Get countdown info
const countdown = KellyLiveSchedule.getCountdown();
// Returns { live: boolean, hours, minutes, seconds, display }

// Get class label
const label = KellyLiveSchedule.getClassLabel(9); // "Morning"
```

### Calendar Export

```javascript
// Download today's lesson as ICS
KellyCalendarExport.downloadTodayLesson('Why do leaves change color?');

// Download daily reminder (recurring)
KellyCalendarExport.downloadDailyReminder('09:00');

// Download all live classes
KellyCalendarExport.downloadLiveClasses();

// Generate custom ICS
const ics = KellyCalendarExport.generateICS({
  title: 'Kelly: Special Lesson',
  description: 'A custom lesson',
  startDate: new Date('2025-12-25T09:00:00'),
  endDate: new Date('2025-12-25T09:05:00'),
  location: 'https://curiouskelly.com/learn.html',
  recurring: false
});
KellyCalendarExport.download(ics, 'special-lesson.ics');
```

### Add to Calendar Buttons

```javascript
// Create dropdown with all calendar options
KellyAddToCalendar.createDropdown('container-id', {
  title: 'Kelly: Daily Lesson',
  description: 'Your 5-minute lesson',
  startDate: new Date(),
  endDate: new Date(Date.now() + 5 * 60 * 1000),
  location: 'https://curiouskelly.com/learn.html'
});

// Get individual links
const event = { title, description, startDate, endDate };
const googleUrl = KellyAddToCalendar.googleCalendar(event);
const outlookUrl = KellyAddToCalendar.outlookWeb(event);
const yahooUrl = KellyAddToCalendar.yahooCalendar(event);
```

### Calendar Subscribe

```javascript
// Create subscribe buttons
KellyCalendarSubscribe.createSubscribeButtons('container-id', {
  type: 'lessons', // or 'live'
  title: "Subscribe to Kelly's Calendar"
});

// Get URLs directly
const webcalUrl = KellyCalendarSubscribe.getWebcalUrl('lessons');
const googleSubUrl = KellyCalendarSubscribe.getGoogleSubscribeUrl('lessons');

// Copy feed URL to clipboard
await KellyCalendarSubscribe.copyFeedUrl('lessons');
```

---

## Live Class Schedule

Kelly runs 5 live classes daily at these local hours:

| Hour | Class Name | Target Audience |
|------|-----------|-----------------|
| 6:00 AM | Early Birds | Early risers |
| 9:00 AM | Morning | Standard class |
| 12:00 PM | Lunch | Midday learners |
| 6:00 PM | Evening | After work |
| 9:00 PM | Night Owls | Late learners |

Each class runs for 15 minutes.

---

## CSS Classes

### Clock

| Class | Description |
|-------|-------------|
| `.kelly-clock` | Container for clock |
| `.kelly-clock.compact` | Compact inline version |
| `.kelly-clock.hero` | Large hero version |
| `.kelly-clock-time` | Time display |
| `.kelly-clock-date` | Date display |
| `.kelly-clock-tz` | Timezone display |

### Countdown

| Class | Description |
|-------|-------------|
| `.kelly-countdown` | Container for countdown |
| `.kelly-countdown.live` | When class is live |
| `.countdown-badge` | LIVE badge |
| `.countdown-timer` | Timer display |
| `.countdown-label` | Class label |
| `.countdown-cta` | Call-to-action button |

### Add to Calendar

| Class | Description |
|-------|-------------|
| `.kelly-add-to-calendar` | Dropdown container |
| `.atc-button` | Trigger button |
| `.atc-dropdown` | Dropdown menu |
| `.atc-dropdown.open` | Open state |

### Subscribe

| Class | Description |
|-------|-------------|
| `.kelly-calendar-subscribe` | Container |
| `.subscribe-btn` | Subscribe button |
| `.subscribe-btn.google` | Google variant |
| `.subscribe-btn.apple` | Apple variant |
| `.subscribe-btn.copy` | Copy URL variant |

---

## API Reference

### GET /api/time

Returns server-authoritative time.

**Response:**
```json
{
  "utc": 1734012345678,
  "iso": "2025-12-12T15:30:45.678Z",
  "unix": 1734012345,
  "formatted": {
    "date": "2025-12-12",
    "time": "15:30:45"
  },
  "server": "kelly-time-authority",
  "version": "1.0.0"
}
```

**Headers:**
- `Cache-Control: no-store, max-age=0`
- `Access-Control-Allow-Origin: *`

### GET /api/calendar/feed

Returns ICS calendar feed.

**Query Parameters:**
- `type` - `lessons` (default) or `live`

**Response:** `text/calendar` (ICS format)

**Subscribe URLs:**
- Google Calendar: `https://calendar.google.com/calendar/r?cid=webcal://curiouskelly.com/api/calendar/feed`
- Apple Calendar: `webcal://curiouskelly.com/api/calendar/feed`
- Direct download: `https://curiouskelly.com/api/calendar/feed`

---

## Integration Points

The Time Authority is integrated into:

- `index.html` - Hero countdown and calendar subscribe section
- `learn.html` - Clock and calendar export
- `live.html` - Live class indicators
- `calendar.html` - Calendar page

---

## Verification Checklist

```bash
# Verify time sync
curl https://curiouskelly.com/api/time
# Should return { utc: ..., iso: "...", unix: ... }

# Verify calendar feed
curl https://curiouskelly.com/api/calendar/feed
# Should return valid ICS content

# Verify live feed
curl "https://curiouskelly.com/api/calendar/feed?type=live"
# Should return ICS with 5 recurring events

# Browser verification
# 1. Open page, check clock updates every second
# 2. Wait until :00:00 of class hour, verify "LIVE NOW" appears
# 3. Click "Add to Calendar", verify dropdown works
# 4. Download .ics file, verify it opens in calendar app
```

---

## Future Enhancements

1. **World Clock Widget** - Show time in multiple zones
2. **Custom Reminder Times** - Let users pick their reminder time
3. **Calendar Deep Links** - Direct links to specific lessons
4. **Push Notification Sync** - Sync reminders with push notifications
5. **Timezone Detection Modal** - Confirm timezone on first visit

---

*Kelly Time Authority - December 12, 2025*
*"When Kelly says 9:00:00, class starts."*




















