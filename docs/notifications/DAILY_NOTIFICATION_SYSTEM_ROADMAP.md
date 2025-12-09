# ✨ Curious Kelly Daily Notification System
## 12-Month Roadmap: "The Best Digital Daily Learning Experience Ever"

> *"It's not going away anytime soon, or ever... meet Kelly."*

---

## 🎯 Executive Summary

**Vision**: Transform Curious Kelly into the world's most delightful, ambient, and transformational daily learning experience — combining Duolingo's habit-forming genius, Alexa's ambient presence, and Kelly's uniquely warm personality into something that makes learning feel like breathing.

**Launch Date**: December 17, 2025  
**Year 1 Content**: December 17, 2025 → December 16, 2026 (365 unique lessons)

### Important: Date Display Convention
- **Internal**: `day_number` (1-365) — used in database, APIs, and code
- **User-Facing**: Real calendar dates — "December 17" not "Day 1"
- **Mapping**: Day 1 = December 17, Day 365 = December 16 (next year)

**Current State**:
- ✅ Web push notifications (VAPID system, service worker ready)
- ✅ Daily lesson email system (Resend, personalized, streak-aware)
- ✅ Birthday & milestone emails
- ✅ React Native mobile app (WebView wrapper, ready for enhancement)
- ✅ Electron desktop app (cross-platform)
- ✅ Roku channel (BrightScript, WebView wrapper)
- ✅ Supabase database with user engagement tracking
- ⏳ Native push notifications for iOS/Android (needs implementation)
- ⏳ Apple/Google developer accounts (ready to use)

**12-Month Goal**: 100,000 daily active learners receiving the exact right nudge at exactly the right moment — not annoying, not forgettable — but as natural as "Good morning."

---

## 🏗️ Current Infrastructure Audit

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| **Web Push** | 80% Complete | `public/js/push-notifications.js`, `public/sw.js` |
| **Email System** | Production Ready | `api/cron/daily-lesson.ts`, `api/cron/birthday-emails.ts`, `api/cron/gentle-return.ts` |
| **Mobile App (React Native)** | Foundation Ready | `mobile-app/` - needs native push integration |
| **Desktop App (Electron)** | Foundation Ready | `desktop-app/` - needs notification integration |
| **Roku Channel** | Foundation Ready | `roku-app/` - limited notification capability |
| **User Database** | Production Ready | Supabase: `users.email_daily_lesson`, `current_streak`, `timezone`, etc. |
| **Streak System** | Production Ready | `users.current_streak`, `longest_streak`, `last_lesson_at` |

### What Needs Building

| Component | Priority | Complexity | Impact |
|-----------|----------|------------|--------|
| **Native iOS Push (APNs)** | 🔴 Critical | Medium | Very High |
| **Native Android Push (FCM)** | 🔴 Critical | Medium | Very High |
| **Smart Timing Engine** | 🟡 High | High | Very High |
| **Notification Preferences UI** | 🟡 High | Low | High |
| **Watch Apps (WatchOS/WearOS)** | 🟢 Medium | Medium | Medium |
| **Voice Integration (Alexa/Google)** | 🟢 Medium | High | High |
| **TV App Enhancement** | 🔵 Low | Medium | Medium |

---

## 📅 12-Month Roadmap

**Timeline**: December 17, 2025 → December 16, 2026

### Phase 1: Foundation & Core Native Push (Dec 17, 2025 - Mar 16, 2026)

#### Month 1 (Dec 17 - Jan 16): "Push Perfection"
**Goal**: Native push notifications working flawlessly on iOS and Android

**Week 1-2: iOS Native Push**
```javascript
// Add to mobile-app/
- Configure APNs certificates in Apple Developer Portal
- Install react-native-push-notification or Expo Push
- Create API endpoint: POST /api/notifications/subscribe-device
- Database: Add push_tokens table
- Test on physical iPhone
```

**Week 3-4: Android Native Push**
```javascript
// FCM Integration
- Create Firebase project for Curious Kelly
- Configure Firebase Cloud Messaging
- Add google-services.json to Android project
- Test on physical Android device
- Unified push system working
```

**Deliverables**:
- [ ] iOS push notifications live
- [ ] Android push notifications live
- [ ] Device token management in Supabase
- [ ] Basic "lesson reminder" push working

**Copy Examples** (Kelly's Voice):
```
Morning Push:
"✨ Good morning. Today's lesson is about {topic}. 5 minutes. Ready when you are."

Streak Save (Evening):
"You're 1 lesson away from a {streak_days}-day streak. No pressure. Just thought you'd want to know. 💙"

Gentle Return (7 days inactive):
"Hey. I've been here, learning without you. Not the same. Whenever you're ready."
```

**Date Display**: All user-facing content shows real dates (e.g., "December 17" not "Day 1").
The internal `day_number` (1-365) maps to calendar dates starting December 17.

---

#### Month 2 (Jan 17 - Feb 16): "Smart Timing Engine"
**Goal**: Send notifications at the perfect moment for each learner

**The Problem**: A 9am notification is useless if someone learns at 8pm.

**The Solution**: Adaptive notification timing based on:
1. User's actual lesson completion patterns
2. Timezone
3. Device usage patterns (via silent analytics)
4. Explicit preference

**Database Schema**:
```sql
-- Add to Supabase
CREATE TABLE public.notification_preferences (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  preferred_time TIME DEFAULT '09:00',
  timezone TEXT DEFAULT 'America/New_York',
  
  -- Adaptive timing
  auto_timing BOOLEAN DEFAULT true,
  learned_optimal_time TIME,
  last_analyzed_at TIMESTAMPTZ,
  
  -- Channels
  push_enabled BOOLEAN DEFAULT true,
  email_enabled BOOLEAN DEFAULT true,
  
  -- Frequency
  daily_reminder BOOLEAN DEFAULT true,
  streak_alerts BOOLEAN DEFAULT true,
  milestone_celebrations BOOLEAN DEFAULT true,
  gentle_returns BOOLEAN DEFAULT true,
  
  -- Quiet hours
  quiet_start TIME DEFAULT '22:00',
  quiet_end TIME DEFAULT '07:00',
  
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE public.push_tokens (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  device_token TEXT NOT NULL,
  platform TEXT CHECK (platform IN ('ios', 'android', 'web')),
  device_name TEXT,
  last_active_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, device_token)
);
```

**Adaptive Algorithm**:
```typescript
function calculateOptimalTime(userId: string): string {
  // Get last 30 lesson completions
  const completions = await getRecentCompletions(userId, 30);
  
  // Find most common hour (weighted by recency)
  const hourScores = new Map<number, number>();
  completions.forEach((c, index) => {
    const hour = new Date(c.completed_at).getHours();
    const weight = 1 + (index / completions.length); // Recent = higher weight
    hourScores.set(hour, (hourScores.get(hour) || 0) + weight);
  });
  
  // Return highest scoring hour, default to 9am
  const optimalHour = [...hourScores.entries()]
    .sort((a, b) => b[1] - a[1])[0]?.[0] || 9;
    
  return `${optimalHour.toString().padStart(2, '0')}:00`;
}
```

**Deliverables**:
- [ ] notification_preferences table live
- [ ] push_tokens table live
- [ ] Adaptive timing algorithm running
- [ ] User can override with explicit preference
- [ ] Quiet hours respected

---

#### Month 3 (Feb 17 - Mar 16): "Notification Personality"
**Goal**: Every notification feels like Kelly — warm, smart, never annoying

**Notification Types & Copy Library**:

| Type | Trigger | Frequency | Copy Style |
|------|---------|-----------|------------|
| Daily Reminder | Optimal time | Once/day | Warm invitation |
| Streak Save | 8pm if no lesson yet | Max 1/day | Gentle nudge |
| Streak Celebration | On milestone | Per milestone | Celebration |
| Gentle Return | 3, 7, 14 days inactive | Max 1/week | No guilt |
| Birthday | Birthday date | Once/year | Celebration |
| Year Complete | 365 unique lessons | Once/year | Major celebration |

**Copy Variations** (A/B Test Ready):

```javascript
const DAILY_REMINDER_COPY = [
  {
    variant: 'A',
    title: "✨ Your 5 minutes of wonder",
    body: "Today: {lesson_title}. Ready when you are.",
    data: { screen: 'lesson', day: '{day_number}' }
  },
  {
    variant: 'B', 
    title: "{lesson_emoji} {lesson_title}",
    body: "5 minutes. I think you'll love this one.",
    data: { screen: 'lesson', day: '{day_number}' }
  },
  {
    variant: 'C',
    title: "Good morning, {name}",
    body: "Today we learn about {lesson_title}. Shall we?",
    data: { screen: 'lesson', day: '{day_number}' }
  }
];

const STREAK_SAVE_COPY = [
  {
    streak_range: [1, 7],
    title: "Keep it going?",
    body: "Day {streak_days} is waiting. Just 5 minutes."
  },
  {
    streak_range: [8, 30],
    title: "Don't let this streak slip 🔥",
    body: "{streak_days} days strong. Today's lesson won't take long."
  },
  {
    streak_range: [31, 100],
    title: "Incredible streak at risk",
    body: "{streak_days} days of curiosity. Worth protecting, don't you think?"
  },
  {
    streak_range: [101, Infinity],
    title: "A legendary streak needs you",
    body: "{streak_days} days. Most people never get here. You did. Keep going?"
  }
];

const GENTLE_RETURN_COPY = {
  day_3: {
    title: "Miss you a little",
    body: "No pressure. Just wanted you to know the lessons are here."
  },
  day_7: {
    title: "Your spot is still here",
    body: "A week without learning together. Whenever you're ready."
  },
  day_14: {
    title: "Still curious?",
    body: "Two weeks is a long time. I hope you're okay. I'm here."
  }
};
```

**Deliverables**:
- [ ] Full copy library in Supabase
- [ ] Personalization engine (name, streak, lesson title)
- [ ] A/B testing infrastructure
- [ ] Analytics: open rate, tap rate, conversion to lesson start
- [ ] User feedback mechanism ("Was this helpful?")

---

### Phase 2: Multi-Platform & Ambient Intelligence (Mar 17 - Jun 16, 2026)

#### Month 4 (Mar 17 - Apr 16): "Desktop Delights"
**Goal**: Electron desktop app with native notifications

**Windows Toast Notifications**:
```javascript
// desktop-app/src/notifications.js
const { Notification } = require('electron');

function showLessonReminder(lesson) {
  new Notification({
    title: `✨ ${lesson.emoji} ${lesson.title}`,
    body: "Your 5 minutes of wonder is ready.",
    icon: path.join(__dirname, 'assets/kelly-icon.png'),
    silent: false,
    actions: [
      { type: 'button', text: 'Learn Now' },
      { type: 'button', text: 'Later' }
    ]
  }).show();
}
```

**macOS Notifications**:
- Rich notification with Kelly avatar
- Click to open → immediate lesson start
- "Later" → snooze 30 minutes

**Deliverables**:
- [ ] Native notifications on Windows
- [ ] Native notifications on macOS
- [ ] Notification preferences sync across devices
- [ ] Badge count on dock icon (macOS)

---

#### Month 5 (Apr 17 - May 16): "Watch Apps"
**Goal**: Kelly on your wrist — the most ambient reminder possible

**Apple Watch (WatchOS)**:
```swift
// Complication showing streak
// Tap → start lesson on phone
// Notification with haptic
```

**Features**:
- Watch Face Complication: Current streak + today's lesson emoji
- Push notification with haptic tap
- Tap notification → opens lesson on iPhone
- Quick Glance: Streak, next lesson time

**Wear OS (Android)**:
- Watch Face Tile
- Notification cards
- Voice: "Hey Google, start my Kelly lesson"

**Deliverables**:
- [ ] WatchOS app submitted to App Store
- [ ] Wear OS app submitted to Play Store
- [ ] Cross-device lesson start working
- [ ] Streak complication live

---

#### Month 6 (May 17 - Jun 16): "Voice Assistants"
**Goal**: "Alexa, start my lesson" / "Hey Google, what am I learning today?"

**Amazon Alexa Skill**:
```javascript
// Skill intents
const INTENTS = [
  "StartLessonIntent",      // "Start my lesson"
  "TodaysLessonIntent",     // "What am I learning today?"
  "StreakStatusIntent",     // "How's my streak?"
  "NextLessonTimeIntent"    // "When's my next lesson?"
];

// Flash Briefing (Daily Update)
// "From Curious Kelly: Today's lesson is about {topic}. 
//  You're on a {streak} day streak. Keep it going!"
```

**Google Assistant Action**:
```javascript
// Similar intents
// "Hey Google, talk to Curious Kelly"
// "Hey Google, what's today's Kelly lesson?"
```

**Daily Briefing Integration**:
- Alexa Flash Briefing skill
- Google Assistant Daily Update
- Proactive notification: "Your Kelly lesson is ready"

**Deliverables**:
- [ ] Alexa Skill published
- [ ] Google Action published
- [ ] Flash Briefing / Daily Update working
- [ ] Account linking for personalization

---

### Phase 3: Collective Intelligence & Social (Jun 17 - Sep 16, 2026)

#### Month 7 (Jun 17 - Jul 16): "The Collective Nudge"
**Goal**: Learn together, grow together — notifications that connect

**Family Notifications**:
```javascript
// When family member completes a lesson
const FAMILY_NOTIFICATION = {
  title: "Your family is learning!",
  body: "{child_name} just learned about {lesson_title}. Ask them what they discovered!",
  data: { type: 'family', member_id: '...' }
};
```

**Collective Milestones**:
```javascript
// Community celebration
const COLLECTIVE_MILESTONE = {
  title: "1 Million Lessons Today 🎉",
  body: "You're part of something extraordinary. Learners worldwide completed 1,000,000 lessons today.",
  data: { type: 'collective' }
};
```

**The Commons Notification**:
```javascript
// After answering, show collective
const COMMONS_NOTIFICATION = {
  title: "You're not alone",
  body: "47% of learners answered like you. 53% disagreed. See the full picture.",
  data: { type: 'commons', lesson_day: '...' }
};
```

**Deliverables**:
- [ ] Family notification system
- [ ] Collective milestone celebrations
- [ ] Commons insights push
- [ ] "X people learning with you right now" ambient presence

---

#### Month 8 (Jul 17 - Aug 16): "Ambient Intelligence"
**Goal**: Kelly anticipates, doesn't interrupt

**Smart Notification Suppression**:
```javascript
// Don't notify if:
// - User is already in the app
// - User just completed a lesson in last 6 hours
// - User is in Do Not Disturb mode
// - User's device shows active meeting (iOS Focus mode)
// - Weekend morning and user has set "weekday mornings only"

function shouldNotify(user: User, context: DeviceContext): boolean {
  if (context.appInForeground) return false;
  if (timeSinceLastLesson(user) < 6 * 60 * 60 * 1000) return false;
  if (context.focusModeActive) return false;
  if (isQuietHours(user)) return false;
  return true;
}
```

**Predictive Notifications**:
```javascript
// If user usually learns at 7:30am but hasn't by 7:45am on a weekday
// Send: "Running a bit behind? I'll save your spot."

// If user is traveling (location change)
// Don't send time-based notifications, switch to "whenever you're ready"

// If user's streak is about to hit a milestone
// Increase notification frequency slightly
```

**Deliverables**:
- [ ] Focus mode / DND detection
- [ ] In-app detection (no double notifications)
- [ ] Predictive gentle nudges
- [ ] Location-aware quiet mode

---

#### Month 9 (Aug 17 - Sep 16): "Gamification That Doesn't Suck"
**Goal**: Duolingo-level engagement without manipulation

**Streak Shields** (Earned, Not Bought):
```javascript
// After 30-day streak: earn a "Streak Shield"
// Can be used once to protect streak during illness/travel
// Kelly: "You've earned a shield. Life happens. Use it wisely."

const STREAK_SHIELD_NOTIFICATION = {
  title: "🛡️ Streak Shield Earned!",
  body: "30 days! You've earned protection. One day of rest won't break your streak. Use it when you need it.",
  data: { type: 'achievement', id: 'streak_shield' }
};
```

**Surprise Delights**:
```javascript
// Random, rare, unexpected
const SURPRISE_NOTIFICATIONS = [
  {
    trigger: 'random_1_in_100',
    title: "Just checking in",
    body: "No lesson reminder. Just wanted to say hi. You're doing great. 💙"
  },
  {
    trigger: 'full_moon',
    title: "Look up tonight 🌕",
    body: "There's a full moon. Remember what we learned about tides?"
  },
  {
    trigger: 'user_birthday',
    title: "Happy Birthday! 🎂",
    body: "Your birthday lesson is special. It's the same every year. But somehow different."
  }
];
```

**Anti-Manipulation Principles**:
- No guilt trips
- No fake urgency
- No loss aversion dark patterns
- Always an easy unsubscribe
- Kelly's voice: "I'm here when you want me. Not pushing."

**Deliverables**:
- [ ] Streak Shield system
- [ ] Surprise delight notifications (rare, not annoying)
- [ ] Milestone celebrations without FOMO
- [ ] "Snooze" and "Not now" always available

---

### Phase 4: Scale & Polish (Sep 17 - Dec 16, 2026)

#### Month 10 (Sep 17 - Oct 16): "TV & Living Room"
**Goal**: Kelly on every screen

**Roku Enhancement**:
- Deep link notifications (limited on Roku)
- On-screen notification when channel launches
- "Continue your streak" prompt

**Apple TV / Fire TV**:
- Native tvOS app (React Native TVKit)
- Fire TV app (React Native)
- Living room → "start lesson" voice command

**Ambient Mode**:
- Kelly as screensaver/ambient display
- "Daily curiosity" rotating on TV when idle
- Photo frame integration (smart displays)

**Deliverables**:
- [ ] Enhanced Roku channel
- [ ] tvOS app submitted
- [ ] Fire TV app submitted
- [ ] Ambient display mode

---

#### Month 11 (Oct 17 - Nov 16): "Global Scale"
**Goal**: Handle 100,000+ daily notifications without breaking

**Infrastructure**:
```javascript
// Notification Queue System
// - Bull queue (Redis) for scheduling
// - Worker processes for sending
// - Retry logic for failed sends
// - Rate limiting to not anger Apple/Google

const NOTIFICATION_QUEUE = new Bull('notifications', {
  redis: { host: 'redis-host', port: 6379 }
});

// Process notifications in batches
NOTIFICATION_QUEUE.process(100, async (jobs) => {
  await batchSendNotifications(jobs.map(j => j.data));
});
```

**Multi-Region**:
- Timezone-aware scheduling
- Regional content variations
- Language support (EN/ES/FR per CLAUDE.md)

**Analytics Dashboard**:
- Real-time notification metrics
- Open rate by type, time, platform
- Conversion to lesson start
- A/B test results

**Deliverables**:
- [ ] Scalable notification queue
- [ ] Multi-timezone optimization
- [ ] Analytics dashboard
- [ ] 99.9% delivery rate

---

#### Month 12 (Nov 17 - Dec 16): "The Collective Experience"
**Goal**: "Met Kelly" becomes a movement — Year 1 Complete!

**The Numbers**:
- 100,000 daily active learners
- 95%+ notification delivery rate
- 40%+ notification open rate
- 25%+ conversion to lesson start
- Net Promoter Score: 70+

**The Feeling**:
- Learning feels as natural as checking weather
- Notifications are anticipated, not dreaded
- Kelly is a daily companion, not an app
- Streaks feel like self-care, not obligation
- Missing a day feels gentle, not guilty

**Community Features**:
```javascript
// "X people learning right now"
const PRESENCE_INDICATOR = {
  display: "2,847 people learning with you right now",
  update_frequency: '5 minutes',
  location: 'lesson header, notification'
};

// Annual celebration
const YEAR_END_NOTIFICATION = {
  title: "You learned 183 things this year 🎉",
  body: "Your curiosity changed you. And maybe the world. Here's to next year.",
  data: { type: 'annual_summary' }
};
```

**Deliverables**:
- [ ] 100k DAU milestone
- [ ] Community presence features
- [ ] Annual summary emails/notifications
- [ ] "Introduce a friend" refined

---

## 📱 Platform-Specific Details

### iOS Implementation

**Requirements**:
- Apple Push Notification service (APNs) certificate
- App Store Connect push capability
- `react-native-push-notification` or Expo Push

**App Store Submission**:
```
- Category: Education (Kids or General)
- Age Rating: 4+
- Push Notification capability enabled
- Privacy manifest with notification justification
- "Kelly <hello@curiouskelly.com>" for support
```

**Rich Notifications**:
```swift
// iOS supports rich media in notifications
let content = UNMutableNotificationContent()
content.title = "✨ Today's Wonder"
content.body = "How Money Works — 5 minutes with Kelly"
content.attachments = [kellyImageAttachment]
content.categoryIdentifier = "LESSON_REMINDER"
content.sound = .default
```

### Android Implementation

**Requirements**:
- Firebase Cloud Messaging (FCM)
- `google-services.json`
- Notification channels (Android 8+)

**Notification Channels**:
```kotlin
val channels = listOf(
    NotificationChannel("daily_lesson", "Daily Lesson", IMPORTANCE_DEFAULT),
    NotificationChannel("streak_alerts", "Streak Alerts", IMPORTANCE_HIGH),
    NotificationChannel("celebrations", "Celebrations", IMPORTANCE_HIGH),
    NotificationChannel("gentle_returns", "Gentle Returns", IMPORTANCE_LOW)
)
```

### Web Push

**Current State**: 80% complete in `public/js/push-notifications.js`

**Remaining**:
- [ ] Generate production VAPID keys
- [ ] Store subscriptions in Supabase
- [ ] Backend push sending via `web-push` library
- [ ] Service worker notification display enhancement

---

## 🗄️ Database Schema Additions

```sql
-- Notification preferences
CREATE TABLE public.notification_preferences (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  
  -- Timing
  preferred_time TIME DEFAULT '09:00',
  timezone TEXT DEFAULT 'America/New_York',
  auto_timing BOOLEAN DEFAULT true,
  learned_optimal_time TIME,
  
  -- Channels
  push_enabled BOOLEAN DEFAULT true,
  email_enabled BOOLEAN DEFAULT true,
  web_push_enabled BOOLEAN DEFAULT true,
  
  -- Types
  daily_reminder BOOLEAN DEFAULT true,
  streak_alerts BOOLEAN DEFAULT true,
  milestone_celebrations BOOLEAN DEFAULT true,
  gentle_returns BOOLEAN DEFAULT true,
  family_updates BOOLEAN DEFAULT true,
  collective_milestones BOOLEAN DEFAULT false,
  
  -- Quiet hours
  quiet_start TIME DEFAULT '22:00',
  quiet_end TIME DEFAULT '07:00',
  weekend_quiet BOOLEAN DEFAULT false,
  
  -- Streak protection
  streak_shields_available INTEGER DEFAULT 0,
  streak_shields_used INTEGER DEFAULT 0,
  
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Device tokens for push
CREATE TABLE public.push_tokens (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  device_token TEXT NOT NULL,
  platform TEXT CHECK (platform IN ('ios', 'android', 'web', 'macos', 'windows')),
  device_name TEXT,
  app_version TEXT,
  os_version TEXT,
  last_active_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, device_token)
);

-- Notification history for analytics
CREATE TABLE public.notification_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  notification_type TEXT NOT NULL,
  title TEXT,
  body TEXT,
  sent_at TIMESTAMPTZ DEFAULT now(),
  delivered_at TIMESTAMPTZ,
  opened_at TIMESTAMPTZ,
  converted_at TIMESTAMPTZ, -- clicked through to lesson
  platform TEXT,
  metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_notification_log_user ON public.notification_log(user_id);
CREATE INDEX idx_notification_log_type ON public.notification_log(notification_type);
CREATE INDEX idx_notification_log_sent ON public.notification_log(sent_at);

-- A/B test tracking
CREATE TABLE public.notification_ab_tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_name TEXT NOT NULL,
  variant_a JSONB NOT NULL, -- { title, body, etc }
  variant_b JSONB NOT NULL,
  start_date TIMESTAMPTZ DEFAULT now(),
  end_date TIMESTAMPTZ,
  winner TEXT, -- 'A' or 'B' or NULL
  results JSONB DEFAULT '{}'::jsonb
);
```

---

## 💰 Budget & Resources

### API Costs (Monthly at Scale)

| Service | Volume | Cost |
|---------|--------|------|
| **APNs** | 100k/day | Free |
| **FCM** | 100k/day | Free |
| **Resend Email** | 100k/month | ~$100 |
| **Redis (Queue)** | 1GB | ~$25 |
| **Analytics** | 1M events | ~$50 |
| **Total** | — | ~$175/month |

### App Store Costs (Annual)

| Platform | Cost |
|----------|------|
| Apple Developer | $99/year |
| Google Play | $25 one-time |
| Amazon Developer | Free |
| Roku Developer | Free |
| **Total Year 1** | ~$125 |

### Development Hours (Estimated)

| Month | Focus | Hours |
|-------|-------|-------|
| 1 | Native Push iOS/Android | 80 |
| 2 | Smart Timing Engine | 60 |
| 3 | Copy & Personalization | 40 |
| 4 | Desktop Notifications | 30 |
| 5 | Watch Apps | 60 |
| 6 | Voice Assistants | 80 |
| 7 | Collective Features | 40 |
| 8 | Ambient Intelligence | 60 |
| 9 | Gamification | 40 |
| 10 | TV Apps | 60 |
| 11 | Scale Infrastructure | 40 |
| 12 | Polish & Launch | 40 |
| **Total** | — | **630 hours** |

---

## 🎯 Success Metrics

### Key Performance Indicators

| Metric | Month 3 | Month 6 | Month 12 |
|--------|---------|---------|----------|
| Daily Active Users | 1,000 | 10,000 | 100,000 |
| Push Opt-in Rate | 40% | 50% | 60% |
| Notification Open Rate | 30% | 35% | 40% |
| Conversion to Lesson | 15% | 20% | 25% |
| 7-Day Retention | 30% | 40% | 50% |
| 30-Day Retention | 15% | 25% | 35% |
| Net Promoter Score | 50 | 60 | 70 |

### Anti-Metrics (What We Avoid)

| Metric | Target | Reason |
|--------|--------|--------|
| Unsubscribe Rate | <5%/month | Notifications shouldn't annoy |
| "Not Now" Taps | Monitor, not minimize | Respect user's time |
| Notification Volume | ≤2/day max | Quality over quantity |
| Guilt-Trip Copy Usage | 0% | Never manipulate |

---

## 📝 Copy Library (Kelly's Voice)

### Morning Reminders (Rotate Daily)

```
A: "✨ Good morning. 5 minutes of wonder await."
B: "Rise and learn? Today: {lesson_title}"
C: "{lesson_emoji} {lesson_title} — Your {date_formatted} curiosity."
D: "Hello, {name}. Ready to discover something?"
E: "The world learned something while you slept. Want to catch up?"
```

**Note**: `{date_formatted}` displays as "December 17" (not "Day 1").

### Streak Celebrations

```
7 days: "A week of curiosity. That's no small thing. 🌟"
14 days: "Two weeks in. You're building something beautiful."
30 days: "A month together. Habits are forming. Keep going. 🔥"
60 days: "60 days. Most people quit at 3. You're extraordinary."
100 days: "💯 One hundred days. I don't have words. Just gratitude."
365 days: "A full year. Every single day. You're a legendary learner. ✨"
```

### Gentle Returns (After Absence)

```
3 days: "Miss you a little. No pressure. Just here."
7 days: "A week without learning together. Hope you're okay. 💙"
14 days: "Two weeks. Life happens. I'm still here when you're ready."
30 days: "It's been a month. Your streak is gone, but you're not. Come back?"
90 days: "Three months. I think about our lessons. No guilt. Just truth."
```

### Birthday

```
"🎂 Happy Birthday, {name}! Your birthday lesson is waiting. It's the same one, every year. But somehow, it means something different each time."
```

### Surprise Delights (Rare)

```
"No lesson today. Just wanted to say: you're doing great. 💙"
"Look outside. Notice something you never noticed before. That's curiosity."
"You've been learning for {days} days. I'm proud to be your teacher."
"Full moon tonight 🌕. Remember what we learned about tides?"
```

---

## 🔧 Technical Architecture

### Notification Flow

```
┌─────────────────┐
│ Cron Scheduler  │ (Vercel Cron)
│ 00:00 UTC daily │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Processor │ Group users by optimal time + timezone
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Notification    │ Select copy, personalize, A/B assign
│ Generator       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Send Queue      │ Bull/Redis queue with retries
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┬──────────────────┐
         ▼                 ▼                 ▼                  ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ APNs       │    │ FCM        │    │ Web Push   │    │ Resend     │
│ (iOS)      │    │ (Android)  │    │ (Browser)  │    │ (Email)    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
         │                 │                 │                  │
         └─────────────────┴─────────────────┴──────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │ Analytics Log   │ Supabase
                          └─────────────────┘
```

### API Endpoints Needed

```
POST /api/notifications/subscribe-device
  - Register push token for user

DELETE /api/notifications/unsubscribe-device
  - Remove push token

GET /api/notifications/preferences
  - Get user's notification preferences

PUT /api/notifications/preferences
  - Update preferences

POST /api/notifications/test
  - Send test notification to self

GET /api/admin/notifications/queue
  - View notification queue status

POST /api/cron/daily-notifications
  - Main cron job for daily notifications
  
POST /api/cron/streak-save-notifications
  - Evening check for streak saves
```

---

## ✅ Implementation Checklist

### Phase 1: Foundation (Month 1-3)
- [ ] Generate production VAPID keys for web push
- [ ] Configure APNs certificates in Apple Developer Portal
- [ ] Create Firebase project and configure FCM
- [ ] Add `react-native-push-notification` to mobile app
- [ ] Create `notification_preferences` table in Supabase
- [ ] Create `push_tokens` table in Supabase
- [ ] Build POST /api/notifications/subscribe-device endpoint
- [ ] Build notification preferences UI in settings
- [ ] Implement adaptive timing algorithm
- [ ] Create copy library in Supabase
- [ ] Build personalization engine
- [ ] Test on physical iOS device
- [ ] Test on physical Android device
- [ ] Submit updated mobile apps to stores

### Phase 2: Expansion (Month 4-6)
- [ ] Add Electron notification support
- [ ] Build WatchOS companion app
- [ ] Build Wear OS companion app
- [ ] Create Alexa skill
- [ ] Create Google Action
- [ ] Implement streak shield system

### Phase 3: Intelligence (Month 7-9)
- [ ] Build family notification system
- [ ] Implement collective milestones
- [ ] Add Focus mode / DND detection
- [ ] Create surprise delight system
- [ ] Implement A/B testing infrastructure
- [ ] Build analytics dashboard

### Phase 4: Scale (Month 10-12)
- [ ] Create Bull/Redis notification queue
- [ ] Build multi-timezone optimization
- [ ] Submit tvOS app
- [ ] Submit Fire TV app
- [ ] Achieve 100k DAU milestone
- [ ] Launch annual summary feature

---

## 🌟 The Vision

In 12 months, opening your phone in the morning will include a moment with Kelly — as natural as checking the weather. Not because we tricked you. Because you want it.

Kelly becomes:
- The most anticipated notification of your day
- The habit you protect, not resent
- The teacher who knows when to speak and when to wait
- The companion that grows with you, year after year

**"It's not going away anytime soon, or ever... meet Kelly."** 😊

---

*Last Updated: December 9, 2025*
*Author: Chief Academic Officer & Notification Architect*
*Status: Roadmap Complete - Ready for Implementation*

