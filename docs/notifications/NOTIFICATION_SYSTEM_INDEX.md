# 🔔 Curious Kelly Notification System

> **"Never miss a class with Kelly!"**

**Launch Date**: December 17, 2025  
**Year 1 Content**: December 17, 2025 → December 16, 2026

### ⚠️ IMPORTANT: Date Display Convention
- **Internal**: `day_number` (1-365) — used in database, APIs, URLs
- **User-Facing**: Real calendar dates — "December 17" not "Day 1"
- **Utility**: `lib/lesson-dates.ts` for all date conversions

This index provides quick navigation to all notification system documentation and code.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**DAILY_NOTIFICATION_SYSTEM_ROADMAP.md**](./DAILY_NOTIFICATION_SYSTEM_ROADMAP.md) | Complete 12-month roadmap for building the best daily learning notification experience |
| [**EMAIL_SYSTEM_COMPLETE.md**](../email/EMAIL_SYSTEM_COMPLETE.md) | Email notification system (Resend) |
| [**LIFETIME_LEARNER_EXPERIENCE.md**](../experience/LIFETIME_LEARNER_EXPERIENCE.md) | Overall learner experience philosophy |

---

## 💻 API Endpoints

### Device Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/notifications/subscribe-device` | POST | Register device for push notifications |
| `/api/notifications/preferences` | GET/PUT | Get or update notification preferences |

### Cron Jobs
| Endpoint | Schedule | Description |
|----------|----------|-------------|
| `/api/cron/daily-push-notifications` | Hourly | Send daily lesson push notifications |
| `/api/cron/daily-lesson` | 12:00 UTC | Send daily lesson emails |
| `/api/cron/birthday-emails` | 00:00 UTC | Send birthday emails |
| `/api/cron/gentle-return` | Daily | Send re-engagement emails |

---

## 🗄️ Database Tables

| Table | Description |
|-------|-------------|
| `notification_preferences` | User's notification settings (timing, channels, types) |
| `push_tokens` | Device tokens for iOS, Android, Web push |
| `notification_log` | Log of all sent notifications |
| `notification_copy` | Kelly's notification copy library |
| `notification_ab_tests` | A/B test configuration |
| `notification_queue` | Scheduled notifications waiting to send |

**Migration**: `supabase/migrations/020_notification_system.sql`

---

## 📱 Platform Support

| Platform | Status | Technology |
|----------|--------|------------|
| iOS | 🟡 Ready for Implementation | APNs via react-native-push-notification |
| Android | 🟡 Ready for Implementation | FCM via Firebase |
| Web | 🟢 Foundation Complete | Web Push (VAPID) |
| Desktop (Windows/Mac/Linux) | 🟡 Planned | Electron native notifications |
| Apple Watch | 🔵 Planned Q2 2026 | WatchOS complications |
| Roku | ⚪ Limited | On-screen only |

---

## 📂 Code Locations

```
UI-TARS-desktop/
├── lib/
│   ├── lesson-dates.ts                 # 📅 Date utilities (CRITICAL)
│   │                                   # Converts day_number ↔ calendar dates
│   │                                   # December 17, 2025 = Day 1
│   └── push-sender.ts                  # 🚀 Push notification sender
│                                       # Web Push, APNs, FCM unified API
├── api/
│   ├── notifications/
│   │   ├── subscribe-device.ts         # Register tokens (auth required)
│   │   ├── web-push-subscribe.ts       # Web push registration (public)
│   │   ├── preferences.ts              # Notification preferences
│   │   └── test-push.ts                # Test push notifications
│   └── cron/
│       ├── daily-push-notifications.ts # Push notification cron (hourly)
│       ├── daily-lesson.ts             # Email cron
│       ├── birthday-emails.ts          # Birthday email cron
│       └── gentle-return.ts            # Re-engagement cron
├── public/
│   ├── js/
│   │   └── push-notifications.js       # Web push client
│   └── sw.js                           # Service worker
├── mobile-app/
│   ├── App.js                          # React Native app with Firebase
│   └── package.json                    # Dependencies including Firebase
├── desktop-app/
│   └── src/
│       └── main.js                     # Electron app
├── supabase/
│   └── migrations/
│       └── 020_notification_system.sql # Database schema
└── docs/
    └── notifications/
        ├── NOTIFICATION_SYSTEM_INDEX.md      # This file
        ├── DAILY_NOTIFICATION_SYSTEM_ROADMAP.md # 12-month roadmap
        └── PUSH_NOTIFICATION_SETUP.md        # Setup guide for FCM/APNs
```

---

## 🎯 Quick Start

### 1. Apply Database Migration
```bash
# Apply the notification system migration
npx supabase migration up
```

### 2. Configure Environment Variables
```env
# Push Notification Keys (add to Vercel)
FIREBASE_PROJECT_ID=your-firebase-project
FIREBASE_PRIVATE_KEY=...
FIREBASE_CLIENT_EMAIL=...
APNS_KEY_ID=...
APNS_TEAM_ID=...
APNS_PRIVATE_KEY=...
VAPID_PUBLIC_KEY=...
VAPID_PRIVATE_KEY=...
```

### 3. Enable Cron Jobs
```json
// Add to vercel.json
{
  "crons": [
    {
      "path": "/api/cron/daily-push-notifications",
      "schedule": "0 * * * *"
    }
  ]
}
```

### 4. Integrate in Mobile App
```javascript
// In mobile-app/App.js
import PushNotificationIOS from '@react-native-community/push-notification-ios';

useEffect(() => {
  // Request permission and register token
  PushNotificationIOS.requestPermissions().then(permission => {
    if (permission.alert) {
      // Get token and send to API
    }
  });
}, []);
```

---

## 📊 Key Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Push Opt-in Rate | 60% | TBD |
| Notification Open Rate | 40% | TBD |
| Conversion to Lesson | 25% | TBD |
| Unsubscribe Rate | <5%/month | TBD |

---

## 🎨 Copy Library Preview

### Daily Reminders
- ✨ Your 5 minutes of wonder
- {emoji} {lesson_title}
- Good morning, {name}

### Streak Saves
- Keep it going? Day {streak_days} is waiting.
- Don't let this streak slip 🔥

### Celebrations
- 🌟 One week of wonder!
- 💯 One hundred days!

### Gentle Returns
- Miss you a little. No pressure.
- Your spot is still here.

---

## 📞 Support

- **Technical Issues**: Check Vercel function logs
- **Delivery Issues**: Monitor APNs/FCM dashboards
- **Questions**: hello@curiouskelly.com

---

*Last Updated: December 9, 2025*

