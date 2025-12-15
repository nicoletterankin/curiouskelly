# Master Backlog - Curious Kelly

> **Last Updated**: December 15, 2025
> **Goal**: Keep learners for life. Track everything. Zero trust.

---

## 🔴 CRITICAL (Blocking Launch)

| ID | Feature | Status | Owner | Notes |
|----|---------|--------|-------|-------|
| C1 | TODAY is free (not Day 1) | ✅ DONE | Claude | Deployed |
| C2 | Stripe checkout working | ✅ DONE | Claude | All plans functional |
| C3 | Welcome email sends | ✅ DONE | Claude | Via Resend |
| C4 | Paywall shows options | ✅ DONE | Claude | Buy lesson + Subscribe |
| C5 | Video fallback to photo+audio | ✅ DONE | Claude | kelly-fallback-engine.js |

---

## 🟠 HIGH PRIORITY (Week 1)

### Revenue & Access
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| H1 | Pay-per-lesson purchase | 🔲 TODO | Stripe product, $1.99 |
| H2 | lesson_purchases table | 🔲 TODO | Track individual buys |
| H3 | Check purchases in canAccessDay() | 🔲 TODO | Wire up |
| H4 | Regional pricing detection | 🔲 TODO | Geo → price tier |

### User Tracking (Foundation)
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| H5 | user_events table | 🔲 TODO | Immutable audit log |
| H6 | Event logging API | 🔲 TODO | POST /api/events |
| H7 | Lesson completion events | 🔲 TODO | Track in frontend |
| H8 | Extend users table | 🔲 TODO | Lifetime stats fields |

---

## 🟡 MEDIUM PRIORITY (Week 2-3)

### Community Features
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| M1 | lesson_comments table | 🔲 TODO | Comments on lessons |
| M2 | Comment posting UI | 🔲 TODO | In lesson player |
| M3 | Comment moderation queue | 🔲 TODO | Admin view |
| M4 | AI moderation filter | 🔲 TODO | Auto-flag inappropriate |

### Contributions
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| M5 | lesson_artwork_submissions table | 🔲 TODO | User-generated art |
| M6 | Artwork upload UI | 🔲 TODO | Image upload flow |
| M7 | Artwork moderation | 🔲 TODO | Approve/reject |
| M8 | Display user artwork in lessons | 🔲 TODO | With attribution |

### Downloads
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| M9 | lesson_downloads table | 🔲 TODO | Track downloads |
| M10 | Download bundle generation | 🔲 TODO | Package lessons |
| M11 | Offline player | 🔲 TODO | PWA or native |
| M12 | Access verification on download | 🔲 TODO | Prevent piracy |

---

## 🟢 FUTURE (Week 4+)

### Live Classes
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| F1 | live_class_sessions table | 🔲 TODO | Schedule + status |
| F2 | live_class_attendance table | 🔲 TODO | Who attended |
| F3 | Live class technology selection | 🔲 RESEARCH | YouTube? Daily.co? |
| F4 | Hourly class scheduling | 🔲 TODO | Every hour on the hour |
| F5 | Free seat for today's lesson | 🔲 TODO | Everyone gets in |
| F6 | Live class recording | 🔲 TODO | For replay |

### Admin & Audit
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| F7 | User audit view | 🔲 TODO | Pull user_id → see everything |
| F8 | Event search/filter | 🔲 TODO | Query event history |
| F9 | Export user data (GDPR) | 🔲 TODO | JSON export |
| F10 | Anomaly detection | 🔲 TODO | Flag unusual activity |

### On-Demand Kelly AI
| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| F11 | Topic request interface | 🔲 FUTURE | "Teach me about X" |
| F12 | AI lesson generation pipeline | 🔲 FUTURE | Generate on demand |
| F13 | Kelly voice synthesis for new content | 🔲 FUTURE | ElevenLabs |
| F14 | Personalized learning paths | 🔲 FUTURE | Based on history |

---

## ⏸️ BACKLOG (Deprioritized)

| ID | Feature | Status | Notes |
|----|---------|--------|-------|
| B1 | Kid video generation (ages 2-12) | ⏸️ BACKLOG | Needs HeyGen avatar |
| B2 | Social media accounts | ⏸️ BACKLOG | Create @CuriousKelly |
| B3 | Roku app | ⏸️ BACKLOG | Low priority |
| B4 | Apple Watch app | ⏸️ BACKLOG | Nice to have |

---

## 📊 Progress Tracker

| Week | Focus | Completed |
|------|-------|-----------|
| Dec 15 | Business model architecture | ✅ |
| Dec 16-22 | Revenue (pay-per-lesson) + Event tracking | 🔲 |
| Dec 23-29 | Comments + Community | 🔲 |
| Dec 30-Jan 5 | Downloads + Offline | 🔲 |
| Jan 6-12 | Live classes | 🔲 |
| Jan 13-19 | Admin audit view | 🔲 |

---

## Database Tables Status

| Table | Exists | Schema Ready | Notes |
|-------|--------|--------------|-------|
| users | ✅ | 🔲 Needs extension | Add lifetime fields |
| core_lessons | ✅ | ✅ | 365 lessons |
| lesson_atoms | ✅ | ✅ | Phase content |
| lesson_shards | ✅ | ✅ | Multilingual |
| kelly_motion_library | ✅ | ✅ | 335 videos |
| revenue_events | ✅ | ✅ | Stripe events |
| user_events | 🔲 | ✅ Designed | Audit log |
| lesson_purchases | 🔲 | ✅ Designed | Pay-per-lesson |
| lesson_comments | 🔲 | ✅ Designed | Comments |
| lesson_artwork_submissions | 🔲 | ✅ Designed | User art |
| lesson_downloads | 🔲 | ✅ Designed | Offline tracking |
| live_class_sessions | 🔲 | ✅ Designed | Live classes |
| live_class_attendance | 🔲 | ✅ Designed | Attendance |
| regional_prices | 🔲 | ✅ Designed | Price tiers |
| user_pricing_tiers | 🔲 | ✅ Designed | User tier |

---

## API Endpoints Status

| Endpoint | Exists | Notes |
|----------|--------|-------|
| POST /api/events | 🔲 | Log user events |
| GET /api/users/:id/events | 🔲 | Audit view (admin) |
| POST /api/lessons/:day/comments | 🔲 | Post comment |
| GET /api/lessons/:day/comments | 🔲 | List comments |
| POST /api/lessons/:day/artwork | 🔲 | Submit artwork |
| POST /api/downloads/bundle | 🔲 | Request download |
| GET /api/live/next | 🔲 | Next live class |
| POST /api/live/:session/join | 🔲 | Join class |
| POST /api/stripe-checkout | ✅ | All plans |
| POST /api/create-checkout | ✅ | Embedded checkout |
| POST /api/webhooks/stripe-revenue | ✅ | Payment events |
| POST /api/send-welcome-email | ✅ | Welcome emails |
| GET /api/health | ✅ | Health check |
| GET /api/kelly-video | ✅ | Video assets |
| GET /api/motion-clip | ✅ | Motion library |

---

## Key Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Daily lesson completions | Unknown | Track with events |
| Subscriber LTV | Unknown | $200+ |
| Streak retention (7 days) | Unknown | 70% |
| Streak retention (30 days) | Unknown | 50% |
| Comment engagement | N/A | 10% of users |
| Live class attendance | N/A | 100+ per session |
| Download usage | N/A | 20% of subscribers |

---

*This is the single source of truth for what needs to be built.*
*Update this document as tasks complete.*
