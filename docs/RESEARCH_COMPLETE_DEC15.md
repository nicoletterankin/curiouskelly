# Research Complete - December 15, 2025

## 1. Supabase Schema Audit

### Existing Tables

| Table | Purpose | Row Count (est) |
|-------|---------|-----------------|
| `core_lessons` | 365 daily lessons | 365 |
| `lesson_atoms` | Phase content (archetype variations) | 21,915 |
| `lesson_shards` | Demographic variants (age/region/tone) | 38,700 |
| `users` | User profiles (extends auth.users) | Variable |
| `user_progress` | Lesson completion tracking | Variable |
| `affiliates` | Affiliate program members | Variable |
| `referrals` | Affiliate referral tracking | Variable |
| `affiliate_applications` | Pending affiliate apps | Variable |
| `enterprise_inquiries` | Enterprise leads | Variable |
| `newsletter_subscribers` | Email list | Variable |
| `analytics_events` | Basic event tracking | Variable |
| `learning_groups` | Group learning (Share feature) | Variable |
| `group_members` | Group membership | Variable |
| `daily_lesson_stats` | Aggregate daily stats | ~365 |
| `kelly_motion_library` | Video motion clips | 420 |
| `kelly_video_assets` | Video assets by day/phase | Variable |

### Current `users` Table Schema

```sql
users (
  id UUID PRIMARY KEY (from auth.users),
  email TEXT NOT NULL,
  name TEXT,
  age INTEGER,
  subscription_tier TEXT DEFAULT 'free', -- 'free', 'annual', 'gift', 'enterprise'
  subscription_status TEXT DEFAULT 'inactive', -- 'active', 'inactive', 'cancelled', 'expired'
  subscription_started_at TIMESTAMPTZ,
  subscription_expires_at TIMESTAMPTZ,
  stripe_customer_id TEXT UNIQUE,
  current_day INTEGER DEFAULT 1,
  streak_days INTEGER DEFAULT 0,
  last_lesson_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
)
```

### Fields to ADD to `users` for Lifetime Tracking

```sql
-- Lifetime engagement metrics
first_lesson_at TIMESTAMPTZ,          -- When they completed first lesson
lifetime_lessons_completed INTEGER DEFAULT 0,
lifetime_contributions INTEGER DEFAULT 0,  -- Comments + artwork
lifetime_value_usd DECIMAL(10,2) DEFAULT 0,
longest_streak INTEGER DEFAULT 0,

-- Preferences
preferred_language VARCHAR(10) DEFAULT 'en',
timezone VARCHAR(50),

-- Acquisition
acquisition_source VARCHAR(100),      -- 'google', 'facebook', 'referral', etc.
acquisition_campaign VARCHAR(100),
referred_by_user_id UUID,
```

### Existing Event Tracking

`analytics_events` table exists but is basic:
- Only has `event_type` and `event_data` (JSONB)
- No immutability protection
- No Kelly → User tracking
- Not designed for audit trail

**Recommendation**: Create new `user_events` table alongside (don't replace)

---

## 2. Stripe Products Audit

### Current Stripe Price IDs (from environment)

| Plan | Price | Mode | Env Variable |
|------|-------|------|--------------|
| Monthly | $7.99/mo | subscription | `STRIPE_PRICE_MONTHLY` |
| Annual | $49.99/yr | subscription | `STRIPE_PRICE_ANNUAL` |
| Lifetime | $199.99 | payment | `STRIPE_PRICE_LIFETIME` |
| Family | $99.99/yr | payment | `STRIPE_PRICE_FAMILY` |
| Gift 3mo | $24.99 | payment | `STRIPE_PRICE_GIFT_3MO` |
| Gift 6mo | $39.99 | payment | `STRIPE_PRICE_GIFT_6MO` |
| Gift 12mo | $49.99 | payment | `STRIPE_PRICE_GIFT_12MO` |
| Gift Lifetime | $149.99 | payment | `STRIPE_PRICE_GIFT_LIFETIME` |

### Missing Stripe Products

| Product Needed | Price | Mode | Notes |
|----------------|-------|------|-------|
| Single Lesson | $1.99 | payment | For pay-per-lesson |
| Single Lesson (emerging) | $0.99 | payment | Regional pricing |
| Download Bundle | TBD | payment | For offline access |

**Action Required**: Create `STRIPE_PRICE_SINGLE_LESSON` product in Stripe Dashboard

---

## 3. Live Class Technology Research

### Requirements
- 24 classes per day (hourly)
- Today's lesson for everyone (free users included)
- 1000+ concurrent viewers per session
- Recording for replay
- Works on web, mobile, Roku

### Options Evaluated

| Platform | Pros | Cons | Cost |
|----------|------|------|------|
| **YouTube Live** | Free, scales infinitely, recording built-in | Less interactive, delay | Free |
| **Daily.co** | Real-time, good SDK | Expensive at scale | $0.004/min/participant |
| **Zoom SDK** | Familiar, reliable | Expensive, complex | Enterprise pricing |
| **Mux Live** | Developer-friendly, good quality | Monthly minimum | $50/mo + per-min |
| **Cloudflare Stream** | Low latency, global | New product | $5/1000 min viewed |

**Recommendation**: Start with **YouTube Live Premieres**
- Free
- Scales infinitely
- Built-in chat
- Auto-recording
- Can schedule 24 premieres for each hour
- Works on all platforms

**Phase 2**: Add interactive layer with Daily.co for premium subscribers

---

## 4. Offline Download Research

### Requirements
- Full 365-lesson bundle
- Works on iOS/Android/Desktop
- Syncs progress when online
- Verifies subscription periodically
- Prevents piracy

### Options Evaluated

| Approach | Pros | Cons |
|----------|------|------|
| **PWA (Progressive Web App)** | Cross-platform, web-native | Storage limits on iOS |
| **Native Apps (Capacitor)** | Full device access | More dev work |
| **Encrypted ZIP** | Simple | No progress sync |
| **SQLite DB** | Structured, queryable | Complex |

**Recommendation**: 
1. **Phase 1**: PWA with Service Worker caching (web)
2. **Phase 2**: Capacitor native apps with SQLite

**DRM Approach**:
- Encrypt video with user-specific key
- Key stored in Supabase, fetched on demand
- Key expires with subscription
- Re-verify subscription every 7 days

---

## 5. Comment System Design

### Moderation Strategy

| Approach | Pros | Cons |
|----------|------|------|
| Pre-moderation (all) | Safe | Doesn't scale |
| Post-moderation (flag-based) | Scales | Risk of bad content |
| AI-first + human review | Scales, catches most | Requires AI service |

**Recommendation**: **AI-first moderation**
- Use OpenAI moderation API for initial filter
- Auto-approve clean content
- Queue flagged content for human review
- Allow Kelly to respond (AI-generated)

### Threading

- Flat comments (no replies) for simplicity
- Phase 2: Single-level replies
- No deep threading (too complex)

---

## 6. Artwork Submission Design

### File Handling

| Aspect | Decision |
|--------|----------|
| Format | PNG, JPEG only |
| Max size | 5MB |
| Dimensions | 1920x1080 min, 4K max |
| Storage | Supabase Storage bucket `lesson-artwork` |
| Moderation | Manual review required |

### Licensing

- User grants perpetual license to use in lessons
- Credit always shown
- User retains copyright
- Can withdraw at any time (but existing uses continue)

---

## 7. Implementation SQL

### Migration: user_events (Zero-Trust Audit)

See: `supabase/migrations/025_user_events_audit_trail.sql`

### Migration: lesson_purchases (Pay-Per-Lesson)

See: `supabase/migrations/026_lesson_purchases.sql`

### Migration: Extend users table

See: `supabase/migrations/027_users_lifetime_tracking.sql`

---

## Next Steps

1. ✅ Float animation removed from Kelly
2. ✅ Research complete
3. 🔲 Run migration SQL in Supabase
4. 🔲 Create Stripe single-lesson product
5. 🔲 Build event logging API
6. 🔲 Wire up lesson completion events

---

*Research completed: December 15, 2025*
