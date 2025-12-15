# Learner Lifecycle Architecture

## The Vision

> **Keep learners for life. Track everything. Zero trust.**

Every interaction a learner has with Kelly—and Kelly has with them—is recorded, auditable, and valuable. This isn't surveillance; it's **relationship memory**. Kelly remembers you because Kelly cares.

---

## Core Principles

### 1. Zero Trust Audit Trail
Every event is:
- **Immutable** - Once recorded, cannot be altered
- **Timestamped** - UTC with millisecond precision
- **Attributable** - Linked to user_id and session_id
- **Queryable** - Can pull complete user history instantly

### 2. Lifetime Value Focus
We're not optimizing for:
- ❌ Daily active users
- ❌ Session length
- ❌ Engagement metrics

We're optimizing for:
- ✅ **Years of learning**
- ✅ **Depth of relationship**
- ✅ **Contribution to community**

### 3. Bidirectional Tracking
Track both:
- **Learner → Kelly**: What did the learner do?
- **Kelly → Learner**: What did Kelly do for them?

---

## Feature Inventory

### 🆓 FREE TIER
| Feature | Status | Notes |
|---------|--------|-------|
| Today's lesson (always free) | ✅ Implemented | Calendar-based |
| Live class seat (today's lesson) | 🔲 NEW | Every hour on the hour |

### 💳 PAID FEATURES
| Feature | Status | Notes |
|---------|--------|-------|
| Pay-per-lesson | 🔲 Designed | $1.99 single purchase |
| All-access subscription | ✅ Stripe ready | Monthly/Annual/Lifetime |
| Download all lessons | 🔲 NEW | Offline bundle |
| Calendar navigation | ✅ Implemented | Past/future lessons |
| Live class priority seating | 🔲 NEW | Premium gets front row? |

### 🎨 CONTRIBUTION FEATURES
| Feature | Status | Notes |
|---------|--------|-------|
| Comment on lessons | 🔲 NEW | Text comments |
| Submit lesson artwork | 🔲 NEW | User-generated visuals |
| Rate/react to lessons | 🔲 NEW | Simple feedback |
| Share lessons | ✅ Exists | Social sharing |

### 📊 TRACKING FEATURES
| Feature | Status | Notes |
|---------|--------|-------|
| Lesson completion events | 🔲 NEW | user_id + day_number + timestamp |
| Comment events | 🔲 NEW | Full text + moderation status |
| Purchase events | ✅ Exists | revenue_events table |
| Artwork submission events | 🔲 NEW | Asset URL + approval status |
| Support interaction events | 🔲 NEW | Tickets, emails, calls |
| Kelly outreach events | 🔲 NEW | Emails/push sent TO user |
| Live class attendance | 🔲 NEW | Joined, duration, participation |

---

## Database Schema: User Lifecycle

### Core User Table (Extended)

```sql
-- Extend existing users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS
  created_at TIMESTAMPTZ DEFAULT NOW(),
  first_lesson_at TIMESTAMPTZ,
  last_active_at TIMESTAMPTZ,
  lifetime_lessons_completed INTEGER DEFAULT 0,
  lifetime_contributions INTEGER DEFAULT 0,
  lifetime_value_usd DECIMAL(10,2) DEFAULT 0,
  preferred_language VARCHAR(10) DEFAULT 'en',
  timezone VARCHAR(50),
  acquisition_source VARCHAR(100),
  acquisition_campaign VARCHAR(100);
```

### Universal Event Log (Zero Trust Audit)

```sql
-- IMMUTABLE audit log - the single source of truth
CREATE TABLE user_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Who
  user_id UUID NOT NULL REFERENCES users(id),
  session_id UUID,  -- Track across session
  
  -- What
  event_type VARCHAR(50) NOT NULL,  -- See event types below
  event_category VARCHAR(30) NOT NULL,  -- 'learner_action', 'kelly_action', 'system'
  
  -- Details
  payload JSONB NOT NULL DEFAULT '{}',  -- Flexible event-specific data
  
  -- When
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  
  -- Context
  ip_address INET,
  user_agent TEXT,
  device_type VARCHAR(20),  -- 'mobile', 'tablet', 'desktop'
  platform VARCHAR(20),  -- 'web', 'ios', 'android', 'roku'
  
  -- Indexing
  day_number INTEGER,  -- For lesson-related events
  
  -- Immutability marker
  checksum VARCHAR(64)  -- SHA-256 of row for tamper detection
);

-- Indexes for common queries
CREATE INDEX idx_user_events_user_id ON user_events(user_id);
CREATE INDEX idx_user_events_type ON user_events(event_type);
CREATE INDEX idx_user_events_created ON user_events(created_at);
CREATE INDEX idx_user_events_day ON user_events(day_number) WHERE day_number IS NOT NULL;

-- Prevent updates/deletes (audit trail is immutable)
CREATE OR REPLACE FUNCTION prevent_event_modification()
RETURNS TRIGGER AS $$
BEGIN
  RAISE EXCEPTION 'user_events table is immutable - modifications not allowed';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_update_events
  BEFORE UPDATE OR DELETE ON user_events
  FOR EACH ROW EXECUTE FUNCTION prevent_event_modification();
```

### Event Types

```
LEARNER → KELLY:
  lesson.started          - Started watching a lesson
  lesson.completed        - Finished entire lesson
  lesson.phase_completed  - Completed one phase
  lesson.skipped          - Skipped a phase
  lesson.paused           - Paused playback
  lesson.resumed          - Resumed playback
  
  comment.posted          - Posted a comment
  comment.edited          - Edited own comment
  comment.deleted         - Deleted own comment
  
  artwork.submitted       - Submitted lesson artwork
  artwork.withdrawn       - Withdrew submission
  
  reaction.added          - Added emoji reaction
  reaction.removed        - Removed reaction
  
  purchase.initiated      - Started checkout
  purchase.completed      - Payment successful
  purchase.failed         - Payment failed
  purchase.refunded       - Refund processed
  
  subscription.started    - New subscription
  subscription.renewed    - Auto-renewed
  subscription.cancelled  - Cancelled (end of period)
  subscription.paused     - Paused subscription
  
  download.requested      - Requested lesson download
  download.completed      - Download finished
  download.bundle         - Downloaded full bundle
  
  liveclass.joined        - Joined live class
  liveclass.left          - Left live class
  liveclass.question      - Asked question in class
  liveclass.answer        - Answered question
  
  support.ticket_opened   - Opened support ticket
  support.message_sent    - Sent support message
  
  settings.updated        - Changed settings
  profile.updated         - Updated profile
  
KELLY → LEARNER:
  kelly.email_sent        - Sent email to user
  kelly.push_sent         - Sent push notification
  kelly.sms_sent          - Sent SMS
  kelly.reminder_sent     - Sent lesson reminder
  
  kelly.streak_celebrated - Celebrated streak milestone
  kelly.welcome_sent      - Sent welcome message
  kelly.comeback_sent     - Re-engagement outreach
  
  kelly.gift_delivered    - Delivered gift subscription
  kelly.birthday_message  - Birthday greeting
  
  moderation.comment_approved  - Approved user comment
  moderation.comment_rejected  - Rejected comment
  moderation.artwork_approved  - Approved artwork
  moderation.artwork_rejected  - Rejected artwork
  
SYSTEM:
  system.session_started  - New session began
  system.session_ended    - Session ended
  system.error            - Error occurred
  system.migration        - Data migration event
```

### Lesson Comments

```sql
CREATE TABLE lesson_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  day_number INTEGER NOT NULL,
  
  -- Content
  content TEXT NOT NULL,
  content_html TEXT,  -- Rendered HTML (sanitized)
  
  -- Threading
  parent_comment_id UUID REFERENCES lesson_comments(id),
  thread_depth INTEGER DEFAULT 0,
  
  -- Moderation
  status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'approved', 'rejected', 'flagged'
  moderated_by UUID REFERENCES users(id),
  moderated_at TIMESTAMPTZ,
  rejection_reason TEXT,
  
  -- Metrics
  upvotes INTEGER DEFAULT 0,
  reports INTEGER DEFAULT 0,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  edited_at TIMESTAMPTZ,
  
  -- Soft delete
  deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_comments_day ON lesson_comments(day_number) WHERE deleted_at IS NULL;
CREATE INDEX idx_comments_user ON lesson_comments(user_id);
CREATE INDEX idx_comments_status ON lesson_comments(status);
```

### User-Contributed Artwork

```sql
CREATE TABLE lesson_artwork_submissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  day_number INTEGER NOT NULL,
  
  -- Asset
  image_url TEXT NOT NULL,
  thumbnail_url TEXT,
  original_filename TEXT,
  file_size_bytes INTEGER,
  dimensions VARCHAR(20),  -- '1920x1080'
  
  -- Metadata
  title VARCHAR(200),
  description TEXT,
  ai_generated BOOLEAN DEFAULT FALSE,
  tools_used TEXT[],  -- ['photoshop', 'midjourney', etc.]
  
  -- Moderation
  status VARCHAR(20) DEFAULT 'pending',
  moderated_by UUID REFERENCES users(id),
  moderated_at TIMESTAMPTZ,
  rejection_reason TEXT,
  
  -- Usage tracking
  times_displayed INTEGER DEFAULT 0,
  selected_as_official BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_artwork_day ON lesson_artwork_submissions(day_number);
CREATE INDEX idx_artwork_user ON lesson_artwork_submissions(user_id);
CREATE INDEX idx_artwork_status ON lesson_artwork_submissions(status);
```

### Live Classes

```sql
CREATE TABLE live_class_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Schedule
  scheduled_at TIMESTAMPTZ NOT NULL,  -- Top of hour
  day_number INTEGER NOT NULL,  -- Which lesson
  
  -- Session info
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  status VARCHAR(20) DEFAULT 'scheduled',  -- 'scheduled', 'live', 'completed', 'cancelled'
  
  -- Capacity
  max_attendees INTEGER DEFAULT 1000,
  actual_attendees INTEGER DEFAULT 0,
  peak_concurrent INTEGER DEFAULT 0,
  
  -- Recording
  recording_url TEXT,
  transcript_url TEXT
);

CREATE TABLE live_class_attendance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES live_class_sessions(id),
  user_id UUID REFERENCES users(id),
  
  -- Timing
  joined_at TIMESTAMPTZ NOT NULL,
  left_at TIMESTAMPTZ,
  duration_seconds INTEGER,
  
  -- Participation
  questions_asked INTEGER DEFAULT 0,
  reactions_sent INTEGER DEFAULT 0,
  
  -- Access type
  access_type VARCHAR(20),  -- 'free_today', 'subscriber', 'purchased'
  
  UNIQUE(session_id, user_id)
);
```

### Lesson Downloads (Offline Access)

```sql
CREATE TABLE lesson_downloads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  
  -- What was downloaded
  download_type VARCHAR(20) NOT NULL,  -- 'single', 'bundle', 'update'
  day_numbers INTEGER[],  -- Array of days included
  
  -- Bundle info
  bundle_version VARCHAR(20),
  file_size_bytes BIGINT,
  
  -- Status
  requested_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,  -- Downloads expire?
  
  -- Access verification
  access_verified BOOLEAN DEFAULT FALSE,  -- Confirmed paid access at download time
  
  -- Device
  device_id VARCHAR(100),
  platform VARCHAR(20)
);
```

---

## API Endpoints Needed

### Event Logging
```
POST /api/events
  - Log any user event
  - Auto-adds session, device, timestamp

GET /api/users/:id/events
  - Full event history for a user
  - Filterable by type, date range
  - Admin only
```

### Comments
```
POST /api/lessons/:day/comments
GET /api/lessons/:day/comments
PATCH /api/comments/:id
DELETE /api/comments/:id
POST /api/comments/:id/report
```

### Artwork
```
POST /api/lessons/:day/artwork
GET /api/lessons/:day/artwork
GET /api/artwork/pending  (admin)
POST /api/artwork/:id/moderate
```

### Downloads
```
POST /api/downloads/bundle
GET /api/downloads/status/:id
GET /api/downloads/my-downloads
```

### Live Classes
```
GET /api/live/next
GET /api/live/schedule
POST /api/live/:session/join
POST /api/live/:session/leave
```

---

## Admin Audit View

When you pull a user_id, you should see:

```
┌─────────────────────────────────────────────────────────────┐
│ USER AUDIT: kelly_learner_42                                │
│ ID: 550e8400-e29b-41d4-a716-446655440000                    │
├─────────────────────────────────────────────────────────────┤
│ LIFETIME STATS                                              │
│ ───────────────────────                                     │
│ Member since: March 15, 2025 (276 days)                     │
│ Lessons completed: 187                                      │
│ Current streak: 23 days                                     │
│ Longest streak: 45 days                                     │
│ Lifetime value: $47.88 (4 months subscription)              │
│ Comments posted: 12                                         │
│ Artwork submitted: 3 (2 approved)                           │
│ Live classes attended: 8                                    │
├─────────────────────────────────────────────────────────────┤
│ KELLY'S OUTREACH                                            │
│ ───────────────────────                                     │
│ Welcome email: March 15, 2025                               │
│ 7-day streak celebration: March 22, 2025                    │
│ Re-engagement (after 5 day gap): June 3, 2025               │
│ Birthday message: August 12, 2025                           │
│ Last push notification: Today, 8:00 AM                      │
├─────────────────────────────────────────────────────────────┤
│ RECENT EVENTS (last 10)                                     │
│ ───────────────────────                                     │
│ 2025-12-15 09:32:14  lesson.completed    Day 349            │
│ 2025-12-15 09:28:01  lesson.started      Day 349            │
│ 2025-12-14 10:15:33  comment.posted      Day 348            │
│ 2025-12-14 10:02:45  lesson.completed    Day 348            │
│ ...                                                         │
├─────────────────────────────────────────────────────────────┤
│ [View Full History] [Export JSON] [Support Ticket]          │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (NOW)
- [ ] Create `user_events` table with immutability trigger
- [ ] Create event logging API endpoint
- [ ] Wire up basic events (lesson.started, lesson.completed)
- [ ] Extend users table with lifetime fields

### Phase 2: Comments & Contributions (Week 1)
- [ ] Create `lesson_comments` table
- [ ] Build comment posting UI
- [ ] Create moderation queue
- [ ] Wire up comment events

### Phase 3: Live Classes (Week 2)
- [ ] Create live class tables
- [ ] Build scheduling system
- [ ] Build join/leave flow
- [ ] Track attendance

### Phase 4: Downloads (Week 3)
- [ ] Create download tracking table
- [ ] Build bundle generation
- [ ] Verify access on download
- [ ] Track offline usage

### Phase 5: Artwork Contributions (Week 4)
- [ ] Create artwork submission table
- [ ] Build upload UI
- [ ] Create moderation flow
- [ ] Display user artwork in lessons

### Phase 6: Admin Dashboard (Week 5)
- [ ] Build user audit view
- [ ] Create event search/filter
- [ ] Export functionality
- [ ] Anomaly detection

---

## Privacy & Compliance

- All tracking disclosed in Terms of Service
- User can request full data export (GDPR Article 15)
- User can request deletion (GDPR Article 17)
- Data retained for 7 years (business records requirement)
- No data sold to third parties (ever)

---

## Next Steps

1. Create the core `user_events` table in Supabase
2. Build the event logging API
3. Wire up lesson completion tracking
4. Build admin audit view

---

*This document is the source of truth for the Learner Lifecycle system.*
*Updated: December 15, 2025*
