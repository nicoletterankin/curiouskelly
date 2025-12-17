# Parent Companion System

## Executive Summary

The Parent Companion System extends Curious Kelly's learning beyond the child into the family. By providing daily summaries, conversation starters, and extension activities, we transform passive screen time into active family learning moments.

**Core Insight:** Parents want to be involved in their children's learning but often don't know what their child learned or how to engage with it. Kelly solves this by providing a daily "pulse" that makes family learning conversations effortless.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PARENT COMPANION FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Child completes lesson] ──────────────────────────────────────────►   │
│          │                                                              │
│          ▼                                                              │
│  ┌───────────────────┐                                                  │
│  │ Generate Parent   │                                                  │
│  │ Pulse content     │                                                  │
│  │ from lesson DNA   │                                                  │
│  └─────────┬─────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│  ┌───────────────────┐     ┌───────────────────┐                        │
│  │ Queue for         │────►│ Deliver at        │                        │
│  │ scheduled time    │     │ preferred time    │                        │
│  └───────────────────┘     │ (e.g., 6 PM)      │                        │
│                            └─────────┬─────────┘                        │
│                                      │                                  │
│                    ┌─────────────────┼─────────────────┐                │
│                    ▼                 ▼                 ▼                │
│              ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│              │  Email   │     │   Push   │     │   SMS    │             │
│              │ (Primary)│     │ (Quick)  │     │ (Future) │             │
│              └──────────┘     └──────────┘     └──────────┘             │
│                                                                         │
│  Parent reads pulse ───► Family conversation at dinner ───► Deeper     │
│                                                              Learning   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Database Schema

```sql
-- Parent companion subscriptions
CREATE TABLE parent_companions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    child_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Parent contact
    parent_email VARCHAR(255) NOT NULL,
    parent_name VARCHAR(100),
    
    -- Delivery preferences
    delivery_preference VARCHAR(20) DEFAULT 'email' 
        CHECK (delivery_preference IN ('email', 'push', 'both', 'none')),
    delivery_time TIME DEFAULT '18:00',
    timezone VARCHAR(50) DEFAULT 'America/Los_Angeles',
    
    -- Content preferences
    extension_level VARCHAR(20) DEFAULT 'full'
        CHECK (extension_level IN ('minimal', 'full', 'deep')),
    
    -- Component toggles (JSONB for flexibility)
    components JSONB DEFAULT '{
        "summary": true,
        "conversationStarter": true,
        "extension": true,
        "bookRecommendation": true,
        "streakUpdate": true
    }',
    
    -- Metadata
    verified_at TIMESTAMP,  -- Email verification
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    unsubscribed_at TIMESTAMP
);

-- Delivery tracking
CREATE TABLE parent_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    parent_companion_id UUID NOT NULL REFERENCES parent_companions(id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL REFERENCES core_lessons(id),
    child_user_id UUID NOT NULL REFERENCES users(id),
    
    -- Delivery data
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('email', 'push', 'sms')),
    content_sent JSONB NOT NULL,  -- Snapshot of what was sent
    
    -- Tracking
    scheduled_at TIMESTAMP NOT NULL,
    delivered_at TIMESTAMP,
    opened_at TIMESTAMP,
    clicked_at TIMESTAMP,
    
    -- Engagement
    clicked_extension BOOLEAN DEFAULT FALSE,
    clicked_book BOOLEAN DEFAULT FALSE,
    conversation_reported BOOLEAN DEFAULT FALSE,  -- Future: parent confirms they talked
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'sent', 'delivered', 'opened', 'bounced', 'failed'))
);

-- Indexes
CREATE INDEX idx_parent_companions_child ON parent_companions(child_user_id);
CREATE INDEX idx_parent_companions_email ON parent_companions(parent_email);
CREATE INDEX idx_parent_deliveries_scheduled ON parent_deliveries(scheduled_at) WHERE status = 'pending';
CREATE INDEX idx_parent_deliveries_parent ON parent_deliveries(parent_companion_id);
```

### TypeScript Types

```typescript
// types/parent-companion.ts

interface ParentCompanion {
  id: string;
  childUserId: string;
  parentEmail: string;
  parentName?: string;
  deliveryPreference: 'email' | 'push' | 'both' | 'none';
  deliveryTime: string;  // HH:MM format
  timezone: string;
  extensionLevel: 'minimal' | 'full' | 'deep';
  components: {
    summary: boolean;
    conversationStarter: boolean;
    extension: boolean;
    bookRecommendation: boolean;
    streakUpdate: boolean;
  };
  verifiedAt?: Date;
  createdAt: Date;
  unsubscribedAt?: Date;
}

interface ParentPulseContent {
  childName: string;
  lessonDate: Date;
  
  // Core content (from lesson DNA)
  topic: string;
  summary: string;
  conversationStarter: string;
  
  // Extension (optional based on preferences)
  extension?: {
    type: 'observation' | 'experiment' | 'discussion' | 'creation' | 'exploration';
    instruction: string;
    question: string;
    scienceNote?: string;
  };
  
  // Book recommendation (optional)
  bookRecommendation?: {
    title: string;
    author: string;
    ageRange: string;
    isbn?: string;
  };
  
  // Progress
  streakDays: number;
  totalLessons: number;
}

interface ParentDelivery {
  id: string;
  parentCompanionId: string;
  lessonId: string;
  channel: 'email' | 'push' | 'sms';
  contentSent: ParentPulseContent;
  scheduledAt: Date;
  deliveredAt?: Date;
  openedAt?: Date;
  status: 'pending' | 'sent' | 'delivered' | 'opened' | 'bounced' | 'failed';
}
```

---

## API Endpoints

### Parent Subscription Management

```typescript
// api/parent-pulse/subscribe.ts
// POST /api/parent-pulse/subscribe
// Creates or updates parent companion subscription

interface SubscribeRequest {
  childUserId: string;
  parentEmail: string;
  parentName?: string;
  deliveryPreference?: 'email' | 'push' | 'both';
  deliveryTime?: string;
  timezone?: string;
}

interface SubscribeResponse {
  success: boolean;
  companionId: string;
  verificationRequired: boolean;
  message: string;
}

// Implementation notes:
// 1. Validate child user exists
// 2. Check parent email isn't already subscribed to this child
// 3. Send verification email
// 4. Create record in parent_companions with verified_at = null
// 5. Return success
```

```typescript
// api/parent-pulse/verify.ts
// GET /api/parent-pulse/verify?token=xxx
// Verifies parent email address

// Token contains: companionId + parentEmail + timestamp
// Validates token signature and expiry
// Sets verified_at timestamp
// Redirects to success page
```

```typescript
// api/parent-pulse/preferences.ts
// GET /api/parent-pulse/preferences?companionId=xxx
// PUT /api/parent-pulse/preferences

interface PreferencesRequest {
  companionId: string;
  deliveryPreference?: 'email' | 'push' | 'both' | 'none';
  deliveryTime?: string;
  timezone?: string;
  extensionLevel?: 'minimal' | 'full' | 'deep';
  components?: Partial<ParentCompanion['components']>;
}
```

```typescript
// api/parent-pulse/unsubscribe.ts
// POST /api/parent-pulse/unsubscribe
// or GET with token for one-click unsubscribe

// Sets unsubscribed_at timestamp
// Keeps record for analytics
```

### Content Delivery

```typescript
// api/parent-pulse/today.ts
// GET /api/parent-pulse/today?companionId=xxx
// Returns today's pulse content (useful for web preview)

// Returns ParentPulseContent
```

```typescript
// api/parent-pulse/history.ts
// GET /api/parent-pulse/history?companionId=xxx&range=7d
// Returns past pulses for dashboard view

interface HistoryResponse {
  pulses: Array<{
    date: Date;
    topic: string;
    summary: string;
    opened: boolean;
  }>;
  totalCount: number;
}
```

---

## Email Templates

### Daily Pulse Email

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Today's Learning Pulse</title>
  <style>
    /* Inline styles for email compatibility */
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: #f5f5f5;
      margin: 0;
      padding: 20px;
    }
    .container {
      max-width: 500px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 16px;
      overflow: hidden;
    }
    .header {
      background: linear-gradient(135deg, #0f0f11 0%, #1a1a2e 100%);
      color: white;
      padding: 24px;
      text-align: center;
    }
    .header h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 500;
    }
    .header .date {
      font-size: 14px;
      opacity: 0.8;
      margin-top: 4px;
    }
    .section {
      padding: 20px 24px;
      border-bottom: 1px solid #eee;
    }
    .section-title {
      font-size: 12px;
      text-transform: uppercase;
      color: #888;
      margin: 0 0 12px 0;
      letter-spacing: 0.5px;
    }
    .topic {
      font-size: 24px;
      font-weight: 600;
      color: #0f0f11;
      margin: 0 0 8px 0;
    }
    .summary {
      font-size: 16px;
      color: #444;
      line-height: 1.5;
      margin: 0;
    }
    .conversation {
      background: #fef7f4;
      border-left: 4px solid #d97757;
      padding: 16px;
      border-radius: 0 8px 8px 0;
      font-size: 16px;
      color: #333;
      font-style: italic;
    }
    .extension {
      background: #f7f9fc;
      border-radius: 12px;
      padding: 16px;
    }
    .extension-instruction {
      font-size: 15px;
      color: #333;
      margin: 0 0 12px 0;
    }
    .extension-question {
      font-weight: 500;
      color: #0f0f11;
      margin: 0 0 8px 0;
    }
    .science-note {
      font-size: 13px;
      color: #666;
      background: white;
      padding: 8px 12px;
      border-radius: 6px;
      margin-top: 12px;
    }
    .book {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .book-cover {
      width: 60px;
      height: 80px;
      background: #ddd;
      border-radius: 4px;
    }
    .book-info h4 {
      margin: 0;
      font-size: 16px;
      color: #0f0f11;
    }
    .book-info p {
      margin: 4px 0 0 0;
      font-size: 14px;
      color: #666;
    }
    .streak {
      text-align: center;
      padding: 16px 24px;
    }
    .streak-badge {
      display: inline-block;
      background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
      color: white;
      padding: 8px 20px;
      border-radius: 20px;
      font-weight: 600;
    }
    .footer {
      text-align: center;
      padding: 20px;
      font-size: 12px;
      color: #888;
    }
    .footer a {
      color: #d97757;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <h1>✨ Today's Learning Pulse</h1>
      <div class="date">{{date}}</div>
    </div>
    
    <!-- What They Learned -->
    <div class="section">
      <p class="section-title">📚 {{childName}} learned about</p>
      <h2 class="topic">{{topic}}</h2>
      <p class="summary">{{summary}}</p>
    </div>
    
    <!-- Conversation Starter -->
    <div class="section">
      <p class="section-title">🗣️ Dinner Conversation Starter</p>
      <div class="conversation">
        "{{conversationStarter}}"
      </div>
    </div>
    
    {{#if extension}}
    <!-- Extension Activity -->
    <div class="section">
      <p class="section-title">🔬 Optional Extension</p>
      <div class="extension">
        <p class="extension-instruction">{{extension.instruction}}</p>
        <p class="extension-question">Ask: "{{extension.question}}"</p>
        {{#if extension.scienceNote}}
        <div class="science-note">
          💡 {{extension.scienceNote}}
        </div>
        {{/if}}
      </div>
    </div>
    {{/if}}
    
    {{#if bookRecommendation}}
    <!-- Book Recommendation -->
    <div class="section">
      <p class="section-title">📖 Book Recommendation</p>
      <div class="book">
        <div class="book-cover"></div>
        <div class="book-info">
          <h4>{{bookRecommendation.title}}</h4>
          <p>by {{bookRecommendation.author}}</p>
          <p>{{bookRecommendation.ageRange}} • Available at your library</p>
        </div>
      </div>
    </div>
    {{/if}}
    
    <!-- Streak -->
    <div class="streak">
      <span class="streak-badge">🔥 {{streakDays}} day streak!</span>
    </div>
    
    <!-- Footer -->
    <div class="footer">
      <p>
        <a href="{{preferencesUrl}}">Update preferences</a> · 
        <a href="{{unsubscribeUrl}}">Unsubscribe</a>
      </p>
      <p>Curious Kelly by Lesson of the Day PBC</p>
      <p>hello@curiouskelly.com</p>
    </div>
  </div>
</body>
</html>
```

### Push Notification Template

```json
{
  "title": "{{childName}} learned about {{topic}} today!",
  "body": "Dinner topic: \"{{conversationStarter}}\"",
  "icon": "/icons/kelly-avatar-small.png",
  "badge": "/icons/badge.png",
  "data": {
    "type": "parent-pulse",
    "lessonId": "{{lessonId}}",
    "companionId": "{{companionId}}"
  },
  "actions": [
    {
      "action": "view",
      "title": "View Details"
    },
    {
      "action": "dismiss",
      "title": "Dismiss"
    }
  ]
}
```

---

## Background Jobs

### Daily Pulse Scheduler

```typescript
// jobs/send-parent-pulses.ts
// Runs every minute to check for scheduled deliveries

async function sendScheduledPulses() {
  const now = new Date();
  
  // Find all companions whose delivery time has passed today
  // and who haven't received today's pulse yet
  const companions = await db.query(`
    SELECT pc.*, u.display_name as child_name, u.total_lessons_completed
    FROM parent_companions pc
    JOIN users u ON u.id = pc.child_user_id
    WHERE pc.delivery_preference IN ('email', 'both')
      AND pc.verified_at IS NOT NULL
      AND pc.unsubscribed_at IS NULL
      AND pc.delivery_time <= $1::time
      AND NOT EXISTS (
        SELECT 1 FROM parent_deliveries pd
        WHERE pd.parent_companion_id = pc.id
          AND DATE(pd.scheduled_at AT TIME ZONE pc.timezone) = CURRENT_DATE
      )
  `, [format(now, 'HH:mm')]);
  
  for (const companion of companions) {
    // Get today's completed lesson for the child
    const lesson = await getCompletedLessonForToday(companion.child_user_id);
    if (!lesson) continue;  // Child hasn't completed today's lesson
    
    // Generate pulse content
    const content = await generatePulseContent(companion, lesson);
    
    // Send email
    await sendPulseEmail(companion, content);
    
    // Record delivery
    await db.insert('parent_deliveries', {
      parent_companion_id: companion.id,
      lesson_id: lesson.id,
      child_user_id: companion.child_user_id,
      channel: 'email',
      content_sent: content,
      scheduled_at: now,
      delivered_at: now,
      status: 'sent'
    });
  }
}
```

---

## Content Generation

### From Lesson DNA to Pulse

```typescript
// lib/generate-pulse-content.ts

async function generatePulseContent(
  companion: ParentCompanion,
  lesson: Lesson
): Promise<ParentPulseContent> {
  const childUser = await getUser(companion.childUserId);
  const lessonDNA = await getLessonDNA(lesson.id);
  
  // Base content from lesson DNA
  const pulse: ParentPulseContent = {
    childName: childUser.displayName || 'Your child',
    lessonDate: new Date(),
    topic: lessonDNA.meta.topic,
    summary: lessonDNA.parentCompanion.summary,
    conversationStarter: lessonDNA.parentCompanion.conversationStarter,
    streakDays: childUser.currentStreak || 0,
    totalLessons: childUser.totalLessonsCompleted || 0
  };
  
  // Extension activity (if enabled)
  if (companion.components.extension && lessonDNA.parentCompanion.extensionActivity) {
    pulse.extension = lessonDNA.parentCompanion.extensionActivity;
  }
  
  // Book recommendation (if enabled and available)
  if (companion.components.bookRecommendation && lessonDNA.parentCompanion.bookRecommendations?.length) {
    pulse.bookRecommendation = lessonDNA.parentCompanion.bookRecommendations[0];
  }
  
  return pulse;
}
```

---

## User Flows

### 1. Parent Subscription Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Child completes first lesson                                        │
│  2. App shows "Share with a parent?" prompt                             │
│  3. Child enters parent's email (or selects from contacts)              │
│  4. System sends verification email to parent                           │
│  5. Parent clicks verify link                                           │
│  6. Parent lands on preferences page, can customize                     │
│  7. Parent starts receiving daily pulses                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Daily Delivery Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Child completes daily lesson (any time)                             │
│  2. System records completion                                           │
│  3. At parent's preferred time (e.g., 6 PM):                            │
│     - Scheduler checks for completed lessons                            │
│     - Generates pulse content from lesson DNA                           │
│     - Sends email (and/or push)                                         │
│  4. Parent opens email at dinner                                        │
│  5. Family has learning conversation                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Preference Management Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Parent clicks "Update preferences" link in email footer             │
│  2. Lands on preferences page (authenticated via token)                 │
│  3. Can adjust:                                                         │
│     - Delivery time                                                     │
│     - Content level (minimal/full/deep)                                 │
│     - Component toggles                                                 │
│     - Pause or unsubscribe                                              │
│  4. Changes take effect next day                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Metrics & Success Criteria

### Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Subscription Rate | 30% of child accounts | Parents subscribed / total children |
| Open Rate | 50%+ | Email opens / emails sent |
| Conversation Rate | TBD | Survey / self-report |

### Secondary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Click-Through Rate | 20%+ | Clicks / opens |
| Extension Engagement | 15%+ | Extension clicks / opens |
| Unsubscribe Rate | <2%/month | Unsubscribes / active subscribers |
| Verification Rate | 60%+ | Verified / sent verification |

### Engagement Signal

We'll ask parents (monthly survey in email):
- "Did you discuss today's topic with your child?" (Y/N)
- "How many times did you use conversation starters this week?" (0-7)
- "Has Curious Kelly improved family learning conversations?" (1-5 scale)

---

## Security & Privacy

### Data Protection
- Parent emails stored encrypted at rest
- Verification tokens expire in 24 hours
- Unsubscribe tokens are single-use
- No child data shared with parents beyond lesson topics

### Consent Model
- Child must initiate parent subscription (not reverse)
- Parent must verify email before receiving content
- Either party can unsubscribe at any time
- Clear data retention policy (delete after 12 months inactive)

### Compliance
- COPPA: Parental involvement, not data collection
- GDPR: Clear consent, easy unsubscribe, data portability
- CAN-SPAM: Physical address in footer, working unsubscribe

---

## Implementation Phases

### Phase 1: MVP (Week 1-2)
- [x] Database schema
- [ ] Subscribe/verify/unsubscribe APIs
- [ ] Daily pulse email template
- [ ] Background scheduler (Vercel Cron or Railway)
- [ ] Basic preferences page

### Phase 2: Enhancement (Week 3-4)
- [ ] Push notifications (web push)
- [ ] Dashboard view for parents
- [ ] History of past pulses
- [ ] Book recommendation integration (OpenLibrary API)

### Phase 3: Analytics (Week 5-6)
- [ ] Open/click tracking
- [ ] Engagement surveys
- [ ] A/B testing subject lines
- [ ] Conversion attribution (pulse → family subscription)

---

## Future Ideas

1. **Multi-Child Support:** One parent receives pulses for multiple children in a single email
2. **Teacher Mode:** Classroom version with weekly digest for educators
3. **Family Progress Dashboard:** Web view showing family learning stats
4. **Voice Summary:** Alexa/Google skill that reads today's pulse
5. **Weekend Recap:** Saturday email summarizing the week's learning

---

*Document created: December 16, 2025*
*Contact: hello@curiouskelly.com*
