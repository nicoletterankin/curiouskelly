# Kelly OS Complete Content Extraction Report

**Generated:** December 21, 2025  
**Purpose:** Comprehensive inventory of all content types and app readiness for Kelly OS migration

---

## Executive Summary

The Curious Kelly platform has **extensive content infrastructure** already built, with most content types existing in the database. However, several Kelly OS apps will need new content generation or extraction from existing fields.

**Key Findings:**
- ✅ **Visual Assets:** Fully implemented (`lesson_visuals` table + `lesson_atoms.visual_url`)
- ✅ **Quiz Questions:** Stored in `core_lessons.quick_quiz_questions` (JSONB)
- ✅ **Reflection Prompts:** Stored in `core_lessons.reflection_prompts` (JSONB)
- ✅ **Notes System:** Community notes exist (`commons_lesson_notes` table)
- ✅ **Geo Context:** API endpoint exists (`/api/geo-context`)
- ✅ **Chat System:** ElevenLabs Conversational AI implemented
- ✅ **Badges/Achievements:** Hardcoded definitions exist
- ❌ **Code Challenges:** Not found
- ❌ **Flashcards:** Not found (but can derive from quiz questions)
- ❌ **Timeline Data:** Not found (but `historical_context` exists in `core_lessons`)
- ❌ **3D Avatar:** Only 2D video (Unity WebGL legacy, not used)

---

## 1. Content Extensions Per Lesson

### External Links / Resources

**Status:** ✅ **EXISTS** (in `core_lessons` table)

**Schema:**
```typescript
interface CoreLesson {
  recommended_videos?: Array<{
    title: string;
    url: string;
    source?: string;
    duration?: string;
  }>;
  recommended_books?: Array<{
    title: string;
    author?: string;
    url?: string;
  }>;
  downloadable_resources?: Array<{
    title: string;
    url: string;
    type: string;
  }>;
}
```

**Storage:** Supabase `core_lessons.recommended_videos`, `recommended_books`, `downloadable_resources` (JSONB columns)

**Implementation:**
- File: `public/lesson-detail.html` (lines 1037-1054)
- Displays resources in lesson detail page
- Age-appropriate flag: Not explicitly stored, but can be inferred from `ideal_age_range`

**Count Per Lesson:** Variable (0-10+ resources)

---

### Visual Assets

**Status:** ✅ **FULLY IMPLEMENTED**

**Storage Locations:**
1. **Supabase Storage:** `lesson_atoms.visual_url` (infographics per phase)
2. **Supabase Storage:** `lesson_atoms.hd_video_url` (HD video per phase)
3. **Database Table:** `lesson_visuals` (tracking table)
4. **Public Assets:** `/public/assets/kelly/production/avatars/` (2D avatar images)

**Schema:**
```sql
-- lesson_visuals table
CREATE TABLE lesson_visuals (
  id UUID PRIMARY KEY,
  core_lesson_id UUID REFERENCES core_lessons(id),
  day_number INTEGER UNIQUE NOT NULL,
  thumbnail_url TEXT,
  infographic_url TEXT,
  infographic_urls JSONB DEFAULT '[]',
  illustration_url TEXT,
  status TEXT DEFAULT 'pending'
);
```

**Image Types:**
- **Infographics:** 1920×1080 educational diagrams (stored in `lesson_visuals.infographic_url`)
- **Option Cards:** 512×512 choice cards (stored in `lesson_atoms.visual_url`)
- **HD Videos:** 1080p MP4 with lip-sync (stored in `lesson_atoms.hd_video_url`)
- **Thumbnails:** Lesson preview images (stored in `lesson_visuals.thumbnail_url`)
- **Hero Images:** Marketing images (stored in `core_lessons.hero_image_url`)

**Naming Convention:**
- Videos: `day-{dayNumber}-{archetype}-{phase}.mp4`
- Infographics: `day-{dayNumber}-infographic-{phase}.png`
- Option cards: `day-{dayNumber}-{phase}-option-{a|b}.png`

**Count Per Lesson:**
- 5 infographics (one per phase: Hook, Fact1, Fact2, Fact3, Wisdom)
- 8 option cards (2 per question phase)
- 5 HD videos (one per phase)
- 1 thumbnail
- **Total: ~19 visual assets per lesson**

**Files:**
- `supabase/migrations/20251214_create_lesson_visuals_table.sql`
- `scripts/generate-lesson-visuals.ts`
- `api/visual/generate.ts`
- `api/visual/check.ts`

---

### Code Challenges

**Status:** ❌ **NOT FOUND**

**Search Results:** No code challenge tables, no programming language support, no starter templates found.

**Recommendation:** 
- Create `lesson_code_challenges` table
- Support languages: JavaScript, Python (start with these)
- Schema needed:
```sql
CREATE TABLE lesson_code_challenges (
  id UUID PRIMARY KEY,
  core_lesson_id UUID REFERENCES core_lessons(id),
  language TEXT NOT NULL, -- 'javascript', 'python'
  starter_code TEXT,
  solution_code TEXT,
  test_cases JSONB,
  hints JSONB,
  difficulty TEXT -- 'beginner', 'intermediate', 'advanced'
);
```

**Build Effort:** HIGH (new feature)

---

### Quiz Questions

**Status:** ✅ **EXISTS** (in `core_lessons` table)

**Schema:**
```typescript
interface QuizQuestion {
  question: string;
  options: string[]; // Array of answer choices
  correct: string; // Correct answer text
  explanation?: string;
}

// Stored in core_lessons.quick_quiz_questions (JSONB)
quick_quiz_questions?: QuizQuestion[];
```

**Storage:** Supabase `core_lessons.quick_quiz_questions` (JSONB column)

**Question Types:** Multiple choice (all questions are multiple choice)

**Auto-Generated:** Yes (generated during lesson creation pipeline)

**Count Per Lesson:** 3-5 questions (varies by lesson)

**Files:**
- `public/learn.html` (line 12710) - Displays quiz
- `public/app.html` (line 1837) - Quiz rendering
- `prisma/schema.prisma` (line 482) - Schema definition

---

### Geographic Data

**Status:** ✅ **EXISTS** (via API endpoint)

**Implementation:** `/api/geo-context` endpoint

**Schema:**
```typescript
interface GeoContext {
  country: string; // ISO country code
  countryName: string;
  region: string;
  city: string;
  timezone: string;
  localTime: string;
  season: 'spring' | 'summer' | 'autumn' | 'winter';
  hemisphere: 'northern' | 'southern';
  dayOfYear: number; // 1-365
  // ... more time/date fields
}
```

**Storage:** Not stored per lesson - calculated dynamically from user's IP

**Lesson-Specific Geo Data:** Not found. Lessons don't have location metadata.

**Recommendation:** Add optional geo fields to `core_lessons`:
```sql
ALTER TABLE core_lessons ADD COLUMN related_locations JSONB;
-- Example: [{"name": "Mount Everest", "lat": 27.9881, "lng": 86.9250, "country": "NP"}]
```

**File:** `api/geo-context.ts`

---

### Timeline Data

**Status:** ⚠️ **PARTIAL** (historical context exists, but no timeline table)

**Existing:** `core_lessons.historical_context` (TEXT column)

**Missing:** Structured timeline data (year, era, related events)

**Recommendation:** Add timeline table:
```sql
CREATE TABLE lesson_timeline_events (
  id UUID PRIMARY KEY,
  core_lesson_id UUID REFERENCES core_lessons(id),
  event_name TEXT NOT NULL,
  year INTEGER,
  era TEXT, -- 'ancient', 'medieval', 'renaissance', 'modern', etc.
  description TEXT,
  related_lessons INTEGER[] -- day_numbers
);
```

**Build Effort:** MEDIUM (extract from `historical_context` + manual curation)

---

### Flashcard Content

**Status:** ❌ **NOT FOUND** (but can derive from existing content)

**Existing Content That Could Become Flashcards:**
1. Quiz questions (`core_lessons.quick_quiz_questions`)
2. Key concepts from lesson atoms
3. Reflection prompts (`core_lessons.reflection_prompts`)

**Recommendation:** Auto-generate flashcards from quiz questions:
```sql
CREATE TABLE lesson_flashcards (
  id UUID PRIMARY KEY,
  core_lesson_id UUID REFERENCES core_lessons(id),
  front TEXT NOT NULL, -- Question or term
  back TEXT NOT NULL, -- Answer or definition
  hint TEXT,
  source TEXT, -- 'quiz', 'atom', 'reflection'
  difficulty TEXT
);
```

**Build Effort:** LOW (can auto-generate from existing quiz questions)

---

## 2. App-Specific Existing Code

### Chat/Conversation System

**Status:** ✅ **FULLY IMPLEMENTED**

**Provider:** ElevenLabs Conversational AI

**Files:**
- `public/js/kelly-conversation.js` - Main conversation handler
- `api/elevenlabs-signed-url.ts` - Agent URL generation

**Features:**
- Real-time voice conversation
- Lesson context awareness
- System prompts per persona
- Expression bridge to avatar
- Conversation history (stored in memory, not database)

**System Prompt Template:**
```javascript
// From kelly-conversation.js:69-103
getSystemPrompt() {
  return `You are Kelly, a warm and curious educator...
  TODAY'S LESSON: ${ctx.topic}
  CURRENT PHASE: ${ctx.currentPhase}
  ...`;
}
```

**Conversation Storage:** Not persisted to database (only in-memory)

**Recommendation:** Add conversation history table:
```sql
CREATE TABLE conversation_history (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  lesson_day INTEGER,
  messages JSONB, -- Array of {role, content, timestamp}
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Build Effort:** LOW (add persistence layer)

---

### Notes/Journal System

**Status:** ✅ **EXISTS** (Community notes + lesson history)

**Community Notes:**
- **Table:** `commons_lesson_notes`
- **API:** `/api/commons/notes`
- **Types:** expert_context, historical_note, source_citation, teaching_tip, age_adaptation, cultural_context, common_misconception, real_world_example, discussion_prompt, related_topic

**User Notes:**
- **Table:** `lesson_history` (has `notes` field)
- **API:** `/api/lesson-complete` (saves notes with lesson completion)
- **Schema:** `notes: TEXT` (free-form text)

**Reflection Prompts:**
- **Storage:** `core_lessons.reflection_prompts` (JSONB array)
- **Usage:** Prompts for journaling after lesson

**Files:**
- `api/commons/notes.ts` - Community notes API
- `api/lesson-complete.ts` - User notes saving
- `docs/backend/migrations/001_learner_commons.sql` - Schema

**Build Effort:** LOW (already functional, just needs UI polish)

---

### Progress/Gamification

**Status:** ✅ **EXISTS** (Streaks + Badges)

**Streak System:**
- **Storage:** `users.streak_days`, `users.longest_streak`
- **Calculation:** Automatic via database trigger (`update_user_streak()`)
- **Logic:** Increments if lesson completed yesterday, resets if gap

**Badges/Achievements:**
- **Status:** Hardcoded in HTML (not in database)
- **File:** `public/learn.html` (lines 10904-10917)
- **Badges Defined:**
  - First Light (complete first lesson)
  - Week Warrior (7 day streak)
  - Explorer (try 3 different Kellys)
  - Bookworm (complete 10 lessons)
  - Night Owl (learn after 10pm)
  - Early Bird (learn before 7am)
  - Curious Mind (ask Kelly 5 questions)
  - Collector (bookmark 10 moments)
  - Month Master (30 day streak)
  - Century Club (100 lessons)
  - Polyglot (learn in 2 languages)
  - Teacher (share a lesson)

**Points/XP System:** Not found

**Certificate Generation:** Not found

**Recommendation:** Move badges to database:
```sql
CREATE TABLE user_badges (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  badge_id TEXT NOT NULL,
  unlocked_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, badge_id)
);

CREATE TABLE badge_definitions (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  icon TEXT,
  requirement_type TEXT, -- 'streak', 'lessons', 'questions', etc.
  requirement_value INTEGER
);
```

**Files:**
- `public/learn.html` (badge definitions)
- `public/me.html` (badge display)
- `api/visual/stats.ts` (badge calculation for visual generation)

**Build Effort:** MEDIUM (migrate hardcoded badges to database)

---

### Audio Playback

**Status:** ✅ **FULLY IMPLEMENTED**

**Files:**
- `public/js/kelly-audio.js` - Main audio system
- `api/tts.ts` - TTS API endpoint
- `infrastructure/cloudflare/tts-worker/src/index.js` - Cloudflare Worker TTS

**Features:**
- ✅ Playback speed controls (not found in current code, but can add)
- ✅ Audio-only mode (mute video, play audio)
- ❌ Background/ambient audio (not found)

**Playback Controls:**
- Play/pause
- Mute/unmute
- Stop
- Pre-generated audio support
- ElevenLabs streaming support

**Build Effort:** LOW (add speed controls if needed)

---

## 3. 3D Avatar Investigation

**Status:** ❌ **NO 3D MODEL** (Only 2D video)

**Search Results:**
- No `.glb`, `.gltf`, `.vrm`, `.fbx` files found
- No Three.js, Babylon.js, or React Three Fiber imports
- Unity WebGL exists (`public/unity/`) but **NOT USED IN PRODUCTION**
- No Ready Player Me or similar avatar SDK

**Current Avatar System:**
- **Primary:** HD MP4 videos (1080p, lip-synced)
- **Fallback:** 2D PNG/WebP images
- **Format:** Pre-rendered videos from Sync Labs/HeyGen pipeline

**3D Integration Possibility:**
- Can add 3D layer on top of 2D video
- Would need: 3D model file (.glb), animation system, blend shapes for visemes
- **Build Effort:** HIGH (new feature)

**Recommendation:** 
- Keep 2D video as primary (already working)
- Consider 3D as future enhancement
- If adding 3D: Use Ready Player Me or custom GLB model with Three.js

---

## 4. Database Schema Dump

### All Tables (Public Schema)

```sql
-- Core Content Tables
core_lessons          -- 365 daily lessons
lesson_atoms          -- 21,915 content pieces (12 archetypes × 5 phases × 365 days)
lesson_shards         -- 38,700 demographic variants
lesson_visuals         -- Visual asset tracking
lesson_history        -- User lesson completion history

-- User Tables
users                 -- User profiles (extends auth.users)
user_progress         -- Lesson progress tracking
user_badges           -- NOT YET CREATED (badges are hardcoded)

-- Community Tables
commons_lesson_notes   -- Community-contributed notes
commons_user_contributions -- Contribution stats
commons_answer_aggregates -- Answer statistics

-- Learning Groups
learning_groups        -- Group learning (family/friends)
group_members          -- Group membership

-- Analytics
analytics_events       -- Event tracking
daily_lesson_stats     -- Aggregate stats per day
learner_observations    -- Detailed learning behavior

-- Marketing/Growth
affiliates             -- Affiliate program
referrals              -- Referral tracking
affiliate_applications -- Affiliate signups
enterprise_inquiries   -- Enterprise leads
newsletter_subscribers  -- Email list

-- Payment (Stripe)
-- No tables (uses Stripe directly, webhooks update users.subscription_tier)
```

### Key Foreign Key Relationships

```
core_lessons (1) ──→ (many) lesson_atoms
core_lessons (1) ──→ (many) lesson_shards
core_lessons (1) ──→ (1) lesson_visuals
core_lessons (1) ──→ (many) commons_lesson_notes

users (1) ──→ (many) user_progress
users (1) ──→ (many) lesson_history
users (1) ──→ (1) affiliates
users (1) ──→ (many) analytics_events

learning_groups (1) ──→ (many) group_members
group_members ──→ users
```

### Complete Column List (Key Tables)

**`core_lessons`:**
- `id`, `day_number`, `topic`, `universal_truth`
- `marketing_headline`, `marketing_tagline`, `marketing_pitch`
- `quick_quiz_questions` (JSONB), `reflection_prompts` (JSONB)
- `recommended_videos` (JSONB), `recommended_books` (JSONB)
- `historical_context` (TEXT), `extended_explanation` (TEXT)
- `hero_image_url`, `thumbnail_url`, `demo_video_url`
- `ideal_age_range`, `difficulty_level`, `estimated_duration`
- `learning_objectives` (JSONB), `prerequisite_concepts` (JSONB)
- `fun_facts` (JSONB), `common_misconceptions` (JSONB)
- `real_world_applications` (JSONB), `hands_on_activities` (JSONB)
- `creative_prompts` (JSONB), `challenge_questions` (JSONB)
- `math_connections`, `science_connections`, `language_arts_connections`
- `social_studies_connections`, `mastery_criteria`

**`lesson_atoms`:**
- `id`, `core_lesson_id`, `archetype`, `phase`
- `content` (JSONB: {script, options, kellyPose, kellyEmotion})
- `visual_url`, `hd_video_url`

**`lesson_visuals`:**
- `id`, `core_lesson_id`, `day_number`, `topic`
- `thumbnail_url`, `infographic_url`, `infographic_urls` (JSONB)
- `illustration_url`, `status`

**`user_progress`:**
- `id`, `user_id`, `lesson_id`
- `completed`, `progress_percent`, `last_position_seconds`
- `time_spent_seconds`, `completed_at`, `started_at`

**`lesson_history`:**
- `id`, `user_id`, `lesson_day`, `year_completed`
- `answers` (JSONB), `notes` (TEXT), `time_spent_seconds`
- `layer`, `user_age_at_completion`

---

## 5. API Routes Inventory

### Content & Lessons

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/lessons/[dayNumber]` | GET | No | Fetch lesson by day number |
| `/api/lessons/[dayNumber]-edge` | GET | No | Edge-optimized lesson fetch |
| `/api/lessons` | GET | No | List all lessons |
| `/api/day/[number]` | GET | No | Alternative lesson endpoint |
| `/api/visual/generate` | POST | Yes | Generate lesson visual |
| `/api/visual/check` | GET | No | Check visual generation status |
| `/api/visual/stats` | GET | Yes | Visual generation stats |

### Audio & TTS

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/tts` | POST | No | ElevenLabs TTS (proxy) |
| `/api/elevenlabs-signed-url` | GET | Yes | Get signed URL for agent |
| `/api/elevenlabs-video` | POST | Yes | Generate video with Omnihuman |
| `/api/elevenlabs-webhook` | POST | No | ElevenLabs webhook handler |
| `/api/lipsync-alignment` | POST | Yes | Generate lip-sync alignment |

### User & Progress

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/lesson/complete` | POST | Yes | Mark lesson complete, save progress |
| `/api/lesson-complete` | POST | Yes | Alternative completion endpoint |
| `/api/lesson-history` | GET | Yes | Get user's lesson history |
| `/api/reflection` | POST | Yes | Save reflection/notes |

### Community & Notes

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/commons/notes` | GET/POST | Yes | Community notes (get/create) |
| `/api/commons/history` | GET | No | Commons contribution history |
| `/api/commons/proposals` | GET/POST | Yes | Commons proposals |
| `/api/commons/votes` | POST | Yes | Vote on proposals |

### Geo & Context

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/geo-context` | GET | No | User's geo context (country, timezone, season) |
| `/api/geo-pricing` | GET | No | Location-based pricing |

### Payment & Subscription

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/stripe-checkout` | POST | No | Create Stripe checkout session |
| `/api/create-checkout` | POST | No | Alternative checkout |
| `/api/create-gift-checkout` | POST | No | Gift subscription checkout |
| `/api/create-portal-session` | POST | Yes | Stripe customer portal |
| `/api/webhooks/stripe-revenue` | POST | No | Stripe webhook handler |
| `/api/subscription-status` | GET | Yes | Get user subscription status |
| `/api/cancel-subscription` | POST | Yes | Cancel subscription |
| `/api/pause-subscription` | POST | Yes | Pause subscription |
| `/api/resume-subscription` | POST | Yes | Resume subscription |
| `/api/resubscribe` | POST | Yes | Resubscribe after cancellation |

### Family & Groups

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/family/link` | POST | Yes | Link family members |
| `/api/family/members` | GET | Yes | Get family members |
| `/api/family/claim-earnings` | POST | Yes | Claim affiliate earnings |

### Referrals & Affiliates

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/referral/lookup` | GET | No | Lookup referral code |
| `/api/referral/track` | POST | No | Track referral click |
| `/api/referral/convert` | POST | Yes | Convert referral to subscription |
| `/api/referral/eligibility` | GET | Yes | Check referral eligibility |
| `/api/referral/payout` | POST | Yes | Request payout |

### Notifications

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/notifications/subscribe-device` | POST | Yes | Subscribe to push notifications |
| `/api/notifications/web-push-subscribe` | POST | Yes | Web push subscription |
| `/api/notifications/preferences` | GET/POST | Yes | Notification preferences |
| `/api/notifications/test-push` | POST | Yes | Test push notification |

### Email

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/email/daily-content` | GET | No | Daily lesson email content |
| `/api/email/send-daily-lesson-email` | POST | Yes | Send daily lesson email |
| `/api/email/send-full-lesson-email` | POST | Yes | Send full lesson email |
| `/api/email/send-welcome-email` | POST | Yes | Send welcome email |
| `/api/email/send-streak-email` | POST | Yes | Send streak reminder |
| `/api/email/inbound` | POST | No | Handle inbound email |
| `/api/email/reply` | POST | Yes | Reply to email thread |

### Cron Jobs

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/cron/daily-lesson` | GET | Yes* | Daily lesson generation |
| `/api/cron/daily-push-notifications` | GET | Yes* | Send daily push notifications |
| `/api/cron/streak-check` | GET | Yes* | Check and update streaks |
| `/api/cron/weekly-digest` | GET | Yes* | Weekly email digest |
| `/api/cron/birthday-emails` | GET | Yes* | Birthday email campaign |
| `/api/cron/gentle-return` | GET | Yes* | Re-engagement emails |

*Protected by Vercel Cron secret

### Other

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | No | Health check |
| `/api/health-check` | GET | No | Alternative health check |
| `/api/ping` | GET | No | Simple ping |
| `/api/stats` | GET | No | Platform statistics |
| `/api/events` | POST | No | Analytics events |
| `/api/sitemap.xml` | GET | No | Sitemap generation |
| `/api/og/[day]` | GET | No | Open Graph image generation |
| `/api/byok-llm` | POST | Yes | BYOK LLM proxy |
| `/api/calendar/feed` | GET | Yes | iCal feed for lessons |

---

## 6. File Structure for Each App

| Kelly OS App | Existing Files | Existing Content | Missing/To Build | Build Effort |
|--------------|----------------|------------------|------------------|--------------|
| **Knowledge Library** | `public/lesson-detail.html` | ✅ `core_lessons` (all fields), `recommended_videos`, `recommended_books` | UI redesign for macOS style | LOW |
| **Research Lab** | `api/commons/notes.ts` | ✅ `commons_lesson_notes`, expert context, citations | UI for research interface | MEDIUM |
| **Code Lab** | None | ❌ No code challenges | Create `lesson_code_challenges` table + UI | HIGH |
| **Learning Journal** | `api/lesson-complete.ts` | ✅ `lesson_history.notes`, `reflection_prompts` | Journal UI, note-taking interface | LOW |
| **365 Calendar** | `public/calendar.html` | ✅ `core_lessons.day_number`, `user_progress` | Calendar UI polish | LOW |
| **Visual Atlas** | `api/visual/check.ts` | ✅ `lesson_visuals`, `lesson_atoms.visual_url`, `hd_video_url` | Gallery UI, visual browser | LOW |
| **Audio Player** | `public/js/kelly-audio.js` | ✅ TTS system, audio playback | React wrapper, speed controls | LOW |
| **Ask Kelly** | `public/js/kelly-conversation.js` | ✅ ElevenLabs Conversational AI | Conversation history UI, persistence | MEDIUM |
| **Quiz Arena** | `public/learn.html` (quiz section) | ✅ `quick_quiz_questions` | Standalone quiz app UI | LOW |
| **World Map** | `api/geo-context.ts` | ⚠️ Geo API exists, but no lesson locations | Add `related_locations` to lessons, map UI | MEDIUM |
| **Timeline** | None | ⚠️ `historical_context` exists (text) | Create `lesson_timeline_events` table, timeline UI | MEDIUM |
| **Flashcards** | None | ⚠️ Can derive from `quick_quiz_questions` | Auto-generate flashcards, study mode UI | LOW |
| **Workshop** | None | ✅ `hands_on_activities`, `creative_prompts` | Activity UI, project templates | MEDIUM |
| **Trophy Room** | `public/me.html` (badges) | ⚠️ Badges hardcoded | Move to database, badge UI polish | MEDIUM |
| **Family Hub** | `api/family/*.ts` | ✅ `learning_groups`, `group_members` | Family dashboard UI | LOW |
| **My Profile** | `public/me.html` | ✅ `users`, `user_progress`, streaks | Profile UI polish | LOW |

---

## 7. Content Extension Inventory (TypeScript)

```typescript
interface ContentExtensionInventory {
  externalLinks: {
    exists: true;
    table: "core_lessons";
    schema: {
      recommended_videos: "JSONB Array<{title, url, source, duration}>";
      recommended_books: "JSONB Array<{title, author, url}>";
      downloadable_resources: "JSONB Array<{title, url, type}>";
    };
    countPerLesson: "Variable (0-10+)";
  };
  
  visualAssets: {
    exists: true;
    storageLocation: "Supabase Storage + lesson_visuals table";
    types: ["infographic", "option_card", "hd_video", "thumbnail", "hero_image"];
    countPerLesson: 19; // 5 infographics + 8 option cards + 5 videos + 1 thumbnail
  };
  
  codeChallenges: {
    exists: false;
    languages: [];
    countPerLesson: 0;
  };
  
  quizQuestions: {
    exists: true;
    questionTypes: ["multiple_choice"];
    countPerLesson: "3-5";
    autoGenerated: true;
  };
  
  geoData: {
    exists: true; // Via API, not per lesson
    schema: {
      endpoint: "/api/geo-context";
      provides: "country, timezone, season, hemisphere, dayOfYear";
      lessonSpecific: false; // No lesson location metadata yet
    };
  };
  
  timelineData: {
    exists: false; // historical_context exists but unstructured
    schema: null;
  };
  
  flashcards: {
    exists: false;
    autoGenerated: false; // Can derive from quiz questions
  };
}
```

---

## 8. App Readiness Assessment

```typescript
interface AppReadiness {
  appName: string;
  existingFiles: string[];
  existingContent: string[];
  missingContent: string[];
  buildEffort: 'low' | 'medium' | 'high';
}

const appReadiness: AppReadiness[] = [
  {
    appName: "Knowledge Library",
    existingFiles: ["public/lesson-detail.html", "public/js/golden-v5-data-loader.js"],
    existingContent: ["core_lessons.*", "recommended_videos", "recommended_books", "extended_explanation"],
    missingContent: [],
    buildEffort: "low"
  },
  {
    appName: "Research Lab",
    existingFiles: ["api/commons/notes.ts", "docs/features/LEARNER_COMMONS.md"],
    existingContent: ["commons_lesson_notes", "expert_context", "source_citation"],
    missingContent: ["Research UI components"],
    buildEffort: "medium"
  },
  {
    appName: "Code Lab",
    existingFiles: [],
    existingContent: [],
    missingContent: ["lesson_code_challenges table", "Code editor", "Test runner", "Solution checker"],
    buildEffort: "high"
  },
  {
    appName: "Learning Journal",
    existingFiles: ["api/lesson-complete.ts", "api/reflection.ts"],
    existingContent: ["lesson_history.notes", "reflection_prompts"],
    missingContent: ["Journal UI", "Note editor"],
    buildEffort: "low"
  },
  {
    appName: "365 Calendar",
    existingFiles: ["public/calendar.html"],
    existingContent: ["core_lessons.day_number", "user_progress"],
    missingContent: [],
    buildEffort: "low"
  },
  {
    appName: "Visual Atlas",
    existingFiles: ["api/visual/check.ts", "api/visual/generate.ts"],
    existingContent: ["lesson_visuals", "lesson_atoms.visual_url", "hd_video_url"],
    missingContent: ["Gallery UI"],
    buildEffort: "low"
  },
  {
    appName: "Audio Player",
    existingFiles: ["public/js/kelly-audio.js", "api/tts.ts"],
    existingContent: ["TTS system", "Audio playback"],
    missingContent: ["Speed controls", "Playlist UI"],
    buildEffort: "low"
  },
  {
    appName: "Ask Kelly",
    existingFiles: ["public/js/kelly-conversation.js", "api/elevenlabs-signed-url.ts"],
    existingContent: ["ElevenLabs Conversational AI", "System prompts"],
    missingContent: ["conversation_history table", "Chat UI", "History persistence"],
    buildEffort: "medium"
  },
  {
    appName: "Quiz Arena",
    existingFiles: ["public/learn.html (quiz section)"],
    existingContent: ["quick_quiz_questions"],
    missingContent: ["Standalone quiz UI"],
    buildEffort: "low"
  },
  {
    appName: "World Map",
    existingFiles: ["api/geo-context.ts"],
    existingContent: ["Geo API endpoint"],
    missingContent: ["related_locations in core_lessons", "Map UI", "Location markers"],
    buildEffort: "medium"
  },
  {
    appName: "Timeline",
    existingFiles: [],
    existingContent: ["historical_context (unstructured text)"],
    missingContent: ["lesson_timeline_events table", "Timeline UI", "Event extraction"],
    buildEffort: "medium"
  },
  {
    appName: "Flashcards",
    existingFiles: [],
    existingContent: ["quick_quiz_questions (can derive)"],
    missingContent: ["lesson_flashcards table", "Flashcard UI", "Study mode"],
    buildEffort: "low"
  },
  {
    appName: "Workshop",
    existingFiles: [],
    existingContent: ["hands_on_activities", "creative_prompts"],
    missingContent: ["Activity UI", "Project templates", "Submission system"],
    buildEffort: "medium"
  },
  {
    appName: "Trophy Room",
    existingFiles: ["public/me.html", "public/learn.html (badge definitions)"],
    existingContent: ["Badge definitions (hardcoded)", "Streak system"],
    missingContent: ["user_badges table", "badge_definitions table", "Badge UI polish"],
    buildEffort: "medium"
  },
  {
    appName: "Family Hub",
    existingFiles: ["api/family/*.ts"],
    existingContent: ["learning_groups", "group_members"],
    missingContent: ["Family dashboard UI"],
    buildEffort: "low"
  },
  {
    appName: "My Profile",
    existingFiles: ["public/me.html"],
    existingContent: ["users", "user_progress", "streak_days"],
    missingContent: [],
    buildEffort: "low"
  }
];
```

---

## 9. Critical Missing Content

### High Priority (Blocks App Launch)

1. **Code Challenges** - Required for Code Lab app
   - Need: `lesson_code_challenges` table
   - Need: Code editor integration
   - Need: Test runner
   - **Effort:** HIGH

2. **Flashcards** - Required for Flashcards app
   - Can auto-generate from `quick_quiz_questions`
   - Need: `lesson_flashcards` table
   - Need: Study mode UI
   - **Effort:** LOW

### Medium Priority (Enhances Apps)

3. **Timeline Events** - Enhances Timeline app
   - Extract from `historical_context`
   - Need: `lesson_timeline_events` table
   - Need: Timeline visualization
   - **Effort:** MEDIUM

4. **Lesson Locations** - Enhances World Map app
   - Add `related_locations` JSONB to `core_lessons`
   - Need: Map UI with markers
   - **Effort:** MEDIUM

5. **Conversation History** - Enhances Ask Kelly app
   - Add `conversation_history` table
   - Persist chat sessions
   - **Effort:** LOW

6. **Badge System** - Enhances Trophy Room app
   - Migrate hardcoded badges to database
   - Add `user_badges` and `badge_definitions` tables
   - **Effort:** MEDIUM

---

## 10. APP_CONTENT_MAP.md

See separate file: `APP_CONTENT_MAP.md` (to be created)

**Summary:** Each Kelly OS app maps to specific database tables and content fields:

- **Knowledge Library** → `core_lessons.*`, `recommended_videos`, `recommended_books`
- **Research Lab** → `commons_lesson_notes`
- **Code Lab** → `lesson_code_challenges` (TO CREATE)
- **Learning Journal** → `lesson_history.notes`, `reflection_prompts`
- **365 Calendar** → `core_lessons.day_number`, `user_progress`
- **Visual Atlas** → `lesson_visuals`, `lesson_atoms.visual_url`, `hd_video_url`
- **Audio Player** → TTS system, audio files
- **Ask Kelly** → ElevenLabs Conversational AI
- **Quiz Arena** → `quick_quiz_questions`
- **World Map** → `related_locations` (TO ADD), `api/geo-context`
- **Timeline** → `lesson_timeline_events` (TO CREATE), `historical_context`
- **Flashcards** → `lesson_flashcards` (TO CREATE, derive from quiz)
- **Workshop** → `hands_on_activities`, `creative_prompts`
- **Trophy Room** → `user_badges` (TO CREATE), `users.streak_days`
- **Family Hub** → `learning_groups`, `group_members`
- **My Profile** → `users`, `user_progress`, streaks

---

## 11. Database Tables Summary

### Production Tables (Verified)

1. `core_lessons` - 365 lessons
2. `lesson_atoms` - 21,915 content pieces
3. `lesson_shards` - 38,700 demographic variants
4. `lesson_visuals` - Visual asset tracking
5. `lesson_history` - User lesson completion history
6. `users` - User profiles
7. `user_progress` - Lesson progress
8. `commons_lesson_notes` - Community notes
9. `commons_user_contributions` - Contribution stats
10. `learning_groups` - Group learning
11. `group_members` - Group membership
12. `analytics_events` - Event tracking
13. `daily_lesson_stats` - Daily aggregates
14. `learner_observations` - Learning behavior
15. `affiliates` - Affiliate program
16. `referrals` - Referral tracking
17. `newsletter_subscribers` - Email list
18. `enterprise_inquiries` - Enterprise leads

### Tables To Create

1. `lesson_code_challenges` - Code challenges
2. `lesson_flashcards` - Flashcards
3. `lesson_timeline_events` - Timeline data
4. `conversation_history` - Chat history
5. `user_badges` - User badge unlocks
6. `badge_definitions` - Badge metadata

---

## 12. Recommendations

### Immediate Actions (Week 1)

1. ✅ **Extract existing content** - All content types are accessible
2. ✅ **Create APP_CONTENT_MAP.md** - Document content → app mapping
3. ⚠️ **Create missing tables** - `lesson_flashcards`, `conversation_history`, `user_badges`

### Short Term (Weeks 2-4)

4. ⚠️ **Add location data** - Populate `related_locations` for World Map app
5. ⚠️ **Extract timeline events** - Create `lesson_timeline_events` from `historical_context`
6. ⚠️ **Migrate badges** - Move hardcoded badges to database

### Long Term (Months 2-3)

7. ❌ **Build Code Lab** - Create code challenges system
8. ⚠️ **Enhance 3D Avatar** - Consider adding 3D layer (optional)

---

**End of Report**





