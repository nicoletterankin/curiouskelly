# Kelly OS Technical Architecture - Complete System Audit

**Date:** December 2025  
**Purpose:** Complete technical audit for v0 integration  
**Status:** Production-ready, hybrid infrastructure

---

## Executive Summary

**Current State:** Hybrid architecture (Vercel + Cloudflare Workers + Supabase)  
**Frontend:** Static HTML/JS served from `public/` directory  
**Backend:** Vercel Edge Functions + Cloudflare Workers  
**Database:** Supabase PostgreSQL  
**Storage:** Supabase Storage + Cloudflare R2  
**Avatar:** 2D HD videos (Unity WebGL legacy, not in production)

**Key Metrics:**
- **365 lessons** (Days 1-365) fully generated
- **20,341 lesson atoms** (12 archetypes × 5 phases × 365 days)
- **~122 HD videos** generated (2.2% of target 5,475)
- **Production URL:** `curiouskelly.com`
- **Database:** Supabase project `tvjalxxsyryjphkforjv`

---

## 1. Repository Structure

### Directory Tree

```
/
├── api/                          # Vercel Edge Functions & Serverless APIs
│   ├── lessons/                  # Lesson data endpoints
│   ├── commons/                  # Community features (notes, votes)
│   ├── visual/                   # Visual asset generation
│   ├── cron/                     # Scheduled jobs (emails, streaks)
│   └── lib/                      # Shared API utilities
│
├── public/                       # Static frontend (served as-is)
│   ├── index.html                # Landing page (marketing)
│   ├── learn.html                # Main lesson player
│   ├── js/                       # Client-side JavaScript
│   │   ├── kelly-audio.js        # Audio playback (ElevenLabs)
│   │   ├── kelly-lipsync.js      # Lip-sync animation
│   │   ├── kelly-conversation.js # Chat system (ElevenLabs AI)
│   │   └── lesson-assets.js      # Asset management
│   ├── css/                      # Stylesheets
│   ├── assets/                    # Static assets (images, fonts)
│   └── unity/                    # Unity WebGL build (legacy)
│
├── components/                   # React/TSX components (macOS-style UI)
│   ├── ui/                       # shadcn/ui components
│   ├── apps/                     # App-specific components
│   └── widgets/                  # Dashboard widgets
│
├── scripts/                      # Build & generation scripts
│   ├── kelly-video-factory/      # Video generation pipeline
│   ├── lesson-factory/           # Lesson content generation
│   └── sync-labs-*.ts           # Sync Labs video generation
│
├── supabase/                     # Database migrations & config
│   └── migrations/               # SQL migration files
│
├── docs/                         # Documentation
│   ├── deployment/                # Deployment guides
│   ├── backend/                  # Backend architecture
│   └── social-media/             # Social media automation
│
├── daily-lesson-marketing/       # Astro marketing site (separate)
│
├── prisma/                       # Prisma schema (database ORM)
│
├── infrastructure/                # Cloudflare Workers
│   └── cloudflare/               # Worker definitions
│
└── vercel.json                   # Vercel deployment config
```

### Key Files Inventory

#### Entry Points
- `public/index.html` - Landing page (redirects returning users to `/learn.html`)
- `public/learn.html` - Main lesson player application
- `public/app.html` - Kelly OS interface (in development)

#### Page Routes (Static HTML)
- `public/index.html` - Marketing homepage
- `public/learn.html` - Lesson player
- `public/calendar.html` - 365-day calendar view
- `public/me.html` - User profile (streaks, badges)
- `public/settings.html` - User settings
- `public/commons.html` - Community notes
- `public/hub.html` - Learning hub

#### API Routes (`api/`)
- `api/lessons/[dayNumber].ts` - Get lesson data
- `api/lessons/[dayNumber]-edge.ts` - Edge-optimized lesson endpoint
- `api/lesson-complete.ts` - Mark lesson complete, update progress
- `api/tts.ts` - ElevenLabs TTS proxy
- `api/elevenlabs-signed-url.ts` - Get signed URL for conversational AI
- `api/visual/generate.ts` - Generate visual assets
- `api/visual/check.ts` - Check visual generation status
- `api/commons/notes.ts` - Community notes CRUD
- `api/create-checkout.ts` - Stripe checkout session
- `api/subscription-status.ts` - Get user subscription status
- `api/cron/*.ts` - Scheduled jobs (emails, streaks, etc.)

#### Database Schemas
- `supabase-schema.sql` - Complete PostgreSQL schema
- `prisma/schema.prisma` - Prisma ORM schema
- `supabase/migrations/*.sql` - Migration history

#### Configuration Files
- `vercel.json` - Vercel deployment config
- `wrangler.toml` - Cloudflare Workers config (disabled)
- `package.json` - Node.js dependencies
- `tsconfig.json` - TypeScript config
- `astro.config.mjs` - Astro config (marketing site)

#### Environment Variables (See `ENV_TEMPLATE.env`)
- `PUBLIC_SUPABASE_URL` - Supabase project URL
- `PUBLIC_SUPABASE_ANON_KEY` - Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key (server-only)
- `ELEVENLABS_API_KEY` - ElevenLabs API key
- `ELEVENLABS_KELLY_VOICE_ID` - Kelly's voice ID (`wAdymQH5YucAkXwmrdL0`)
- `STRIPE_SECRET_KEY` - Stripe secret key
- `STRIPE_WEBHOOK_SECRET` - Stripe webhook secret
- `SYNC_LABS_API_KEY` - Sync Labs API key (video generation)
- `REPLICATE_API_TOKEN` - Replicate API token
- `HUGGINGFACE_API_KEY` - HuggingFace API key (optional)

---

## 2. Current Deployment Architecture

### Vercel Setup

**Project Name:** `curiouskelly-marketing` (or root project)  
**Production URL:** `https://curiouskelly.com`  
**Preview URLs:** Auto-generated per PR/branch

**Configuration (`vercel.json`):**
```json
{
  "buildCommand": "echo static",
  "outputDirectory": "public",
  "installCommand": "npm install",
  "framework": null,
  "functions": {
    "api/**/*.ts": {
      "memory": 256,
      "maxDuration": 30
    },
    "api/**/*-edge.ts": {
      "runtime": "@vercel/edge",
      "memory": 128,
      "maxDuration": 10
    }
  }
}
```

**Build Settings:**
- **Output Directory:** `public/` (static files)
- **Framework:** None (static HTML/JS)
- **Edge Functions:** Enabled for `*-edge.ts` files
- **Serverless Functions:** Standard Node.js runtime for other APIs

**Vercel Features Enabled:**
- ✅ Edge Functions (`@vercel/edge` runtime)
- ✅ Serverless Functions (Node.js)
- ✅ Automatic HTTPS/SSL
- ✅ Preview deployments (per branch/PR)
- ✅ Environment variables (per environment)
- ✅ Custom headers (cache control for HTML)

**Environment Variables Configured:**
- Production: All required env vars (see `ENV_TEMPLATE.env`)
- Preview: Same vars with test values
- Development: Local `.env` file

### Cloudflare Integration

**Current Status:** Partial (Workers exist but not primary)

**Cloudflare Workers:**
1. **Unity CDN Worker** (`infrastructure/cloudflare/unity-cdn-worker/`)
   - Purpose: Serve Unity WebGL assets from R2
   - Status: Legacy (Unity not in production)
   - Route: `unity.curiouskelly.com/*`

2. **Lessons API Worker** (`infrastructure/cloudflare/lessons-api-worker/`)
   - Purpose: Mirror lesson data to D1 (edge database)
   - Status: Partial (not primary)
   - Route: `api.curiouskelly.com/lessons/*`

**Cloudflare R2 Storage:**
- **Bucket:** `curious-kelly-media` (Unity assets)
- **Bucket:** `curious-kelly-videos` (video storage)
- **Status:** Used for Unity assets (legacy), not primary video storage

**Cloudflare D1 Database:**
- **Database:** `lessons-db` (mirror of Supabase `core_lessons`)
- **Purpose:** Edge caching for lesson data
- **Status:** Partial implementation

**What Needs to Migrate to Vercel:**
- ✅ Already on Vercel: All APIs, static frontend
- ⚠️ Cloudflare Workers: Can be migrated to Vercel Edge Functions
- ⚠️ Cloudflare R2: Migrate to Vercel Blob Storage or Supabase Storage
- ⚠️ Cloudflare D1: Not needed (use Supabase + Edge caching)

**DNS/Proxy Configuration:**
- **Domain:** `curiouskelly.com` (managed in Vercel)
- **DNS:** Pointed to Vercel
- **SSL:** Automatic via Vercel

### Database

**Supabase Connection:**
- **Project URL:** `https://tvjalxxsyryjphkforjv.supabase.co`
- **Database:** PostgreSQL (Supabase managed)
- **Auth:** Supabase Auth (extends `auth.users`)

**All Tables (Public Schema):**

1. **`users`** - User profiles (extends `auth.users`)
2. **`core_lessons`** - 365 daily lessons
3. **`lesson_atoms`** - 20,341 lesson phases (12 archetypes × 5 phases × 365 days)
4. **`lesson_shards`** - Age/language variants
5. **`lesson_visuals`** - Visual asset tracking
6. **`kelly_video_assets`** - Video generation tracking
7. **`lesson_video_generation_status`** - Video generation status
8. **`user_progress`** - Learning progress
9. **`lesson_history`** - Lesson completion history
10. **`commons_lesson_notes`** - Community notes
11. **`affiliates`** - Affiliate program
12. **`referrals`** - Referral tracking
13. **`newsletter_subscribers`** - Email list
14. **`analytics_events`** - Event tracking
15. **`learning_groups`** - Family/group accounts
16. **`group_members`** - Group membership
17. **`daily_lesson_stats`** - Daily statistics

**RLS Policies:**
- Users can only access their own data
- Public lessons are readable by all
- Service role has full access

**Functions/Triggers:**
- `handle_new_user()` - Creates `public.users` on signup
- `update_updated_at()` - Auto-updates timestamps
- `get_kelly_video_url()` - Video URL lookup helper

**Data Volume:**
- **core_lessons:** 365 rows
- **lesson_atoms:** 20,341 rows
- **lesson_shards:** ~6,570 rows (6 ages × 3 langs × 365 days)
- **users:** ~100+ (production)
- **user_progress:** ~500+ rows

---

## 3. Kelly Avatar System

### Unity WebGL Export

**Status:** ⚠️ **LEGACY (Not in Production)**

**Location:** `public/unity/` directory

**File Size:** Unknown (needs measurement)

**Communication:**
- Uses `postMessage` API for bidirectional communication
- Web app sends commands: `{ type: 'speak', text: '...' }`
- Unity sends events: `{ type: 'speaking', status: 'start' }`

**Parameters:**
- `speak(text)` - Trigger speech animation
- `setExpression(expression)` - Change facial expression
- `setPose(pose)` - Change body pose
- `playAnimation(animation)` - Play specific animation

**Current Usage:** Not actively used (2D videos preferred)

### Character Creator / iClone Setup

**Software:** Reallusion iClone (legacy), now using:
- **Image Generation:** Flux + Kelly LoRA (`CuriousKellycom/curious-kelly-lora`)
- **Video Generation:** Sync Labs `lipsync-2-pro`, MiniMax Video-01
- **Motion:** HeyGen motion library (deprecated)

**Source Files:**
- LoRA: `https://huggingface.co/CuriousKellycom/curious-kelly-lora`
- Motion library: `scripts/kelly-video-factory/motion-library.json`

**Animations Available:**
- Expressive gestures (teaching)
- Listening poses
- Celebrating reactions
- Thinking poses
- Encouraging gestures

**Expressions/Morphs:**
- Happy/excited
- Thoughtful/explaining
- Encouraging/supportive
- Celebrating/success
- Listening/attentive

### Video Segments

**Storage Location:**
- **Primary:** Supabase Storage (`kelly-videos` bucket)
- **Path Format:** `videos/day-{dayNumber}/{archetype}/{phase}.mp4`
- **Example:** `videos/day-001/explorer/hook_main.mp4`

**Naming Convention:**
```
day-{dayNumber}/{archetype}/{phase}_{type}.mp4

Types:
- main (script video)
- response_a (Option A response)
- response_b (Option B response)
- response_c (Option C response)
```

**Manifest/Index:**
- Database table: `kelly_video_assets` (tracks all videos)
- Fields: `video_public_url`, `video_storage_path`, `status`, `quality_score`

**Metadata Per Video:**
- Duration (ms)
- File size (bytes)
- Resolution (e.g., "1080x1920")
- Generation source (ElevenLabs, Sync Labs, etc.)
- Quality scores (lip sync, video quality)
- Approval status

**Total Storage Size:** Unknown (needs calculation)
**Current Count:** ~122 videos generated, 0 uploaded (blocked by missing bucket)

### Audio / TTS

**ElevenLabs Configuration:**
- **Voice ID:** `wAdymQH5YucAkXwmrdL0` (Kelly's trained voice)
- **Model:** `eleven_multilingual_v2` (for TTS)
- **Conversational AI:** ElevenLabs Conversational AI (for chat)

**Voice Clone:**
- ✅ Trained voice model exists
- Voice ID: `wAdymQH5YucAkXwmrdL0`
- Used for all Kelly speech

**Audio Storage:**
- **Pre-generated:** Supabase Storage (`kelly-audio` bucket)
- **On-demand:** Generated via `/api/tts` endpoint
- **Format:** MP3, 16kHz sample rate

**Generation Pipeline:**
1. Script text → ElevenLabs TTS API
2. Generate audio with Kelly voice
3. Store in Supabase Storage (optional caching)
4. Return URL to frontend
5. Frontend plays via `KellyAudio` class

**Files:**
- `public/js/kelly-audio.js` - Audio playback system
- `api/tts.ts` - TTS API endpoint
- `api/elevenlabs-signed-url.ts` - Conversational AI setup

---

## 4. Content System

### Lesson Data

**Storage:** Supabase `core_lessons` and `lesson_atoms` tables

**Schema:**

```typescript
interface CoreLesson {
  id: string;
  day_number: number; // 1-365
  topic: string;
  universal_truth: string;
  ideal_age_range: string;
  difficulty_level: string;
  estimated_duration: number; // minutes
  
  // Content extensions (JSONB)
  quick_quiz_questions?: QuizQuestion[];
  reflection_prompts?: string[];
  recommended_videos?: Resource[];
  recommended_books?: Resource[];
  interactive_simulations?: Resource[];
  downloadable_resources?: Resource[];
  discussion_questions?: string[];
  hands_on_activities?: Activity[];
  creative_prompts?: string[];
  challenge_questions?: string[];
  historical_context?: string;
  
  // Media
  hero_image_url?: string;
  thumbnail_url?: string;
  
  created_at: string;
  updated_at: string;
}

interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string; // 'The Explorer', 'The Rebel', 'The Scientist', etc.
  phase: string; // 'Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'
  
  content: {
    script: string;
    script_video_url?: string; // Main script video
    options?: Array<{
      letter: string; // 'A', 'B', 'C'
      text: string;
      quality: string; // 'good', 'best', 'redirect'
      response: string;
      response_video_url?: string; // Response video
    }>;
  };
  
  visual_url?: string; // Infographic URL
  hd_video_url?: string; // HD video URL
  
  created_at: string;
}
```

**Organization:**
- **Day → Phases → Atoms**
- Each day has 5 phases (Hook, Fact1, Fact2, Fact3, Wisdom)
- Each phase has 12 archetype variants (The Explorer, The Rebel, etc.)
- Total: 365 days × 5 phases × 12 archetypes = 21,915 atoms

**Content Types Per Atom:**
- Script text
- Options (A/B/C choices)
- Response videos (per option)
- Visual infographics
- HD videos

**Archetype Variants:**
1. The Explorer
2. The Rebel
3. The Scientist
4. The Architect
5. The Mystic
6. The Provider
7. The Artist
8. The Scholar
9. The Warrior
10. The Sage
11. The Dreamer
12. The Builder

### Curriculum Structure

**TypeScript Interfaces:**

```typescript
interface Day {
  day_number: number; // 1-365
  topic: string;
  universal_truth: string;
  ideal_age_range: string;
  difficulty_level: 'beginner' | 'intermediate' | 'advanced';
  estimated_duration: number; // minutes
  
  // Content
  quick_quiz_questions: QuizQuestion[];
  reflection_prompts: string[];
  recommended_videos: Resource[];
  recommended_books: Resource[];
  
  // Media
  hero_image_url?: string;
  thumbnail_url?: string;
  
  // Atoms (phases × archetypes)
  atoms: LessonAtom[];
}

interface Phase {
  phase: 'Hook' | 'Fact1' | 'Fact2' | 'Fact3' | 'Wisdom';
  script: string;
  script_video_url?: string;
  options?: Option[];
  visual_url?: string;
  hd_video_url?: string;
}

interface Atom {
  archetype: string;
  phase: string;
  content: {
    script: string;
    script_video_url?: string;
    options?: Option[];
  };
  visual_url?: string;
  hd_video_url?: string;
}
```

### Media Assets

**Images:**
- **Storage:** Supabase Storage (`lesson-visuals` bucket)
- **Types:** Infographics, option cards, thumbnails, hero images
- **CDN:** Supabase CDN (automatic)

**Videos:**
- **Storage:** Supabase Storage (`kelly-videos` bucket)
- **Types:** HD videos (1080p), response videos
- **CDN:** Supabase CDN

**Audio:**
- **Storage:** Supabase Storage (`kelly-audio` bucket)
- **Types:** Pre-generated TTS audio
- **CDN:** Supabase CDN

**Total Asset Size:** Unknown (needs calculation)

---

## 5. Current Home Page & App

### What's Built

**Home Page (`public/index.html`):**
- Marketing landing page
- Hero section with Kelly introduction
- Value proposition ("One lesson a day")
- Social proof
- Pricing/CTA
- Auto-redirects returning users to `/learn.html`

**Main App (`public/learn.html`):**
- Lesson player interface
- Kelly avatar display (2D video)
- Lesson content (script, options)
- Progress tracking
- Streak display
- Badge showcase

**User Flows:**
1. **New User:** Landing → Sign up → Onboarding → First lesson
2. **Returning User:** Auto-redirect to `/learn.html` → Continue lesson
3. **Lesson Flow:** Hook → Fact1 (Q&A) → Fact2 (Q&A) → Fact3 (Q&A) → Wisdom

**What Works Well:**
- ✅ Lesson content loading
- ✅ Video playback
- ✅ Progress tracking
- ✅ Streak calculation
- ✅ Badge system
- ✅ Responsive design

**What's Broken/Incomplete:**
- ⚠️ Video upload blocked (missing storage bucket)
- ⚠️ Unity WebGL not integrated (legacy)
- ⚠️ Some visual assets missing
- ⚠️ Kelly OS interface incomplete (`app.html`)

### Components Worth Keeping

**From `components/` (React/TSX):**
- `components/ui/*` - shadcn/ui components (well-tested, polished)
- `components/apps/calendar.tsx` - Calendar widget
- `components/widgets/*` - Dashboard widgets

**From `public/js/`:**
- `kelly-audio.js` - Audio playback system (production-ready)
- `kelly-lipsync.js` - Lip-sync animation (production-ready)
- `kelly-conversation.js` - Chat system (production-ready)
- `lesson-assets.js` - Asset management (useful)

**Dependencies:**
- shadcn/ui components (React)
- Supabase JS client
- Web Audio API
- Canvas API (for lip-sync)

### Components to Discard

**Legacy/Unused:**
- Unity WebGL integration (`public/unity/`) - Not in production
- Old lesson player versions (`public/lesson-player-*.html`)
- Test files (`public/test-*.html`)
- Archived components (`_archive/`, `_archived_legacy/`)

**Why:**
- Unity WebGL: Too heavy, 2D videos preferred
- Old versions: Superseded by current implementation
- Test files: Not needed in production

---

## 6. Authentication & Users

### Current Auth Setup

**Provider:** Supabase Auth

**User Data Stored:**
- `auth.users` (Supabase Auth table)
- `public.users` (extends auth.users with app-specific fields)

**Schema:**
```typescript
interface User {
  id: string; // UUID from auth.users
  email: string;
  name?: string;
  age?: number;
  subscription_tier: 'free' | 'annual' | 'gift' | 'enterprise';
  subscription_status: 'active' | 'inactive' | 'cancelled' | 'expired';
  stripe_customer_id?: string;
  current_day: number; // 1-365
  streak_days: number;
  last_lesson_at?: string;
  created_at: string;
  updated_at: string;
}
```

**Session Management:**
- Supabase Auth handles sessions
- JWT tokens stored in cookies/localStorage
- RLS policies enforce data access

**Files:**
- `api/supabase-auth-webhook.ts` - Auth webhook handler
- `public/js/*` - Client-side auth (Supabase JS client)

### User Progress

**Progress Tracking:**
- Table: `user_progress`
- Fields: `completed`, `progress_percent`, `last_position_seconds`, `time_spent_seconds`

**Streaks:**
- Stored in: `users.streak_days`
- Calculated by: `api/cron/streak-check.ts` (daily cron)
- Logic: Consecutive days with lesson completion

**Achievements:**
- Hardcoded in: `public/learn.html` (BADGES array)
- Types: 'First Light', 'Week Warrior', 'Explorer', etc.
- **TODO:** Migrate to database table

**Journal Entries:**
- Table: `commons_lesson_notes`
- Fields: `lesson_id`, `user_id`, `type`, `title`, `content`
- Types: `expert_context`, `historical_note`, `discussion_prompt`

---

## 7. API & Integrations

### Internal APIs

**Lesson APIs:**
- `GET /api/lessons/[dayNumber]` - Get lesson data
- `GET /api/lessons/[dayNumber]-edge` - Edge-optimized lesson endpoint
- `POST /api/lesson-complete` - Mark lesson complete
- `GET /api/lesson-history` - Get user's lesson history

**Visual APIs:**
- `POST /api/visual/generate` - Generate visual assets
- `GET /api/visual/check` - Check generation status
- `GET /api/visual/stats` - Visual generation stats

**Community APIs:**
- `GET /api/commons/notes` - Get community notes
- `POST /api/commons/notes` - Create note
- `GET /api/commons/votes` - Get votes
- `POST /api/commons/votes` - Vote on note

**User APIs:**
- `GET /api/subscription-status` - Get subscription status
- `POST /api/create-checkout` - Create Stripe checkout
- `POST /api/create-portal-session` - Stripe customer portal

**Cron Jobs:**
- `GET /api/cron/daily-lesson` - Daily lesson email
- `GET /api/cron/streak-check` - Update streaks
- `GET /api/cron/weekly-digest` - Weekly email digest

### External Services

**HuggingFace:**
- **Models:** Kelly LoRA (`CuriousKellycom/curious-kelly-lora`)
- **Endpoints:** Model inference API
- **Usage:** Image generation for Kelly avatar

**Sync.so (Sync Labs):**
- **API:** `lipsync-2-pro` endpoint
- **Usage:** Video lip-sync enhancement
- **Integration:** `scripts/sync-labs-*.ts`

**Replicate:**
- **Models:** Flux LoRA, Wav2Lip, Sadtalker
- **Usage:** Image/video generation
- **Integration:** `api/replicate-lipsync.ts`

**HeyGen:**
- **Avatar ID:** Multiple Kelly avatars (deprecated)
- **Templates:** Motion library
- **Status:** Deprecated (queue issues)

**ElevenLabs:**
- **Voice ID:** `wAdymQH5YucAkXwmrdL0` (Kelly)
- **API:** TTS + Conversational AI
- **Usage:** All Kelly speech
- **Integration:** `api/tts.ts`, `api/elevenlabs-signed-url.ts`

**Stripe:**
- **Usage:** Payment processing
- **Integration:** `api/create-checkout.ts`, `api/webhooks/stripe-revenue.ts`

**Supabase:**
- **Usage:** Database, Auth, Storage
- **Integration:** `@supabase/supabase-js` client

---

## 8. Offline Capability

### Current State

**Service Workers:**
- File: `public/sw.js`
- Status: Basic implementation
- Caching: Static assets, lesson data

**What's Cached:**
- Static HTML/CSS/JS
- Lesson JSON data
- Kelly avatar images
- Audio files (pre-generated)

**IndexedDB Usage:**
- Not extensively used
- Some lesson data cached locally

### Planned Architecture

**For 2027 Hardware Vision:**
- Full offline lesson playback
- Offline progress tracking
- Sync when online
- Pre-download lessons for offline

**Sync Strategy:**
- Background sync API
- Conflict resolution (server wins)
- Queue offline actions
- Sync on reconnect

---

## 9. Migration Checklist

### Move to Vercel

**Already on Vercel:**
- ✅ Static frontend (`public/`)
- ✅ All API routes (`api/`)
- ✅ Edge Functions (`*-edge.ts`)

**Can Migrate from Cloudflare:**
- ⚠️ Unity CDN Worker → Vercel Blob Storage + Edge Function
- ⚠️ Lessons API Worker → Already have Vercel Edge Function
- ⚠️ R2 Storage → Vercel Blob Storage or Supabase Storage

**Not Needed:**
- Cloudflare D1 (use Supabase + Edge caching)

### Environment Variables

See `ENV_TEMPLATE.env` for complete list.

### DNS/Domain Changes

**Current:**
- Domain: `curiouskelly.com`
- DNS: Pointed to Vercel
- SSL: Automatic

**No Changes Needed:** Already configured correctly

---

## 10. The Classroom Feel

### What Makes a Classroom?

**Visual Elements:**
- Warm colors (not sterile white)
- Natural textures (wood, paper, fabric)
- Hand-drawn elements (chalkboard, sketches)
- Personal touches (Kelly's personality)
- Curiosity-inducing visuals (wonder, discovery)

**Brand Personality:**
- Warm and inviting
- Curious and playful
- Age-appropriate (2-102)
- Not corporate/sterile
- Educational but fun

**Emotions to Evoke:**
- Wonder and curiosity
- Comfort and safety
- Excitement to learn
- Belonging and community
- Achievement and progress

**Reference Apps/Sites:**
- Duolingo (gamification, but warmer)
- Khan Academy Kids (playful, colorful)
- Scratch (creative, inviting)
- Notion (personal, customizable)
- Apple Education (clean but warm)

### Lens Renaming

**Current → Suggested:**

| Current | Suggested | Rationale |
|---------|-----------|-----------|
| Calendar | **Adventure Map** | More playful, journey-focused |
| Library | **Discovery Shelf** | Invites exploration |
| Atlas | **Wonder Gallery** | More magical, less clinical |
| Journal | **Memory Book** | Personal, warm |
| Chat | **Ask Kelly** | Already good, keep |
| Quiz | **Check Your Thinking** | Less test-like |
| Achievements | **Milestones** | Less gamified, more meaningful |
| Settings | **Your Space** | Personal, inviting |

**Design Principles:**
- Use warm, inviting language
- Avoid corporate/sterile terms
- Make it feel personal and safe
- Encourage curiosity and exploration
- Age-appropriate for 2-102

---

## 11. Next Steps for v0

### Immediate Priorities

1. **Understand Current UI:**
   - Review `public/learn.html` (main lesson player)
   - Review `public/index.html` (landing page)
   - Understand component structure

2. **Set Up Development:**
   - Clone repository
   - Install dependencies (`npm install`)
   - Set up `.env` file (see `ENV_TEMPLATE.env`)
   - Run local dev server (`npm run serve`)

3. **Review Architecture:**
   - Read `KELLY_OS_CONTENT_EXTRACTION_REPORT.md`
   - Read `KELLY_SEGMENT_INVENTORY_QUALITY_ANALYSIS.md`
   - Understand database schema

4. **Plan Kelly OS Interface:**
   - Review `CLASSROOM_DESIGN_BRIEF.md`
   - Understand app structure (16 apps)
   - Plan component migration

### Key Files to Review

- `SYSTEM_ARCHITECTURE.md` (this file)
- `MIGRATION_PLAN.md` (migration guide)
- `COMPONENT_INVENTORY.json` (component mapping)
- `KELLY_INTEGRATION.md` (avatar integration)
- `CONTENT_SCHEMA.sql` (database schema)
- `CLASSROOM_DESIGN_BRIEF.md` (design direction)

---

**Document Status:** ✅ Complete  
**Last Updated:** December 2025  
**Next Review:** After v0 integration





