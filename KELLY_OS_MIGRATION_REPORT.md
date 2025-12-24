# Kelly OS Migration Analysis Report

**Generated:** December 21, 2025  
**Target:** Migration to macOS-style browser OS template  
**Source:** curiouskelly.com frontend

---

## Executive Summary

The curiouskelly.com frontend is a **hybrid architecture** combining:
- **Marketing Site:** Astro-based (`daily-lesson-marketing/`) deployed to Vercel
- **Lesson Player:** Static HTML/JS (`public/`) with Supabase backend
- **API Layer:** Vercel Edge Functions (`api/`) for serverless operations

**Key Finding:** The codebase is **NOT** a traditional Next.js app. It's primarily static HTML with Astro for marketing pages, making migration to a new template more straightforward than a full React/Next.js migration.

---

## 1. Architecture Audit

### Framework & Structure

```typescript
framework: {
  name: "Hybrid (Astro + Static HTML)",
  version: "Astro 5.0.0",
  router: "static" // File-based for Astro, manual routing for HTML
}
```

**Top-Level Directories:**
- `public/` - Static HTML lesson player (main app)
- `daily-lesson-marketing/` - Astro marketing site
- `api/` - Vercel serverless functions (TypeScript)
- `scripts/` - Build/generation scripts
- `docs/` - Documentation
- `content/` - Lesson content manifests
- `curious-kellly/` - Legacy lesson player implementations
- `components/` - Shared UI components (minimal)
- `hooks/` - React hooks (unused in current frontend)

**Routing Structure:**
- **Marketing:** Astro file-based (`src/pages/*.astro`)
- **Lesson Player:** Manual routing via `window.location` and hash routing
- **API:** Vercel function routing (`api/**/*.ts`)

**State Management:**
- **No formal state management library** (no Zustand, Redux, Context)
- **Vanilla JS classes** with internal state (`KellyOS`, `KellyAudio`, `KellyConversation`)
- **LocalStorage/SessionStorage** for user preferences
- **Supabase client** for server state

**Styling System:**
- **No Tailwind** (only in `daa-app/` which is separate)
- **Custom CSS/SCSS** (`public/css/`, `public/styles/`)
- **SCSS** in Astro project (`daily-lesson-marketing/src/styles/`)
- **CSS Modules:** Not used
- **Styled Components:** Not used

### External Dependencies

**API Integrations:**
1. **ElevenLabs** - TTS (Text-to-Speech)
   - Voice ID: `wAdymQH5YucAkXwmrdL0` (Kelly's trained voice)
   - Endpoint: `https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
   - BYOK support: Users can provide own API key

2. **Supabase** - Database & Auth
   - Tables: `core_lessons`, `lesson_atoms`, `user_progress`, `users`
   - Auth: Google OAuth, Apple OAuth, Magic Link
   - Storage: Video/audio assets

3. **Stripe** - Payments
   - Products: Monthly ($7.99), Annual ($49.99), Lifetime ($199.99)
   - Webhooks: `/api/webhooks/stripe-revenue`
   - Checkout: `/api/stripe-checkout`

4. **OpenAI/Anthropic** - LLM (via BYOK)
   - User-provided keys stored in localStorage
   - Proxy endpoint: `/api/byok-llm`

5. **HeyGen** - Video generation (backend only)
   - Photo Avatar API for lip-sync videos

6. **Sync Labs** - Lip-sync (backend)
   - `lipsync-2-pro` model

**API Key Management:**
- **Environment Variables:** `.env` (never committed)
- **Required Vars:**
  - `PUBLIC_SUPABASE_URL`
  - `PUBLIC_SUPABASE_ANON_KEY`
  - `ELEVENLABS_API_KEY`
  - `ELEVENLABS_KELLY_VOICE_ID`
  - `STRIPE_SECRET_KEY`
  - `STRIPE_WEBHOOK_SECRET`
- **BYOK:** User keys stored in `localStorage` via `BYOKManager`

**Database/Backend:**
- **Supabase** (PostgreSQL)
- **Schema:** See `docs/backend/SUPABASE_SCHEMA.md`
- **Key Tables:**
  - `core_lessons` (365 lessons)
  - `lesson_atoms` (21,915 content pieces)
  - `user_progress` (completion tracking)
  - `users` (profiles)

**Authentication:**
- **Provider:** Supabase Auth
- **Methods:** Google OAuth, Apple OAuth, Magic Link, Guest Mode
- **Session:** PKCE flow, auto-refresh tokens
- **Storage Key:** `curious-kelly-auth`

---

## 2. Avatar System Deep Dive

### Avatar Format & Storage

**Format:** Multiple formats supported:
1. **HD Video** (Primary) - MP4 files from Sync Labs/HeyGen pipeline
   - Stored in Supabase Storage
   - Path: `lesson_atoms.hd_video_url`
   - Format: 1080p MP4 with lip-sync

2. **2D Images** - PNG/WebP
   - Path: `/assets/kelly/production/avatars/{expression}/`
   - Expressions: `curious`, `explaining`, `celebrating`, `listening`, `wisdom`
   - Sizes: 64px, 128px, 256px, 512px

3. **Unity WebGL** (Legacy) - `/public/unity/`
   - Not actively used in production

**Asset Paths:**
```
public/assets/kelly/production/
├── avatars/{expression}/kelly-{expression}-{size}.webp
├── hero/kelly-hero-{size}.{webp|jpg}
└── social/og-image.jpg

Supabase Storage:
└── lesson_atoms.hd_video_url (HD video URLs)
```

### TTS Integration

**File:** `public/js/kelly-audio.js`

**How it works:**
1. **Pre-generated audio check** - Looks for `options.audioUrl`
2. **ElevenLabs API call** - `/api/tts` or direct ElevenLabs API (if BYOK)
3. **Audio playback** - HTML5 Audio element
4. **Lip-sync signaling** - Calls `KellyLipSync` on audio events

**Key Code:**
```javascript
// public/js/kelly-audio.js:240-260
async _speakWithElevenLabs(text, options) {
  // BYOK: Try user's key first
  if (window.BYOKManager?.hasProvider('elevenlabs')) {
    return await this._speakWithByokElevenLabs(text, userKey, options);
  }
  // Fallback to platform API
  const response = await fetch('/api/tts', {
    method: 'POST',
    body: JSON.stringify({ text, voiceId: 'wAdymQH5YucAkXwmrdL0' })
  });
}
```

**Voice IDs:**
- **Kelly Voice:** `wAdymQH5YucAkXwmrdL0` (default)
- **Configurable:** Via `window.ELEVENLABS_VOICE_ID`

### Lip-Sync Logic

**File:** `public/js/kelly-lipsync.js`

**Method:** Real-time amplitude-based lip-sync
- **Streaming Mode:** For ElevenLabs WebSocket streaming
- **Audio Element Mode:** For pre-rendered audio files
- **Frequency Analysis:** Uses Web Audio API analyser
- **Unity Bridge:** Sends viseme data to Unity WebGL (if loaded)
- **2D Avatar:** Updates mouth state via `KellyPoseManager`

**Key Code:**
```javascript
// public/js/kelly-lipsync.js:228-254
async addAudioChunk(audioData) {
  // Decode audio chunk
  const audioBuffer = await this.audioContext.decodeAudioData(chunk);
  // Analyze amplitude
  const amplitude = this._calculateAmplitude(audioBuffer);
  // Map to mouth shape
  const mouthShape = AMPLITUDE_TO_MOUTH[amplitude];
  // Send to Unity or 2D avatar
  this._updateMouth(mouthShape);
}
```

**Lip-Sync States:**
- `idle` - Mouth closed
- `speaking` - Active lip movement
- `paused` - Hold current shape
- `stopped` - Return to idle

### Avatar States

**States (from `KellyConversation`):**
- `idle` - Default resting state
- `listening` - User is interacting
- `thinking` - Processing response
- `speaking` - Active speech
- `celebrating` - Correct answer
- `curious` - Question phase
- `explaining` - Teaching moment
- `wisdom` - Closing reflection

**File Paths:**
- Avatar Component: `public/js/kelly-2d-avatar.js`
- Conversation Handler: `public/js/kelly-conversation.js`
- Performance Engine: `public/js/kelly-performance-engine.js`
- Lip-Sync: `public/js/kelly-lipsync.js`
- Audio: `public/js/kelly-audio.js`

---

## 3. Lesson System Extraction

### Lesson Data Source

**Primary:** Supabase `core_lessons` + `lesson_atoms`
**Fallback:** Static JSON files (`public/data/day-XXX-complete.js`)

**Fetching Logic:**
```javascript
// public/js/golden-v5-data-loader.js:31-58
async loadLesson(dayNumber, archetype) {
  // 1. Load core lesson
  const { data: coreLesson } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();
  
  // 2. Load atoms (phases)
  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', coreLesson.id)
    .eq('archetype', archetype)
    .order('phase');
}
```

### Lesson Data Schema

```typescript
interface CoreLesson {
  id: UUID;
  day_number: number; // 1-365
  topic: string;
  universal_truth: string;
  marketing_headline: string;
  marketing_tagline?: string;
  marketing_pitch?: string;
  quick_quiz_questions?: JSONB;
  reflection_prompts?: JSONB;
  mastery_criteria?: string;
}

interface LessonAtom {
  id: UUID;
  core_lesson_id: UUID;
  archetype: string; // "The Explorer", "The Scientist", etc.
  phase: string; // "Hook", "Fact1", "Fact2", "Fact3", "Wisdom"
  content: {
    script: string;
    options?: Array<{text: string, response: string}>;
    kellyPose?: string;
    kellyEmotion?: string;
    optionIntro?: string;
    hintSystem?: object;
  };
  visual_url?: string;
  hd_video_url?: string; // HD video from pipeline
}
```

### 365-Day Curriculum Map

**Implementation:**
- **Database:** `core_lessons.day_number` (1-365)
- **Calendar Component:** `public/calendar.html`
- **Day Calculation:** Based on day of year
```javascript
const now = new Date();
const start = new Date(now.getFullYear(), 0, 0);
const diff = now - start;
const dayNumber = Math.floor(diff / (1000 * 60 * 60 * 24)); // 1-365
```

### 12 Personas Implementation

**Archetypes (from `docs/GOLDEN_THREE_ARCHETYPES.md`):**
1. The Explorer
2. The Scientist
3. The Rebel
4. The Architect
5. The Diplomat
6. The Empath
7. The MacGyver
8. The Mystic
9. The Provider
10. The Storyteller
11. The Strategist
12. The Survivor

**Implementation:**
- **Database:** `lesson_atoms.archetype` column
- **Content:** Each archetype has unique script/voice pattern
- **Selection:** User selects "tone" → maps to archetype
- **File:** `public/js/kelly-personas.js` - Contains all 12 persona definitions with metadata, icons, colors, and image URLs
- **Persona Metadata:** Includes id, name, icon, tagline, description, color, and Supabase CDN image paths

**Persona Count:** 12 archetypes × 5 phases × 365 days = 21,900 atoms

### Age Slider Implementation

**How it works:**
- **Client-side filtering:** Age stored in `localStorage` (`kelly_age_setting`)
- **Content adaptation:** Database has `lesson_shards` table (age variants)
- **UI:** Slider in lesson player (2-102 years)
- **Age Buckets:** Maps to ranges: `2-5`, `6-12`, `13-17`, `18-35`, `36-60`, `61+`
- **Effect:** Changes script complexity and vocabulary, not archetype
- **Debounced:** Age changes trigger content reload after 400ms delay

**Files:** Age logic is inline in:
- `public/learn.html` - Main lesson player
- `public/app.html` - Alternative app interface
- `app/script.js` - Unified lesson app (if using that version)

### Language Switching

**Implementation:**
- **Precomputed:** All content has EN + ES/FR in database
- **Storage:** `lesson_atoms.content` JSONB contains multilingual scripts
- **UI:** Language selector in settings
- **Storage Key:** `kelly_language_setting` (default: 'en')

**Files:**
- Language Switcher: `daily-lesson-marketing/src/components/LanguageSwitcher.astro`
- Content: Multilingual in `lesson_atoms.content` JSONB

---

## 4. User System

### Auth Flow

**Sign Up:**
1. User clicks "Sign in with Google/Apple"
2. Redirects to Supabase OAuth
3. Callback: `/auth/callback` → creates `auth.users` record
4. Webhook: `/api/supabase-auth-webhook` → creates `public.users` record

**Sign In:**
- Same as sign up (OAuth handles both)
- Session stored in localStorage (`curious-kelly-auth`)

**Session Management:**
- **Auto-refresh:** Supabase handles token refresh
- **Check:** `supabase.auth.getSession()` on page load
- **Storage:** PKCE flow, secure token storage

**Files:**
- Auth Logic: `daily-lesson-marketing/src/lib/auth.ts`
- Callback: `daily-lesson-marketing/src/pages/auth/callback.astro`
- Webhook: `api/supabase-auth-webhook.ts`

### User Profile Schema

```typescript
interface User {
  id: UUID; // Matches auth.users.id
  email: string;
  name?: string;
  age?: number;
  subscription_tier: 'free' | 'scholar' | 'pro' | 'lifetime';
  subscription_expires_at?: Timestamp;
  current_day?: number; // Progress tracking
  streak_days?: number;
  created_at: Timestamp;
}
```

**Table:** `public.users` (Supabase)

### Progress Tracking

**Implementation:**
- **Table:** `user_progress`
- **Schema:**
```typescript
interface UserProgress {
  id: UUID;
  user_id: UUID;
  lesson_id: UUID;
  completed: boolean;
  progress_percent: number; // 0-100
  last_position_seconds: number;
  time_spent_seconds: number;
  completed_at?: Timestamp;
  started_at: Timestamp;
}
```

**API Endpoint:** `/api/lesson/complete.ts`

**Storage:** Supabase `user_progress` table

### Subscription/Payment Integration

**Stripe Products:**
- Monthly: $7.99/month
- Annual: $49.99/year
- Lifetime: $199.99 one-time
- Family: $99.99/year

**Checkout Flow:**
1. User clicks "Subscribe" → `/api/stripe-checkout`
2. Creates Stripe Checkout Session
3. Redirects to Stripe hosted page
4. Success → `/welcome.html?session_id={CHECKOUT_SESSION_ID}`
5. Webhook: `/api/webhooks/stripe-revenue` → updates `users.subscription_tier`

**Files:**
- Checkout: `api/stripe-checkout.ts`
- Webhook: `api/webhooks/stripe-revenue.ts`
- Portal: `api/create-portal-session.ts`

### BYOK (Bring Your Own Keys)

**Implementation:**
- **Storage:** `localStorage` via `BYOKManager`
- **Providers:** ElevenLabs, OpenAI, Anthropic
- **UI:** Settings panel in `learn.html`
- **API:** `/api/byok-llm.ts` (proxy for LLM calls)

**File:** `public/js/kelly-byok-prompt-generator.js`

---

## 5. Reusable Assets Inventory

### Must Extract (Critical)

**Avatar Assets:**
- [x] HD Video files (Supabase Storage URLs)
- [x] 2D Avatar images (`public/assets/kelly/production/avatars/`)
- [x] Hero images (`public/assets/kelly/production/hero/`)

**Code:**
- [x] `public/js/kelly-audio.js` - ElevenLabs TTS integration
- [x] `public/js/kelly-lipsync.js` - Lip-sync system
- [x] `public/js/kelly-conversation.js` - Conversation handler
- [x] `public/js/kelly-2d-avatar.js` - 2D avatar component
- [x] `public/js/golden-v5-data-loader.js` - Lesson data loader
- [x] `api/tts.ts` - TTS API endpoint
- [x] `api/lessons/[dayNumber].ts` - Lesson API

**Data:**
- [x] Supabase schema (`docs/backend/SUPABASE_SCHEMA.md`)
- [x] Lesson content (365 days in `core_lessons` + `lesson_atoms`)
- [x] Persona definitions (`docs/GOLDEN_THREE_ARCHETYPES.md`)

**Auth:**
- [x] `daily-lesson-marketing/src/lib/auth.ts` - Auth system
- [x] `api/supabase-auth-webhook.ts` - User creation webhook

**Payment:**
- [x] `api/stripe-checkout.ts` - Checkout creation
- [x] `api/webhooks/stripe-revenue.ts` - Subscription webhooks

### Should Extract (Valuable)

**UI Components:**
- `public/js/kelly-performance-engine.js` - Performance orchestration
- `public/js/kelly-resilience.js` - Error handling
- `daily-lesson-marketing/src/components/KellyAvatar.astro` - Avatar component
- `daily-lesson-marketing/src/components/AgeAdaptiveDemo.astro` - Age demo

**Utilities:**
- `public/js/lib/supabase.js` - Supabase client singleton
- `api/lib/supabase.ts` - Server-side Supabase client
- `api/lib/static-lessons.ts` - Static lesson fallback

**Type Definitions:**
- `types/lesson-runtime.ts` - TypeScript interfaces

### Can Rebuild (Technical Debt)

**Legacy Components:**
- Unity WebGL integration (`public/unity/`) - Not used in production
- Old lesson player (`curious-kellly/lesson-player-v2/`) - Superseded
- Multiple HTML files in `public/` - Consolidate into single app

**Patterns to Improve:**
- Manual routing → Use proper router
- Vanilla JS classes → Consider React/Vue for better state management
- Scattered localStorage keys → Centralize in config

**Dependencies to Drop:**
- jQuery (if not needed)
- Legacy Bootstrap (if migrating to new design system)

---

## 6. Migration Risk Assessment

### Hard Dependencies

**Critical:**
1. **Supabase** - Database, Auth, Storage (cannot change without data migration)
2. **ElevenLabs** - TTS provider (voice ID tied to trained model)
3. **Stripe** - Payment processing (webhooks configured)
4. **Vercel** - Deployment platform (Edge Functions)

**Medium:**
1. **Astro** - Marketing site (can rebuild, but content migration needed)
2. **Static HTML** - Lesson player (easy to migrate, but large codebase)

### Data Migration Requirements

**User Accounts:**
- Export from Supabase `auth.users` + `public.users`
- Import to new system maintaining UUIDs
- **Risk:** Low (Supabase supports export)

**Progress Data:**
- Export `user_progress` table
- Map lesson IDs if schema changes
- **Risk:** Medium (need to verify lesson ID mapping)

**Content:**
- Export `core_lessons` + `lesson_atoms` (21,900+ rows)
- Verify JSONB structure compatibility
- **Risk:** Low (PostgreSQL export/import)

### Third-Party Integrations

**Need Reconfiguration:**
1. **Stripe Webhooks** - Update endpoint URLs
2. **Supabase Auth** - Update redirect URLs
3. **ElevenLabs** - API keys remain same
4. **Vercel** - New project setup

**No Change Needed:**
- ElevenLabs voice IDs
- Supabase database (if keeping same project)
- Stripe products/prices

### Environment Variables

**Must Transfer:**
```bash
PUBLIC_SUPABASE_URL=https://xxx.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJ...
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_MONTHLY=price_...
STRIPE_PRICE_ANNUAL=price_...
STRIPE_PRICE_LIFETIME=price_...
```

**Optional:**
- `CRM_CUSTOMERIO_SITE_ID` (email automation)
- `TURNSTILE_SECRET_KEY` (bot protection)
- `RECAPTCHA_SECRET_KEY` (alternative bot protection)

---

## 7. Migration Report (TypeScript Interface)

```typescript
interface MigrationReport {
  framework: {
    name: "Hybrid (Astro + Static HTML)";
    version: "Astro 5.0.0";
    router: "static";
  };
  
  stateManagement: [
    "Vanilla JS classes",
    "LocalStorage",
    "Supabase client state"
  ];
  
  styling: "Custom CSS/SCSS (no framework)";
  
  avatar: {
    format: "HD MP4 (primary), 2D PNG/WebP (fallback)";
    assetPaths: [
      "public/assets/kelly/production/avatars/",
      "public/assets/kelly/production/hero/",
      "Supabase Storage (hd_video_url)"
    ];
    ttsIntegration: {
      provider: "ElevenLabs";
      filePath: "public/js/kelly-audio.js";
    };
    lipSyncMethod: "Real-time amplitude analysis (Web Audio API)";
    states: [
      "idle",
      "listening",
      "thinking",
      "speaking",
      "celebrating",
      "curious",
      "explaining",
      "wisdom"
    ];
  };
  
  lessons: {
    dataSource: "Supabase (core_lessons + lesson_atoms)";
    schema: {
      core_lessons: {
        id: "UUID";
        day_number: "number (1-365)";
        topic: "string";
        universal_truth: "string";
      };
      lesson_atoms: {
        id: "UUID";
        core_lesson_id: "UUID";
        archetype: "string (12 archetypes)";
        phase: "string (Hook|Fact1|Fact2|Fact3|Wisdom)";
        content: "JSONB {script, options, kellyPose, kellyEmotion}";
        hd_video_url: "string (optional)";
      };
    };
    personaCount: 12;
    personaImplementation: "Database column (lesson_atoms.archetype)";
  };
  
  auth: {
    provider: "Supabase Auth";
    userSchema: {
      id: "UUID";
      email: "string";
      subscription_tier: "string";
      current_day: "number";
      streak_days: "number";
    };
    progressTracking: "Supabase user_progress table";
  };
  
  extractionFiles: {
    critical: [
      "public/js/kelly-audio.js",
      "public/js/kelly-lipsync.js",
      "public/js/kelly-conversation.js",
      "public/js/kelly-2d-avatar.js",
      "public/js/golden-v5-data-loader.js",
      "api/tts.ts",
      "api/lessons/[dayNumber].ts",
      "api/stripe-checkout.ts",
      "api/webhooks/stripe-revenue.ts",
      "daily-lesson-marketing/src/lib/auth.ts"
    ];
    valuable: [
      "public/js/kelly-performance-engine.js",
      "public/js/kelly-resilience.js",
      "api/lib/supabase.ts",
      "types/lesson-runtime.ts"
    ];
    rebuild: [
      "public/unity/ (Unity WebGL - not used)",
      "curious-kellly/lesson-player-v2/ (legacy)",
      "Multiple HTML files (consolidate)"
    ];
  };
  
  envVars: [
    "PUBLIC_SUPABASE_URL",
    "PUBLIC_SUPABASE_ANON_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_KELLY_VOICE_ID",
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "STRIPE_PRICE_MONTHLY",
    "STRIPE_PRICE_ANNUAL",
    "STRIPE_PRICE_LIFETIME"
  ];
  
  technicalDebt: [
    "Manual routing (should use router)",
    "Vanilla JS classes (consider React/Vue)",
    "Scattered localStorage keys",
    "Multiple HTML files (consolidate)",
    "Legacy Unity integration (unused)"
  ];
  
  migrationRisks: [
    "Supabase data export/import (low risk)",
    "Stripe webhook URL updates (medium risk)",
    "Lesson ID mapping if schema changes (medium risk)",
    "Environment variable transfer (low risk)",
    "Asset URL migration (low risk if using CDN)"
  ];
}
```

---

## 8. Recommended Migration Strategy

### Phase 1: Extract Core Systems (Week 1)
1. Extract avatar system (`kelly-audio.js`, `kelly-lipsync.js`, `kelly-conversation.js`)
2. Extract lesson loader (`golden-v5-data-loader.js`)
3. Extract auth system (`auth.ts`)
4. Extract payment system (`stripe-checkout.ts`, webhooks)

### Phase 2: Data Migration (Week 1-2)
1. Export Supabase data (users, lessons, progress)
2. Verify data integrity
3. Set up new Supabase project (if needed)
4. Import data to new system

### Phase 3: API Migration (Week 2)
1. Migrate Vercel Edge Functions
2. Update webhook URLs (Stripe, Supabase)
3. Test all API endpoints

### Phase 4: Frontend Migration (Week 2-3)
1. Build new macOS-style OS template
2. Integrate extracted systems
3. Migrate UI components
4. Test lesson player functionality

### Phase 5: Testing & Launch (Week 3-4)
1. End-to-end testing
2. User acceptance testing
3. Performance optimization
4. Gradual rollout

---

## 9. Key Files Reference

### Avatar System
- `public/js/kelly-audio.js` - TTS integration
- `public/js/kelly-lipsync.js` - Lip-sync engine
- `public/js/kelly-conversation.js` - Conversation handler
- `public/js/kelly-2d-avatar.js` - 2D avatar component
- `public/js/kelly-performance-engine.js` - Performance orchestration
- `public/js/kelly-personas.js` - Persona definitions and metadata

### Lesson System
- `public/js/golden-v5-data-loader.js` - Lesson data loader
- `api/lessons/[dayNumber].ts` - Lesson API endpoint
- `api/lib/static-lessons.ts` - Static lesson fallback

### Auth System
- `daily-lesson-marketing/src/lib/auth.ts` - Auth class
- `api/supabase-auth-webhook.ts` - User creation webhook
- `daily-lesson-marketing/src/pages/auth/callback.astro` - OAuth callback

### Payment System
- `api/stripe-checkout.ts` - Checkout creation
- `api/webhooks/stripe-revenue.ts` - Subscription webhooks
- `api/create-portal-session.ts` - Customer portal

### Configuration
- `docs/backend/SUPABASE_SCHEMA.md` - Database schema
- `docs/GOLDEN_THREE_ARCHETYPES.md` - Persona definitions
- `vercel.json` - Deployment config
- `daily-lesson-marketing/astro.config.mjs` - Astro config

---

---

## 10. Additional Notes & Corrections

### Astro Version
- **Actual Version:** Astro 5.0.0 (not 4.x as initially stated)
- **Source:** `daily-lesson-marketing/package.json`

### Age Adaptation Details
- **Age Buckets:** 6 ranges (2-5, 6-12, 13-17, 18-35, 36-60, 61+)
- **Implementation:** Inline in HTML files, not separate module
- **Debounce:** 400ms delay before content reload to prevent DB hammering
- **Storage:** `localStorage` key `kelly_age_setting`

### Persona System
- **File:** `public/js/kelly-personas.js` exists and exports `PERSONAS` array
- **CDN:** Uses Supabase Storage CDN for persona images
- **Metadata:** Each persona has id, name, icon, tagline, description, color, and image paths

### Missing Files Clarification
- **`age-adaptive.js`:** Does not exist as separate file - logic is inline
- **Age adaptation:** Handled in `learn.html`, `app.html`, and `app/script.js`

---

**End of Report**

