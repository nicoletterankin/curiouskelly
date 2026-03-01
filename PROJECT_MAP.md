# PROJECT MAP — Complete Reference for AI Tools

> **Purpose:** This file gives v0.app, Claude Code Desktop, Cursor, and any other AI tool
> the actual locations of every database, API, service, config, content file, and asset
> in this repository. If you're lost, start here.
>
> **Last updated:** 2026-02-08
> **Workspace root:** `c:\Users\user\UI-TARS-desktop`

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Supabase (Primary Database)](#2-supabase-primary-database)
3. [Database Tables (Complete List)](#3-database-tables-complete-list)
4. [Supabase Storage Buckets](#4-supabase-storage-buckets)
5. [Supabase Edge Functions](#5-supabase-edge-functions)
6. [Supabase Client Libraries (How to Connect)](#6-supabase-client-libraries)
7. [Environment Variables](#7-environment-variables)
8. [Project Structure (Top-Level)](#8-project-structure)
9. [The Two Active Apps](#9-the-two-active-apps)
10. [API Routes — Vercel Serverless Functions](#10-api-routes-vercel-serverless)
11. [API Routes — Next.js App Router](#11-api-routes-nextjs-app-router)
12. [Cloudflare Workers](#12-cloudflare-workers)
13. [Lesson Content & Data Files](#13-lesson-content-data-files)
14. [Audio Assets](#14-audio-assets)
15. [Video Assets](#15-video-assets)
16. [CDN URL Patterns](#16-cdn-url-patterns)
17. [Schema & ORM Files](#17-schema-orm-files)
18. [Scripts & Tooling](#18-scripts-tooling)
19. [Configuration Files](#19-configuration-files)
20. [Deployment](#20-deployment)
21. [Documentation Index](#21-documentation-index)
22. [Key NPM Scripts](#22-key-npm-scripts)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Product** | Curious Kelly / The Daily Lesson / Lesson of the Day PBC |
| **Repo name** | `daily-lesson-platform` (root package.json) |
| **Production URL** | `https://curiouskelly.com` |
| **Marketing site** | `https://www.thedailylesson.com` |
| **Supabase project ref** | `tvjalxxsyryjphkforjv` |
| **Supabase URL** | `https://tvjalxxsyryjphkforjv.supabase.co` |
| **Vercel Blob Storage** | `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com` |
| **Cloudflare R2** | `https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev` |
| **TTS custom domain** | `https://tts.curiouskelly.com` |

---

## 2. Supabase (Primary Database)

```
Project Ref:  tvjalxxsyryjphkforjv
API URL:      https://tvjalxxsyryjphkforjv.supabase.co
Anon Key:     eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI
Publishable:  sb_publishable_kLMlC14ckEp-XoL8RX5liw_cMdGs8lR
MCP Config:   .cursor/mcp.json  →  project_ref=tvjalxxsyryjphkforjv
```

**Service Role Key:** In `.env` file (never committed). Variable name: `SUPABASE_SERVICE_ROLE_KEY`

---

## 3. Database Tables (Complete List)

All tables live in the `public` schema. Here is every table as of 2026-02-08:

### Content Tables
| Table | Purpose |
|---|---|
| `core_lessons` | 365 daily lessons (day_number, topic, universal_truth, etc.) |
| `lesson_atoms` | ~21,915 content pieces (archetype-specific dialog) |
| `lesson_shards` | ~38,700 demographic variants (age/region/tone) |
| `lessons` | Legacy/alternate lesson storage |
| `content_atoms` | Content atom records |
| `content_generation` | Content generation tracking |
| `content_health` | Content health metrics |
| `content_history` | Content revision history |
| `content_validation_results` | Validation results |
| `lesson_age_hooks` | Age-specific lesson hooks |
| `lesson_scripts` | Kelly's speaking scripts per lesson |
| `lesson_translations` | Multilingual translations |
| `lesson_assets` | Lesson asset references |
| `archetype_dialog_templates` | Dialog templates by archetype |
| `curriculum_suggestions` | Curriculum improvement suggestions |
| `prompt_templates` | AI prompt templates |

### Kelly Video/Audio Tables
| Table | Purpose |
|---|---|
| `kelly_video_assets` | Video assets by day/phase |
| `kelly_motion_library` | Motion clips (~420 rows) |
| `kelly_assets` | General Kelly assets |
| `kelly_generated_assets` | AI-generated assets |
| `kelly_generation_jobs` | Generation job queue |
| `kelly_generation_runs` | Generation run records |
| `kelly_keys` | Kelly API keys |
| `kelly_lesson_assets` | Per-lesson Kelly assets |
| `kelly_production_assets` | Production-ready assets |
| `kelly_production_progress` | Production progress tracking |
| `kelly_prompts` | Kelly AI prompts |
| `kelly_prompt_suggestions` | Prompt suggestions |
| `kelly_factory_dashboard` | Factory dashboard data |
| `lipsync_alignments` | Lip sync alignment data |
| `lesson_video_generation_status` | Video generation status |

### User Tables
| Table | Purpose |
|---|---|
| `users` | User profiles (extends auth.users) |
| `users_with_age` | Users view with age calculation |
| `user_progress` | Lesson completion tracking |
| `user_learning_journey` | Learning journey tracking |
| `byok_keys` | Bring-your-own-key storage |
| `byok_providers` | BYOK provider configs |
| `push_tokens` | Push notification tokens |

### Business / Revenue Tables
| Table | Purpose |
|---|---|
| `affiliates` | Affiliate program members |
| `affiliate_applications` | Pending affiliate applications |
| `affiliate_payouts` | Payout records |
| `referrals` | Referral tracking |
| `referral_clicks` | Click tracking |
| `commission_tiers` | Commission tier definitions |
| `commission_transactions` | Commission transactions |
| `revenue_events` | Revenue event log |
| `payment_events` | Payment event log |
| `payouts` | Payout records |
| `enterprise_inquiries` | Enterprise leads |
| `newsletter_subscribers` | Email list |
| `contact_submissions` | Contact form submissions |
| `earnings_compliance_log` | Earnings compliance |
| `family_earnings_summary` | Family earnings rollup |
| `minor_earnings_ledger` | Minor (child) earnings |
| `financial_alerts` | Financial alert rules |
| `financial_snapshots` | Point-in-time financial snapshots |
| `generation_costs` | Cost tracking for AI generation |

### Analytics & Engagement Tables
| Table | Purpose |
|---|---|
| `analytics_events` | Event tracking |
| `happy_learner_events` | Positive learner events |
| `lesson_completions` | Completion records |
| `lesson_history` | Lesson view history |
| `lesson_impacts` | Impact measurements |
| `lesson_audits` | Lesson quality audits |
| `lesson_comments` | User comments on lessons |
| `student_feedback` | Student feedback records |
| `learner_insights` | Learner insight data |
| `learner_observations` | Observation records |
| `milestones` | User milestone tracking |

### Community Tables
| Table | Purpose |
|---|---|
| `community_contributions` | Community contributions |
| `commons_answers` | Commons Q&A answers |
| `commons_notes` | Commons shared notes |
| `commons_note_reactions` | Note reactions |
| `commons_proposals` | Community proposals |
| `commons_votes` | Voting records |
| `comment_replies` | Comment reply threads |
| `comment_votes` | Comment voting |
| `phase_comments` | Phase-specific comments |
| `visual_commons` | Shared visual assets |
| `visual_contexts` | Visual context data |
| `impact_classifications` | Impact classification taxonomy |

### Operations Tables
| Table | Purpose |
|---|---|
| `video_jobs` | Video generation job queue |
| `generation_queue` | General generation queue |
| `generation_status` | Generation status tracking |
| `improvement_queue` | Content improvement queue |
| `incoherencies` | Detected incoherencies |
| `slop_issue_types` | Content quality issue types |
| `audit_log` | System audit log |
| `heygen_performance_logs` | HeyGen API performance |
| `founder_notifications` | Founder alert notifications |
| `notification_ab_tests` | Notification A/B tests |
| `notification_copy` | Notification copy variants |
| `notification_log` | Notification delivery log |
| `notification_preferences` | User notification prefs |
| `notification_queue` | Notification send queue |

### Views (prefixed with `v_`)
| View | Purpose |
|---|---|
| `v_affiliate_performance` | Affiliate performance rollup |
| `v_contribution_summary` | Contribution summary |
| `v_contributions_by_provider` | Contributions grouped by provider |
| `v_current_mrr` | Current MRR calculation |
| `v_daily_revenue` | Daily revenue aggregation |
| `v_subscription_health` | Subscription health metrics |
| `v_user_cohorts` | User cohort analysis |

---

## 4. Supabase Storage Buckets

| Bucket | Public | Purpose |
|---|---|---|
| `kelly-videos` | Yes | Primary video assets (MP4) |
| `kelly-templates` | Yes | Template images, audio, personas |
| `lesson-visuals` | Yes | Infographics, backgrounds (WebP) |
| `images` | Yes | General image storage |
| `visuals` | Yes | Visual assets |
| `submissions` | No | Private user submissions |

**Base URL pattern:** `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/{bucket}/{path}`

**Video path pattern:** `kelly-videos/day-{dayNumber}/{archetype}/{phase}.mp4`
**Video production pattern:** `kelly-videos/production/day_{dayNumber}/day_{dayNumber}_{phase}_{archetype}_dynamic.mp4`
**Audio path pattern (in kelly-templates):** `kelly-templates/heygen/audio/day_{day}_{phase}_{timestamp}.mp3`

---

## 5. Supabase Edge Functions

All located in `supabase/functions/`:

| Function | Path |
|---|---|
| `feedback-complete` | `supabase/functions/feedback-complete/index.ts` |
| `feedback-heartbeat` | `supabase/functions/feedback-heartbeat/index.ts` |
| `feedback-vote` | `supabase/functions/feedback-vote/index.ts` |
| `get-lesson` | `supabase/functions/get-lesson/index.ts` |
| `get-progress` | `supabase/functions/get-progress/index.ts` |
| `loop-analyze` | `supabase/functions/loop-analyze/index.ts` |

---

## 6. Supabase Client Libraries

These are the files that create Supabase client connections. **Use these when you need database access.**

| File | Context | Type |
|---|---|---|
| `api/lib/supabase.ts` | **Primary** — Vercel serverless functions | Admin + Public client factory |
| `2_6_2026/lib/supabase/server.ts` | Next.js SSR (server components) | Server client |
| `2_6_2026/lib/supabase/client.ts` | Next.js browser (client components) | Browser client |
| `TEMPLATES/v0/lib/supabase.ts` | v0 template hooks & queries | Template reference |
| `daily-lesson-marketing/src/lib/supabase.ts` | Marketing site (Astro) | Server client |
| `public/js/lib/supabase.js` | Static public JS | Browser client |

**Environment variables used:**
- `PUBLIC_SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL` → `https://tvjalxxsyryjphkforjv.supabase.co`
- `PUBLIC_SUPABASE_ANON_KEY` or `NEXT_PUBLIC_SUPABASE_ANON_KEY` → the anon key above
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_SERVICE_KEY` → in `.env` (never committed)

---

## 7. Environment Variables

**Files containing env configs:**
| File | Purpose |
|---|---|
| `.env` | Root environment (gitignored) |
| `.env.local` | Local overrides (gitignored) |
| `.env.example` | Template with variable names |
| `daily-lesson-marketing/.env` | Marketing site env |
| `daily-lesson-marketing/.env.local` | Marketing local overrides |

**Required variables:**
```
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=<anon key>
NEXT_PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
SUPABASE_SERVICE_ROLE_KEY=<service role key>
ELEVENLABS_API_KEY=<elevenlabs key>
KELLY_VOICE_ID=pFZP5JQG7iQjIQuC4Bku
STRIPE_SECRET_KEY=<stripe secret>
STRIPE_PUBLISHABLE_KEY=<stripe publishable>
STRIPE_WEBHOOK_SECRET=<stripe webhook secret>
OPENAI_API_KEY=<openai key>
HEYGEN_API_KEY=<heygen key>
RESEND_API_KEY=<resend key>
FAL_KEY=<fal.ai key>
```

---

## 8. Project Structure

```
c:\Users\user\UI-TARS-desktop\
│
├── 2_6_2026/                  ← ACTIVE Next.js App (the main product)
│   ├── app/                   ← Next.js App Router (pages + API routes)
│   │   ├── api/               ← ~200+ API route handlers
│   │   ├── page.tsx           ← Main page
│   │   └── ...
│   ├── components/            ← React components (Kelly, lessons, UI)
│   ├── hooks/                 ← Custom React hooks
│   ├── lib/                   ← Utilities, Supabase clients, types
│   └── package.json           ← (if exists, otherwise uses root)
│
├── api/                       ← Vercel Serverless Functions (~110 files)
│   ├── admin/                 ← Admin operations
│   ├── audio/                 ← Audio generation (batch.ts, generate.ts)
│   ├── cfo/                   ← Financial metrics & payouts
│   ├── commons/               ← Community features
│   ├── cron/                  ← Scheduled jobs
│   ├── day/                   ← Day content endpoint
│   ├── email/                 ← Email operations
│   ├── family/                ← Family features
│   ├── feedback/              ← User feedback
│   ├── lesson/                ← Lesson endpoints
│   ├── lessons/               ← Lessons list endpoints
│   ├── lib/                   ← Shared libraries (supabase.ts, rate-limit.ts)
│   ├── notifications/         ← Push notifications
│   ├── pipeline/              ← Content pipeline
│   ├── referral/              ← Referral system
│   ├── video-jobs/            ← Video job queue
│   ├── visual/                ← Visual generation
│   ├── webhooks/              ← External webhooks (Stripe, HeyGen, FAL)
│   └── *.ts                   ← Individual endpoints
│
├── public/                    ← Static assets served by Vercel
│   ├── lessons/               ← 365+ lesson JSON files (day-1.json to day-365.json)
│   │   ├── year2/             ← Year 2 lessons
│   │   ├── special/           ← Special lessons
│   │   └── audio/             ← Audio metadata
│   ├── assets/kelly/          ← Kelly avatar assets
│   ├── js/lib/                ← Client-side JS (supabase.js)
│   ├── ziggurat/              ← Ziggurat investor app assets
│   └── data/                  ← Static data files
│
├── content/                   ← Content source files
│   ├── translations/          ← Multilingual content (es/, pt/)
│   ├── email-summary-video/   ← 349 email summary manifests
│   ├── visual-plans/          ← Visual planning files
│   └── gold-standard/         ← Gold standard lesson references
│
├── lessons/                   ← Lesson metadata & curriculum
│   ├── 365_day_calendar.json  ← Master calendar
│   ├── year1-foundations/     ← Year 1 monthly curricula
│   ├── year2-ai-fluency/     ← Year 2 monthly curricula
│   └── manifests/             ← Lesson manifests
│
├── scripts/                   ← Build & generation scripts
│   ├── lesson-factory/        ← Unified lesson factory
│   ├── kelly-visual-identity/ ← Visual identity tools
│   └── *.ts / *.js            ← Individual scripts
│
├── lib/                       ← Shared TypeScript libraries
│   └── engines/               ← Engine definitions
│
├── supabase/                  ← Supabase project config
│   └── functions/             ← 6 Edge Functions
│
├── prisma/                    ← Prisma ORM
│   ├── schema.prisma          ← Schema definition
│   └── migrations/            ← Migration SQL
│
├── daily-lesson-marketing/    ← Astro marketing site (thedailylesson.com)
│   ├── src/                   ← Astro source
│   └── astro.config.mjs       ← Config (port 4321)
│
├── infrastructure/            ← Infrastructure configs
│   └── cloudflare/            ← Cloudflare Workers
│       ├── lessons-api-worker/ ← Lesson API worker
│       └── tts-worker/        ← TTS worker (tts.curiouskelly.com)
│
├── schemas/                   ← JSON schemas
│   └── lesson-dna-v3-schema.json
│
├── generated-audio/           ← Generated audio manifests
├── generated-videos/          ← Generated video manifests & metadata
├── mobile-app/                ← React Native mobile app
├── desktop-app/               ← Electron desktop app
├── capacitor/                 ← Capacitor mobile wrapper
├── daa-app/                   ← DAA Next.js app
├── TEMPLATES/v0/              ← v0.app template reference
├── docs/                      ← Documentation
├── _archive/                  ← Archived code
├── _archived_legacy/          ← Legacy archived code
└── _quarantine/               ← Quarantined files
```

---

## 9. The Two Active Apps

### App 1: Next.js App (Main Product) — `2_6_2026/`
- **Framework:** Next.js (App Router)
- **Entry:** `2_6_2026/app/page.tsx`
- **API Routes:** `2_6_2026/app/api/` (~200+ route.ts files)
- **Components:** `2_6_2026/components/`
- **Hooks:** `2_6_2026/hooks/`
- **Supabase client:** `2_6_2026/lib/supabase/server.ts` and `client.ts`
- **Config:** Root `next.config.mjs`

### App 2: Vercel Static + Serverless API — Root `api/` + `public/`
- **Static files:** `public/` directory
- **API Routes:** `api/` (~110 files, Vercel serverless)
- **Config:** `vercel.json`
- **Supabase client:** `api/lib/supabase.ts`

### Also: Marketing Site — `daily-lesson-marketing/`
- **Framework:** Astro
- **Config:** `daily-lesson-marketing/astro.config.mjs`
- **Dev port:** 4321
- **Production:** `https://www.thedailylesson.com`

---

## 10. API Routes — Vercel Serverless Functions (`api/`)

Every `.ts` file in the `api/` directory is a Vercel serverless function accessible at `https://curiouskelly.com/api/{path}`.

### Core Endpoints
| Endpoint | File | Purpose |
|---|---|---|
| `/api/health` | `api/health.ts` | Health check |
| `/api/ping` | `api/ping.ts` | Simple ping |
| `/api/time` | `api/time.ts` | Server time |
| `/api/lessons` | `api/lessons.ts` | Get lessons list |
| `/api/day/[number]` | `api/day/[number].ts` | Get lesson by day number |
| `/api/lesson/[day]/[phase]` | `api/lesson/[day]/[phase].ts` | Get specific lesson phase |
| `/api/lesson-complete` | `api/lesson-complete.ts` | Mark lesson complete |
| `/api/lesson-history` | `api/lesson-history.ts` | Get lesson history |

### Audio
| Endpoint | File |
|---|---|
| `/api/audio/generate` | `api/audio/generate.ts` |
| `/api/audio/batch` | `api/audio/batch.ts` |
| `/api/tts` | `api/tts.ts` |

### Video
| Endpoint | File |
|---|---|
| `/api/kelly-video` | `api/kelly-video.ts` |
| `/api/kelly-jobs` | `api/kelly-jobs.ts` |
| `/api/video/url` | `api/video/url.ts` |
| `/api/video-jobs/submit` | `api/video-jobs/submit.ts` |
| `/api/video-jobs/status` | `api/video-jobs/status.ts` |
| `/api/video-jobs/approve` | `api/video-jobs/approve.ts` |
| `/api/video-jobs/queue` | `api/video-jobs/queue.ts` |
| `/api/video-jobs/compare` | `api/video-jobs/compare.ts` |

### Payments & Billing
| Endpoint | File |
|---|---|
| `/api/create-checkout` | `api/create-checkout.ts` |
| `/api/stripe-checkout` | `api/stripe-checkout.ts` |
| `/api/subscription-status` | `api/subscription-status.ts` |
| `/api/create-portal-session` | `api/create-portal-session.ts` |
| `/api/webhooks/stripe-revenue` | `api/webhooks/stripe-revenue.ts` |

### Email
| Endpoint | File |
|---|---|
| `/api/email/daily-content` | `api/email/daily-content.ts` |
| `/api/email/send-daily-duo` | `api/email/send-daily-duo.ts` |
| `/api/send-welcome-email` | `api/send-welcome-email.ts` |
| `/api/subscribe-email` | `api/subscribe-email.ts` |

### Referrals & Affiliates
| Endpoint | File |
|---|---|
| `/api/referral/track` | `api/referral/track.ts` |
| `/api/referral/convert` | `api/referral/convert.ts` |
| `/api/referral/lookup` | `api/referral/lookup.ts` |
| `/api/cfo/affiliate-payouts` | `api/cfo/affiliate-payouts.ts` |
| `/api/cfo/metrics` | `api/cfo/metrics.ts` |

### Cron Jobs
| Endpoint | File |
|---|---|
| `/api/cron/daily-lesson` | `api/cron/daily-lesson.ts` |
| `/api/cron/streak-check` | `api/cron/streak-check.ts` |
| `/api/cron/video-jobs` | `api/cron/video-jobs.ts` |
| `/api/cron/email-digest` | `api/cron/email-digest.ts` |
| `/api/cron/heygen-monitor` | `api/cron/heygen-monitor.ts` |

### Webhooks
| Endpoint | File |
|---|---|
| `/api/webhooks/stripe-revenue` | `api/webhooks/stripe-revenue.ts` |
| `/api/webhooks/heygen` | `api/webhooks/heygen.ts` |
| `/api/webhooks/fal` | `api/webhooks/fal.ts` |
| `/api/webhooks/sync-so` | `api/webhooks/sync-so.ts` |

---

## 11. API Routes — Next.js App Router (`2_6_2026/app/api/`)

Every `route.ts` file in `2_6_2026/app/api/` is a Next.js route handler. Key categories:

### HeyGen Video Pipeline (largest section, ~50 routes)
All under `2_6_2026/app/api/heygen/`. Key ones:
- `generate/route.ts` — Generate a single video
- `batch-generate/route.ts` — Batch generation
- `status/route.ts` — Check generation status
- `sync-completed/route.ts` — Sync completed videos
- `pipeline/route.ts` — Full pipeline control
- `credits/route.ts` — Check HeyGen credits

### FAL.ai Lipsync
All under `2_6_2026/app/api/fal/`:
- `lipsync/route.ts` — Single lipsync job
- `batch-day/route.ts` — Batch lipsync for a day
- `mass-lipsync/route.ts` — Mass lipsync

### Admin
All under `2_6_2026/app/api/admin/`:
- `day-status/route.ts` — Day production status
- `generate-day-34/route.ts` — Generate specific day
- `regenerate-day/route.ts` — Regenerate a day
- `video-stats/route.ts` — Video statistics

### Lesson & Content
- `2_6_2026/app/api/lesson/[day]/route.ts` — Get lesson by day
- `2_6_2026/app/api/lesson/today/route.ts` — Get today's lesson
- `2_6_2026/app/api/lessons/by-day/route.ts` — Lessons by day
- `2_6_2026/app/api/sync/core-lessons/route.ts` — Sync core lessons

### Auth
- `2_6_2026/app/api/auth/login/route.ts`
- `2_6_2026/app/api/auth/register/route.ts`
- `2_6_2026/app/api/auth/me/route.ts`
- `2_6_2026/app/api/auth/logout/route.ts`
- `2_6_2026/app/auth/callback/route.ts` — OAuth callback

### Audio
- `2_6_2026/app/api/audio/generate/route.ts`
- `2_6_2026/app/api/audio/batch/route.ts`
- `2_6_2026/app/api/audio/tts/route.ts`
- `2_6_2026/app/api/audio/generate-language/route.ts`

### Payments
- `2_6_2026/app/api/checkout/route.ts`
- `2_6_2026/app/api/subscription/route.ts`
- `2_6_2026/app/api/webhooks/stripe/route.ts`

### Learner Progress
- `2_6_2026/app/api/learner/progress/route.ts`
- `2_6_2026/app/api/learner/streak/route.ts`
- `2_6_2026/app/api/learner/phase-complete/route.ts`

---

## 12. Cloudflare Workers

### Lessons API Worker
- **Path:** `infrastructure/cloudflare/lessons-api-worker/`
- **Entry:** `src/index.js`
- **Config:** `wrangler.toml`
- **D1 Database:** `kelly-lessons-mirror`
- **Routes:** `/health`, `/lesson/:day`, `/lessons`, `/sync/status`

### TTS Worker
- **Path:** `infrastructure/cloudflare/tts-worker/`
- **Entry:** `src/index.js`
- **Config:** `wrangler.toml`
- **R2 Bucket:** `curious-kelly-audio-cache`
- **Custom Domain:** `tts.curiouskelly.com`
- **Routes:** `POST /tts`

### D1 Schema Files
- `lessons/schema.sql`
- `sql/d1-schema.sql`

---

## 13. Lesson Content & Data Files

### Static Lesson JSONs (the actual lesson data served to users)
```
public/lessons/day-1.json through day-365.json     ← 365 Year 1 lessons
public/lessons/year2/*.json                         ← 365 Year 2 lessons
public/lessons/special/*.json                       ← 20 special lessons
```

### Curriculum Structure
```
lessons/365_day_calendar.json                       ← Master 365-day calendar
lessons/year1-foundations/*_curriculum.json          ← 12 monthly curriculum files
lessons/year2-ai-fluency/*_curriculum.json          ← 12 monthly curriculum files
```

### Generated Content
```
generated/                                          ← 366 generated JSON files
local/                                              ← 202 local JSON files
curious-kellly/golden-v2/deploy/                    ← 1,244 deploy-ready JSON files
```

### Translations
```
content/translations/es/day-*.json                  ← 50 Spanish translations
content/translations/pt/day-*.json                  ← 50 Portuguese translations
public/locales/en/lessons.json                      ← English locale strings
public/locales/es/lessons.json                      ← Spanish locale strings
public/locales/pt/lessons.json                      ← Portuguese locale strings
```

### Metadata & Manifests
```
public/data/lessons-metadata.json                   ← Lesson metadata index
schemas/lesson-dna-v3-schema.json                   ← Lesson DNA schema
lessons/manifests/*-manifest.json                   ← 10 lesson manifests
content/email-summary-video/day-*-summary-manifest.json  ← 349 email manifests
```

---

## 14. Audio Assets

### Supabase Storage
- **Bucket:** `lesson-audio` (referenced in code but may need creation)
- **Bucket:** `kelly-templates` (contains audio at `heygen/audio/`)
- **Path pattern:** `day-{dayNumber}/{language}/{archetype}/{phase}.mp3`

### Vercel Blob Storage
- **URL pattern:** `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/audio/2026/{lang}/day-{day}/{phase}-age{age}.mp3`

### Local Directories
```
generated-audio/                                    ← Generated audio manifests
public/lessons/audio/                               ← Audio metadata per lesson
```

### Audio Generation Scripts
| Script | Purpose |
|---|---|
| `scripts/generate-audio-only.ts` | Generate audio files |
| `scripts/generate-full-lesson-audio.ts` | Full lesson audio |
| `scripts/generate-day-audio-elevenlabs.ts` | ElevenLabs audio |
| `api/audio/generate.ts` | Audio generation API |
| `api/audio/batch.ts` | Batch audio generation |

---

## 15. Video Assets

### Supabase Storage
- **Bucket:** `kelly-videos` (public)
- **Path pattern:** `day-{dayNumber}/{archetype}/{phase}.mp4`
- **Production path:** `production/day_{dayNumber}/day_{dayNumber}_{phase}_{archetype}_dynamic.mp4`

### Vercel Blob Storage
- **URL pattern:** `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/{age}-{archetype}.mp4`

### Cloudflare R2
- **URL pattern:** `https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/ziggurat/...`

### Local Directories
```
generated-videos/                                   ← Video manifests & metadata
generated-videos/golden-lesson-hd/                  ← HD lesson videos
generated-videos/heygen-production/                 ← HeyGen production videos
generated-videos/sync-labs-production/              ← Sync Labs videos
kelly-pipeline/videos/                              ← Pipeline videos
```

### Video Generation Scripts
| Script | Purpose |
|---|---|
| `scripts/generate-day-videos-heygen.ts` | HeyGen video generation |
| `scripts/generate-kelly-videos.ts` | Kelly video generation |
| `scripts/batch-lipsync-pipeline.ts` | Batch lipsync |
| `2_6_2026/app/api/heygen/generate/route.ts` | HeyGen API route |
| `2_6_2026/app/api/fal/lipsync/route.ts` | FAL lipsync API |

---

## 16. CDN URL Patterns

| Provider | Base URL | Usage |
|---|---|---|
| **Supabase Storage** | `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/{bucket}/{path}` | Primary video/image/audio storage |
| **Vercel Blob** | `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/{path}` | Audio and base video files |
| **Cloudflare R2** | `https://pub-ae8248f6a4f44c61a5de0d2f19b8dcd1.r2.dev/{path}` | Ziggurat assets |
| **HeyGen** | `https://files.heygen.ai/video/v1/{uuid}/{uuid}.mp4` | Generated video source |
| **Google Cloud** | `https://storage.googleapis.com/curious-kelly-assets/kelly/...` | Kelly character assets |
| **TTS CDN** | `https://tts.curiouskelly.com` | Cloudflare TTS worker |

---

## 17. Schema & ORM Files

### Prisma
```
prisma/schema.prisma                                ← Full Prisma schema
prisma/migrations/                                  ← Migration SQL files
```
- **Provider:** PostgreSQL
- **Multi-schema:** `auth` + `public`
- **Connection:** `DATABASE_URL` env var
- **Key models:** `core_lessons`, `lesson_atoms`, `lesson_shards`, `users`, `affiliates`, `referrals`

### SQL Schema Files
```
supabase-schema.sql                                 ← Full Supabase SQL schema
supabase-schema-clean.sql                           ← Clean version
lessons/schema.sql                                  ← Cloudflare D1 schema
sql/d1-schema.sql                                   ← D1 mirror schema
scripts/migrations/001_core_schema.sql              ← Core migration
```

### Schema Documentation
```
docs/backend/SUPABASE_SCHEMA.md                     ← Production schema reference
docs/V0_SCHEMA_CONFIRMATION.md                      ← v0 schema confirmation
docs/architecture/LESSON_SCHEMA_V5_CANONICAL.md     ← Canonical lesson schema
docs/phasedna/PHASEDNA_V2_SCHEMA_SUMMARY.md        ← PhaseDNA schema summary
```

---

## 18. Scripts & Tooling

### Lesson Factory (Primary content pipeline)
```
scripts/lesson-factory/unified-factory.ts           ← Main factory script
scripts/lesson-factory/preflight-check.ts           ← Pre-flight validation
```

### Content Generation
```
scripts/bulk-seed-database.ts                       ← Seed database with lessons
scripts/bulk-load-all-days.ts                       ← Load all days
scripts/generate-audio-only.ts                      ← Audio generation
scripts/generate-full-lesson-audio.ts               ← Full lesson audio
scripts/generate-day-audio-elevenlabs.ts            ← ElevenLabs audio
scripts/generate-lesson-visuals.ts                  ← Visual generation
scripts/generate-all-365-visuals.ts                 ← All 365 visuals
scripts/generate-day-videos-heygen.ts               ← HeyGen videos
scripts/eval-lesson-quality.ts                      ← Quality evaluation
```

### Database & Sync
```
scripts/sync-day34-to-db.cjs                        ← Sync specific day to DB
scripts/full-status-check.cjs                       ← Full status check
scripts/sync-all-processing.cjs                     ← Sync all processing
```

### Quality Tools
```
scripts/slop-detector.ts                            ← Content quality checker
scripts/fix-headlines.ts                            ← Headline fixer
scripts/update-atom-visual-urls.ts                  ← Visual URL updater
```

---

## 19. Configuration Files

### Root Level
| File | Purpose |
|---|---|
| `package.json` | Root monorepo config (`daily-lesson-platform`) |
| `tsconfig.json` | Root TypeScript config |
| `vercel.json` | Vercel deployment config |
| `next.config.mjs` | Next.js config |
| `wrangler.toml` | Root Cloudflare Workers config |
| `.env` | Environment variables (gitignored) |
| `.env.example` | Env template |
| `CLAUDE.md` | AI assistant operating rules |
| `PROJECT_MAP.md` | This file |

### Cursor IDE
```
.cursor/mcp.json                                    ← MCP server config (Supabase)
.cursor/worktrees.json                              ← Worktree setup
.cursor/skills/curious-kelly-founder/SKILL.md       ← Founder skill
```

### Deployment
```
vercel.json                                         ← Vercel config
infrastructure/cloudflare/lessons-api-worker/wrangler.toml
infrastructure/cloudflare/tts-worker/wrangler.toml
deployment/setup-cloud.sh                           ← Cloud setup script
deployment/cloudrun/Dockerfile                      ← Docker config
deployment/cloudrun/cloudbuild.yaml                 ← Cloud Build config
capacitor/capacitor.config.ts                       ← Mobile app config (com.curiouskelly.app)
```

---

## 20. Deployment

| Platform | What | Config |
|---|---|---|
| **Vercel** | Main app + API + static | `vercel.json` |
| **Vercel** | Next.js app (`2_6_2026/`) | `next.config.mjs` |
| **Cloudflare Workers** | Lessons API mirror | `infrastructure/cloudflare/lessons-api-worker/wrangler.toml` |
| **Cloudflare Workers** | TTS service | `infrastructure/cloudflare/tts-worker/wrangler.toml` |
| **Cloudflare D1** | Lesson data mirror | `lessons/schema.sql` |
| **Cloudflare R2** | Asset storage | `tts-worker/wrangler.toml` (bucket: `curious-kelly-audio-cache`) |
| **Supabase** | Database + Auth + Storage + Edge Functions | Project ref: `tvjalxxsyryjphkforjv` |
| **Capacitor** | iOS/Android wrapper | `capacitor/capacitor.config.ts` |
| **Google Cloud Run** | Legacy gateway/classroom (archived) | `deployment/cloudrun/` |

---

## 21. Documentation Index

### Architecture & Schema
```
docs/backend/SUPABASE_SCHEMA.md
docs/backend/SUPABASE_MCP_SETUP.md
docs/architecture/LESSON_SCHEMA_V5_CANONICAL.md
docs/V0_SCHEMA_CONFIRMATION.md
docs/phasedna/PHASEDNA_V2_SCHEMA_SUMMARY.md
```

### Plans & Strategy
```
CURIOUS_KELLLY_EXECUTION_PLAN.md
CURIOUS_KELLLY_INDEX.md
TECHNICAL_ALIGNMENT_MATRIX.md
BUILD_PLAN.md
MIGRATION_PLAN.md
START_HERE.md
```

### Deployment
```
docs/deployment/DEPLOYMENT_CHECKLIST.md
docs/deployment/VERCEL_SETUP_GUIDE.md
docs/deployment/CLOUDFLARE_PAGES_SETUP.md
docs/deployment/DEPLOYMENT_ARCHITECTURE.md
```

### Social Media & Brand
```
docs/social-media/SOCIAL_MEDIA_INDEX.md
docs/social-media/SOCIAL_MEDIA_STRATEGY.md
docs/social-media/SOCIAL_MEDIA_BRAND_GUIDELINES.md
docs/social-media/LOGO_DECISION.md
docs/brand/FORBIDDEN_WORD_FREE.md
```

### Billing & Business
```
docs/billing/GLOBAL_ROADMAP.md
docs/reinmaker/API_OVERVIEW.md
docs/web/SITE_MAP.md
```

### Trust & Safety
```
docs/trust-safety/TRUST_AND_SAFETY_INDEX.md
docs/trust-safety/SIMULATED_SOCIAL_CONTENT.md
docs/trust-safety/USER_CONTROLS.md
```

---

## 22. Key NPM Scripts

```bash
# Development
npm run serve              # Serve public/ on port 3000
npm run dev                # Run dev stack (gateway + classroom)

# Testing
npm run test               # Unit + integration tests
npm run eval               # Run all evals
npm run test:simulation    # Learner simulation test

# Content Pipeline
npm run factory:day -- 34  # Generate content for day 34
npm run factory:range -- --from 1 --to 365  # Generate range
npm run factory:check      # Pre-flight check

# Audio
npm run headlines:preview  # Preview headline fixes
npm run headlines:fix      # Fix all headlines

# Visuals
npm run visuals:dry-run    # Dry run visual generation
npm run visuals:day -- 34  # Generate visuals for day 34
npm run visuals:missing    # Generate missing visuals

# Quality
npm run slop:detect        # Detect content quality issues
npm run slop:report        # Quality report only
npm run audit:lessons      # Audit all lessons

# Database
npm run prisma:generate    # Generate Prisma client

# Deployment
npm run deploy             # Quick deploy script
npm run build:seo          # Build SEO files
```

---

## Quick Start for AI Tools

**To query the database:**
```typescript
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  process.env.SUPABASE_SERVICE_ROLE_KEY // or anon key for public access
)
// Example: Get today's lesson
const { data } = await supabase.from('core_lessons').select('*').eq('day_number', 34).single()
```

**To get a lesson JSON file:**
```
File: public/lessons/day-34.json
URL:  https://curiouskelly.com/lessons/day-34.json
```

**To get a Kelly video:**
```
URL: https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-videos/day-34/explorer/hook_main.mp4
```

**To get audio:**
```
URL: https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/audio/2026/en/day-34/hook-age8.mp3
```

---

*This file is the single source of truth for locating anything in this repository. Keep it updated when adding new services, tables, or major directories.*
