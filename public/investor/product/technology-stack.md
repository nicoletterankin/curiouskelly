# Technology Stack

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Web App    │  │   iOS App    │  │    Android App       │  │
│  │   (Astro)    │  │ (React Native)│  │   (React Native)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│                         API LAYER                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Vercel Edge Functions                       │   │
│  │         (Auth, Lessons, Progress, Billing)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Supabase   │  │  Cloudflare  │  │       Stripe         │  │
│  │  (Postgres)  │  │    (CDN)     │  │     (Payments)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│                      AI/ML LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  ElevenLabs  │  │   HeyGen     │  │   Local Inference    │  │
│  │   (Voice)    │  │  (Lip Sync)  │  │    (RTX 5090)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Frontend Stack

### Web Application

| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Astro** | Static site generation | Fast, SEO-friendly, modern |
| **React** | Interactive components | Ecosystem, hiring pool |
| **TypeScript** | Type safety | Reduces bugs, better DX |
| **Tailwind CSS** | Styling | Rapid development |

**Performance Targets:**
- First Contentful Paint: <1.5s
- Time to Interactive: <3s
- Lighthouse Score: 95+

### Mobile Applications

| Technology | Platform | Status |
|-----------|----------|--------|
| **React Native** | iOS & Android | In Development |
| **Expo** | Build toolchain | Configured |
| **React Navigation** | Routing | Implemented |

**Key Decisions:**
- Cross-platform to maximize reach
- Offline-first architecture
- Native audio/video playback
- Push notifications for habit formation

---

## Backend Stack

### Supabase (Primary Database)

**Why Supabase:**
- Postgres reliability + modern DX
- Built-in auth, realtime, storage
- Edge functions for serverless
- Self-hostable if needed

**Schema Highlights:**

```sql
-- Core tables
users              -- Auth, preferences, subscription
lessons            -- 365+ lesson definitions
user_progress      -- Completion tracking
streaks            -- Habit formation data

-- Indexes optimized for:
- Daily lesson lookup (user + date)
- Progress queries (user + lesson range)
- Analytics aggregation
```

### Vercel (Deployment & Edge)

**Why Vercel:**
- Zero-config deployment
- Edge functions (low latency)
- Preview deployments (fast iteration)
- Excellent Astro support

**Edge Functions:**
- `/api/lessons` - Lesson delivery
- `/api/progress` - Progress tracking
- `/api/auth` - Authentication
- `/api/billing` - Stripe integration

### Cloudflare (CDN & Protection)

**Services Used:**
- CDN for static assets
- DDoS protection
- SSL/TLS termination
- R2 storage for media

---

## AI/ML Pipeline

### Voice Synthesis (ElevenLabs)

```
Script → ElevenLabs API → Audio File → Cache
         │
         └── Custom Kelly voice model
             (trained on 60+ min samples)
```

**Capabilities:**
- Human-quality speech synthesis
- Emotion control (curious, encouraging, etc.)
- Multi-language support
- <2s generation latency

**Cost:** ~$0.30/1000 characters

### Lip Sync (HeyGen / MuseTalk)

```
Audio + Base Video → Lip Sync → Final Video → CDN
                      │
                      ├── HeyGen (cloud, high quality)
                      └── MuseTalk (local, fast iteration)
```

**Current Status:**
- 87 base expression videos uploaded
- Lip sync pipeline in development
- Target: 1,825 lesson videos (5 years × 365)

### Local Inference (Future)

**Hardware:** NVIDIA RTX 5090 (24GB VRAM)

**Use Cases:**
- Real-time conversation mode
- On-device privacy
- Cost reduction at scale
- Faster iteration

**Stack:**
- Ollama for LLM serving
- Whisper for speech-to-text
- Custom LoRA for Kelly personality

---

## Content Pipeline

### Phase DNA System

Every lesson encoded as structured JSON:

```json
{
  "lessonId": "day-001",
  "topic": "What is AI?",
  "phases": [
    {
      "id": "welcome",
      "duration": 30,
      "script": "Good morning! I'm Kelly...",
      "voiceEmotion": "excited"
    },
    {
      "id": "teach_1",
      "duration": 60,
      "script": "Let me ask you something...",
      "interaction": "multiple_choice",
      "options": ["Yes", "No", "Not sure"]
    }
  ],
  "translations": {
    "es": { /* Spanish version */ },
    "fr": { /* French version */ }
  }
}
```

**Benefits:**
- Deterministic content
- Easy localization
- A/B testing built-in
- Version control

### Content Generation Workflow

```
Curriculum Design → Script Writing → Voice Generation → 
Video Generation → QA Review → Deploy
```

---

## Infrastructure Costs

### Current (Pre-Scale)

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| Supabase | $25 | Pro tier |
| Vercel | $20 | Pro tier |
| ElevenLabs | $99 | Creator tier |
| HeyGen | $89 | Creator tier |
| Cloudflare | $0 | Free tier sufficient |
| **Total** | **$233** | |

### Projected (10K Users)

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| Supabase | $75 | Scale tier |
| Vercel | $40 | Team tier |
| ElevenLabs | $330 | Pro tier |
| HeyGen | $179 | Business tier |
| Cloudflare | $20 | Pro tier |
| **Total** | **$644** | |

### Unit Economics

| Metric | Value |
|--------|-------|
| Infrastructure cost per user | ~$0.06/month |
| Gross margin (at $15/user) | 99.6% |

---

## Security & Privacy

### Data Protection

- All data encrypted at rest (AES-256)
- TLS 1.3 for all traffic
- No PII shared with third parties
- GDPR/CCPA compliant architecture

### Authentication

- Supabase Auth (email, social)
- Row Level Security on all tables
- JWT tokens (short-lived)
- Refresh token rotation

### Privacy-First Design

- Local inference option for sensitive data
- Data export (user right)
- Account deletion (user right)
- Minimal data collection

---

## Scalability Plan

### Phase 1: 0-10K Users
- Current stack sufficient
- Optimize queries
- CDN for static content

### Phase 2: 10K-100K Users
- Read replicas for Supabase
- Video CDN optimization
- Edge caching for lessons

### Phase 3: 100K-1M Users
- Multi-region deployment
- Custom CDN solution
- Database sharding if needed

---

## Development Practices

| Practice | Tools |
|---------|-------|
| Version Control | Git + GitHub |
| CI/CD | GitHub Actions |
| Code Review | Required for all PRs |
| Testing | Vitest, Playwright |
| Monitoring | Vercel Analytics, Supabase Dashboard |
| Error Tracking | Planned: Sentry |

---

## Key Technical Decisions

### Why Not Mobile-First Native?
- Cross-platform reaches more users faster
- React Native quality is sufficient
- Team expertise in web technologies
- Native can come later if needed

### Why Supabase Over Firebase?
- Postgres > proprietary database
- Better pricing at scale
- Self-hosting option
- SQL familiarity

### Why Pre-Generated Content Over Live AI?
- Consistency (Kelly is always Kelly)
- Quality control
- Cost predictability
- No latency issues

### Why Edge Functions Over Traditional Backend?
- Global low latency
- Scales automatically
- No server management
- Cost-efficient
