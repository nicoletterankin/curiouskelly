# SYSTEM DIRECTIVE: Vercel Edge-First Performance Optimization
**Date:** December 23, 2025  
**Status:** ACTIVE DIRECTIVE  
**Priority:** CRITICAL - Maximum Performance Architecture

---

## 🎯 MISSION

Transform Curious Kelly into a **world-class edge-optimized learning platform** using Vercel's edge-native capabilities. Achieve **<20ms TTFB globally**, **zero buffering**, and **full offline support** for deep lesson use cases.

---

## 📋 DEEP USE CASE UNDERSTANDING

### Lesson Complexity (LOCKED - DO NOT SIMPLIFY)

Each lesson is a **multi-dimensional media experience**:

```
Day N Lesson Structure:
├── Core Lesson (1)
│   ├── Topic, headline, universal truth
│   └── Metadata (emoji, category)
│
├── Atoms (7 phases × 12 archetypes × 3 languages = 252 variants)
│   ├── Hook phase
│   │   ├── Script (text)
│   │   ├── HD Video (1080p lipsync, ~5-10MB)
│   │   ├── Audio (ElevenLabs MP3, ~500KB)
│   │   ├── Visual (infographic, ~2MB)
│   │   └── Options (2 choices + responses)
│   ├── Question phase (same structure)
│   ├── Context phase (same structure)
│   ├── Choice phase (same structure)
│   ├── Reflection phase (same structure)
│   ├── Wisdom phase (same structure)
│   └── Action phase (same structure)
│
├── Shards (age variants)
│   └── 6 age buckets × personalized content
│
└── Grow Track (optional)
    └── AI fluency content
```

**Per-Lesson Asset Count:**
- **Videos:** 7 phases × 3 archetypes (primary) = 21 HD videos (~150MB total)
- **Audio:** 7 phases × 12 archetypes × 3 languages = 252 audio files (~125MB total)
- **Visuals:** 5 infographics + 14 option cards = 19 images (~40MB total)
- **Data:** Lesson JSON + atoms + shards = ~500KB

**Total per lesson:** ~315MB of assets

### Critical Requirements

1. **Zero Buffering:** Users expect instant phase transitions
2. **Offline-First:** Lessons must work without internet
3. **Global Scale:** Serve millions of users globally
4. **Complex Fallbacks:** 4-level cascading fallback system
5. **Preloading Critical:** Next 2 phases must be ready before user reaches them

---

## 🏗️ ARCHITECTURE DIRECTIVE

### Core Principle: **Edge-First, Offline-First, Preload-Everything**

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERCEL EDGE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Edge       │───▶│   Edge       │───▶│   Edge       │      │
│  │   Middleware │    │   Functions  │    │   Config     │      │
│  │   (Routing)  │    │   (APIs)     │    │   (Metadata) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VERCEL BLOB STORAGE                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ Videos   │  │ Audio    │  │ Visuals  │              │  │
│  │  │ (CDN)    │  │ (CDN)    │  │ (CDN)    │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ISR (Incremental Static Regeneration)        │  │
│  │  Pre-generate lesson pages at build time                 │  │
│  │  Revalidate on-demand when content updates                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              EDGE CACHING STRATEGY                        │  │
│  │  • Lesson metadata: 1 year (immutable)                    │  │
│  │  • Videos: 1 year (versioned URLs)                        │  │
│  │  • Audio: 1 year (versioned URLs)                         │  │
│  │  • Visuals: 1 year (versioned URLs)                      │  │
│  │  • User state: No cache (personalized)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 IMPLEMENTATION DIRECTIVE

### PHASE 1: Edge Middleware - Smart Routing & Preloading

#### 1.1 Create Edge Middleware

**File:** `middleware.ts` (root of project)

```typescript
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  
  // Lesson route: /learn?day=161
  if (pathname === '/learn.html' || pathname.startsWith('/learn')) {
    const day = request.nextUrl.searchParams.get('day') || getTodayDayNumber();
    const archetype = request.nextUrl.searchParams.get('archetype') || 'The Scientist';
    
    const response = NextResponse.next();
    
    // CRITICAL: Preload headers for instant asset loading
    const preloadLinks = [
      // Preload lesson metadata (Edge Config)
      `</api/lessons/${day}?archetype=${archetype}>; rel=preload; as=fetch; crossorigin`,
      
      // Preload first phase video (Hook) - CRITICAL for zero buffering
      `</blob/videos/day-${day}/scientist/hook.mp4>; rel=preload; as=video`,
      
      // Preload first phase audio
      `</blob/audio/day-${day}/scientist/hook.mp3>; rel=preload; as=audio`,
      
      // Preload first phase visual
      `</blob/visuals/day-${day}/hook-infographic.png>; rel=preload; as=image`,
      
      // Preload next 2 phases (Question, Context) - ZERO BUFFERING
      `</blob/videos/day-${day}/scientist/question.mp4>; rel=prefetch; as=video`,
      `</blob/videos/day-${day}/scientist/context.mp4>; rel=prefetch; as=video`,
      
      // Preload adjacent days (for calendar navigation)
      `</api/lessons/${parseInt(day) + 1}>; rel=prefetch; as=fetch`,
      `</api/lessons/${parseInt(day) - 1}>; rel=prefetch; as=fetch`,
    ];
    
    response.headers.set('Link', preloadLinks.join(', '));
    
    // Add cache headers
    response.headers.set('Cache-Control', 'public, s-maxage=3600, stale-while-revalidate=86400');
    
    return response;
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ['/learn.html', '/learn/:path*'],
};
```

**Success Criteria:**
- ✅ Preload headers added to all lesson routes
- ✅ Next 2 phases preloaded automatically
- ✅ Adjacent lessons prefetched
- ✅ Zero client-side API calls for initial load

---

### PHASE 2: Edge Config - Instant Metadata

#### 2.1 Create Edge Config Project

**Action:** Set up Vercel Edge Config in dashboard

**Schema:**
```typescript
// Edge Config stores lightweight lesson metadata
interface LessonMetadata {
  day: number;
  topic: string;
  emoji: string;
  category: string;
  headline: string;
  hasLearn: boolean;
  hasGrow: boolean;
  phases: string[]; // ['hook', 'question', 'context', 'choice', 'reflection', 'wisdom', 'action']
  archetypes: string[]; // Available archetypes for this lesson
}
```

#### 2.2 Build Sync Worker

**File:** `api/sync-edge-config/route.ts`

```typescript
import { get, set } from '@vercel/edge-config';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

export async function POST(request: Request) {
  const { secret, day } = await request.json();
  
  if (secret !== process.env.EDGE_CONFIG_SYNC_SECRET) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
  
  // Fetch lesson from Supabase
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('*, lesson_atoms(archetype, phase)')
    .eq('day_number', day)
    .single();
  
  if (!lesson) {
    return Response.json({ error: 'Lesson not found' }, { status: 404 });
  }
  
  // Transform to lightweight metadata
  const metadata = {
    day: lesson.day_number,
    topic: lesson.topic,
    emoji: lesson.emoji || '📚',
    category: lesson.category || '',
    headline: lesson.marketing_headline || lesson.headline || '',
    hasLearn: true,
    hasGrow: !!lesson.grow_track_id,
    phases: [...new Set(lesson.lesson_atoms.map(a => a.phase))],
    archetypes: [...new Set(lesson.lesson_atoms.map(a => a.archetype))],
  };
  
  // Store in Edge Config
  await set(`lesson:${day}:meta`, metadata);
  
  return Response.json({ success: true, day, metadata });
}
```

#### 2.3 Update Lesson API to Use Edge Config

**File:** `api/lessons/[day]/route.ts`

```typescript
import { get } from '@vercel/edge-config';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';
export const revalidate = 3600; // 1 hour

export async function GET(
  request: Request,
  { params }: { params: { day: string } }
) {
  const day = parseInt(params.day);
  const { searchParams } = new URL(request.url);
  const archetype = searchParams.get('archetype') || 'The Scientist';
  const track = searchParams.get('track') || 'learn';
  
  // CRITICAL: Try Edge Config first (<5ms reads globally)
  const cacheKey = `lesson:${day}:meta`;
  const cached = await get(cacheKey);
  
  if (cached) {
    return Response.json(cached, {
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
        'CDN-Cache-Control': 'public, s-maxage=3600',
        'X-Data-Source': 'edge-config',
      }
    });
  }
  
  // Fallback to Supabase (slower, but works)
  const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
  
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', day)
    .single();
  
  return Response.json(lesson, {
    headers: {
      'Cache-Control': 'public, s-maxage=300',
      'X-Data-Source': 'supabase',
    }
  });
}
```

**Success Criteria:**
- ✅ Edge Config storing all lesson metadata
- ✅ <5ms metadata reads globally
- ✅ 99.9% cache hit rate
- ✅ Webhook syncing updates automatically

---

### PHASE 3: Vercel Blob Storage - Media Delivery

#### 3.1 Migrate Assets to Blob Storage

**Action:** Move all videos, audio, visuals from Supabase Storage to Vercel Blob

**File:** `scripts/migrate-to-blob.ts`

```typescript
import { put } from '@vercel/blob';
import { createClient } from '@supabase/supabase-js';
import fs from 'fs';

async function migrateAssets() {
  const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);
  
  // Fetch all video assets
  const { data: videos } = await supabase
    .from('kelly_video_assets')
    .select('*')
    .eq('asset_type', 'video')
    .not('public_url', 'is', null);
  
  for (const video of videos) {
    // Download from Supabase
    const { data: blob } = await supabase.storage
      .from('kelly-videos')
      .download(video.storage_path);
    
    if (!blob) continue;
    
    // Upload to Vercel Blob
    const buffer = Buffer.from(await blob.arrayBuffer());
    const vercelBlob = await put(
      `videos/day-${video.day_number}/${video.archetype}/${video.phase}.mp4`,
      buffer,
      {
        access: 'public',
        addRandomSuffix: false,
        cacheControlMaxAge: 31536000, // 1 year
      }
    );
    
    // Update database with new URL
    await supabase
      .from('kelly_video_assets')
      .update({ vercel_blob_url: vercelBlob.url })
      .eq('id', video.id);
    
    console.log(`Migrated: ${video.storage_path} → ${vercelBlob.url}`);
  }
}
```

#### 3.2 Update Asset URLs in Codebase

**Action:** Replace Supabase Storage URLs with Vercel Blob URLs

**Pattern:**
- Old: `https://[project].supabase.co/storage/v1/object/public/kelly-videos/...`
- New: `https://[project].blob.vercel-storage.com/videos/...`

**Files to Update:**
- `public/js/kelly-lesson-loader.js` (hd_video_url references)
- `public/js/kelly-unified-lesson-service.js` (video extraction)
- `public/learn.html` (video player URLs)
- Any other files referencing video/audio/visual URLs

**Success Criteria:**
- ✅ All assets migrated to Vercel Blob
- ✅ URLs updated in codebase
- ✅ CDN delivery working globally
- ✅ Zero egress fees

---

### PHASE 4: ISR - Static Generation

#### 4.1 Convert to Next.js App Router

**Action:** Migrate from static HTML to Next.js App Router

**File Structure:**
```
app/
├── layout.tsx
├── page.tsx (homepage)
├── learn/
│   ├── [day]/
│   │   └── page.tsx (lesson page)
│   └── page.tsx (learn index)
└── api/
    └── lessons/
        └── [day]/
            └── route.ts
```

**File:** `app/learn/[day]/page.tsx`

```typescript
import { notFound } from 'next/navigation';
import { get } from '@vercel/edge-config';
import LessonPlayer from '@/components/LessonPlayer';

export const revalidate = 3600; // Revalidate every hour
export const dynamicParams = false; // Only generate 365 days

// Pre-generate all 365 lesson pages at build time
export async function generateStaticParams() {
  return Array.from({ length: 365 }, (_, i) => ({
    day: String(i + 1)
  }));
}

export default async function LessonPage({ 
  params 
}: { 
  params: { day: string } 
}) {
  const day = parseInt(params.day);
  
  if (day < 1 || day > 365) {
    notFound();
  }
  
  // Try Edge Config first (instant)
  const metadata = await get(`lesson:${day}:meta`);
  
  if (!metadata) {
    // Fallback to Supabase
    const supabase = createClient(...);
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', day)
      .single();
    
    if (!lesson) {
      notFound();
    }
    
    return <LessonPlayer lesson={lesson} />;
  }
  
  // Load full lesson data (atoms, videos, etc.)
  const fullLesson = await getFullLesson(day, metadata);
  
  return <LessonPlayer lesson={fullLesson} />;
}
```

#### 4.2 On-Demand Revalidation

**File:** `app/api/revalidate/route.ts`

```typescript
import { revalidatePath } from 'next/cache';
import { set } from '@vercel/edge-config';

export async function POST(request: Request) {
  const { secret, day } = await request.json();
  
  if (secret !== process.env.REVALIDATION_SECRET) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  // Revalidate specific lesson page
  revalidatePath(`/learn/${day}`);
  
  // Also update Edge Config
  await syncLessonToEdgeConfig(day);
  
  return Response.json({ revalidated: true, day });
}
```

**Success Criteria:**
- ✅ All 365 lesson pages pre-rendered
- ✅ Instant HTML delivery (<20ms TTFB)
- ✅ Automatic content updates
- ✅ Perfect SEO

---

### PHASE 5: Client-Side Preloading

#### 5.1 Phase Preloading Component

**File:** `components/PhasePreloader.tsx`

```typescript
'use client';

import { useEffect } from 'react';

interface PhasePreloaderProps {
  atoms: Array<{
    phase: string;
    hd_video_url?: string;
    audio_url?: string;
    visual_url?: string;
  }>;
  currentPhase: number;
}

export function PhasePreloader({ atoms, currentPhase }: PhasePreloaderProps) {
  useEffect(() => {
    // Preload next 2 phases ahead (ZERO BUFFERING)
    const phasesToPreload = atoms.slice(
      currentPhase,
      Math.min(currentPhase + 2, atoms.length)
    );
    
    phasesToPreload.forEach((atom, index) => {
      const phaseIndex = currentPhase + index;
      
      // Preload video
      if (atom.hd_video_url) {
        const videoLink = document.createElement('link');
        videoLink.rel = 'preload';
        videoLink.as = 'video';
        videoLink.href = atom.hd_video_url;
        videoLink.crossOrigin = 'anonymous';
        document.head.appendChild(videoLink);
      }
      
      // Preload audio
      if (atom.audio_url) {
        const audioLink = document.createElement('link');
        audioLink.rel = 'preload';
        audioLink.as = 'audio';
        audioLink.href = atom.audio_url;
        document.head.appendChild(audioLink);
      }
      
      // Preload visual
      if (atom.visual_url) {
        const visualLink = document.createElement('link');
        visualLink.rel = 'preload';
        visualLink.as = 'image';
        visualLink.href = atom.visual_url;
        document.head.appendChild(visualLink);
      }
    });
  }, [atoms, currentPhase]);
  
  return null; // No UI
}
```

#### 5.2 Enhanced Service Worker

**File:** `public/sw.js` (update existing)

```javascript
const LESSON_CACHE_VERSION = 'lessons-v3';
const ASSET_CACHE_VERSION = 'assets-v3';

// Preload today's lesson + next 7 days
self.addEventListener('install', async (event) => {
  event.waitUntil(
    (async () => {
      const lessonCache = await caches.open(LESSON_CACHE_VERSION);
      const assetCache = await caches.open(ASSET_CACHE_VERSION);
      const today = getTodayDayNumber();
      
      // Preload lesson data for next week
      const preloadPromises = [];
      for (let i = 0; i < 7; i++) {
        const day = today + i;
        if (day > 365) break;
        
        // Preload lesson metadata
        preloadPromises.push(
          fetch(`/api/lessons/${day}`)
            .then(res => res.ok && lessonCache.put(`/api/lessons/${day}`, res.clone()))
        );
        
        // Preload first phase video (Hook) for each day
        preloadPromises.push(
          fetch(`/blob/videos/day-${day}/scientist/hook.mp4`)
            .then(res => res.ok && assetCache.put(`/blob/videos/day-${day}/scientist/hook.mp4`, res.clone()))
            .catch(() => {}) // Non-blocking
        );
      }
      
      await Promise.allSettled(preloadPromises);
      await self.skipWaiting();
    })()
  );
});

// Cache-first for assets, network-first for APIs
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Assets: Cache-first (offline support)
  if (url.pathname.startsWith('/blob/')) {
    event.respondWith(
      caches.match(request).then(cached => {
        if (cached) return cached;
        return fetch(request).then(response => {
          if (response.ok) {
            const cache = url.pathname.includes('/videos/') || url.pathname.includes('/audio/')
              ? ASSET_CACHE_VERSION
              : LESSON_CACHE_VERSION;
            caches.open(cache).then(c => c.put(request, response.clone()));
          }
          return response;
        });
      })
    );
    return;
  }
  
  // APIs: Network-first with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request).catch(() => caches.match(request))
    );
    return;
  }
});
```

**Success Criteria:**
- ✅ Next 2 phases preloaded automatically
- ✅ Zero buffering between phases
- ✅ Offline lesson access
- ✅ Reduced network usage

---

### PHASE 6: Vercel KV - User Progress

#### 6.1 Progress API Migration

**File:** `app/api/progress/route.ts`

```typescript
import { kv } from '@vercel/kv';

export const runtime = 'edge';

export async function POST(request: Request) {
  const { userId, day, completed } = await request.json();
  
  // Store in Vercel KV (edge-native, <1ms reads)
  await kv.set(`progress:${userId}:${day}`, {
    completed,
    timestamp: Date.now(),
  });
  
  // Also update Supabase (async, non-blocking)
  fetch(`${process.env.SUPABASE_URL}/rest/v1/user_progress`, {
    method: 'POST',
    headers: {
      'apikey': process.env.SUPABASE_SERVICE_ROLE_KEY!,
      'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY!}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      day_number: day,
      status: 'completed',
      completed_at: new Date().toISOString(),
    })
  }).catch(() => {}); // Non-blocking
  
  return Response.json({ success: true });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const userId = searchParams.get('userId');
  
  if (!userId) {
    return Response.json({ error: 'Missing userId' }, { status: 400 });
  }
  
  // Read from KV (instant, <1ms)
  const keys = await kv.keys(`progress:${userId}:*`);
  const progress = await Promise.all(
    keys.map(key => kv.get(key))
  );
  
  return Response.json(progress);
}
```

**Success Criteria:**
- ✅ Instant progress reads (<1ms)
- ✅ Offline-first (KV syncs async)
- ✅ No database load for reads
- ✅ Better user experience

---

## 📊 PERFORMANCE TARGETS (MANDATORY)

### Must Achieve

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **TTFB** | 200-500ms | **<20ms** | Vercel Analytics |
| **Lesson Load** | 1-3s | **<200ms** | Performance API |
| **Video Start** | 2-5s | **<500ms** | Video element events |
| **Phase Transition** | 1-2s | **<100ms** | Custom timing |
| **Cache Hit Rate** | 0% | **99.5%** | Edge Config metrics |
| **Offline Support** | Partial | **Full** | Service worker tests |

### Stretch Goals

- ✅ <10ms TTFB (stretch)
- ✅ <100ms lesson load (stretch)
- ✅ 99.9% cache hit rate (stretch)

---

## ✅ IMPLEMENTATION CHECKLIST

### Week 1: Foundation
- [ ] Set up Vercel Edge Config
- [ ] Create sync worker (Supabase → Edge Config)
- [ ] Migrate videos to Vercel Blob
- [ ] Migrate audio to Vercel Blob
- [ ] Migrate visuals to Vercel Blob
- [ ] Update asset URLs in codebase
- [ ] Test CDN delivery globally

### Week 2: Edge Functions
- [ ] Create Edge Middleware with preload headers
- [ ] Migrate lesson API to Edge Function
- [ ] Implement Edge Config caching
- [ ] Migrate progress API to Vercel KV
- [ ] Test all APIs
- [ ] Validate performance targets

### Week 3: ISR
- [ ] Convert to Next.js App Router
- [ ] Implement `generateStaticParams()` for 365 days
- [ ] Set up on-demand revalidation
- [ ] Test static generation
- [ ] Validate SEO

### Week 4: Preloading
- [ ] Implement PhasePreloader component
- [ ] Enhance service worker
- [ ] Add video streaming support
- [ ] Test preloading effectiveness
- [ ] Validate zero buffering

### Week 5: Testing & Optimization
- [ ] Load testing (1000+ concurrent users)
- [ ] Global latency testing
- [ ] Cache hit rate validation
- [ ] Bundle optimization
- [ ] Cost analysis
- [ ] Documentation

---

## 🚨 CRITICAL RULES

### DO NOT

1. **DO NOT** simplify lesson structure (7 phases × 12 archetypes is required)
2. **DO NOT** remove offline-first capability
3. **DO NOT** skip preloading (zero buffering is mandatory)
4. **DO NOT** cache user-specific data at edge
5. **DO NOT** break cascading fallback system

### MUST DO

1. **MUST** preload next 2 phases ahead
2. **MUST** use Edge Config for lesson metadata
3. **MUST** use Vercel Blob for all media
4. **MUST** implement ISR for lesson pages
5. **MUST** achieve <20ms TTFB globally
6. **MUST** maintain offline-first architecture

---

## 🎯 SUCCESS CRITERIA

### Technical Excellence
- ✅ <20ms TTFB globally
- ✅ <200ms lesson load time
- ✅ <500ms video start time
- ✅ <100ms phase transitions
- ✅ 99.5% cache hit rate
- ✅ Zero buffering
- ✅ Full offline support

### User Experience
- ✅ Instant lesson start
- ✅ Zero buffering between phases
- ✅ Smooth video playback
- ✅ Offline lesson access
- ✅ Perfect mobile experience

### Business Impact
- ✅ 10x faster page loads
- ✅ 20-40% cost reduction
- ✅ Global scale capability
- ✅ Better developer experience

---

## 📚 REFERENCE ARCHITECTURE

### Data Flow

```
User Request
    │
    ▼
Edge Middleware (preload headers)
    │
    ▼
ISR Page (pre-rendered HTML)
    │
    ├──▶ Edge Config (metadata, <5ms)
    ├──▶ Vercel Blob (videos/audio, CDN)
    ├──▶ Vercel KV (user progress, <1ms)
    └──▶ Edge Functions (APIs, <50ms)
    │
    ▼
Client (preloaded assets ready)
    │
    ├──▶ Service Worker (offline cache)
    ├──▶ Preload next phases
    └──▶ Background sync progress
```

### Caching Strategy

| Asset Type | Cache Duration | Cache Location | Invalidation |
|------------|----------------|----------------|--------------|
| **Lesson Metadata** | 1 year | Edge Config | Webhook |
| **Videos** | 1 year | Vercel Blob CDN | Versioned URLs |
| **Audio** | 1 year | Vercel Blob CDN | Versioned URLs |
| **Visuals** | 1 year | Vercel Blob CDN | Versioned URLs |
| **User Progress** | No cache | Vercel KV | Real-time |
| **Lesson Pages** | 1 hour | ISR | On-demand |

---

## 🔧 TECHNICAL SPECIFICATIONS

### Edge Config Schema

```typescript
// Edge Config key: `lesson:{day}:meta`
{
  day: number;
  topic: string;
  emoji: string;
  category: string;
  headline: string;
  hasLearn: boolean;
  hasGrow: boolean;
  phases: string[]; // ['hook', 'question', ...]
  archetypes: string[]; // ['The Scientist', ...]
}
```

### Vercel Blob Structure

```
blob.vercel-storage.com/
├── videos/
│   └── day-{day}/
│       └── {archetype}/
│           └── {phase}.mp4
├── audio/
│   └── day-{day}/
│       └── {archetype}/
│           └── {phase}.mp3
└── visuals/
    └── day-{day}/
        ├── {phase}-infographic.png
        └── {phase}-option-{a|b}.png
```

### Vercel KV Schema

```
Key: `progress:{userId}:{day}`
Value: {
  completed: boolean;
  timestamp: number;
  phase?: number; // Last completed phase
}
```

---

## 📋 DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] All assets migrated to Vercel Blob
- [ ] Edge Config populated with metadata
- [ ] Edge Functions tested locally
- [ ] ISR pages generated successfully
- [ ] Service worker updated
- [ ] Performance targets validated

### Deployment
- [ ] Deploy to Vercel production
- [ ] Verify Edge Config reads
- [ ] Test Blob CDN delivery
- [ ] Validate ISR pages
- [ ] Test preloading
- [ ] Monitor performance

### Post-Deployment
- [ ] Monitor cache hit rates
- [ ] Track performance metrics
- [ ] Validate cost savings
- [ ] User testing
- [ ] Iterate based on data

---

## 🎓 DEEP USE CASE UNDERSTANDING

### Why This Architecture?

1. **Multi-Asset Lessons:** Edge Config + Blob Storage handles 315MB per lesson
2. **Offline-First:** Service worker + KV ensures offline access
3. **Zero Buffering:** Preloading next 2 phases eliminates wait time
4. **Global Scale:** Edge-native architecture serves millions
5. **Complex Fallbacks:** Edge Config → Supabase → Static → Emergency

### Key Differentiators

- **Edge Config:** <5ms metadata reads (vs 200-500ms database)
- **Vercel Blob:** Zero egress fees (vs variable costs)
- **ISR:** Pre-rendered pages (vs dynamic rendering)
- **Edge Functions:** Zero cold starts (vs 500-2000ms)
- **Vercel KV:** <1ms progress reads (vs 200-500ms database)

---

## 🚀 QUICK START

### Immediate Actions (Today)

1. **Set up Vercel Edge Config**
   ```bash
   # In Vercel Dashboard
   Storage → Edge Config → Create
   ```

2. **Create Blob Storage buckets**
   ```bash
   # In Vercel Dashboard
   Storage → Blob → Create buckets:
   - curious-kelly-videos
   - curious-kelly-audio
   - curious-kelly-visuals
   ```

3. **Create Edge Middleware**
   ```bash
   # Create middleware.ts in root
   # Add preload headers for lesson routes
   ```

4. **Test Edge Config**
   ```bash
   # Create sync worker
   # Test <5ms reads
   ```

---

## 📊 MONITORING & VALIDATION

### Key Metrics to Track

1. **Performance:**
   - TTFB (target: <20ms)
   - Lesson load time (target: <200ms)
   - Video start time (target: <500ms)
   - Phase transition time (target: <100ms)

2. **Caching:**
   - Edge Config hit rate (target: 99.5%)
   - Blob CDN hit rate (target: 99%)
   - ISR cache hit rate (target: 95%)

3. **Costs:**
   - Blob storage costs
   - Edge Function invocations
   - KV operations
   - Bandwidth usage

---

## 🏆 FINAL VERDICT

**Architecture:** Vercel Edge-First Platform  
**Timeline:** 5 weeks  
**Performance:** 10x improvement  
**Cost:** 20-40% reduction  
**Status:** ✅ Ready for Implementation

---

**Directive Status:** ACTIVE  
**Priority:** CRITICAL  
**Next Session:** Begin Week 1 implementation  
**Owner:** Engineering Team

---

**This directive supersedes all previous optimization plans. Follow this architecture for maximum performance.**

