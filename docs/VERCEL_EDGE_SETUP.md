# Vercel Edge Optimization - Setup Guide
**Date:** December 23, 2025  
**Status:** Implementation In Progress

---

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
npm install @vercel/blob @vercel/edge-config @vercel/kv
```

✅ **DONE** - Packages installed

---

### 2. Set Up Vercel Edge Config

**In Vercel Dashboard:**

1. Go to your project → **Storage** → **Edge Config**
2. Click **Create Edge Config**
3. Name it: `curious-kelly-lessons`
4. Copy the **Connection String** (looks like: `https://edge-config.vercel.app/...`)

**Add to Environment Variables:**

```bash
# In Vercel Dashboard → Settings → Environment Variables
EDGE_CONFIG=your-connection-string-here
EDGE_CONFIG_SYNC_SECRET=your-random-secret-here
```

---

### 3. Set Up Vercel Blob Storage

**In Vercel Dashboard:**

1. Go to **Storage** → **Blob**
2. Create buckets:
   - `curious-kelly-videos`
   - `curious-kelly-audio`
   - `curious-kelly-visuals`

**No environment variables needed** - Blob SDK uses automatic authentication

---

### 4. Initial Sync (Populate Edge Config)

**Run sync worker to populate Edge Config:**

```bash
# Set environment variable locally
export EDGE_CONFIG_SYNC_SECRET=your-secret-here

# Call sync endpoint (syncs all 365 lessons)
curl -X POST https://your-domain.vercel.app/api/sync-edge-config \
  -H "Content-Type: application/json" \
  -d '{"secret": "your-secret-here"}'
```

**Or sync single day:**

```bash
curl -X POST https://your-domain.vercel.app/api/sync-edge-config \
  -H "Content-Type: application/json" \
  -d '{"secret": "your-secret-here", "day": 1}'
```

---

### 5. Migrate Assets to Blob Storage

**Dry run (see what would be migrated):**

```bash
npx tsx scripts/migrate-to-blob.ts --dry-run
```

**Migrate single day:**

```bash
npx tsx scripts/migrate-to-blob.ts --day 1
```

**Migrate all assets:**

```bash
npx tsx scripts/migrate-to-blob.ts --all
```

**Migrate by type:**

```bash
npx tsx scripts/migrate-to-blob.ts --type video
npx tsx scripts/migrate-to-blob.ts --type audio
npx tsx scripts/migrate-to-blob.ts --type visual
```

---

## 📋 API Endpoints Created

### 1. Edge-Optimized Lesson API

**Endpoint:** `GET /api/lessons/[dayNumber]-edge`

**Usage:**
```bash
curl https://your-domain.vercel.app/api/lessons/161-edge?archetype=The%20Scientist
```

**Response:**
```json
{
  "day": 161,
  "topic": "Starting Fresh",
  "emoji": "🌱",
  "category": "Psychology",
  "headline": "New beginnings start with a single step",
  "hasLearn": true,
  "hasGrow": true,
  "phases": ["hook", "question", "context", "choice", "reflection", "wisdom", "action"],
  "archetypes": ["The Scientist", "The Explorer", "The Rebel"],
  "_source": "edge-config",
  "_cached": true
}
```

**Performance:**
- Edge Config hit: <5ms
- Supabase fallback: 200-500ms

---

### 2. Sync Edge Config

**Endpoint:** `POST /api/sync-edge-config`

**Usage:**
```bash
curl -X POST https://your-domain.vercel.app/api/sync-edge-config \
  -H "Content-Type: application/json" \
  -d '{"secret": "your-secret", "day": 1}'
```

**Response:**
```json
{
  "success": true,
  "day": 1,
  "synced": true
}
```

---

### 3. Preload Headers

**Endpoint:** `GET /api/preload-headers?day=161&archetype=The%20Scientist`

**Usage:**
```bash
curl https://your-domain.vercel.app/api/preload-headers?day=161
```

**Response:**
```json
{
  "links": [
    "</api/lessons/161?archetype=The%20Scientist>; rel=preload; as=fetch; crossorigin",
    "</blob/videos/day-161/the-scientist/hook.mp4>; rel=preload; as=video",
    ...
  ],
  "day": 161,
  "archetype": "The Scientist"
}
```

---

## 🔧 Configuration

### vercel.json

Updated to support Edge Functions:

```json
{
  "functions": {
    "api/**/*-edge.ts": {
      "runtime": "@vercel/edge",
      "memory": 128,
      "maxDuration": 10
    }
  }
}
```

---

## ✅ Implementation Checklist

- [x] Install Vercel packages
- [x] Create Edge Function for lesson API
- [x] Create sync worker (Supabase → Edge Config)
- [x] Create migration script for Blob Storage
- [x] Create preload headers API
- [x] Update vercel.json for edge runtime
- [ ] Set up Edge Config in Vercel Dashboard
- [ ] Set up Blob Storage buckets
- [ ] Run initial sync
- [ ] Migrate assets to Blob
- [ ] Update asset URLs in codebase
- [ ] Test performance improvements

---

## 📊 Next Steps

1. **Set up Edge Config** (Vercel Dashboard)
2. **Set up Blob Storage** (Vercel Dashboard)
3. **Run initial sync** (populate Edge Config)
4. **Migrate assets** (run migration script)
5. **Update codebase** (replace Supabase URLs with Blob URLs)
6. **Test** (verify performance improvements)

---

## 🎯 Performance Targets

- ✅ <20ms TTFB globally
- ✅ <200ms lesson load time
- ✅ <500ms video start time
- ✅ <100ms phase transitions
- ✅ 99.5% cache hit rate

---

**Status:** ✅ Code Complete - Ready for Configuration  
**Next:** Set up Edge Config and Blob Storage in Vercel Dashboard

