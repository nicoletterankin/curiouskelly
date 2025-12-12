# 🏗️ Curious Kelly - Production Architecture

## Current State Assessment

### ✅ What's Live
- **Domain:** curiouskelly.com (Vercel-hosted)
- **Static Pages:** 9 HTML pages deployed to `public/`
- **Git:** All code pushed to `main` branch
- **Auto-Deploy:** Vercel connected to GitHub

### ⚠️ What's Broken
- **Image 404s:** Lesson player looking for images at wrong paths
- **No OAuth:** Login buttons are mocked
- **No Backend:** Forms log to console only
- **No Database Connection:** Supabase not wired up

---

## 🎯 Production Service Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Vercel)                        │
│  curiouskelly.com - Static HTML/CSS/JS                      │
│  • index.html (Login Portal)                                │
│  • about.html (Marketing/Campus)                            │
│  • dashboard.html (Authenticated App) ← BUILD THIS          │
│  • 7 footer pages (careers, privacy, etc.)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTPS/REST
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              BACKEND API (Render/Railway)                    │
│  api.curiouskelly.com                                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Authentication Service                               │  │
│  │  • POST /api/auth/google                             │  │
│  │  • POST /api/auth/apple                              │  │
│  │  • POST /api/auth/github                             │  │
│  │  • GET  /api/auth/session                            │  │
│  │  • POST /api/auth/logout                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Lesson Service                                       │  │
│  │  • GET  /api/lessons/today                           │  │
│  │  • GET  /api/lessons/:id                             │  │
│  │  • GET  /api/lessons/calendar                        │  │
│  │  • POST /api/lessons/:id/progress                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  User Service                                         │  │
│  │  • GET  /api/user/profile                            │  │
│  │  • PUT  /api/user/profile                            │  │
│  │  • GET  /api/user/progress                           │  │
│  │  • GET  /api/user/streak                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Affiliate Service                                    │  │
│  │  • POST /api/affiliate/apply                         │  │
│  │  • GET  /api/affiliate/dashboard                     │  │
│  │  • GET  /api/affiliate/stats                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Enterprise Service                                   │  │
│  │  • POST /api/enterprise/contact                      │  │
│  │  • POST /api/newsletter/subscribe                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Supabase Client
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                  SUPABASE (Database + Auth)                  │
│  https://tvjalxxsyryjphkforjv.supabase.co                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Auth (Built-in)                                      │  │
│  │  • Google OAuth                                       │  │
│  │  • Apple OAuth                                        │  │
│  │  • GitHub OAuth                                       │  │
│  │  • JWT tokens                                         │  │
│  │  • Row Level Security                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Database Tables                                      │  │
│  │                                                        │  │
│  │  users                                                │  │
│  │  ├─ id (uuid, pk)                                     │  │
│  │  ├─ email (text)                                      │  │
│  │  ├─ name (text)                                       │  │
│  │  ├─ age (int)                                         │  │
│  │  ├─ subscription_tier (text)                          │  │
│  │  ├─ subscription_expires_at (timestamp)               │  │
│  │  └─ created_at (timestamp)                            │  │
│  │                                                        │  │
│  │  lessons                                              │  │
│  │  ├─ id (uuid, pk)                                     │  │
│  │  ├─ day_number (int)                                  │  │
│  │  ├─ title (text)                                      │  │
│  │  ├─ content (jsonb) ← PhaseDNA                       │  │
│  │  ├─ audio_url (text)                                  │  │
│  │  ├─ duration_seconds (int)                            │  │
│  │  └─ created_at (timestamp)                            │  │
│  │                                                        │  │
│  │  user_progress                                        │  │
│  │  ├─ id (uuid, pk)                                     │  │
│  │  ├─ user_id (uuid, fk → users)                       │  │
│  │  ├─ lesson_id (uuid, fk → lessons)                   │  │
│  │  ├─ completed (boolean)                               │  │
│  │  ├─ progress_percent (int)                            │  │
│  │  ├─ last_position_seconds (int)                       │  │
│  │  └─ completed_at (timestamp)                          │  │
│  │                                                        │  │
│  │  affiliates                                           │  │
│  │  ├─ id (uuid, pk)                                     │  │
│  │  ├─ user_id (uuid, fk → users)                       │  │
│  │  ├─ referral_code (text, unique)                     │  │
│  │  ├─ tier (text) ← Scholar/Fellow/Ambassador          │  │
│  │  ├─ commission_rate (decimal)                         │  │
│  │  ├─ total_referrals (int)                             │  │
│  │  ├─ active_referrals (int)                            │  │
│  │  └─ lifetime_earnings (decimal)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Storage Buckets                                      │  │
│  │  • images/ (Kelly assets, lesson images)             │  │
│  │  • audio/ (Lesson audio files from ElevenLabs)       │  │
│  │  • avatars/ (User profile pictures)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Plan

### **STEP 1: Fix Image Paths (5 minutes)**
```bash
# Copy Kelly images to correct public path
cp -r assets/kelly_canonical/core/chair/* public/images/kelly/
git add public/images/kelly/
git commit -m "Fix: Add Kelly chair images to public directory"
git push origin main
```

### **STEP 2: Set Up Supabase Auth (30 minutes)**

#### 2.1 Configure OAuth Providers in Supabase Dashboard
```
1. Go to: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv/auth/providers
2. Enable Google OAuth:
   - Client ID: [from Google Cloud Console]
   - Client Secret: [from Google Cloud Console]
   - Redirect URL: https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback
3. Enable Apple OAuth (similar)
4. Enable GitHub OAuth (similar)
```

#### 2.2 Update Frontend Auth Flow
```javascript
// index.html - Replace mock login with real Supabase auth
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  'YOUR_ANON_KEY'
)

async function handleLogin(provider) {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: provider, // 'google', 'apple', 'github'
    options: {
      redirectTo: 'https://curiouskelly.com/dashboard.html'
    }
  })
  
  if (error) console.error('Auth error:', error)
}
```

### **STEP 3: Create Database Schema (15 minutes)**

```sql
-- Run in Supabase SQL Editor

-- Users table (extends auth.users)
CREATE TABLE public.users (
  id UUID REFERENCES auth.users PRIMARY KEY,
  email TEXT NOT NULL,
  name TEXT,
  age INTEGER,
  subscription_tier TEXT DEFAULT 'free',
  subscription_expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Users can only read/update their own data
CREATE POLICY "Users can view own data" ON public.users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own data" ON public.users
  FOR UPDATE USING (auth.uid() = id);

-- Lessons table
CREATE TABLE public.lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER UNIQUE NOT NULL,
  title TEXT NOT NULL,
  content JSONB NOT NULL, -- PhaseDNA structure
  audio_url TEXT,
  duration_seconds INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Public read access to lessons
ALTER TABLE public.lessons ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can view lessons" ON public.lessons
  FOR SELECT USING (true);

-- User progress table
CREATE TABLE public.user_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users NOT NULL,
  lesson_id UUID REFERENCES public.lessons NOT NULL,
  completed BOOLEAN DEFAULT false,
  progress_percent INTEGER DEFAULT 0,
  last_position_seconds INTEGER DEFAULT 0,
  completed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, lesson_id)
);

ALTER TABLE public.user_progress ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own progress" ON public.user_progress
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own progress" ON public.user_progress
  FOR ALL USING (auth.uid() = user_id);

-- Affiliates table
CREATE TABLE public.affiliates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users NOT NULL,
  referral_code TEXT UNIQUE NOT NULL,
  tier TEXT DEFAULT 'scholar', -- scholar, fellow, ambassador
  commission_rate DECIMAL(5,2) DEFAULT 20.00,
  total_referrals INTEGER DEFAULT 0,
  active_referrals INTEGER DEFAULT 0,
  lifetime_earnings DECIMAL(10,2) DEFAULT 0.00,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.affiliates ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own affiliate data" ON public.affiliates
  FOR SELECT USING (auth.uid() = user_id);
```

### **STEP 4: Build Backend API (2 hours)**

```javascript
// curious-kellly/backend/src/index.js

import express from 'express'
import cors from 'cors'
import { createClient } from '@supabase/supabase-js'

const app = express()
app.use(cors({ origin: 'https://curiouskelly.com' }))
app.use(express.json())

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY // Server-side key
)

// ============================================
// AUTH ENDPOINTS
// ============================================

// Verify session (called by frontend on page load)
app.get('/api/auth/session', async (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '')
  
  const { data: { user }, error } = await supabase.auth.getUser(token)
  
  if (error) return res.status(401).json({ error: 'Unauthorized' })
  
  res.json({ user })
})

// ============================================
// LESSON ENDPOINTS
// ============================================

// Get today's lesson (based on user's day in curriculum)
app.get('/api/lessons/today', async (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '')
  const { data: { user } } = await supabase.auth.getUser(token)
  
  if (!user) return res.status(401).json({ error: 'Unauthorized' })
  
  // Get user's progress to determine current day
  const { data: progress } = await supabase
    .from('user_progress')
    .select('lesson_id')
    .eq('user_id', user.id)
    .eq('completed', true)
  
  const currentDay = (progress?.length || 0) + 1
  
  // Get lesson for current day
  const { data: lesson, error } = await supabase
    .from('lessons')
    .select('*')
    .eq('day_number', currentDay)
    .single()
  
  if (error) return res.status(404).json({ error: 'Lesson not found' })
  
  res.json({ lesson, day: currentDay })
})

// Get specific lesson by ID
app.get('/api/lessons/:id', async (req, res) => {
  const { data: lesson, error } = await supabase
    .from('lessons')
    .select('*')
    .eq('id', req.params.id)
    .single()
  
  if (error) return res.status(404).json({ error: 'Lesson not found' })
  
  res.json({ lesson })
})

// Get full calendar
app.get('/api/lessons/calendar', async (req, res) => {
  const { data: lessons, error } = await supabase
    .from('lessons')
    .select('id, day_number, title, duration_seconds')
    .order('day_number')
  
  if (error) return res.status(500).json({ error: 'Failed to fetch calendar' })
  
  res.json({ lessons })
})

// Update progress
app.post('/api/lessons/:id/progress', async (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '')
  const { data: { user } } = await supabase.auth.getUser(token)
  
  if (!user) return res.status(401).json({ error: 'Unauthorized' })
  
  const { progress_percent, last_position_seconds, completed } = req.body
  
  const { data, error } = await supabase
    .from('user_progress')
    .upsert({
      user_id: user.id,
      lesson_id: req.params.id,
      progress_percent,
      last_position_seconds,
      completed,
      completed_at: completed ? new Date().toISOString() : null,
      updated_at: new Date().toISOString()
    })
    .select()
    .single()
  
  if (error) return res.status(500).json({ error: 'Failed to update progress' })
  
  res.json({ progress: data })
})

// ============================================
// AFFILIATE ENDPOINTS
// ============================================

app.post('/api/affiliate/apply', async (req, res) => {
  const { name, email, platform, url, audience, focus, why } = req.body
  
  // Store application in Supabase
  const { data, error } = await supabase
    .from('affiliate_applications')
    .insert({
      name,
      email,
      platform,
      url,
      audience,
      focus,
      why,
      status: 'pending'
    })
    .select()
    .single()
  
  if (error) return res.status(500).json({ error: 'Application failed' })
  
  // TODO: Send notification email to admin
  
  res.json({ success: true, application: data })
})

// ============================================
// ENTERPRISE ENDPOINTS
// ============================================

app.post('/api/enterprise/contact', async (req, res) => {
  const { organization, name, email, phone, org_type, size, use_case, timeline } = req.body
  
  const { data, error } = await supabase
    .from('enterprise_inquiries')
    .insert({
      organization,
      name,
      email,
      phone,
      org_type,
      size,
      use_case,
      timeline,
      status: 'new'
    })
    .select()
    .single()
  
  if (error) return res.status(500).json({ error: 'Inquiry failed' })
  
  res.json({ success: true, inquiry: data })
})

// ============================================
// NEWSLETTER ENDPOINT
// ============================================

app.post('/api/newsletter/subscribe', async (req, res) => {
  const { email } = req.body
  
  const { data, error } = await supabase
    .from('newsletter_subscribers')
    .insert({ email })
    .select()
    .single()
  
  if (error) {
    if (error.code === '23505') { // Duplicate email
      return res.status(400).json({ error: 'Already subscribed' })
    }
    return res.status(500).json({ error: 'Subscription failed' })
  }
  
  res.json({ success: true })
})

const PORT = process.env.PORT || 3000
app.listen(PORT, () => {
  console.log(`🚀 API server running on port ${PORT}`)
})
```

### **STEP 5: Deploy Backend (30 minutes)**

```bash
# Deploy to Render.com
1. Go to render.com
2. New → Web Service
3. Connect GitHub repo: nicoletterankin/curiouskelly
4. Root directory: curious-kellly/backend
5. Build command: npm install
6. Start command: node src/index.js
7. Add environment variables:
   - SUPABASE_URL
   - SUPABASE_SERVICE_KEY
   - NODE_ENV=production
8. Deploy

# Update DNS
Add CNAME record: api.curiouskelly.com → your-app.onrender.com
```

### **STEP 6: Build Dashboard (1 hour)**

Create `dashboard.html` - the authenticated app experience where users access lessons after login.

---

## 📋 Immediate Action Items

1. ✅ **Fix image 404s** - Copy assets to public/images/kelly/
2. ✅ **Get Supabase keys** - From dashboard
3. ✅ **Configure OAuth** - Google, Apple, GitHub in Supabase
4. ✅ **Run SQL schema** - Create tables
5. ✅ **Deploy backend** - Render.com
6. ✅ **Wire up frontend** - Replace mocks with real API calls
7. ✅ **Build dashboard.html** - Post-login experience
8. ✅ **Test end-to-end** - Login → See lesson → Track progress

---

## 🎯 Success Criteria

- [ ] User can log in with Google/Apple/GitHub
- [ ] User redirected to dashboard.html after auth
- [ ] Dashboard shows "Today's Lesson" from Supabase
- [ ] Lesson player loads audio and tracks progress
- [ ] Progress saved to database
- [ ] Affiliate form submissions stored
- [ ] Enterprise inquiries captured
- [ ] No 404 errors in console

---

**I understand my job: Build production-grade authentication, wire up Supabase, deploy a real backend API, and create the authenticated dashboard experience. Let's execute.**
























