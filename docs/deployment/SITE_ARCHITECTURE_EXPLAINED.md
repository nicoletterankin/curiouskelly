# CuriousKelly.com - Site Architecture Explained

**Simple, clear explanation of how the marketing site and lesson player work together.**

---

## 🏠 The Big Picture

```
curiouskelly.com
├── 🏡 Marketing Site (Homepage, About, Pricing)
│   └── Hosted on: Vercel
│   └── Framework: Astro
│   └── Purpose: Convert visitors to users
│
└── 🎓 Lesson Player (Interactive Learning)
    └── Hosted on: Cloudflare Pages
    └── Framework: Static HTML/JS/CSS
    └── Purpose: Deliver lessons to learners
```

---

## 🎯 How It Works: User Journey

### Step 1: User Visits Homepage
**URL:** `https://curiouskelly.com`

**What they see:**
- Marketing homepage (Astro site on Vercel)
- Hero section with Kelly's introduction
- Lead capture form
- Call-to-action buttons
- Navigation menu

**Purpose:** 
- Introduce Curious Kelly
- Capture email addresses
- Explain the product
- Convert visitors to users

---

### Step 2: User Clicks "Try Lesson" or "Start Learning"
**URL:** `https://curiouskelly.com/lesson-player`

**What happens:**
- User is routed to the lesson player
- Lesson player loads (hosted on Cloudflare Pages)
- Interactive lesson interface appears

**What they see:**
- Age slider (2-102 years)
- Kelly's avatar/video
- Lesson content
- Audio playback controls
- Interactive choices

**Purpose:**
- Deliver the actual learning experience
- Let users interact with Kelly
- Demonstrate age-adaptive content

---

## 🗺️ Site Map (Simple Version)

```
curiouskelly.com (Root Domain)
│
├── / (Homepage)
│   └── Marketing site - Hero, features, CTA
│
├── /adults
│   └── Marketing page for adult learners
│
├── /children
│   └── Marketing page for children
│
├── /teachers
│   └── Marketing page for teachers
│
├── /schools
│   └── Marketing page for schools
│
├── /lesson-player ⭐ (THE ACTUAL APP)
│   ├── /lesson-player/index.html
│   ├── /lesson-player/script.js
│   ├── /lesson-player/styles.css
│   └── /lesson-player/videos/audio/
│
├── /privacy
│   └── Privacy policy page
│
└── /thank-you
    └── Thank you page after form submission
```

---

## 🔧 Technical Architecture

### Two Separate Deployments

#### 1. Marketing Site (Vercel)
- **Location:** `curiouskelly-marketing-site/` or `daily-lesson-marketing/`
- **Platform:** Vercel Pages
- **Domain:** `curiouskelly.com` (root)
- **Framework:** Astro
- **Build:** `npm run build` → creates `dist/` folder
- **Pages:** Home, Adults, Children, Teachers, Schools, Privacy, etc.

#### 2. Lesson Player (Cloudflare Pages)
- **Location:** `lesson-player/`
- **Platform:** Cloudflare Pages
- **Domain:** `curiouskelly.com/lesson-player` (subdirectory)
- **Framework:** Static HTML/JS/CSS (no build needed)
- **Build:** None (files are ready to deploy)
- **Files:** `index.html`, `script.js`, `styles.css`, audio files

---

## 🔀 How Routing Works

### Option 1: Separate Domains/Subdomains (Current Setup)
```
Marketing Site:  curiouskelly.com (Vercel)
Lesson Player:   curiouskelly.com/lesson-player (Cloudflare Pages)
```

**How it works:**
- Cloudflare DNS routes `curiouskelly.com` → Vercel (marketing site)
- Cloudflare DNS routes `curiouskelly.com/lesson-player` → Cloudflare Pages (lesson player)
- OR: Marketing site links to `/lesson-player` which is proxied to Cloudflare Pages

### Option 2: Unified Routing (Recommended for Production)
```
Everything: curiouskelly.com (Single Platform)
```

**How it would work:**
- Vercel serves marketing pages (`/`, `/adults`, etc.)
- Vercel rewrites `/lesson-player/*` → Cloudflare Pages
- OR: Deploy lesson-player as part of Vercel project

---

## 📋 Current Deployment Status

### ✅ What's Deployed

1. **Lesson Player** ✅
   - **URL:** `curiouskelly-lessons-v2.pages.dev` (Cloudflare preview)
   - **Status:** Successfully deployed
   - **Files:** 17 files uploaded
   - **Ready:** Yes, fully functional

2. **Marketing Site** ⏳
   - **Status:** Not yet deployed
   - **Location:** `curiouskelly-marketing-site/` or `daily-lesson-marketing/`
   - **Action Needed:** Create Vercel project and deploy

---

## 🎨 User Experience Flow

### Scenario: New Visitor

1. **Arrives at:** `curiouskelly.com`
   - Sees marketing homepage
   - Reads about Kelly
   - Fills out lead form

2. **Clicks:** "Try a Lesson" button
   - Redirected to: `curiouskelly.com/lesson-player`
   - Lesson player loads
   - Can interact with Kelly

3. **Uses Lesson Player:**
   - Adjusts age slider
   - Watches/listens to lesson
   - Answers questions
   - Completes lesson

4. **Returns:**
   - Can bookmark `curiouskelly.com/lesson-player`
   - Can return to homepage via navigation
   - Can sign up for full access

---

## 🔗 How They Connect

### Marketing Site → Lesson Player

**Link in Marketing Site:**
```html
<a href="/lesson-player">Try a Lesson</a>
```

**What happens:**
- User clicks link
- Browser navigates to `/lesson-player`
- Cloudflare Pages serves the lesson player
- Lesson player loads and works

### Lesson Player → Marketing Site

**Link in Lesson Player:**
```html
<a href="/">Back to Home</a>
```

**What happens:**
- User clicks link
- Browser navigates to `/`
- Vercel serves the marketing homepage
- Marketing site loads

---

## 🏗️ File Structure

```
curiouskelly/ (GitHub Repository)
│
├── curiouskelly-marketing-site/  (Marketing Site)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── [[...slug]].astro  (Homepage, Adults, Children, etc.)
│   │   │   └── api/               (API routes)
│   │   ├── components/            (LeadForm, HeroCountdown, etc.)
│   │   └── layouts/               (SiteLayout)
│   ├── package.json
│   └── vercel.json                (Vercel config)
│
├── lesson-player/                 (Lesson Player)
│   ├── index.html                 (Main HTML)
│   ├── script.js                  (Lesson logic)
│   ├── styles.css                 (Styling)
│   ├── components/                (Right-rail, read-along)
│   ├── videos/audio/              (MP3 files)
│   └── README.md
│
└── .github/workflows/
    ├── deploy-vercel.yml          (Deploys marketing site)
    └── deploy-cloudflare.yml      (Deploys lesson player)
```

---

## 🌐 Domain Configuration

### Current Setup (After Deployment)

```
curiouskelly.com (Cloudflare DNS)
│
├── Root (/) → Vercel (Marketing Site)
│   └── Homepage, marketing pages
│
└── /lesson-player → Cloudflare Pages (Lesson Player)
    └── Interactive lesson player
```

### DNS Records Needed

1. **A Record or CNAME:**
   - `curiouskelly.com` → Vercel IP/CNAME
   - `www.curiouskelly.com` → Vercel IP/CNAME (optional)

2. **Cloudflare Pages:**
   - Custom domain: `curiouskelly.com`
   - Path: `/lesson-player`
   - OR: Subdomain: `lessons.curiouskelly.com`

---

## 🎯 Key Points to Remember

### 1. Two Separate Apps
- **Marketing Site** = Sales & Marketing (Vercel)
- **Lesson Player** = Product & Learning (Cloudflare Pages)

### 2. They Work Together
- Marketing site drives traffic
- Lesson player delivers value
- Both share the same domain

### 3. Different Technologies
- Marketing: Astro (static site generator)
- Lesson Player: Vanilla HTML/JS/CSS (no framework)

### 4. Different Hosts
- Marketing: Vercel
- Lesson Player: Cloudflare Pages

---

## 🚀 Quick Reference

### Marketing Site
- **URL:** `curiouskelly.com`
- **Host:** Vercel
- **Framework:** Astro
- **Purpose:** Convert visitors
- **Status:** ⏳ Not deployed yet

### Lesson Player
- **URL:** `curiouskelly.com/lesson-player`
- **Host:** Cloudflare Pages
- **Framework:** Static HTML
- **Purpose:** Deliver lessons
- **Status:** ✅ Deployed successfully

---

## 📝 Next Steps

1. **Deploy Marketing Site:**
   - Create Vercel project
   - Connect to GitHub
   - Deploy `curiouskelly-marketing-site/`

2. **Configure Routing:**
   - Set up DNS to route root → Vercel
   - Configure `/lesson-player` → Cloudflare Pages
   - OR: Use Vercel rewrites to proxy to Cloudflare

3. **Add Navigation:**
   - Add "Try Lesson" button on marketing site
   - Add "Back to Home" link in lesson player
   - Ensure smooth user flow

---

## ❓ Common Questions

### Q: Why two separate deployments?
**A:** 
- Marketing site needs Astro build process
- Lesson player is simple static files
- Different hosting optimizes for each use case
- Easier to update independently

### Q: Can they share the same domain?
**A:** 
- Yes! Both can use `curiouskelly.com`
- Marketing site: `/` (root)
- Lesson player: `/lesson-player` (subdirectory)

### Q: How do users navigate between them?
**A:**
- Marketing site has links to `/lesson-player`
- Lesson player has links back to `/`
- Standard web navigation (no special setup needed)

### Q: What's the homepage?
**A:**
- **Homepage** = Marketing site at `curiouskelly.com/`
- Shows hero, features, lead form
- **Lesson Player** = Separate app at `curiouskelly.com/lesson-player`
- Shows interactive lesson interface

---

**Last Updated:** 2025-01-11  
**Status:** Architecture documented  
**Next:** Deploy marketing site and configure routing

