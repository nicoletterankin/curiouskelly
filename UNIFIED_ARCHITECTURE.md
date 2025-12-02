# 🎯 Curious Kelly - Unified Architecture

## The Two-Page Experience

### Page 1: Login (`/index.html`)
```
┌────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    [Kelly Image - Hero]                             │
│                                                                     │
│                    ✨ Curious Kelly                                 │
│                                                                     │
│                    Curious? Always.                                 │
│                    The AI for lifelong learners.                    │
│                                                                     │
│                 ┌─────────────────────────┐                         │
│                 │  G  Continue with Google │                        │
│                 └─────────────────────────┘                         │
│                 ┌─────────────────────────┐                         │
│                 │   Continue with Apple   │                        │
│                 └─────────────────────────┘                         │
│                                                                     │
│                 ┌─────────────────────────┐                         │
│                 │  ✨ Start Learning Now  │  (Guest)               │
│                 └─────────────────────────┘                         │
│                                                                     │
│                 No account needed. Upgrade anytime.                 │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│  Explore        About           Social         Download             │
│  • Missions     • About         • Twitter      • App Store          │
│  • Curriculum   • Careers       • Instagram    • Google Play        │
│  • Gifts        • Newsroom      • YouTube      • Amazon             │
│  • Enterprise   • Privacy       • LinkedIn     • Roku               │
│                 • Terms                                             │
│                 • Diversity                                         │
├────────────────────────────────────────────────────────────────────┤
│  © 2025 Curious Kelly PBC               Made on Earth 🌍           │
└────────────────────────────────────────────────────────────────────┘
```

### Page 2: Kelly OS (`/app.html`)
```
┌────────────────────────────────────────────────────────────────────┐
│ ┌──────────┐ ┌───────────────────────────────────────────────────┐ │
│ │ Calendar │ │                                                   │ │
│ │          │ │                                                   │ │
│ │ TODAY    │ │              ┌─────────────────┐                  │ │
│ │ • Day 1  │ │              │                 │                  │ │
│ │          │ │              │   KELLY IMAGE   │                  │ │
│ │ UPCOMING │ │              │  (fallback)     │                  │ │
│ │ • Day 2  │ │              │                 │                  │ │
│ │ • Day 3  │ │              │   or UNITY 3D   │                  │ │
│ │          │ │              │  (when loaded)  │                  │ │
│ │ PROGRESS │ │              │                 │                  │ │
│ │ 🔥 3 day │ │              └─────────────────┘                  │ │
│ │   streak │ │                                                   │ │
│ │          │ │  ┌─────────────────────────────────────────────┐  │ │
│ │ SETTINGS │ │  │         LESSON CONTENT OVERLAY              │  │ │
│ │ Sign Out │ │  │  Question / Wisdom / Interactive Elements   │  │ │
│ └──────────┘ │  └─────────────────────────────────────────────┘  │ │
│              │  [Type your answer or ask Kelly for a hint...]    │ │
│              └───────────────────────────────────────────────────┘ │
│  [Footer: Privacy | Terms | Help]                                  │
└────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. 16:9 Aspect Ratio Lock
- Main content area maintains 16:9
- Mobile: Full width, aspect-ratio maintained
- Desktop: Centered with max-width

### 2. Kelly Always Visible
```javascript
// Load order:
1. Kelly IMAGE loads immediately (static, <100KB)
2. Unity WebGL loads in background
3. When Unity ready: fade from image to 3D
4. If Unity fails: image stays (graceful degradation)
```

### 3. Mobile-First Responsive
```
Mobile (<768px):    Stacked layout, sidebar becomes drawer
Tablet (768-1024):  Side-by-side, compact
Desktop (>1024):    Full 16:9 experience
```

## File Structure (Canonical)
```
public/
├── index.html         ← Login page (OAuth + Footer)
├── app.html           ← Kelly OS (single unified experience)
├── js/
│   ├── auth.js        ← Supabase auth utilities
│   ├── calendar.js    ← Calendar/lesson navigation
│   └── kelly-loader.js ← Image→3D transition logic
├── css/
│   └── unified.css    ← Single stylesheet for both pages
├── images/
│   └── kelly/
│       └── kelly-directors-chair-*.png  ← Fallback images
├── data/
│   └── 365_day_calendar.json  ← Lesson data
└── unity/
    └── kelly-live/    ← Unity WebGL build (hosted on CDN)
```

## Kelly Image → 3D Transition

```javascript
class KellyPresence {
  constructor() {
    this.kellyImage = document.getElementById('kelly-image');
    this.unityContainer = document.getElementById('unity-container');
    this.isUnityReady = false;
  }

  init() {
    // 1. Show Kelly image immediately
    this.showKellyImage();
    
    // 2. Start Unity loading in background
    this.loadUnity();
  }

  showKellyImage() {
    this.kellyImage.style.opacity = '1';
    this.kellyImage.src = '/images/kelly/kelly-directors-chair-neutral.png';
  }

  async loadUnity() {
    try {
      // Load Unity (may take 5-30 seconds)
      await createUnityInstance(this.unityContainer, unityConfig);
      this.isUnityReady = true;
      
      // Crossfade from image to Unity
      this.crossfadeToUnity();
    } catch (error) {
      console.log('Unity unavailable, Kelly image remains');
      // Image stays visible - graceful degradation
    }
  }

  crossfadeToUnity() {
    // Smooth crossfade over 500ms
    this.kellyImage.style.transition = 'opacity 0.5s ease';
    this.kellyImage.style.opacity = '0';
    
    this.unityContainer.style.transition = 'opacity 0.5s ease';
    this.unityContainer.style.opacity = '1';
  }
}
```

## Lesson Loading Strategy

1. **Primary**: Supabase `lesson_shards` or `core_lessons` table
2. **Fallback**: Local `/data/365_day_calendar.json`
3. **Fallback**: Local `/lessons/{slug}-dna.json`
4. **Last Resort**: Built-in sample lesson

## Authentication Flow

```
User visits / → Check session
  ├─ Has session → Redirect to /app.html
  └─ No session → Show login options
       ├─ Google OAuth → Supabase → /app.html
       ├─ Apple OAuth → Supabase → /app.html
       └─ Guest Mode → localStorage flag → /app.html
```

## Redirects Configuration

### Supabase URL Configuration:
- Site URL: `https://curiouskelly.com`
- Redirect URLs:
  - `https://curiouskelly.com/app.html`
  - `https://www.curiouskelly.com/app.html`
  - `https://curiouskelly.com/`
  - `https://www.curiouskelly.com/`

## Unity Hosting Strategy

Due to 227MB file size (exceeds GitHub 100MB limit):
- **Option A**: Git LFS (increases storage costs)
- **Option B**: Cloudflare R2 / S3 bucket with CDN
- **Option C**: Netlify Drop (current temporary solution)

Recommended: Cloudflare R2 with custom domain subdomain:
- `unity.curiouskelly.com/kelly-live/Build/...`

## Version Consolidation

All other versions are now archived:
- `daily-lesson-marketing/` → Reference only
- `curious-kellly/` → Archived
- `app/` → Archived
- `_archive/` → Archived

**Canonical source**: `public/` folder









