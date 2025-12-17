# AUDIT: index.html & learn.html Migration Map

**Generated:** December 13, 2025  
**Purpose:** Comprehensive audit for UI consolidation

---

## 📋 AUDIT: index.html (Marketing Landing Page)

### 1. Sections/Components

| Component | Lines | Description |
|-----------|-------|-------------|
| **Header** | 3237-3257 | Fixed nav: Logo, Curriculum, Pricing, Calendar, About, Start Learning CTA |
| **Hero** | 3330-3444 | 40/60 split - Content left, Kelly video right |
| **Value Strip** | 3449-3469 | Stats: 365 lessons, 12 personas, 2+ ages, 5 min/day |
| **Platform Depth** | 3474-3506 | 4 cards: 365 Lessons, 12 Styles, Every Learner, Calendar Sync |
| **Today's Lesson** | 3513-3582 | Mini calendar + Today's lesson card |
| **Video Demo** | 3587-3599 | "Watch Kelly Teach" - archetype toggles |
| **Pricing** | ~1975-2077 | 3 price cards (implied structure) |
| **Gifts** | ~2080-2186 | Collapsible gift grid |
| **Careers** | ~2190-2365 | Earnings calculator, tier grid |
| **Enterprise** | ~2368-2400 | Collapsible enterprise features |
| **About Kelly** | ~2404-2460 | Mission grid, Kelly large avatar |
| **Newsroom** | ~2464-2508 | Collapsible press releases |
| **Footer** | 2512-2720 | 6-column grid, brand, social, app badges |

### 2. Interactive Elements

| Element | Location | Action | ID/Class |
|---------|----------|--------|----------|
| Mobile hamburger | Header | Toggle nav | `.mobile-menu-btn` |
| Start Learning btn | Header | Scroll to hero | `.btn-primary` |
| Today's lesson CTA | Hero | Navigate to learn.html | `a[href="/learn.html"]` |
| View curriculum | Hero | Navigate to curriculum | `a[href="/curriculum.html"]` |
| Google signup | Hero | OAuth login | `#hero-btn-google` |
| Email signup | Hero | Show modal | `#hero-btn-email` |
| Age pills | Hero Kelly | Change Kelly age preview | `#persona-age-pills` |
| Teaching style select | Hero Kelly | Change persona | `#persona-tone-select` |
| Persona thumbnails | Hero Kelly | Switch persona | `#personas-strip` |
| Calendar nav | Today's Lesson | Navigate months | `navigateMonth()` |
| Calendar days | Today's Lesson | Select day | `.cal-day` |
| Archetype toggle | Video Demo | Switch video | `.archetype-btn` |
| Collapsible headers | Gifts/Enterprise/Newsroom | Expand/collapse | `.collapsible-header` |
| Price cards | Pricing | (display only) | `.price-card` |
| Proof strip toggle | Hero (mobile) | Accordion | `toggleProofStrip()` |

### 3. External Links

| Link | Destination |
|------|-------------|
| `/curriculum.html` | Full curriculum browser |
| `/calendar.html` | Calendar view |
| `/learn.html` | Lesson player (learn.html) |
| `/trust.html` | COPPA/Trust policy |
| Google Fonts | fonts.googleapis.com |
| Supabase CDN | tvjalxxsyryjphkforjv.supabase.co |

### 4. Kelly-Related UI

| Element | Description | Location |
|---------|-------------|----------|
| Hero Kelly video | Autoplay video with poster | Right column (60%) |
| Age pills | kid/teen/adult/elder | `.kelly-age-pills` |
| Persona dropdown | 12 teaching styles | `#persona-tone-select` |
| Persona thumbnail strip | Visual persona selector | `#personas-strip` |
| Kelly in About section | Large avatar 280x280 | `.kelly-avatar-large` |
| Kelly home presence | Fixed bottom-right bubble | `.kelly-home-presence` |

### 5. Marketing Copy

| Element | Content |
|---------|---------|
| Eyebrow | "365 DAYS. 365 LESSONS. ONE TEACHER." |
| Headline | "A year from now, you'll know 365 things you don't know today." |
| Today's Question | Dynamic (e.g., "Why do leaves change color?") |
| CTA Primary | "Start today's lesson →" |
| CTA Secondary | "View all 365 lessons" |
| Subtext | "5 minutes. No signup. Just learning." |
| Welcome | "Whether you have 5 minutes or 50..." |
| Auth note | "Create your account to track your 365-day journey" |
| Trust badges | "🔒 COPPA compliant", "👨‍👩‍👧‍👦 Family safe", "📱 Works offline" |

---

## 📋 AUDIT: learn.html (Kelly OS App)

### Scenes Overview

| Scene ID | Purpose | Nav Access |
|----------|---------|------------|
| `scene-character` | Choose Your Kelly carousel | Default/Change |
| `scene-lesson` | Lesson player | Play button |
| `scene-journey` | Progress calendar | Journey tab |
| `scene-settings` | Settings panel | Settings tab |
| `scene-complete` | Lesson completion | After lesson |
| `scene-achievements` | Badge collection | Via Settings |

---

### Scene: CHARACTER SELECT (`#scene-character`)

**UI Elements:**
| Element | ID/Class | Description |
|---------|----------|-------------|
| Scene title | `.scene-title` | "Choose Your Kelly" |
| Kelly clock | `#kelly-clock` | Live time display |
| Carousel track | `#carousel-track` | 3D card carousel |
| Carousel arrows | `#carousel-prev`, `#carousel-next` | Navigation |
| Carousel dots | `#carousel-dots` | Indicator dots |
| Start button | `#btn-start` | "Let's Learn →" |
| Match quiz | `.btn-secondary` | "Take the Match Quiz" |

**Data Displayed:**
- 12 Kelly personas with: name, tagline, speech bubble, avatar, personality stats (curiosity, warmth, energy, structure)

**Actions:**
- Swipe/click carousel to select Kelly
- Click start to begin lesson

**Missing:**
- ⚠️ Match Quiz not implemented
- ⚠️ No "recommended for you" logic

---

### Scene: LESSON PLAYER (`#scene-lesson`)

**UI Elements:**
| Element | ID/Class | Description |
|---------|----------|-------------|
| Day badge | `#day-badge` | Date + topic + LIVE indicator |
| Playback controls | `.header-playback-controls` | ⏮️ ⏸️ ⏭️ + Auto toggle |
| Phase bar | `#phase-bar` | 5 clickable phases (Hook→Wisdom) |
| Kelly container | `.lesson-kelly-container` | Responsive Kelly display |
| Kelly image | `#lesson-kelly-img` | Current archetype image |
| Tap zones | `.tap-zones` | Prev/Next tap areas |
| Side actions | `.side-actions` | Bookmark, Infographic, Ask |
| Caption area | `.caption-area` | Phase icon + transcript text |
| Reactions | `.reactions` | "💡 Got it", "🤯 Wow", "💭 More" |

**Data Displayed:**
- Lesson date (from KellyTime)
- Lesson topic (from core_lessons)
- Phase content (from lesson_atoms)
- Kelly archetype image

**Actions:**
- Navigate phases (prev/next/tap)
- Toggle auto-advance
- Bookmark moment
- View infographic
- Ask Kelly (💬)
- Click reactions

**Missing:**
- ⚠️ Audio playback not visible (TTS or pre-rendered)
- ⚠️ Infographic overlay mostly placeholder
- ⚠️ "Ask Kelly" not connected to AI

---

### Scene: JOURNEY (`#scene-journey`)

**UI Elements:**
| Element | ID/Class | Description |
|---------|----------|-------------|
| Stats row | `.stats-row` | Streak, Completed, Time |
| View toggle | `.view-toggle` | Week / Calendar |
| Week view | `#week-view` | 7-day list with thumbnails |
| Week nav | `.week-nav` | ← → buttons |
| Grid view | `#grid-view` | Monthly calendar grid |

**Data Displayed:**
- Streak count
- Total completed lessons
- Time spent learning
- Weekly lesson list with thumbnails
- Calendar grid with completion markers

**Actions:**
- Toggle week/grid view
- Navigate weeks
- Click day to load lesson

**Missing:**
- ⚠️ Time tracking is approximate
- ⚠️ No month navigation in grid view

---

### Scene: SETTINGS (`#scene-settings`)

**UI Elements:**
| Section | Items |
|---------|-------|
| **Your Kelly** | Preview avatar, name, tagline, "Change" button |
| **Learning** | Teaching Age slider (2-102), Kelly Size slider (0/50/100), Language, Voice Speed |
| **Reminders** | Daily Reminder toggle, Reminder Time |
| **Progress** | Achievements, Bookmarks, Downloaded Lessons |
| **Account** | Profile, Family, Subscription |
| **Support** | About, Help & FAQ, Feedback |

**Data Displayed:**
- Current Kelly selection
- Age setting with emoji
- Kelly presence mode (Zen/Balanced/Immersive)
- Subscription status

**Actions:**
- Change Kelly → scene-character
- Adjust age slider
- Toggle Kelly size
- Toggle reminders
- Navigate to achievements

**Missing:**
- ⚠️ Language picker not fully wired
- ⚠️ Voice speed picker not fully wired
- ⚠️ Family management not implemented
- ⚠️ Offline downloads placeholder

---

### Scene: COMPLETE (`#scene-complete`)

**UI Elements:**
| Element | ID/Class | Description |
|---------|----------|-------------|
| Kelly avatar | `#complete-kelly-img` | Celebration image |
| Title | `.complete-title` | "Beautiful!" |
| Message | `#complete-message` | Completion message |
| Streak display | `.streak-display` | 🔥 streak count |
| Tomorrow teaser | `.tomorrow-teaser` | Next lesson preview |
| Continue button | `#btn-next-lesson` | "Continue Journey →" |
| Review button | `[data-scene="lesson"]` | "Review This Lesson" |

**Data Displayed:**
- Current streak
- Tomorrow's topic

**Actions:**
- Continue to next lesson
- Review current lesson

---

### Scene: ACHIEVEMENTS (`#scene-achievements`)

**UI Elements:**
| Element | ID/Class | Description |
|---------|----------|-------------|
| Back button | `[data-scene="settings"]` | Return to settings |
| Progress | `#achievements-progress` | "0 of 24 unlocked" |
| Badge grid | `#badge-grid` | 3-column badge display |

**Data Displayed:**
- 12 predefined badges with unlock status

**Missing:**
- ⚠️ Badges are placeholder (unlock logic incomplete)

---

### Overlays

| Overlay | ID | Purpose |
|---------|-----|---------|
| Infographic | `#overlay-infographic` | Visual learning display |
| Picker | `#overlay-picker` | Language/voice/time selection |
| Parental Gate | `#parental-gate` | COPPA math challenge |
| Paywall | `#paywall` | IAP subscription options |

---

## 🔄 MIGRATION CHECKLIST

| Item | From | To | Action | Priority |
|------|------|----|--------|----------|
| Kelly persona selector (age pills) | index.html hero | Settings → Your Kelly (learn.html) | **Move** | 🔴 High |
| Teaching style dropdown | index.html hero | Settings → Learning (learn.html) | **Move** | 🔴 High |
| Persona thumbnail strip | index.html hero | Settings → Your Kelly (learn.html) | **Move** | 🔴 High |
| Today's question hook | index.html hero | lesson scene header (learn.html) | **Move** | 🟡 Medium |
| Mini calendar | index.html | Journey scene (learn.html) | **Consolidate** | 🟡 Medium |
| Trust badges | index.html hero | Settings → About (learn.html) | **Move** | 🟢 Low |
| Video demo section | index.html | Keep on index only | **Keep** | — |
| Pricing section | index.html | Keep on index only | **Keep** | — |
| Google/Email auth | index.html hero | Settings → Account (learn.html) | **Add auth** | 🔴 High |
| Kelly home presence bubble | index.html | Remove (learn.html has full Kelly) | **Remove** | 🟢 Low |
| Live indicator | Both files | Consolidate to shared CSS | **DRY** | 🟡 Medium |
| Phase bar styles | learn.html | Extract to shared CSS | **Refactor** | 🟢 Low |
| Supabase client init | Both files | Extract to shared JS | **DRY** | 🔴 High |
| Kelly image helper | Both files | Extract to kelly-presence.js | **DRY** | 🔴 High |
| Age bucket mapping | Both files | Extract to kelly-presence.js | **DRY** | 🔴 High |
| COPPA parental gate | learn.html | Add to index.html (on sign up) | **Add** | 🟡 Medium |

---

## 🔁 REDUNDANCIES (Should be consolidated)

### Duplicated Code

| Pattern | Locations | Recommendation |
|---------|-----------|----------------|
| **Supabase client creation** | index.html (3x), learn.html (1x) | Extract to `/js/kelly-supabase.js` |
| **Kelly persona data (FALLBACK_KELLYS)** | learn.html only | Move to `/js/kelly-personas.js`, import in both |
| **Age bucket logic (getAgeBucket)** | learn.html | Move to `/js/kelly-presence.js` |
| **PERSONA_TO_ARCHETYPE mapping** | learn.html | Move to `/js/kelly-personas.js` |
| **Live indicator animation CSS** | index.html, learn.html | Move to `/styles/kelly-foundation.css` |
| **Hero date badge styling** | index.html | Merge with learn.html `.day-badge` |
| **Kelly image URL builder** | learn.html | Move to `/js/kelly-presence.js` |
| **Proof/trust strip** | index.html hero, learn.html settings | Single source for copy |
| **Mobile hamburger menu** | index.html | learn.html uses bottom nav (OK) |
| **Google Fonts loading** | index.html only | Add to learn.html if needed |

### Duplicated Concepts

| Concept | index.html | learn.html | Resolution |
|---------|------------|------------|------------|
| Kelly selector UI | Persona strip + dropdown | Carousel + settings | **Carousel is richer** → use in settings |
| Calendar display | Mini calendar widget | Journey grid view | **Consolidate to Journey** |
| Progress tracking | localStorage keys differ | `kellyState` vs `kelly_progress` | **Already bridged** |
| Age adaptation | Pills in hero | Slider in settings | **Slider is better UX** → remove pills |
| Kelly presence toggle | Fixed bubble | Slider + FAB | **Slider is better UX** |

---

## 📊 RECOMMENDATIONS

### Immediate (Pre-Launch)

1. **Extract shared JS modules:**
   - `/js/kelly-supabase.js` — single Supabase client factory
   - `/js/kelly-personas.js` — FALLBACK_KELLYS + archetype mappings
   - Move age bucket + image URL helpers to existing `/js/kelly-presence.js`

2. **Remove redundant hero selectors:**
   - Age pills → Settings only
   - Persona dropdown → Settings only  
   - Keep hero clean: just Kelly video + CTA

3. **Add auth to learn.html:**
   - Currently no sign-in flow in learn.html
   - Add Google/Email buttons to Settings → Account

### Short-Term (Post-Launch)

4. **Consolidate calendar:**
   - index.html mini calendar → link to learn.html Journey
   - Avoid maintaining two calendar renderers

5. **Single CSS for Kelly elements:**
   - Extract `.kelly-*` classes to `/styles/kelly-components.css`
   - Import in both pages

6. **Match Quiz implementation:**
   - Button exists but no logic
   - Would improve onboarding

### Long-Term (Refactor)

7. **Consider SPA architecture:**
   - Two large HTML files with overlapping code
   - Could benefit from Astro/React component extraction

---

## ✅ SUMMARY

| Metric | index.html | learn.html |
|--------|------------|------------|
| Lines | ~5182 | ~5590 |
| Sections | 12 | 6 scenes |
| Interactive elements | ~25 | ~40 |
| Kelly UI components | 6 | 8 |
| Shared JS dependencies | 6 | 8 |
| CSS (inline) | ~3200 lines | ~2200 lines |

**Key insight:** Both files duplicate ~30% of their JavaScript logic and ~20% of their CSS. Extracting shared modules would reduce maintenance burden significantly.





