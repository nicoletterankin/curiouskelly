# KELLY EXPERT MODE: FINAL PUSH
## December 12, 2025 — 5 Days to Launch

---

# AGENT 1: CORE EXPERIENCE

## MISSION
Make every lesson load instantly with full content. Remove all gates for testing. Ensure the complete learning journey works flawlessly.

---

## TASK 1: DISABLE PAYWALL FOR TESTING

Find and disable the paywall modal. It's blocking the experience.

```bash
# Find paywall code
grep -rn "paywall\|Unlock 365\|pricing\|subscribe" public/
grep -rn "showPaywall\|openPaywall\|paywallModal" public/
```

**Option A: Comment out the trigger**
```javascript
// Find where paywall is triggered (likely in learn.html or a JS file)
// Comment out or wrap in a feature flag:

const TESTING_MODE = true; // Set false for production

if (!TESTING_MODE && !userHasAccess()) {
  showPaywall();
}
```

**Option B: Add testing bypass**
```javascript
// Check URL param: ?bypass=testing
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('bypass') === 'testing') {
  localStorage.setItem('kelly_bypass_paywall', 'true');
}

// In paywall check:
if (localStorage.getItem('kelly_bypass_paywall') === 'true') {
  return; // Skip paywall
}
```

**Option C: Set unlimited access in localStorage**
```javascript
// Console command for testing:
localStorage.setItem('kelly_subscription_status', 'lifetime');
localStorage.setItem('kelly_access_level', 'unlimited');
```

**Deliverable:** Paywall disabled. Full app accessible for testing.

---

## TASK 2: LESSON LOADING PIPELINE

Ensure every calendar date loads its lesson correctly.

### 2.1 Verify kelly-lesson-loader.js is working

```javascript
// Test in browser console:
KellyLessonLoader.init(supabase);

// Test Day 1
const day1 = await KellyLessonLoader.getLesson(1, { archetype: 'The Scientist', age: 30 });
console.log('Day 1:', day1.title, day1.greeting, day1.script?.substring(0, 100));

// Test Day 347 (today)
const today = await KellyLessonLoader.getToday({ archetype: 'The Explorer', age: 8 });
console.log('Today:', today.title, today.greeting);

// Test all archetypes
const archetypes = ['The Scientist', 'The Explorer', 'The Nerd', 'The Artist', 'The Coach', 
                    'The Storyteller', 'The Philosopher', 'The Optimist', 'The Mystic', 
                    'The Maverick', 'The Guardian', 'The Sage'];
for (const arch of archetypes) {
  const lesson = await KellyLessonLoader.getLesson(1, { archetype: arch });
  console.log(`${arch}:`, lesson.greeting?.substring(0, 50) || 'NO GREETING');
}
```

### 2.2 Fix learn.html lesson display

The lesson modal/view should show:
1. **Topic title** (from core_lessons.title)
2. **Hook question** (from core_lessons.hook_question or marketing_hook)
3. **Kelly greeting** (from lesson_atoms where dialog_type = 'greeting')
4. **Lesson content** (from lesson_shards.script_content)
5. **Image** (from lesson_assets or fallback)
6. **Audio** (from lesson_assets or ElevenLabs URL)

```javascript
// In learn.html, the loadLesson function should:
async function loadLesson(dayNumber) {
  const kellyId = localStorage.getItem('kelly_current_persona') || 'scientist';
  const age = parseInt(localStorage.getItem('kelly_teaching_age') || '30');
  
  // Normalize archetype name
  const archetype = 'The ' + kellyId.charAt(0).toUpperCase() + kellyId.slice(1);
  
  // Fetch complete lesson
  const lesson = await KellyLessonLoader.getLesson(dayNumber, { archetype, age });
  
  // Update UI
  document.querySelector('.lesson-title').textContent = lesson.title;
  document.querySelector('.lesson-subtitle').textContent = lesson.subtitle || lesson.hookQuestion;
  document.querySelector('.lesson-image').src = lesson.imageUrl || '/images/fallback.png';
  
  // Kelly speaks the greeting
  if (lesson.greeting) {
    showKellyDialog(lesson.greeting, lesson.emotion || 'friendly');
  }
  
  // Render main content
  const contentArea = document.querySelector('.lesson-content');
  contentArea.innerHTML = formatContent(lesson.script || lesson.content);
  
  // Load audio
  if (lesson.audioUrl) {
    const audio = document.getElementById('lesson-audio');
    audio.src = lesson.audioUrl;
    audio.load();
  }
  
  // Track view
  trackLessonView(dayNumber, kellyId);
}
```

### 2.3 Calendar click → Lesson load

When user taps a date on the calendar, it should:
1. Close calendar view
2. Load that day's lesson
3. Show lesson content
4. Auto-play Kelly greeting (optional)

```javascript
// Calendar date click handler
function handleCalendarDateClick(dayNumber) {
  // Close calendar panel if open
  closePanel('calendar');
  
  // Show loading state
  showLoadingState();
  
  // Load lesson
  loadLesson(dayNumber).then(() => {
    hideLoadingState();
    
    // Optional: auto-play audio greeting
    if (localStorage.getItem('kelly_autoplay') === 'true') {
      playKellyGreeting();
    }
  });
}
```

**Deliverable:** Clicking any calendar date loads that lesson with full personalization.

---

## TASK 3: AUDIO PIPELINE

Kelly needs to speak. Verify audio loads and plays.

### 3.1 Check audio sources

```sql
-- In Supabase, check if lessons have audio
SELECT day_number, title, audio_url 
FROM core_lessons 
WHERE audio_url IS NOT NULL 
LIMIT 20;

-- Check lesson_assets for audio
SELECT lesson_id, asset_type, public_url 
FROM lesson_assets 
WHERE asset_type = 'audio' 
LIMIT 20;
```

### 3.2 Audio player implementation

```javascript
const KellyAudio = {
  player: null,
  
  init() {
    this.player = document.getElementById('kelly-audio') || this.createPlayer();
  },
  
  createPlayer() {
    const audio = document.createElement('audio');
    audio.id = 'kelly-audio';
    audio.preload = 'auto';
    document.body.appendChild(audio);
    return audio;
  },
  
  async play(url) {
    if (!url) return;
    
    this.player.src = url;
    this.player.load();
    
    try {
      await this.player.play();
    } catch (e) {
      console.log('Audio autoplay blocked, user interaction required');
    }
  },
  
  pause() {
    this.player.pause();
  },
  
  // Generate TTS if no pre-recorded audio
  async generateTTS(text, voice = 'kelly-scientist') {
    // If using ElevenLabs:
    const response = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice })
    });
    
    if (response.ok) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      this.play(url);
    }
  }
};
```

**Deliverable:** Audio plays for lessons that have it. Graceful fallback for those without.

---

## TASK 4: IMAGE PIPELINE

Every lesson should have a visual.

### 4.1 Check image sources

```sql
-- In Supabase
SELECT day_number, title, hero_image_url, thumbnail_url, hook_image_url
FROM core_lessons
WHERE hero_image_url IS NOT NULL
LIMIT 20;

-- Check lesson_assets
SELECT la.*, cl.day_number, cl.title
FROM lesson_assets la
JOIN core_lessons cl ON la.lesson_id = cl.id
WHERE la.asset_type = 'image'
LIMIT 20;
```

### 4.2 Image loading with fallbacks

```javascript
function getLessonImage(lesson, assets) {
  // Priority order:
  // 1. lesson.hero_image_url
  // 2. lesson_assets with type='image' 
  // 3. Cloudinary/storage path
  // 4. Generated fallback
  
  if (lesson.hero_image_url) return lesson.hero_image_url;
  
  const imageAsset = assets?.find(a => a.asset_type === 'image');
  if (imageAsset?.public_url) return imageAsset.public_url;
  
  // Storage path pattern
  const dayStr = String(lesson.day_number).padStart(3, '0');
  const storagePath = `https://your-storage.com/lessons/day-${dayStr}/hero.png`;
  
  // Fallback
  return `/images/lessons/fallback-${lesson.category || 'general'}.png`;
}
```

### 4.3 Lazy loading for performance

```html
<!-- In lesson display -->
<img 
  class="lesson-image"
  loading="lazy"
  decoding="async"
  src="/images/placeholder.png"
  data-src=""
  alt=""
>
```

```javascript
// Intersection Observer for lazy loading
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      imageObserver.unobserve(img);
    }
  });
});

document.querySelectorAll('img[data-src]').forEach(img => {
  imageObserver.observe(img);
});
```

**Deliverable:** Images load for all lessons. Fallbacks work. No broken images.

---

## TASK 5: PROGRESS TRACKING

Save user progress without requiring login.

### 5.1 Local progress storage

```javascript
const KellyProgress = {
  STORAGE_KEY: 'kelly_progress',
  
  getProgress() {
    const data = localStorage.getItem(this.STORAGE_KEY);
    return data ? JSON.parse(data) : { completed: [], streaks: { current: 0, best: 0 }, lastDate: null };
  },
  
  saveProgress(progress) {
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(progress));
  },
  
  markComplete(dayNumber) {
    const progress = this.getProgress();
    
    if (!progress.completed.includes(dayNumber)) {
      progress.completed.push(dayNumber);
    }
    
    // Update streak
    const today = new Date().toISOString().split('T')[0];
    if (progress.lastDate !== today) {
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      
      if (progress.lastDate === yesterday) {
        progress.streaks.current++;
      } else {
        progress.streaks.current = 1;
      }
      
      if (progress.streaks.current > progress.streaks.best) {
        progress.streaks.best = progress.streaks.current;
      }
      
      progress.lastDate = today;
    }
    
    this.saveProgress(progress);
    return progress;
  },
  
  isComplete(dayNumber) {
    return this.getProgress().completed.includes(dayNumber);
  },
  
  getStats() {
    const progress = this.getProgress();
    return {
      completed: progress.completed.length,
      streak: progress.streaks.current,
      bestStreak: progress.streaks.best
    };
  }
};
```

### 5.2 Visual feedback on calendar

```javascript
// Mark completed days on calendar
function updateCalendarProgress() {
  const progress = KellyProgress.getProgress();
  
  document.querySelectorAll('.calendar-day').forEach(dayEl => {
    const dayNum = parseInt(dayEl.dataset.day);
    
    if (progress.completed.includes(dayNum)) {
      dayEl.classList.add('completed');
      dayEl.querySelector('.completion-check')?.classList.remove('hidden');
    }
  });
}
```

**Deliverable:** Progress saves locally. Calendar shows completed days. Streaks calculate correctly.

---

## TASK 6: REMOVE LOGOUT BUTTON

Kelly is always there. No logout.

```bash
# Find logout references
grep -rn "logout\|sign.out\|signOut\|Log Out\|Sign Out" public/

# Common locations:
# - settings panel
# - user menu
# - header nav
```

**Replace with:**
```html
<!-- Instead of Sign Out, show Sync option -->
<div class="account-section">
  <h3>Your Progress</h3>
  <p>You've completed <span id="completed-count">0</span> lessons!</p>
  
  <button class="btn-secondary" onclick="showSyncModal()">
    Sync to Cloud →
  </button>
  <p class="hint">Access your progress on any device</p>
</div>
```

**Deliverable:** No logout button anywhere. "Sync Progress" available for those who want accounts.

---

## VERIFICATION CHECKLIST

Run through this entire flow:

```
□ Open curiouskelly.com/learn.html
□ Paywall does NOT appear (disabled for testing)
□ Today's lesson loads automatically
□ Topic title displays correctly
□ Kelly greeting shows (personalized to selected archetype)
□ Lesson content renders (from lesson_shards)
□ Image loads (or fallback shows)
□ Audio plays when available
□ Click calendar icon → calendar opens
□ Click a different date → that lesson loads
□ Complete a lesson → progress saves
□ Refresh page → progress persists
□ Check Settings → no Logout button
□ Change Kelly → greeting changes on next lesson
□ Change Age → content adapts
□ Test Days 1, 100, 200, 300, 365 → all load
```

---

# AGENT 2: SEO & HOMEPAGE

## MISSION
Rewrite the homepage for marketing. Create SEO infrastructure so Google indexes all 365 lessons.

---

## TASK 1: HOMEPAGE REWRITE

The homepage should sell the product, not demo technical features.

### 1.1 New index.html structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Curious Kelly — Learn Something New Every Day</title>
  <meta name="description" content="365 daily lessons with 12 AI teachers. Personalized learning for ages 2-102. 5 minutes a day that change everything.">
  
  <!-- Open Graph -->
  <meta property="og:title" content="Curious Kelly — Daily Learning for Curious Minds">
  <meta property="og:description" content="365 lessons. 12 unique teachers. Ages 2-102. Start free today.">
  <meta property="og:image" content="https://curiouskelly.com/images/og-home.png">
  <meta property="og:url" content="https://curiouskelly.com">
  <meta property="og:type" content="website">
  
  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@curiouskelly">
  
  <!-- Favicon -->
  <link rel="icon" type="image/png" href="/favicon.png">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  
  <!-- Fonts & Styles -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/styles/home.css">
  
  <!-- Schema.org -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Curious Kelly",
    "description": "Daily learning platform with AI-powered personalized lessons",
    "url": "https://curiouskelly.com",
    "applicationCategory": "EducationalApplication",
    "operatingSystem": "Web",
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD",
      "description": "Day 1 free forever"
    }
  }
  </script>
</head>
<body>
  <!-- NAV -->
  <nav class="nav">
    <a href="/" class="nav-logo">
      <img src="/images/logo.svg" alt="Curious Kelly" height="32">
    </a>
    <div class="nav-links">
      <a href="#kellys">Meet the Kellys</a>
      <a href="#how-it-works">How It Works</a>
      <a href="/learn.html" class="btn-primary">Start Free →</a>
    </div>
  </nav>

  <!-- HERO -->
  <header class="hero">
    <div class="hero-content">
      <p class="hero-badge">
        <span class="live-dot"></span>
        <span id="today-date">Friday, December 12, 2025</span>
      </p>
      
      <h1 class="hero-title">Today's Lesson</h1>
      <h2 class="hero-topic" id="today-topic">Loading...</h2>
      <p class="hero-hook" id="today-hook"></p>
      
      <div class="hero-cta">
        <a href="/learn.html" class="btn-primary btn-xl">
          Start Learning — Free
        </a>
        <p class="cta-subtext">No account required. 5 minutes.</p>
      </div>
    </div>
    
    <div class="hero-visual">
      <img src="/images/kelly-hero.png" alt="Kelly" class="hero-kelly">
      <div class="hero-stats">
        <div class="stat">
          <span class="stat-number">365</span>
          <span class="stat-label">Daily Lessons</span>
        </div>
        <div class="stat">
          <span class="stat-number">12</span>
          <span class="stat-label">AI Teachers</span>
        </div>
        <div class="stat">
          <span class="stat-number">2-102</span>
          <span class="stat-label">Ages Served</span>
        </div>
      </div>
    </div>
  </header>

  <!-- KELLYS -->
  <section id="kellys" class="section kellys">
    <h2 class="section-title">Meet Your Teachers</h2>
    <p class="section-subtitle">Same lesson. Your style. Choose the Kelly that speaks to you.</p>
    
    <div class="kellys-grid" id="kellys-grid">
      <!-- Populated by JS or hardcoded -->
      <div class="kelly-card">
        <img src="/images/kellys/scientist.png" alt="The Scientist">
        <h3>The Scientist</h3>
        <p>Data-driven precision</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/explorer.png" alt="The Explorer">
        <h3>The Explorer</h3>
        <p>Adventure awaits</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/nerd.png" alt="The Nerd">
        <h3>The Nerd</h3>
        <p>Deep knowledge</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/artist.png" alt="The Artist">
        <h3>The Artist</h3>
        <p>Creative expression</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/coach.png" alt="The Coach">
        <h3>The Coach</h3>
        <p>Motivational support</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/storyteller.png" alt="The Storyteller">
        <h3>The Storyteller</h3>
        <p>Narrative learning</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/philosopher.png" alt="The Philosopher">
        <h3>The Philosopher</h3>
        <p>Big questions</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/optimist.png" alt="The Optimist">
        <h3>The Optimist</h3>
        <p>Positive vibes</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/mystic.png" alt="The Mystic">
        <h3>The Mystic</h3>
        <p>Spiritual wisdom</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/maverick.png" alt="The Maverick">
        <h3>The Maverick</h3>
        <p>Break the rules</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/guardian.png" alt="The Guardian">
        <h3>The Guardian</h3>
        <p>Safe and steady</p>
      </div>
      <div class="kelly-card">
        <img src="/images/kellys/sage.png" alt="The Sage">
        <h3>The Sage</h3>
        <p>Ancient wisdom</p>
      </div>
    </div>
  </section>

  <!-- HOW IT WORKS -->
  <section id="how-it-works" class="section how-it-works">
    <h2 class="section-title">How It Works</h2>
    
    <div class="steps">
      <div class="step">
        <span class="step-number">1</span>
        <h3>Open</h3>
        <p>Visit anytime. Today's lesson is waiting.</p>
      </div>
      <div class="step">
        <span class="step-number">2</span>
        <h3>Learn</h3>
        <p>5 minutes with Kelly. One topic. Your way.</p>
      </div>
      <div class="step">
        <span class="step-number">3</span>
        <h3>Grow</h3>
        <p>365 days. 365 topics. A year of discovery.</p>
      </div>
    </div>
  </section>

  <!-- CALENDAR PREVIEW -->
  <section class="section calendar-preview">
    <h2 class="section-title">Your Year of Learning</h2>
    <p class="section-subtitle">Every day brings something new</p>
    
    <div class="months-preview">
      <div class="month">
        <h4>January</h4>
        <p class="month-theme">New Beginnings</p>
      </div>
      <div class="month">
        <h4>February</h4>
        <p class="month-theme">Love & Connection</p>
      </div>
      <div class="month">
        <h4>March</h4>
        <p class="month-theme">Growth & Change</p>
      </div>
      <!-- ... rest of months ... -->
    </div>
    
    <a href="/learn.html#calendar" class="btn-secondary">View Full Calendar →</a>
  </section>

  <!-- FINAL CTA -->
  <section class="section final-cta">
    <h2>Ready to get curious?</h2>
    <p>5 minutes a day. Every day. For life.</p>
    <a href="/learn.html" class="btn-primary btn-xl">Start Your Journey →</a>
  </section>

  <!-- FOOTER -->
  <footer class="footer">
    <div class="footer-content">
      <div class="footer-brand">
        <img src="/images/logo-white.svg" alt="Curious Kelly" height="24">
        <p>Made with 💛 for curious minds everywhere.</p>
      </div>
      <nav class="footer-links">
        <a href="/about">About</a>
        <a href="/privacy">Privacy</a>
        <a href="/terms">Terms</a>
        <a href="mailto:hello@curiouskelly.com">Contact</a>
      </nav>
    </div>
    <p class="copyright">© 2025 Curious Kelly. Patent pending.</p>
  </footer>

  <!-- SCRIPTS -->
  <script src="/js/config.js"></script>
  <script src="/js/supabase-client.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', async () => {
      // Update date
      const dateEl = document.getElementById('today-date');
      const now = new Date();
      dateEl.textContent = now.toLocaleDateString('en-US', { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
      });
      
      // Fetch today's lesson
      try {
        const dayOfYear = Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000);
        
        const { data: lesson } = await supabase
          .from('core_lessons')
          .select('title, marketing_hook')
          .eq('day_number', dayOfYear)
          .single();
        
        if (lesson) {
          document.getElementById('today-topic').textContent = lesson.title;
          document.getElementById('today-hook').textContent = lesson.marketing_hook || '';
        }
      } catch (e) {
        document.getElementById('today-topic').textContent = 'Visual Learning';
        document.getElementById('today-hook').textContent = 'A visual representation of today\'s lesson';
      }
    });
  </script>
</body>
</html>
```

### 1.2 Home CSS (/styles/home.css)

```css
/* Core variables */
:root {
  --color-bg: #0a0a1a;
  --color-surface: #1a1a2e;
  --color-primary: #4361ee;
  --color-primary-hover: #3a56d4;
  --color-accent: #ffd700;
  --color-text: #ffffff;
  --color-text-muted: #8888aa;
  --font-main: 'Inter', system-ui, sans-serif;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-main);
  background: var(--color-bg);
  color: var(--color-text);
  line-height: 1.6;
}

/* Navigation */
.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 40px;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  background: rgba(10, 10, 26, 0.9);
  backdrop-filter: blur(10px);
}

.nav-links {
  display: flex;
  gap: 32px;
  align-items: center;
}

.nav-links a {
  color: var(--color-text-muted);
  text-decoration: none;
  transition: color 0.2s;
}

.nav-links a:hover {
  color: var(--color-text);
}

/* Buttons */
.btn-primary {
  display: inline-block;
  background: var(--color-primary);
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  transition: background 0.2s, transform 0.2s;
}

.btn-primary:hover {
  background: var(--color-primary-hover);
  transform: translateY(-2px);
}

.btn-primary.btn-xl {
  padding: 16px 32px;
  font-size: 18px;
}

.btn-secondary {
  display: inline-block;
  background: transparent;
  color: var(--color-primary);
  padding: 12px 24px;
  border: 2px solid var(--color-primary);
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  transition: background 0.2s, color 0.2s;
}

.btn-secondary:hover {
  background: var(--color-primary);
  color: white;
}

/* Hero */
.hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 120px 80px 80px;
  gap: 80px;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--color-accent);
  font-size: 14px;
  margin-bottom: 16px;
}

.live-dot {
  width: 8px;
  height: 8px;
  background: #ff4444;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.hero-title {
  font-size: 24px;
  font-weight: 500;
  color: var(--color-text-muted);
  margin-bottom: 8px;
}

.hero-topic {
  font-size: 56px;
  font-weight: 700;
  margin-bottom: 16px;
  line-height: 1.2;
}

.hero-hook {
  font-size: 20px;
  color: var(--color-text-muted);
  margin-bottom: 32px;
  max-width: 500px;
}

.hero-cta {
  margin-bottom: 24px;
}

.cta-subtext {
  color: var(--color-text-muted);
  font-size: 14px;
  margin-top: 12px;
}

.hero-visual {
  flex-shrink: 0;
}

.hero-kelly {
  max-width: 400px;
  border-radius: 24px;
}

.hero-stats {
  display: flex;
  gap: 32px;
  margin-top: 24px;
}

.stat {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 32px;
  font-weight: 700;
  color: var(--color-accent);
}

.stat-label {
  font-size: 14px;
  color: var(--color-text-muted);
}

/* Sections */
.section {
  padding: 100px 80px;
}

.section-title {
  font-size: 40px;
  font-weight: 700;
  text-align: center;
  margin-bottom: 16px;
}

.section-subtitle {
  font-size: 18px;
  color: var(--color-text-muted);
  text-align: center;
  margin-bottom: 60px;
}

/* Kellys Grid */
.kellys-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.kelly-card {
  background: var(--color-surface);
  border-radius: 16px;
  padding: 24px;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
  cursor: pointer;
}

.kelly-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 40px rgba(67, 97, 238, 0.2);
}

.kelly-card img {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  margin-bottom: 16px;
}

.kelly-card h3 {
  font-size: 18px;
  margin-bottom: 4px;
}

.kelly-card p {
  font-size: 14px;
  color: var(--color-text-muted);
}

/* Steps */
.steps {
  display: flex;
  justify-content: center;
  gap: 80px;
  max-width: 900px;
  margin: 0 auto;
}

.step {
  text-align: center;
  flex: 1;
}

.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  background: var(--color-primary);
  border-radius: 50%;
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 16px;
}

.step h3 {
  font-size: 24px;
  margin-bottom: 8px;
}

.step p {
  color: var(--color-text-muted);
}

/* Footer */
.footer {
  background: var(--color-surface);
  padding: 60px 80px 40px;
}

.footer-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 40px;
}

.footer-links {
  display: flex;
  gap: 32px;
}

.footer-links a {
  color: var(--color-text-muted);
  text-decoration: none;
}

.footer-links a:hover {
  color: var(--color-text);
}

.copyright {
  text-align: center;
  color: var(--color-text-muted);
  font-size: 14px;
}

/* Responsive */
@media (max-width: 768px) {
  .nav {
    padding: 16px 24px;
  }
  
  .nav-links a:not(.btn-primary) {
    display: none;
  }
  
  .hero {
    flex-direction: column;
    padding: 100px 24px 60px;
    text-align: center;
  }
  
  .hero-topic {
    font-size: 36px;
  }
  
  .hero-hook {
    margin: 0 auto 32px;
  }
  
  .section {
    padding: 60px 24px;
  }
  
  .steps {
    flex-direction: column;
    gap: 40px;
  }
  
  .footer {
    padding: 40px 24px;
  }
  
  .footer-content {
    flex-direction: column;
    gap: 32px;
  }
}
```

**Deliverable:** Clean marketing homepage that sells the product.

---

## TASK 2: SEO API ROUTES

Create Vercel serverless functions for SEO.

### 2.1 /api/day/[number].ts

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY!
);

export const config = {
  runtime: 'edge',
};

export default async function handler(request: Request) {
  const url = new URL(request.url);
  const pathParts = url.pathname.split('/');
  const number = pathParts[pathParts.length - 1];
  const dayNum = parseInt(number);

  if (isNaN(dayNum) || dayNum < 1 || dayNum > 366) {
    return new Response('Day not found', { status: 404 });
  }

  const { data: lesson, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNum)
    .single();

  if (error || !lesson) {
    return new Response('Lesson not found', { status: 404 });
  }

  const date = new Date(2025, 0, dayNum);
  const dateStr = date.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${lesson.title} | Day ${dayNum} | Curious Kelly</title>
  <meta name="description" content="${lesson.marketing_hook || lesson.subtitle || lesson.title}">
  
  <meta property="og:title" content="${lesson.title} - Day ${dayNum}">
  <meta property="og:description" content="${lesson.marketing_hook || lesson.title}">
  <meta property="og:image" content="${lesson.hero_image_url || 'https://curiouskelly.com/images/og-lesson.png'}">
  <meta property="og:url" content="https://curiouskelly.com/day/${dayNum}">
  <meta property="og:type" content="article">
  
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${lesson.title}">
  <meta name="twitter:description" content="${lesson.marketing_hook || lesson.title}">
  
  <link rel="canonical" href="https://curiouskelly.com/day/${dayNum}">
  
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "LearningResource",
    "name": "${lesson.title}",
    "description": "${lesson.marketing_hook || lesson.title}",
    "educationalLevel": "All ages",
    "learningResourceType": "Lesson",
    "timeRequired": "PT${lesson.duration_estimate || 5}M",
    "datePublished": "${date.toISOString().split('T')[0]}",
    "provider": {
      "@type": "Organization",
      "name": "Curious Kelly",
      "url": "https://curiouskelly.com"
    },
    "url": "https://curiouskelly.com/day/${dayNum}"
  }
  </script>
  
  <style>
    body { font-family: system-ui; max-width: 600px; margin: 60px auto; padding: 24px; text-align: center; background: #0a0a1a; color: white; }
    .date { color: #888; margin-bottom: 24px; }
    h1 { margin-bottom: 16px; }
    p { color: #aaa; margin-bottom: 32px; }
    .cta { display: inline-block; background: #4361ee; color: white; padding: 16px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; }
    .cta:hover { background: #3a56d4; }
  </style>
</head>
<body>
  <p class="date">Day ${dayNum} • ${dateStr}</p>
  <h1>${lesson.title}</h1>
  <p>${lesson.marketing_hook || lesson.subtitle || 'Learn something new today'}</p>
  <a href="/learn.html?day=${dayNum}" class="cta">Start This Lesson →</a>
  
  <script>
    if (!/bot|crawl|spider/i.test(navigator.userAgent)) {
      setTimeout(() => { window.location.href = '/learn.html?day=${dayNum}'; }, 1500);
    }
  </script>
</body>
</html>`;

  return new Response(html, {
    headers: {
      'Content-Type': 'text/html',
      'Cache-Control': 's-maxage=86400, stale-while-revalidate',
    },
  });
}
```

### 2.2 /api/sitemap.xml.ts

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY!
);

export const config = {
  runtime: 'edge',
};

export default async function handler() {
  const baseUrl = 'https://curiouskelly.com';
  const today = new Date().toISOString().split('T')[0];

  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('day_number, updated_at')
    .order('day_number');

  const kellys = ['scientist', 'explorer', 'nerd', 'artist', 'coach', 'storyteller',
                  'philosopher', 'optimist', 'mystic', 'maverick', 'guardian', 'sage'];

  let xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${baseUrl}/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>${baseUrl}/learn.html</loc>
    <lastmod>${today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>`;

  for (const lesson of lessons || []) {
    const lastmod = lesson.updated_at 
      ? new Date(lesson.updated_at).toISOString().split('T')[0]
      : today;
    xml += `
  <url>
    <loc>${baseUrl}/day/${lesson.day_number}</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>`;
  }

  for (const kelly of kellys) {
    xml += `
  <url>
    <loc>${baseUrl}/kelly/${kelly}</loc>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`;
  }

  xml += `
</urlset>`;

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml',
      'Cache-Control': 's-maxage=86400, stale-while-revalidate',
    },
  });
}
```

### 2.3 /api/mcp/lessons.json.ts

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY!
);

export const config = {
  runtime: 'edge',
};

export default async function handler() {
  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('day_number, title, subtitle, category, difficulty, duration_estimate, marketing_hook')
    .order('day_number');

  const response = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    "name": "Curious Kelly Daily Lessons",
    "description": "365 daily learning experiences for ages 2-102",
    "url": "https://curiouskelly.com",
    "numberOfItems": lessons?.length || 365,
    "itemListElement": (lessons || []).map((lesson, i) => ({
      "@type": "ListItem",
      "position": i + 1,
      "item": {
        "@type": "LearningResource",
        "identifier": `day-${lesson.day_number}`,
        "name": lesson.title,
        "description": lesson.marketing_hook || lesson.subtitle,
        "category": lesson.category,
        "educationalLevel": lesson.difficulty || "beginner",
        "timeRequired": `PT${lesson.duration_estimate || 5}M`,
        "url": `https://curiouskelly.com/day/${lesson.day_number}`
      }
    }))
  };

  return new Response(JSON.stringify(response, null, 2), {
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 's-maxage=3600, stale-while-revalidate',
    },
  });
}
```

**Deliverable:** All 365 lessons indexed by Google. AI agents can read the catalog.

---

## TASK 3: ROBOTS.TXT & META

### robots.txt

```
User-agent: *
Allow: /
Allow: /day/
Allow: /kelly/

Sitemap: https://curiouskelly.com/api/sitemap.xml

# Disallow admin/internal
Disallow: /admin/
Disallow: /api/internal/
```

### vercel.json rewrites

```json
{
  "rewrites": [
    { "source": "/day/:number", "destination": "/api/day/:number" },
    { "source": "/kelly/:name", "destination": "/api/kelly/:name" },
    { "source": "/sitemap.xml", "destination": "/api/sitemap.xml" },
    { "source": "/robots.txt", "destination": "/robots.txt" }
  ]
}
```

---

## VERIFICATION CHECKLIST

```
HOMEPAGE
□ Opens quickly (< 2 seconds)
□ Shows today's topic
□ Shows date
□ 12 Kelly cards visible
□ "Start Learning" button works
□ Mobile responsive

SEO
□ /day/1 returns HTML with meta tags
□ /day/365 works
□ /api/sitemap.xml returns valid XML
□ /api/mcp/lessons.json returns valid JSON-LD
□ robots.txt exists

INTEGRATION
□ Homepage links to /learn.html
□ Kelly cards link to /learn.html?kelly=X
□ SEO pages redirect to app
```

---

# DEPLOY COMMAND

```bash
cd C:\Users\user\UI-TARS-desktop
git add -A
git commit -m "Phase 4: Full experience - paywall bypass, homepage rewrite, SEO pages"
git push
vercel --prod
```

---

*December 12, 2025 — 5 days to launch*
*Expert Mode: Complete the experience*
