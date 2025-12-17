# 🌍 INTERNATIONALIZATION MASTER PLAN

## Kelly Speaks Your Language, Knows Your Time, Feels Your Season

**Created:** December 17, 2025  
**Status:** ✅ FOUNDATION BUILT — Ready for Use  
**Priority:** Critical for Global Launch

### ✅ What's Built (December 17, 2025)

| Component | Status | Files |
|-----------|--------|-------|
| **i18n Core Engine** | ✅ Complete | `/public/js/i18n/i18n-core.js` |
| **Kelly Personality Layer** | ✅ Complete | `/public/js/i18n/i18n-kelly.js` |
| **Language Selector** | ✅ Complete | `/public/js/i18n/language-selector.js` |
| **Geo-Context API** | ✅ Complete | `/api/geo-context.ts` |
| **English UI** | ✅ Complete | `/public/locales/en/` (6 files) |
| **Spanish UI** | ✅ Complete | `/public/locales/es/` (6 files) |
| **Portuguese UI** | ✅ Complete | `/public/locales/pt/` (6 files) |
| **Language Manifest** | ✅ Complete | `/public/locales/manifest.json` |

---

## 🎯 Vision

> "Kelly always knows what time it is and what season it is where you are and what the weather is and she is basically a calendar app disguised as a learning app."

Kelly isn't just translated — she's **localized**. She understands that:
- It's summer in December in Australia
- Diwali matters in India, Lunar New Year in China
- "Good morning" means 6am in NYC but the user might be in Tokyo
- €49.99 feels different than ₹1,999 even if the value is similar

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KELLY LOCALIZATION ENGINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   LANGUAGE   │  │   REGION     │  │   CONTEXT    │               │
│  │   LAYER      │  │   LAYER      │  │   LAYER      │               │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤               │
│  │ • UI Strings │  │ • Currency   │  │ • Time/Date  │               │
│  │ • Lessons    │  │ • Pricing    │  │ • Season     │               │
│  │ • Kelly Voice│  │ • Payment    │  │ • Weather    │               │
│  │ • Alt Text   │  │   Methods    │  │ • Holidays   │               │
│  │ • Errors     │  │ • Tax        │  │ • Day/Night  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│         │                 │                 │                        │
│         └─────────────────┼─────────────────┘                        │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    UNIFIED CONTEXT                           │    │
│  │  { language, region, timezone, season, timeOfDay, weather } │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    KELLY PERSONALITY                         │    │
│  │  "Good evening, Maria! Perfect autumn weather for learning  │    │
│  │   about leaves today. Ready to explore Day 285?"             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure

```
/public/
├── locales/                          # UI Translations
│   ├── en/
│   │   ├── common.json               # Shared UI strings
│   │   ├── lessons.json              # Lesson UI (phases, buttons)
│   │   ├── settings.json             # Settings page
│   │   ├── paywall.json              # Payment/subscription
│   │   ├── onboarding.json           # First-time user
│   │   └── kelly.json                # Kelly's personality phrases
│   ├── es/
│   │   └── ... (same structure)
│   ├── pt/
│   │   └── ... (same structure)
│   ├── fr/
│   │   └── ... (same structure)
│   ├── de/
│   │   └── ... (same structure)
│   └── hi/
│       └── ... (same structure)
│
├── js/
│   ├── i18n/
│   │   ├── i18n-core.js              # Translation engine
│   │   ├── i18n-context.js           # Time/season/weather context
│   │   ├── i18n-calendar.js          # Holiday/calendar awareness
│   │   └── i18n-kelly.js             # Kelly's localized personality
│   └── geo-pricing.js                # (Already built)
│
/api/
├── geo-context.ts                    # Returns full context (time, season, weather)
├── geo-pricing.ts                    # (Already built)
│
/data/
├── lessons/
│   ├── en/
│   │   ├── day-001.json              # Lesson content in English
│   │   ├── day-002.json
│   │   └── ... (365 files)
│   ├── es/
│   │   └── ... (365 files, Spanish)
│   ├── pt/
│   │   └── ... (365 files, Portuguese)
│   └── fr/
│       └── ... (365 files, French)
│
/supabase/
└── tables/
    ├── lesson_translations           # Lesson content by language
    ├── ui_translations               # UI strings by language
    └── user_preferences              # User's language/timezone prefs
```

---

## 🌐 Layer 1: Language System

### Supported Languages (Priority Order)

| Code | Language | Status | Coverage Target |
|------|----------|--------|-----------------|
| `en` | English | ✅ Complete | 100% UI, 100% Lessons |
| `es` | Spanish | 🚧 Partial | 100% by Jan 15 |
| `pt` | Portuguese | 🚧 Partial | 100% by Jan 15 |
| `fr` | French | 📋 Planned | 100% by Feb 1 |
| `de` | German | 📋 Planned | 100% by Feb 1 |
| `hi` | Hindi | 📋 Planned | 100% by Mar 1 |
| `zh` | Chinese (Simplified) | 📋 Planned | Q2 2026 |
| `ja` | Japanese | 📋 Planned | Q2 2026 |
| `ko` | Korean | 📋 Planned | Q2 2026 |
| `ar` | Arabic (RTL!) | 📋 Planned | Q3 2026 |

### UI Translation Schema

```json
// /public/locales/en/common.json
{
  "app": {
    "name": "Curious Kelly",
    "tagline": "Learn something new every day"
  },
  "greeting": {
    "morning": "Good morning",
    "afternoon": "Good afternoon",
    "evening": "Good evening",
    "night": "Good night"
  },
  "nav": {
    "home": "Home",
    "journey": "Journey",
    "settings": "Settings",
    "today": "Today's Lesson"
  },
  "lesson": {
    "day": "Day {{number}}",
    "topic": "Today's Topic",
    "start": "Start Learning",
    "continue": "Continue",
    "completed": "Completed!",
    "phases": {
      "welcome": "Welcome",
      "explore": "Explore",
      "question": "Question",
      "wisdom": "Wisdom"
    }
  },
  "time": {
    "today": "Today",
    "yesterday": "Yesterday",
    "tomorrow": "Tomorrow",
    "thisWeek": "This Week",
    "daysAgo": "{{count}} days ago"
  }
}
```

```json
// /public/locales/es/common.json
{
  "app": {
    "name": "Kelly Curiosa",
    "tagline": "Aprende algo nuevo cada día"
  },
  "greeting": {
    "morning": "Buenos días",
    "afternoon": "Buenas tardes",
    "evening": "Buenas noches",
    "night": "Buenas noches"
  },
  "nav": {
    "home": "Inicio",
    "journey": "Mi Viaje",
    "settings": "Configuración",
    "today": "Lección de Hoy"
  },
  // ... etc
}
```

### i18n Core Engine

```javascript
// /public/js/i18n/i18n-core.js

window.KellyI18n = (function() {
  'use strict';
  
  const SUPPORTED_LANGUAGES = ['en', 'es', 'pt', 'fr', 'de', 'hi'];
  const DEFAULT_LANGUAGE = 'en';
  const CACHE_KEY = 'kelly_i18n_cache';
  
  let _translations = {};
  let _currentLanguage = DEFAULT_LANGUAGE;
  let _fallbackLanguage = DEFAULT_LANGUAGE;
  
  // Detect user's preferred language
  function detectLanguage() {
    // 1. Check saved preference
    const saved = localStorage.getItem('kelly_language');
    if (saved && SUPPORTED_LANGUAGES.includes(saved)) {
      return saved;
    }
    
    // 2. Check browser language
    const browserLang = navigator.language.split('-')[0];
    if (SUPPORTED_LANGUAGES.includes(browserLang)) {
      return browserLang;
    }
    
    // 3. Default to English
    return DEFAULT_LANGUAGE;
  }
  
  // Load translations for a language
  async function loadLanguage(lang) {
    if (_translations[lang]) return _translations[lang];
    
    try {
      const [common, lessons, settings, paywall, kelly] = await Promise.all([
        fetch(`/locales/${lang}/common.json`).then(r => r.json()),
        fetch(`/locales/${lang}/lessons.json`).then(r => r.json()),
        fetch(`/locales/${lang}/settings.json`).then(r => r.json()),
        fetch(`/locales/${lang}/paywall.json`).then(r => r.json()),
        fetch(`/locales/${lang}/kelly.json`).then(r => r.json()),
      ]);
      
      _translations[lang] = { common, lessons, settings, paywall, kelly };
      return _translations[lang];
    } catch (e) {
      console.warn(`[i18n] Failed to load ${lang}, falling back to ${_fallbackLanguage}`);
      if (lang !== _fallbackLanguage) {
        return loadLanguage(_fallbackLanguage);
      }
      return {};
    }
  }
  
  // Get translation with interpolation
  // Usage: t('lesson.day', { number: 42 }) => "Day 42"
  function t(key, params = {}) {
    const keys = key.split('.');
    let value = _translations[_currentLanguage];
    
    for (const k of keys) {
      value = value?.[k];
      if (!value) break;
    }
    
    // Fallback to English
    if (!value && _currentLanguage !== _fallbackLanguage) {
      value = _translations[_fallbackLanguage];
      for (const k of keys) {
        value = value?.[k];
        if (!value) break;
      }
    }
    
    // Still no value? Return key
    if (!value) return key;
    
    // Interpolate {{params}}
    return value.replace(/\{\{(\w+)\}\}/g, (_, param) => params[param] ?? '');
  }
  
  // Set language and reload translations
  async function setLanguage(lang) {
    if (!SUPPORTED_LANGUAGES.includes(lang)) {
      console.warn(`[i18n] Unsupported language: ${lang}`);
      return false;
    }
    
    await loadLanguage(lang);
    _currentLanguage = lang;
    localStorage.setItem('kelly_language', lang);
    
    // Update all [data-i18n] elements
    applyTranslations();
    
    // Dispatch event
    window.dispatchEvent(new CustomEvent('languagechanged', { detail: { language: lang } }));
    
    return true;
  }
  
  // Apply translations to DOM
  function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const params = el.dataset.i18nParams ? JSON.parse(el.dataset.i18nParams) : {};
      el.textContent = t(key, params);
    });
    
    // Update placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    
    // Update aria-labels
    document.querySelectorAll('[data-i18n-aria]').forEach(el => {
      el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria')));
    });
  }
  
  // Initialize
  async function init() {
    _currentLanguage = detectLanguage();
    await loadLanguage(_currentLanguage);
    await loadLanguage(_fallbackLanguage); // Always load English as fallback
    applyTranslations();
    return _currentLanguage;
  }
  
  return {
    init,
    t,
    setLanguage,
    getLanguage: () => _currentLanguage,
    getSupportedLanguages: () => SUPPORTED_LANGUAGES,
    applyTranslations,
  };
})();
```

---

## 🕐 Layer 2: Context System (Time/Season/Weather)

### Geo-Context API

```typescript
// /api/geo-context.ts
import type { VercelRequest, VercelResponse } from '@vercel/node';

interface GeoContext {
  // Location
  country: string;
  countryName: string;
  region: string;
  city: string;
  
  // Time
  timezone: string;
  localTime: string;
  localDate: string;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  hour: number;
  
  // Calendar
  dayOfWeek: string;
  dayOfYear: number;
  weekOfYear: number;
  month: string;
  isWeekend: boolean;
  
  // Season (hemisphere-aware!)
  season: 'spring' | 'summer' | 'autumn' | 'winter';
  hemisphere: 'northern' | 'southern';
  
  // Weather (optional, from external API)
  weather?: {
    condition: string;
    temp: number;
    tempUnit: 'C' | 'F';
    icon: string;
  };
  
  // Holidays (if applicable)
  holidays?: Array<{
    name: string;
    date: string;
    isToday: boolean;
  }>;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const country = (req.headers['x-vercel-ip-country'] as string) || 'US';
  const city = (req.headers['x-vercel-ip-city'] as string) || '';
  const timezone = (req.headers['x-vercel-ip-timezone'] as string) || 'America/New_York';
  
  // Get local time in user's timezone
  const now = new Date();
  const localTime = now.toLocaleString('en-US', { timeZone: timezone });
  const localDate = now.toLocaleDateString('en-US', { timeZone: timezone });
  const hour = parseInt(now.toLocaleString('en-US', { timeZone: timezone, hour: 'numeric', hour12: false }));
  
  // Determine time of day
  let timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  if (hour >= 5 && hour < 12) timeOfDay = 'morning';
  else if (hour >= 12 && hour < 17) timeOfDay = 'afternoon';
  else if (hour >= 17 && hour < 21) timeOfDay = 'evening';
  else timeOfDay = 'night';
  
  // Determine hemisphere and season
  const southernHemisphere = ['AU', 'NZ', 'AR', 'CL', 'ZA', 'BR', 'PE', 'UY', 'PY', 'BO'];
  const hemisphere = southernHemisphere.includes(country) ? 'southern' : 'northern';
  
  const month = now.getMonth(); // 0-11
  let season: 'spring' | 'summer' | 'autumn' | 'winter';
  if (hemisphere === 'northern') {
    if (month >= 2 && month <= 4) season = 'spring';
    else if (month >= 5 && month <= 7) season = 'summer';
    else if (month >= 8 && month <= 10) season = 'autumn';
    else season = 'winter';
  } else {
    // Flip for southern hemisphere
    if (month >= 2 && month <= 4) season = 'autumn';
    else if (month >= 5 && month <= 7) season = 'winter';
    else if (month >= 8 && month <= 10) season = 'spring';
    else season = 'summer';
  }
  
  // Day of week
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const dayOfWeek = days[now.getDay()];
  const isWeekend = now.getDay() === 0 || now.getDay() === 6;
  
  // Day of year
  const startOfYear = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - startOfYear.getTime();
  const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24));
  
  const context: GeoContext = {
    country,
    countryName: getCountryName(country),
    region: req.headers['x-vercel-ip-country-region'] as string || '',
    city,
    timezone,
    localTime,
    localDate,
    timeOfDay,
    hour,
    dayOfWeek,
    dayOfYear,
    weekOfYear: Math.ceil(dayOfYear / 7),
    month: now.toLocaleString('en-US', { month: 'long' }),
    isWeekend,
    season,
    hemisphere,
  };
  
  res.setHeader('Cache-Control', 'public, max-age=300'); // 5 min cache
  res.json(context);
}
```

### Context-Aware Kelly Greetings

```javascript
// /public/js/i18n/i18n-kelly.js

window.KellyPersonality = (function() {
  'use strict';
  
  let _context = null;
  
  async function loadContext() {
    try {
      const response = await fetch('/api/geo-context');
      _context = await response.json();
    } catch (e) {
      console.warn('[Kelly] Context fetch failed, using defaults');
      _context = {
        timeOfDay: 'afternoon',
        season: 'spring',
        isWeekend: false,
      };
    }
    return _context;
  }
  
  // Generate personalized greeting
  function getGreeting(userName) {
    if (!_context) return `Hello${userName ? `, ${userName}` : ''}!`;
    
    const t = window.KellyI18n.t;
    const timeGreeting = t(`greeting.${_context.timeOfDay}`);
    
    let greeting = userName ? `${timeGreeting}, ${userName}!` : `${timeGreeting}!`;
    
    // Add seasonal/contextual flair
    if (_context.isWeekend) {
      greeting += ' ' + t('kelly.weekend');
    }
    
    return greeting;
  }
  
  // Get seasonal lesson commentary
  function getSeasonalComment(topic) {
    if (!_context) return '';
    
    const comments = {
      winter: "Perfect weather to stay curious indoors!",
      spring: "New season, new things to learn!",
      summer: "Summer days are made for discovery!",
      autumn: "Cozy up and learn something new today.",
    };
    
    return comments[_context.season] || '';
  }
  
  // Get time-appropriate encouragement
  function getEncouragement() {
    if (!_context) return "Let's learn!";
    
    if (_context.timeOfDay === 'morning') {
      return "Great way to start your day!";
    } else if (_context.timeOfDay === 'evening') {
      return "Wind down with some curiosity!";
    } else if (_context.timeOfDay === 'night') {
      return "Night owl learning? I like it!";
    }
    return "Perfect time to learn something new!";
  }
  
  return {
    loadContext,
    getGreeting,
    getSeasonalComment,
    getEncouragement,
    getContext: () => _context,
  };
})();
```

---

## 📚 Layer 3: Lesson Content Translation

### Lesson Translation Schema

```json
// /data/lessons/en/day-001.json
{
  "day": 1,
  "language": "en",
  "topic": "The Science of Bubbles",
  "headline": "Why do bubbles always become spheres?",
  "thumbnail": "/images/lessons/day-001-thumb.jpg",
  
  "phases": {
    "welcome": {
      "kelly_says": "Hi there! Have you ever wondered why bubbles are always round?",
      "audio_url": "/audio/en/day-001-welcome.mp3"
    },
    "explore": {
      "content": "Bubbles form as spheres because a sphere has the smallest surface area for any given volume. It's nature's way of being efficient! Surface tension pulls the soap film into the most compact shape possible.",
      "fun_fact": "The largest free-floating soap bubble ever made was 96 feet long!",
      "audio_url": "/audio/en/day-001-explore.mp3"
    },
    "question": {
      "question": "What force makes bubbles round?",
      "options": [
        { "id": "a", "text": "Gravity", "correct": false },
        { "id": "b", "text": "Surface tension", "correct": true },
        { "id": "c", "text": "Air pressure", "correct": false },
        { "id": "d", "text": "Magnetism", "correct": false }
      ],
      "explanation": "Surface tension is created by water molecules pulling toward each other, creating the thinnest possible film."
    },
    "wisdom": {
      "kelly_says": "Next time you blow bubbles, you're watching physics in action! Nature always finds the most efficient shape.",
      "audio_url": "/audio/en/day-001-wisdom.mp3"
    }
  },
  
  "related_days": [15, 89, 234],
  "tags": ["physics", "nature", "everyday-science"],
  "age_adaptations": {
    "child": { "headline": "Why are bubbles round and not square?" },
    "teen": { "headline": "The physics behind soap bubbles" },
    "adult": { "headline": "Surface tension and minimal surfaces" }
  }
}
```

### Translation Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                 LESSON TRANSLATION PIPELINE                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. SOURCE CONTENT (English)                                 │
│     └─→ /data/lessons/en/day-XXX.json                        │
│                                                               │
│  2. MACHINE TRANSLATION (First Pass)                         │
│     └─→ GPT-4 / DeepL API                                    │
│     └─→ Output to /data/lessons/{lang}/day-XXX.json          │
│                                                               │
│  3. HUMAN REVIEW (Quality Pass)                              │
│     └─→ Native speaker reviews translations                  │
│     └─→ Cultural adaptation (examples, idioms)               │
│     └─→ Age-appropriate language verification                │
│                                                               │
│  4. AUDIO GENERATION                                         │
│     └─→ ElevenLabs TTS in target language                    │
│     └─→ /audio/{lang}/day-XXX-{phase}.mp3                    │
│                                                               │
│  5. QUALITY ASSURANCE                                        │
│     └─→ Automated length checks                              │
│     └─→ Sync marker validation                               │
│     └─→ Native speaker audio review                          │
│                                                               │
│  6. PUBLISH                                                   │
│     └─→ Upload to Supabase                                   │
│     └─→ Invalidate CDN cache                                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Supabase Schema for Translations

```sql
-- Lesson translations table
CREATE TABLE lesson_translations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER NOT NULL,
  language VARCHAR(5) NOT NULL,
  
  -- Core content
  topic TEXT NOT NULL,
  headline TEXT NOT NULL,
  
  -- Phase content (JSONB for flexibility)
  phases JSONB NOT NULL,
  
  -- Audio URLs
  audio_urls JSONB,
  
  -- Metadata
  translator_id UUID REFERENCES users(id),
  review_status VARCHAR(20) DEFAULT 'pending', -- pending, reviewed, approved
  reviewed_by UUID REFERENCES users(id),
  reviewed_at TIMESTAMP,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  UNIQUE(day_number, language)
);

-- UI translations table
CREATE TABLE ui_translations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  language VARCHAR(5) NOT NULL,
  namespace VARCHAR(50) NOT NULL, -- 'common', 'lessons', 'settings', etc.
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  UNIQUE(language, namespace, key)
);

-- User language preferences
ALTER TABLE users ADD COLUMN preferred_language VARCHAR(5) DEFAULT 'en';
ALTER TABLE users ADD COLUMN timezone VARCHAR(50);
ALTER TABLE users ADD COLUMN date_format VARCHAR(20) DEFAULT 'MM/DD/YYYY';
ALTER TABLE users ADD COLUMN use_24_hour BOOLEAN DEFAULT FALSE;
```

---

## 🗓️ Layer 4: Calendar & Holiday Awareness

### Holiday Database

```javascript
// /public/js/i18n/i18n-calendar.js

const HOLIDAYS = {
  // Global/Western
  "01-01": { name: "New Year's Day", global: true },
  "02-14": { name: "Valentine's Day", regions: ['US', 'CA', 'GB', 'AU'] },
  "12-25": { name: "Christmas Day", global: true },
  "12-31": { name: "New Year's Eve", global: true },
  
  // US
  "07-04": { name: "Independence Day", regions: ['US'] },
  "11-28": { name: "Thanksgiving", regions: ['US'], moveable: true },
  
  // India
  "01-26": { name: "Republic Day", regions: ['IN'] },
  "08-15": { name: "Independence Day", regions: ['IN'] },
  // Diwali, Holi - moveable, need lunar calendar
  
  // China
  // Lunar New Year - moveable
  
  // Brazil
  "09-07": { name: "Independence Day", regions: ['BR'] },
  // Carnival - moveable
  
  // And many more...
};

// Get holidays for today based on user's region
function getTodaysHolidays(region) {
  const today = new Date();
  const key = `${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  
  return Object.entries(HOLIDAYS)
    .filter(([date]) => date === key)
    .filter(([_, holiday]) => holiday.global || holiday.regions?.includes(region))
    .map(([_, holiday]) => holiday.name);
}
```

---

## 📅 Implementation Timeline

### Phase 1: Foundation (Week 1 - Dec 17-24)
- [x] Geo-pricing API and frontend
- [ ] Create `/public/locales/` structure
- [ ] Build i18n-core.js translation engine
- [ ] Create English translation files (extract from HTML)
- [ ] Add `data-i18n` attributes to learn.html

### Phase 2: Spanish & Portuguese (Week 2 - Dec 25-31)
- [ ] Translate all UI strings to Spanish
- [ ] Translate all UI strings to Portuguese
- [ ] Machine translate first 50 lessons to Spanish
- [ ] Machine translate first 50 lessons to Portuguese
- [ ] Human review translations

### Phase 3: Context Awareness (Week 3 - Jan 1-7)
- [ ] Build geo-context API
- [ ] Build i18n-kelly.js personality layer
- [ ] Add seasonal/time-aware greetings
- [ ] Hemisphere-aware season detection
- [ ] Weekend/weekday awareness

### Phase 4: Full Lesson Translation (Ongoing)
- [ ] Complete Spanish lesson translations (1-365)
- [ ] Complete Portuguese lesson translations (1-365)
- [ ] Generate Spanish audio via ElevenLabs
- [ ] Generate Portuguese audio via ElevenLabs

### Phase 5: French & German (Feb 2026)
- [ ] UI translations
- [ ] Lesson translations
- [ ] Audio generation

### Phase 6: Hindi & Asian Languages (Q2 2026)
- [ ] UI translations
- [ ] Lesson translations
- [ ] Audio generation
- [ ] RTL support for Arabic (future)

---

## 🔧 Implementation Checklist (Your Backlog)

### Stripe Work (Already Documented)
- [ ] Create 36 multi-currency prices
- [ ] Enable 18 payment methods
- [ ] Add 36 environment variables
- [ ] Test international checkout

### i18n Setup
- [ ] Create `/public/locales/en/` directory structure
- [ ] Extract all UI strings from learn.html
- [ ] Create common.json, lessons.json, settings.json, paywall.json, kelly.json
- [ ] Add data-i18n attributes to HTML
- [ ] Build and integrate i18n-core.js

### Spanish Translation
- [ ] Translate common.json to Spanish
- [ ] Translate all other UI files
- [ ] Machine translate lessons 1-50
- [ ] Find native speaker for review

### Portuguese Translation
- [ ] Same as Spanish

### Context API
- [ ] Build /api/geo-context.ts
- [ ] Integrate with Kelly greetings
- [ ] Test hemisphere season detection
- [ ] Add holiday awareness

---

## 📊 Success Metrics

| Metric | Current | Target (Q1 2026) |
|--------|---------|------------------|
| Languages supported | 1 | 5 (EN, ES, PT, FR, DE) |
| Lessons translated | 365 EN | 365 × 3 languages |
| UI translation coverage | 0% | 100% for top 5 languages |
| International conversion rate | ~0.5% | 2%+ |
| Time-aware greetings | No | Yes |
| Season-aware content | No | Yes |
| Holiday awareness | No | Major holidays |

---

## 🎯 Key Principles

1. **Always fall back gracefully** - If Spanish isn't available, show English
2. **Never break the lesson** - Missing translation = English fallback
3. **Context is king** - Kelly should feel local, not translated
4. **Audio is essential** - Lessons need native-language audio
5. **Human review required** - Machine translation is a first pass only
6. **One source of truth** - All translations in Supabase, synced to CDN

---

*Plan created: December 17, 2025*
*"Kelly speaks your language, knows your time, feels your season."*
