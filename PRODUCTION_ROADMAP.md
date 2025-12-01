# Production Roadmap - Kelly Global Launch

## Mission: Make Kelly Accessible Everywhere, In Every Language

**Target**: Production-ready global platform  
**Timeline**: Today  
**Status**: IN PROGRESS

---

## 1. REALITY CHECK - Tone Down Promises ✅

### Current Issues
- "ages 2-102" - overpromise
- "Founding member badge" - not implemented
- "VIP support" - not staffed
- Implied features not built yet

### Fixes
- Change to "ages 5-55+" (realistic range with content)
- Remove "Founding member badge" or clarify it's cosmetic
- Change "VIP support" to "Priority support"
- Add "Coming soon" labels where appropriate
- Honest about beta/launch status

---

## 2. CALENDAR VIEWS - Multiple Perspectives

### Month View (Current)
- ✅ Already implemented
- Shows all 12 months
- Collapsible month cards

### Week View (NEW)
- 7-day horizontal scroll
- Current week highlighted
- Quick navigation

### List View (NEW)
- Vertical list of all 366 lessons
- Search/filter capability
- Jump to date

### Implementation
```javascript
const calendarViews = {
    month: 'default',
    week: 'compact',
    list: 'detailed'
};

function switchCalendarView(view) {
    // Toggle between views
    // Save preference to localStorage
}
```

---

## 3. CONTENT PREVIEW SYSTEM

### Lesson Detail Modal
- Click any lesson → modal overlay
- Shows: Title, Description, Duration, Prerequisites
- Preview first 30 seconds of audio
- "Start Lesson" CTA

### Quick Preview Cards
- Hover over lesson → tooltip preview
- Shows hook for selected age
- Estimated completion time

---

## 4. DESKTOP APP DOWNLOADS

### Platforms
- Windows (x64)
- macOS (Intel + Apple Silicon)
- Linux (AppImage/Snap)

### Download Page Section
```html
<section id="downloads">
    <h2>Download Kelly</h2>
    <p>Learn offline. Sync across devices.</p>
    
    <div class="download-grid">
        <div class="download-card">
            <h3>🪟 Windows</h3>
            <p>Windows 10/11 (64-bit)</p>
            <button>Download .exe</button>
            <a href="#">View system requirements</a>
        </div>
        
        <div class="download-card">
            <h3>🍎 macOS</h3>
            <p>macOS 11+ (Intel & Apple Silicon)</p>
            <button>Download .dmg</button>
            <a href="#">View system requirements</a>
        </div>
        
        <div class="download-card">
            <h3>🐧 Linux</h3>
            <p>Ubuntu, Fedora, Arch</p>
            <button>Download AppImage</button>
            <a href="#">View system requirements</a>
        </div>
    </div>
    
    <p class="note">Web version works on any device. Desktop apps coming Q1 2026.</p>
</section>
```

---

## 5. MOBILE APP DOWNLOADS

### App Store Presence
- iOS App Store (iPhone/iPad)
- Google Play Store (Android)
- Huawei AppGallery (China)

### Current Status
- Apps in development
- Web app is PWA-ready
- Can "Add to Home Screen" now

### Honest Messaging
```html
<section id="mobile-apps">
    <h2>Kelly on Mobile</h2>
    
    <div class="app-status">
        <div class="status-badge beta">Web App (Available Now)</div>
        <p>Add Kelly to your home screen for app-like experience</p>
        <button onclick="promptPWAInstall()">Add to Home Screen</button>
    </div>
    
    <div class="app-status">
        <div class="status-badge coming-soon">Native Apps (Coming Q1 2026)</div>
        <p>Full offline support, faster performance, native features</p>
        <div class="app-badges disabled">
            <img src="/images/app-store-badge.svg" alt="Coming to App Store" />
            <img src="/images/google-play-badge.svg" alt="Coming to Google Play" />
        </div>
        <button onclick="notifyWhenReady()">Notify me when available</button>
    </div>
</section>
```

---

## 6. INTERNATIONALIZATION (i18n)

### Language Selector
- Top-right header
- Flag icons + language names
- Persists to localStorage
- Reloads UI in selected language

### Initial Languages (Priority Order)
1. 🇺🇸 English (en) - DEFAULT
2. 🇪🇸 Spanish (es) - 559M speakers
3. 🇫🇷 French (fr) - 280M speakers
4. 🇩🇪 German (de) - 134M speakers
5. 🇨🇳 Chinese Simplified (zh-CN) - 1.1B speakers
6. 🇯🇵 Japanese (ja) - 125M speakers
7. 🇵🇹 Portuguese (pt) - 258M speakers
8. 🇷🇺 Russian (ru) - 258M speakers
9. 🇮🇳 Hindi (hi) - 602M speakers
10. 🇸🇦 Arabic (ar) - 274M speakers

### Implementation Strategy
```javascript
// Simple i18n object
const translations = {
    en: {
        'nav.curriculum': 'Curriculum',
        'nav.pricing': 'Pricing',
        'hero.title': 'Curious? Always.',
        'hero.subtitle': 'The AI for lifelong learners.',
        // ... 200+ strings
    },
    es: {
        'nav.curriculum': 'Currículum',
        'nav.pricing': 'Precios',
        'hero.title': '¿Curioso? Siempre.',
        'hero.subtitle': 'La IA para aprendices de por vida.',
        // ...
    },
    // ... other languages
};

function t(key, lang = currentLang) {
    return translations[lang][key] || translations['en'][key] || key;
}

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('kelly-lang', lang);
    document.documentElement.lang = lang;
    updateUI();
}
```

### UI Translation Points
- Navigation (5 items)
- Hero section (3 strings)
- Pricing cards (20 strings)
- Footer (15 strings)
- Buttons (10 strings)
- Form labels (8 strings)
- Error messages (5 strings)

**Total**: ~200 UI strings to translate

---

## 7. POLISH IMPROVEMENTS

### Visual Polish
- Add subtle animations on scroll
- Improve button hover states
- Add micro-interactions
- Consistent spacing/alignment

### Content Polish
- Professional copywriting
- Clear CTAs
- Remove marketing fluff
- Add social proof (when real)

### Performance Polish
- Lazy load images
- Optimize fonts
- Minify CSS/JS
- Enable caching

---

## 8. DEPLOYMENT STRATEGY

### Phase 1: Core Improvements (Today)
1. Tone down promises
2. Add calendar view switcher
3. Add download section (honest about status)
4. Add language selector UI
5. Implement basic i18n for English/Spanish/French

### Phase 2: Full i18n (This Week)
1. Translate all UI strings to 10 languages
2. Add RTL support for Arabic
3. Test all languages
4. Add language-specific meta tags

### Phase 3: Apps (Q1 2026)
1. Build Electron desktop apps
2. Build React Native mobile apps
3. Submit to app stores
4. Update download page with real links

---

## HONEST MESSAGING EXAMPLES

### Before (Overpromise)
> "Join 1.2M learners worldwide"

### After (Reality)
> "Join thousands of curious learners"

---

### Before (Vague)
> "Download our app"

### After (Honest)
> "Web app available now. Native apps coming Q1 2026."

---

### Before (Unrealistic)
> "Learn anything, anytime, anywhere"

### After (Specific)
> "366 daily lessons, personalized for your age"

---

## SUCCESS METRICS

### Launch Readiness
- [ ] All promises are achievable
- [ ] Download section is honest about status
- [ ] Language selector works for 3+ languages
- [ ] Calendar has 2+ view options
- [ ] Content previews work
- [ ] Mobile responsive
- [ ] No broken links
- [ ] Fast load times (<2s)

### Post-Launch
- Track actual user counts (don't inflate)
- Monitor app install requests
- Track language usage
- Measure engagement by view type

---

**Status**: Ready to implement. Let's build this properly.



