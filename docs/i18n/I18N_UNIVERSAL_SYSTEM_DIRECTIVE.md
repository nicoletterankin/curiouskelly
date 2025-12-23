# 🌍 UNIVERSAL I18N SYSTEM DIRECTIVE
## Complete Integration: Language + Country + Pricing + Lesson Universality

**Created:** December 23, 2025  
**Status:** 🚀 READY TO IMPLEMENT  
**Priority:** CRITICAL — Unlocks Global Market

---

## 🎯 MISSION

**Create a unified, visible, interactive system that demonstrates Kelly's universality:**
- ✅ Switch languages (EN, ES, PT) with instant UI updates
- ✅ Switch countries/currencies and see live pricing changes
- ✅ See how lessons adapt by age, tone, and language
- ✅ Complete Stripe integration for international checkout
- ✅ One cohesive experience that "just works"

**The Goal:** When a user visits `learn.html`, they immediately see:
1. **Language selector** (top-right) — "🇺🇸 English | 🇪🇸 Español | 🇵🇹 Português"
2. **Country/Currency selector** (in Settings → Billing) — "🇺🇸 United States ($) | 🇪🇺 Germany (€) | 🇮🇳 India (₹)"
3. **Live pricing updates** — Prices change instantly when country changes
4. **Lesson adaptation demo** — Show how Day 1 looks different in Spanish vs English, Child vs Adult
5. **Universal lesson badge** — "This lesson adapts: 🌍 3 languages • 👶 3 age groups • 🎭 3 tones"

---

## 📋 PHASE 1: STRIPE BACKEND SETUP (45-60 min)

### Task 1.1: Complete Stripe Batch Work
**Reference:** `docs/billing/STRIPE_BATCH_WORK.md`

**Action Items:**
- [ ] Create 36 multi-currency prices in Stripe Dashboard
  - EUR: Monthly, Annual, Family, Lifetime (4)
  - GBP: Monthly, Annual, Family, Lifetime (4)
  - CAD: Monthly, Annual, Family, Lifetime (4)
  - AUD: Monthly, Annual, Family, Lifetime (4)
  - INR: Monthly, Annual, Lifetime (3) — PPP
  - BRL: Monthly, Annual, Lifetime (3) — PPP
  - MXN: Monthly, Annual, Lifetime (3) — PPP
  - PLN: Monthly, Annual, Lifetime (3) — PPP
  - EUR Gifts: 3mo, 6mo, 12mo, Lifetime (4)
  - GBP Gifts: 3mo, 6mo, 12mo, Lifetime (4)
- [ ] Enable 18 payment methods (iDEAL, SEPA, PIX, UPI, etc.)
- [ ] Collect all Price IDs
- [ ] Add 36 environment variables to Vercel
- [ ] Test with `?force_country=DE` / `?force_country=IN`

**Deliverable:** All Stripe prices exist, payment methods enabled, env vars set.

---

## 📋 PHASE 2: UNIFIED LANGUAGE + COUNTRY SWITCHER (2-3 hours)

### Task 2.1: Create Universal Switcher Component
**File:** `public/js/i18n/universal-switcher.js`

**Features:**
- Language selector (EN, ES, PT) with flags
- Country/Currency selector (US, DE, GB, IN, BR, MX, etc.) with flags
- Both persist to localStorage
- Both trigger instant UI updates
- Both work together (language + currency = localized experience)

**API Integration:**
- Language changes → Load translations from `/locales/{lang}/`
- Country changes → Fetch pricing from `/api/geo-pricing?country={code}`
- Both changes → Update lesson content, pricing, UI strings simultaneously

**UI Design:**
```
┌─────────────────────────────────────────┐
│  🌍 Language: [🇺🇸 EN ▼] [🇪🇸 ES] [🇵🇹 PT] │
│  💰 Country:  [🇺🇸 US ($) ▼] [🇪🇺 DE (€)] │
└─────────────────────────────────────────┘
```

**Implementation:**
```javascript
// public/js/i18n/universal-switcher.js
window.UniversalSwitcher = {
  async init() {
    // Load saved preferences
    const savedLang = localStorage.getItem('kelly_language') || 'en';
    const savedCountry = localStorage.getItem('kelly_country') || 'US';
    
    // Initialize i18n
    await window.KellyI18n.setLanguage(savedLang);
    
    // Initialize geo-pricing
    await window.GeoPricing.setCountry(savedCountry);
    
    // Render switcher UI
    this.renderSwitcher();
    
    // Listen for changes
    this.attachHandlers();
  },
  
  async switchLanguage(lang) {
    localStorage.setItem('kelly_language', lang);
    await window.KellyI18n.setLanguage(lang);
    await this.updateLessonContent(); // Reload lesson in new language
    this.updateSwitcherUI();
  },
  
  async switchCountry(country) {
    localStorage.setItem('kelly_country', country);
    await window.GeoPricing.setCountry(country);
    await this.updatePricing(); // Update all prices
    this.updateSwitcherUI();
  },
  
  async updateLessonContent() {
    // Reload current lesson in new language
    const dayNumber = window.KellyTime?.dayNumber || 1;
    const lang = window.KellyI18n.getLanguage();
    // Trigger lesson reload with language param
    window.location.href = `/learn.html?day=${dayNumber}&lang=${lang}`;
  },
  
  async updatePricing() {
    // Update all pricing displays
    const pricing = await window.GeoPricing.getCurrentPricing();
    document.querySelectorAll('[data-price-monthly]').forEach(el => {
      el.textContent = pricing.prices.monthly;
    });
    // ... update all price elements
  }
};
```

### Task 2.2: Integrate Switcher into learn.html
**File:** `public/learn.html`

**Location:** Top-right header, next to Settings button

**HTML:**
```html
<!-- Universal Switcher -->
<div id="universal-switcher" class="universal-switcher">
  <div class="switcher-group">
    <label>🌍</label>
    <select id="language-selector" class="switcher-select">
      <option value="en">🇺🇸 English</option>
      <option value="es">🇪🇸 Español</option>
      <option value="pt">🇵🇹 Português</option>
    </select>
  </div>
  <div class="switcher-group">
    <label>💰</label>
    <select id="country-selector" class="switcher-select">
      <option value="US">🇺🇸 United States ($)</option>
      <option value="DE">🇪🇺 Germany (€)</option>
      <option value="GB">🇬🇧 United Kingdom (£)</option>
      <option value="IN">🇮🇳 India (₹)</option>
      <option value="BR">🇧🇷 Brazil (R$)</option>
      <option value="MX">🇲🇽 Mexico (MX$)</option>
      <!-- ... more countries ... -->
    </select>
  </div>
</div>
```

**Styling:** Match brand blue, compact, visible but not intrusive.

**Deliverable:** Users can switch language and country from header, see instant updates.

---

## 📋 PHASE 3: LIVE PRICING UPDATES (1-2 hours)

### Task 3.1: Update Billing Section with Dynamic Pricing
**File:** `public/learn.html` (Billing section)

**Current State:** Hardcoded USD prices

**Target State:** Prices update based on selected country

**Implementation:**
```javascript
// In renderCheckoutPanel() function
async function renderCheckoutPanel() {
  const pricing = await window.GeoPricing.getCurrentPricing();
  const currency = pricing.currency;
  const symbol = pricing.symbol;
  
  slot.innerHTML = `
    <div class="checkout-panel">
      <!-- Show current country/currency -->
      <div class="pricing-context">
        <span>🌍 ${pricing.countryName}</span>
        <span>💰 ${currency} ${symbol}</span>
        ${pricing.isPPP ? '<span class="ppp-badge">PPP Pricing</span>' : ''}
      </div>
      
      <!-- Dynamic prices -->
      <button class="checkout-plan" data-plan="monthly">
        <div class="name">Monthly</div>
        <div class="price">${symbol}${pricing.prices.monthly}</div>
      </button>
      <!-- ... etc ... -->
    </div>
  `;
}
```

### Task 3.2: Add Country Switcher to Billing Panel
**Location:** Inside checkout panel, above pricing options

**UI:**
```
┌─────────────────────────────────────┐
│  Billing                            │
│  ─────────────────────────────────  │
│  💰 Change Country:                 │
│  [🇺🇸 US] [🇪🇺 DE] [🇬🇧 GB] [🇮🇳 IN] │
│  ─────────────────────────────────  │
│  Monthly — €7.99/mo                │
│  Annual — €49.99/yr                 │
└─────────────────────────────────────┘
```

**Behavior:** Clicking a country flag updates prices instantly, no page reload.

**Deliverable:** Pricing section shows live currency conversion, users can flip through countries.

---

## 📋 PHASE 4: LESSON UNIVERSALITY DEMONSTRATION (2-3 hours)

### Task 4.1: Create "Universal Lesson Badge"
**Location:** Lesson player header, next to "Day X"

**Shows:**
- Available languages: "🌍 3 languages"
- Available age groups: "👶 3 age groups"
- Available tones/archetypes: "🎭 3 tones"

**Implementation:**
```javascript
function renderUniversalBadge(lessonData) {
  const languages = lessonData.meta?.languages || ['en'];
  const ageBuckets = lessonData.meta?.ageBuckets || ['adult'];
  const archetypes = lessonData.meta?.archetypes || ['explorer'];
  
  return `
    <div class="universal-badge">
      <span>🌍 ${languages.length} languages</span>
      <span>👶 ${ageBuckets.length} age groups</span>
      <span>🎭 ${archetypes.length} tones</span>
      <button onclick="showAdaptationDemo()">See how it adapts →</button>
    </div>
  `;
}
```

### Task 4.2: Create "Adaptation Demo" Modal
**Trigger:** Click "See how it adapts" button

**Shows:**
- Side-by-side comparison of same lesson in different languages
- Age-adapted versions (Child vs Adult wording)
- Tone variations (Explorer vs Scientist vs Rebel)

**UI:**
```
┌─────────────────────────────────────────────────────────┐
│  How Day 1 Adapts                                      │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  Language: [🇺🇸 EN] [🇪🇸 ES] [🇵🇹 PT]                    │
│  Age:      [👶 Child] [🧒 Teen] [👤 Adult]              │
│  Tone:     [🔍 Explorer] [🔬 Scientist] [🎸 Rebel]     │
│                                                         │
│  ┌─────────────┬─────────────┬─────────────┐          │
│  │ English     │ Spanish     │ Portuguese  │          │
│  │ Child       │ Niño        │ Criança     │          │
│  │ Explorer    │ Explorador  │ Explorador  │          │
│  │             │             │             │          │
│  │ "Why are    │ "¿Por qué   │ "Por que as │          │
│  │  bubbles    │  las        │  bolhas são │          │
│  │  round?"    │  burbujas   │  redondas?" │          │
│  │             │  son        │             │          │
│  │             │  redondas?" │             │          │
│  └─────────────┴─────────────┴─────────────┘          │
│                                                         │
│  [Close] [Try This Lesson]                            │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**
- Load lesson JSONs for all languages
- Show phase-by-phase comparison
- Highlight differences (word choice, complexity, tone)
- Allow switching between variants live

**Deliverable:** Users can see how lessons adapt, understand universality.

---

## 📋 PHASE 5: LESSON CONTENT LANGUAGE SWITCHING (2-3 hours)

### Task 5.1: Update Lesson Loader for Multi-Language
**File:** `public/js/kelly-lesson-loader.js`

**Current:** Loads `/lessons/day-001.json` (English only)

**Target:** Loads `/lessons/{lang}/day-001.json` or falls back to English

**Implementation:**
```javascript
// In KellyLessonLoader
async loadLesson(dayNumber, options = {}) {
  const lang = options.language || window.KellyI18n?.getLanguage() || 'en';
  
  // Try language-specific JSON first
  let jsonUrl = `/lessons/${lang}/day-${String(dayNumber).padStart(3, '0')}.json`;
  let response = await fetch(jsonUrl);
  
  // Fallback to English if not found
  if (!response.ok && lang !== 'en') {
    jsonUrl = `/lessons/en/day-${String(dayNumber).padStart(3, '0')}.json`;
    response = await fetch(jsonUrl);
  }
  
  const lessonData = await response.json();
  
  // Apply language to all text content
  return this.localizeLesson(lessonData, lang);
}
```

### Task 5.2: Create Language Fallback System
**Strategy:**
1. Try requested language: `/lessons/es/day-001.json`
2. If missing, try English: `/lessons/en/day-001.json`
3. Show badge: "⚠️ This lesson is in English (Spanish coming soon)"

**Deliverable:** Lessons load in selected language, graceful fallback.

---

## 📋 PHASE 6: INTEGRATION & POLISH (1-2 hours)

### Task 6.1: Connect All Systems
**Flow:**
1. User selects language → UI updates → Lesson reloads
2. User selects country → Pricing updates → Currency changes
3. Both changes → Full localization (language + currency + payment methods)

### Task 6.2: Add Visual Feedback
- Loading states when switching language/country
- Success toast: "Switched to Español (Spain)"
- Error handling: "Spanish not available for this lesson yet"

### Task 6.3: Persist Preferences
- Save language + country to localStorage
- Save to user profile (if logged in)
- Remember across sessions

**Deliverable:** Seamless, polished experience.

---

## 📋 PHASE 7: STRIPE CHECKOUT INTEGRATION (1-2 hours)

### Task 7.1: Update create-checkout.ts
**File:** `api/create-checkout.ts`

**Current:** Uses hardcoded USD price IDs

**Target:** Selects price ID based on country/currency

**Implementation:**
```typescript
// Get currency from request
const country = req.body.country || req.headers['x-vercel-ip-country'] || 'US';
const currency = getCurrencyForCountry(country);

// Get price ID for this currency + plan
const priceId = process.env[`STRIPE_PRICE_${planType.toUpperCase()}_${currency}`];

if (!priceId) {
  return res.status(400).json({ error: `Price not configured for ${currency}` });
}

// Create checkout session with correct price
const session = await stripe.checkout.sessions.create({
  mode: planType === 'lifetime' ? 'payment' : 'subscription',
  line_items: [{ price: priceId, quantity: 1 }],
  currency: currency.toLowerCase(),
  // ... rest of config
});
```

### Task 7.2: Update Frontend Checkout Flow
**File:** `public/learn.html` (`startEmbeddedCheckoutFlow`)

**Send country/currency to API:**
```javascript
const country = localStorage.getItem('kelly_country') || 'US';
const currency = await window.GeoPricing.getCurrencyForCountry(country);

const resp = await fetch('/api/create-checkout', {
  method: 'POST',
  body: JSON.stringify({
    planType: __selectedPlanType,
    country: country,
    currency: currency,
    // ... rest
  })
});
```

**Deliverable:** Checkout uses correct currency, shows local payment methods.

---

## 🎯 SUCCESS METRICS

### Immediate (Week 1)
- [ ] Language switcher visible and functional
- [ ] Country switcher visible and functional
- [ ] Prices update when country changes
- [ ] Lessons load in selected language (with fallback)

### Short-term (Month 1)
- [ ] 10% of users switch language
- [ ] 5% of users switch country
- [ ] International checkout conversion rate increases 2x
- [ ] Zero errors when switching language/country

### Long-term (Q1 2026)
- [ ] 30% of users outside US use local currency
- [ ] Spanish/Portuguese lessons have 50%+ completion rate
- [ ] International revenue = 25% of total

---

## 🚨 CRITICAL REQUIREMENTS

### 1. Never Break the Lesson
- **Rule:** If Spanish lesson doesn't exist, show English with badge
- **Rule:** If pricing fails, show USD with warning
- **Rule:** Always have fallback

### 2. Instant Updates
- **Rule:** Language switch = < 500ms UI update
- **Rule:** Country switch = < 1s pricing update
- **Rule:** No page reloads (use SPA navigation)

### 3. Clear Visual Feedback
- **Rule:** Show loading state during switch
- **Rule:** Show success/error toast
- **Rule:** Highlight active language/country

### 4. Universal Lesson Badge
- **Rule:** Always show what's available (languages, ages, tones)
- **Rule:** Make it clickable → opens demo
- **Rule:** Update badge when lesson loads

---

## 📁 FILE STRUCTURE

```
/public/
├── js/
│   ├── i18n/
│   │   ├── i18n-core.js              ✅ EXISTS
│   │   ├── i18n-kelly.js            ✅ EXISTS
│   │   ├── language-selector.js     ✅ EXISTS
│   │   └── universal-switcher.js    🆕 CREATE
│   └── geo-pricing.js               ✅ EXISTS
│
├── locales/
│   ├── en/                           ✅ EXISTS
│   ├── es/                           ✅ EXISTS
│   └── pt/                           ✅ EXISTS
│
/api/
├── geo-pricing.ts                    ✅ EXISTS
└── create-checkout.ts                ✅ EXISTS (needs update)

/public/
└── learn.html                        ✅ EXISTS (needs integration)
```

---

## 🔄 IMPLEMENTATION ORDER

1. **Day 1:** Complete Stripe batch work (36 prices, 18 payment methods)
2. **Day 2:** Build universal-switcher.js + integrate into learn.html
3. **Day 3:** Update pricing displays + add country switcher to billing
4. **Day 4:** Build adaptation demo modal + universal badge
5. **Day 5:** Update lesson loader for multi-language + fallbacks
6. **Day 6:** Connect Stripe checkout to currency selection
7. **Day 7:** Polish, testing, bug fixes

---

## 💡 BONUS IDEAS

### 1. "Try It Free" Demo Mode
- Let users switch language/country without account
- Show preview of lesson in new language
- "Sign up to unlock full experience"

### 2. Language Progress Badge
- "You've completed 5 lessons in Spanish!"
- Encourage language learning through Kelly

### 3. Currency Comparison Tool
- "In your currency, that's ₹1,999 (50% off for India)"
- Show savings vs USD

### 4. Regional Payment Methods Highlight
- "Popular in India: UPI" badge
- "Popular in Brazil: PIX" badge

---

## ✅ FINAL CHECKLIST

Before marking complete:
- [ ] Language switcher works in learn.html
- [ ] Country switcher works in learn.html + billing panel
- [ ] Prices update instantly when country changes
- [ ] Lessons load in selected language (with fallback badge)
- [ ] Universal badge shows available variants
- [ ] Adaptation demo modal works
- [ ] Stripe checkout uses correct currency
- [ ] All 36 Stripe prices created
- [ ] All 18 payment methods enabled
- [ ] All 36 env vars set in Vercel
- [ ] Tested with ?force_country=DE, ?force_country=IN
- [ ] No console errors when switching
- [ ] Preferences persist across sessions

---

**"Kelly speaks your language, knows your time, feels your season, and respects your currency."**

*Directive created: December 23, 2025*

