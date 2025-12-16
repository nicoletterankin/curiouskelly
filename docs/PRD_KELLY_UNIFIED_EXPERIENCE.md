# 🎯 PRD: Kelly Unified Experience
## Kelly as Master of Ceremonies for Everything

**Version:** 1.0  
**Created:** December 16, 2025  
**Priority:** CRITICAL (Launch is December 17)  
**Owner:** AI Assistant  

---

## 📋 EXECUTIVE SUMMARY

### The Problem
1. **Confusing pricing messaging** — Users don't understand what's free vs paid
2. **Checkout leaves Kelly** — Gift purchases go to external Stripe pages
3. **No unified presence** — Kelly isn't the guide for payments, settings, onboarding
4. **Too many options** — Monthly, Annual, Family, Lifetime, Gifts, Enterprise, Single Lesson... overwhelming
5. **"Free tier" unclear** — "7 days to explore" doesn't communicate there's always a free option

### The Vision
**Kelly is ALWAYS visible.** She is the master of ceremonies for:
- Onboarding new users
- Daily lessons
- Payments & upgrades
- Settings changes
- Lesson completion celebrations
- Creating custom lessons (future)
- Learners Commons interactions

**Nothing happens without Kelly being present.** The user never leaves her.

---

## 📊 CURRENT STATE AUDIT

### What Exists Today

| Tier | Price | Exposed Where | Status |
|------|-------|---------------|--------|
| **Free** | $0 | "7 days to explore" | ⚠️ Confusing — implies trial, not permanent free tier |
| **Monthly** | $7.99/mo | pricing.html, learn.html | ✅ Working |
| **Annual** | $49.99/yr | pricing.html, learn.html | ✅ Working |
| **Family** | $99.99/yr | docs only | ❌ NOT on pricing page |
| **Lifetime** | $199.99 | pricing.html, learn.html | ✅ Working |
| **Single Lesson** | $1.99 | API only | ❌ Hidden, exists in DB |
| **Gifts** | $24.99-$149.99 | pricing.html | ⚠️ External Stripe links! |
| **Enterprise** | Custom | enterprise.html | ✅ Contact form |

### Checkout Flows Today

| Flow | Experience | Kelly Present? |
|------|------------|----------------|
| Monthly/Annual/Lifetime | Embedded Stripe in learn.html | ✅ Yes |
| Single Lesson | API → Stripe redirect | ❌ No |
| Gifts | External Stripe payment link | ❌ No — user leaves site! |
| Enterprise | Contact form → email | Partially |

---

## 🎯 PROPOSED TIERS (Simplified)

### Tier 1: FREE (Always Available)
**"Yours to Keep"**
- Today's lesson, every day
- All 12 Kelly personas
- Basic age adaptation
- **No credit card. No trial. Forever free.**

### Tier 2: KELLY+ ($7.99/mo or $49.99/yr)
**"Unlock Everything"**
- All 365 lessons (past, present, future)
- Multi-language support (EN, ES, PT)
- Progress sync across devices
- Family sharing (up to 5 members)
- Priority support
- **Coming soon:** Generate X custom lessons/month

### Tier 3: KELLY LIFETIME ($199.99)
**"Once, Forever Yours"**
- Everything in Kelly+
- Founding member badge
- VIP support
- All future features included
- Never pay again

### Enterprise & Education
**"Contact Us"**
- Schools, districts, libraries
- Custom pricing per seat
- Admin dashboard
- Usage reporting

---

## 🏛️ ARCHITECTURE: KELLY AS MC

### The Kelly Frame
Kelly is ALWAYS visible as a background/sidebar presence. All interactions happen IN FRONT of her.

```
┌─────────────────────────────────────────────┐
│                                             │
│                KELLY VISIBLE                │
│              (background layer)             │
│                                             │
│   ┌─────────────────────────────────┐      │
│   │                                 │      │
│   │      CONTENT PANEL              │      │
│   │   (lessons, checkout, etc)      │      │
│   │                                 │      │
│   │   This slides/fades in/out      │      │
│   │   Kelly never disappears        │      │
│   │                                 │      │
│   └─────────────────────────────────┘      │
│                                             │
└─────────────────────────────────────────────┘
```

### State Machine: Kelly's Modes

| Mode | Kelly's Role | Visual |
|------|-------------|--------|
| **ONBOARDING** | "Welcome! Let me show you around..." | Warm smile, eye contact |
| **TEACHING** | Active lesson delivery | Gestures, expressions |
| **CELEBRATING** | "You did it!" | Excited, applauding |
| **UPGRADING** | "Ready to unlock more?" | Encouraging, not pushy |
| **WAITING** | Idle, breathing, occasional blink | Peaceful, patient |
| **THINKING** | User is processing | Thoughtful expression |
| **SETTINGS** | "Let me help you customize..." | Attentive |

---

## 🛒 CHECKOUT REDESIGN

### Requirements
1. **Never leave learn.html** — all checkout happens in modals/panels
2. **Kelly visible** — checkout panel overlays, Kelly still in background
3. **Gift flow embedded** — no external Stripe links
4. **Simple 3-step flow:**
   - Step 1: Choose plan (visual cards, Kelly recommends annual)
   - Step 2: Enter email (if not logged in)
   - Step 3: Payment (Stripe Elements embedded)

### Implementation: Unified Checkout Panel

```javascript
// All checkout flows use same panel
function showCheckoutPanel(options) {
  const { 
    type,           // 'subscription' | 'gift' | 'single_lesson'
    plan,           // 'monthly' | 'annual' | 'lifetime'
    lessonDay,      // for single lesson
    giftDuration,   // for gifts
    recipientEmail  // for gifts
  } = options;
  
  // Panel slides in from right
  // Kelly stays visible on left
  // Stripe Embedded Checkout loads in panel
}
```

### Gift Flow (Fixed)
Currently: `window.open('https://buy.stripe.com/...')` — LEAVES SITE

New flow:
1. User clicks "Give 3 Months"
2. Gift panel slides in (Kelly visible)
3. Enter recipient's email
4. Add personal message (optional)
5. Pay with embedded Stripe
6. Kelly celebrates: "Gift sent! 🎁"

---

## 📱 PAGE-BY-PAGE AUDIT

### Homepage (index.html)
**Current:** Full marketing page with Kelly hero  
**Status:** ✅ Good — Kelly is prominent

### Learn Page (learn.html)
**Current:** Main app, lessons, checkout  
**Status:** ✅ Good — Kelly always visible, embedded checkout works

**Needed:**
- Add single lesson purchase to paywall (not just subscription)
- Show "FREE" badge on today's lesson more clearly

### Pricing Page (pricing.html)
**Current:** Standalone marketing page  
**Status:** ⚠️ Not used — routes to learn.html

**Decision:** 
- OPTION A: Keep for SEO, but redirect all buttons to learn.html (current)
- OPTION B: Remove entirely, pricing lives only in learn.html
- **RECOMMENDATION:** Option A — SEO value, but all purchases in learn.html

### Enterprise Page (enterprise.html)
**Current:** Contact form  
**Status:** ✅ Acceptable — leads are email-based

**Enhancement:** Add Kelly to page header

### Gift Flow
**Current:** External Stripe links  
**Status:** ❌ BROKEN — must fix before launch

**Fix:** Embed gift checkout in learn.html panel

---

## 🚀 LAUNCH CRITICAL (Dec 17)

### Must Have

| Task | Effort | Impact |
|------|--------|--------|
| Change "7 days to explore" → "Yours to keep" messaging | 30min | High |
| Add "FREE" badge to today's lesson | 30min | High |
| Fix gift checkout (embed in learn.html) | 2-3hr | Critical |
| Add single lesson purchase option to paywall | 1hr | Medium |

### Nice to Have (Post-Launch)

| Task | Effort | Impact |
|------|--------|--------|
| Kelly mode transitions (celebrate, think, wait) | 4hr | Delight |
| Family plan UI | 2hr | Revenue |
| Custom lesson generation | 8hr+ | Future |

---

## 📐 MESSAGING CHEAT SHEET

### ❌ DON'T SAY
- "Free trial"
- "7 days free"
- "Try before you buy"
- "Premium"
- "Pro"

### ✅ DO SAY
- "Today's lesson is yours"
- "Unlock the full library"
- "Kelly+"
- "One subscription, whole family"
- "Learn with Kelly"

### Price Anchoring
- Annual: "$4.17/month" or "$0.14/day"
- Monthly: "Cancel anytime"
- Lifetime: "Pay once, learn forever"

---

## 🔧 TECHNICAL IMPLEMENTATION

### 1. Gift Checkout (Priority)

Replace external links in pricing.html:

```javascript
// OLD (broken)
window.handleGift = function(duration) {
  window.open('https://buy.stripe.com/...'); // LEAVES SITE
};

// NEW (embedded)
window.handleGift = function(duration) {
  window.location.href = '/learn.html?gift=' + duration;
};
```

In learn.html, detect `?gift=` param and show gift panel:
```javascript
const urlParams = new URLSearchParams(window.location.search);
const giftDuration = urlParams.get('gift');
if (giftDuration) {
  showGiftCheckoutPanel(giftDuration);
}
```

### 2. API Changes Needed

```
POST /api/create-gift-checkout
{
  "giftDuration": "3-month" | "6-month" | "12-month" | "lifetime",
  "recipientEmail": "friend@example.com",
  "senderEmail": "buyer@example.com",
  "message": "Happy Birthday!"
}

Response:
{
  "clientSecret": "cs_xxx" // For Stripe Embedded Checkout
}
```

### 3. Single Lesson Purchase UI

Add to paywall modal:
```html
<div class="purchase-options">
  <button onclick="purchaseSingleLesson({{DAY}})">
    Unlock this lesson — $1.99
  </button>
  <div class="divider">— or —</div>
  <button onclick="showCheckoutPanel('annual')">
    Unlock all 365 — $49.99/year
  </button>
</div>
```

---

## 📊 SUCCESS METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Checkout abandonment | Unknown | <30% |
| Gift purchases | ~0 (broken) | Track |
| Free → Paid conversion | Unknown | 5% |
| Time to first lesson | Unknown | <60 seconds |

---

## 🎬 KELLY MODE SPECIFICATIONS (Future)

### Mode: CELEBRATING
**Trigger:** User completes a lesson  
**Duration:** 3-5 seconds  
**Animation:** 
- Kelly smiles broadly
- Subtle confetti in background
- Eyes sparkle
- Optional: brief applause gesture

### Mode: UPGRADING  
**Trigger:** User hits paywall  
**Tone:** Encouraging, not pushy  
**Script variants:**
- "Looks like you're curious about more! I have 364 other lessons waiting..."
- "Want to explore the whole year? I'd love to teach you more."

### Mode: WAITING  
**Trigger:** User idle for >30 seconds  
**Animation:**
- Subtle breathing
- Occasional blink
- Maybe a slight smile

---

## ✅ CHECKLIST FOR LAUNCH

- [ ] Messaging: "7 days" → "Yours" everywhere
- [ ] Gift checkout embedded (not external)
- [ ] Single lesson purchase visible in paywall
- [ ] Kelly avatar on pricing page hero
- [ ] Kelly avatar on about page hero  
- [ ] Family plan visible (or hidden until ready)
- [ ] Enterprise page has Kelly
- [ ] All checkout stays on curiouskelly.com

---

## 📎 APPENDIX

### Related Documents
- `PRICING_LOCKED.md` — Canonical prices
- `docs/billing/PRICING_STRATEGY_BIBLE.md` — Promotions, affiliates
- `supabase/migrations/026_lesson_purchases.sql` — DB schema

### Stripe Products Required
- `kelly_plus_monthly` — $7.99/mo recurring
- `kelly_plus_annual` — $49.99/yr recurring  
- `kelly_plus_lifetime` — $199.99 one-time
- `kelly_gift_3mo` — $24.99 one-time
- `kelly_gift_6mo` — $39.99 one-time
- `kelly_gift_12mo` — $49.99 one-time
- `kelly_gift_lifetime` — $149.99 one-time
- `kelly_single_lesson` — $1.99 one-time

---

*This PRD is a living document. Update as we ship.*

