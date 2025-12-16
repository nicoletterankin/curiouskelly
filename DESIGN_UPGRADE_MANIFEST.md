# CURIOUS KELLY — DESIGN UPGRADE MANIFEST
## One-Shot Polish Sprint

**Generated:** 2025-12-16
**Goal:** Transform curiouskelly.com from "functional" to "delightful first impression"

---

## CRITICAL FIX #1: Restore the Landing Page

### Problem
`public/index.html` has TWO HTML documents concatenated:
- Lines 1-27: Redirect stub (what users see)
- Lines 28-5217: Beautiful landing page (DEAD CODE)

### Solution
Delete lines 1-27 (the redirect stub). The landing page already has intelligent routing:
- Returns users (with `kelly-started` or `kellyState` in localStorage) → auto-redirect to `/learn.html`
- New visitors → see the full landing page with Kelly, value prop, and CTA

### Why This Matters
New visitors currently see a flash of "Opening Curious Kelly..." then land in a complex lesson player with no context. They need to:
1. See Kelly's face
2. Understand the value prop ("Learn something new every day")
3. Choose to start

---

## CRITICAL FIX #2: Unify Typography

### Current State (fragmented)
| Page | Headline | Body |
|------|----------|------|
| index.html (landing) | Inter | Inter |
| learn.html | Georgia | system-ui |
| pricing.html | Newsreader | Inter |
| about.html | Newsreader | Inter |

### Target State (unified)
| Element | Font | Rationale |
|---------|------|-----------|
| Headlines | `Newsreader` | Warm, editorial, timeless — matches Kelly's teaching persona |
| Body | `Inter` | Clean, highly legible, professional |
| Fallback | System stack | Bulletproof rendering |

### Implementation
Create `/css/typography.css`:
```css
:root {
  --font-display: 'Newsreader', Georgia, 'Times New Roman', serif;
  --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

h1, h2, h3, .headline { font-family: var(--font-display); }
body, p, .body-text { font-family: var(--font-body); }
```

Apply to: index.html, learn.html, pricing.html, about.html, settings.html

---

## CRITICAL FIX #3: Unify Color System

### Current State
- Kelly Blue: `#2563eb` (learn.html)
- Accent Blue: `#3b82f6` (some places)
- Orange Glow: `#f97316` (pricing.html)
- Random variations everywhere

### Target State
All pages use the same CSS custom properties from `/css/brand-tokens.css`:

```css
:root {
  /* Kelly Blue — Primary Brand */
  --kelly-blue: #2563eb;
  --kelly-blue-light: #3b82f6;
  --kelly-blue-dark: #1d4ed8;
  --kelly-blue-glow: rgba(37, 99, 235, 0.15);
  
  /* Surfaces (Dark Theme) */
  --surface-deepest: #09090b;
  --surface-deep: #0f0f13;
  --surface-base: #18181b;
  --surface-elevated: #1c1c24;
  --surface-raised: #252530;
  
  /* Text */
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  
  /* Borders */
  --border-default: #27272a;
  --border-hover: #3f3f46;
  
  /* Semantic */
  --success: #22c55e;
  --error: #ef4444;
  --gold-wisdom: #f59e0b;
}
```

Remove orange glow from pricing. Use `--kelly-blue-glow` instead.

---

## CRITICAL FIX #4: Navigation Consistency

### Current State
- `learn.html`: Minimal top nav + bottom tab bar (immersive app)
- `pricing.html`: Traditional top nav with links
- `about.html`: Traditional top nav with links
- `settings.html`: Different bottom nav layout

### Target State
Two modes, intentionally chosen:

**Mode A: Immersive (lesson experience)**
- `learn.html` only
- Minimal chrome: just "🏠 Curious Kelly" top-left
- Bottom nav: Home / Journey / Play / Settings
- Kelly is the focus

**Mode B: Marketing (everything else)**
- Shared header component across: index, pricing, about, curriculum, etc.
- Links: Learn | Pricing | About | Start Learning (CTA button)
- Consistent footer with legal links

### Implementation
Create shared header/footer partials and include in all marketing pages.

---

## POLISH #1: Add Kelly to Marketing Pages

### Problem
Pricing and About pages have no Kelly presence. The product is a digital human teacher, but marketing pages show no human.

### Solution
Add a small Kelly avatar or illustration to:
1. **Pricing page**: Kelly peeking from the side with a thought bubble about value
2. **About page**: Kelly portrait in hero section

Use existing assets from `/public/assets/kelly/` or `/public/images/`.

---

## POLISH #2: Improve First-Time Experience

### Current Flow (broken)
1. User visits curiouskelly.com
2. Instant redirect to /learn.html
3. User lands in lesson player with paywall dialog
4. User is confused

### Target Flow
1. User visits curiouskelly.com
2. **Sees landing page** with:
   - Kelly's face (hero right side, 60% width)
   - Today's date and lesson topic
   - Clear CTA: "Start Today's Lesson"
   - Social proof / testimonials below fold
3. User clicks CTA → goes to /learn.html
4. `kelly-started` flag is set
5. Future visits → auto-redirect to /learn.html

This is already implemented in the dead code — we just need to remove the redirect stub.

---

## POLISH #3: Quiet the Console

### Problem
Production site logs many debug warnings:
- `[LocalPack] Day 017 loaded`
- `🆘 Emergency lessons loaded`
- `⏸️ DEV MODE: Auto-advance DISABLED`

### Solution
Wrap debug logs in a check:
```javascript
const DEBUG = window.location.search.includes('debug=true');
if (DEBUG) console.warn(...);
```

Apply to:
- `/js/kelly-lesson-loader.js`
- `/data/day-017-complete.js`
- `/data/emergency-lessons.js`
- `/js/kelly-fallback-engine.js`

---

## POLISH #4: Fix Paywall Dialog UX

### Problem
Paywall dialog appears immediately when loading a non-today lesson, before user has any context.

### Solution
Delay paywall by 2 seconds OR wait until user tries to advance past first phase. Let them "taste" the lesson before asking for payment.

---

## EXECUTION ORDER

1. **Fix index.html** — Remove redirect stub (lines 1-27)
2. **Update fonts** — Apply Newsreader + Inter consistently
3. **Unify colors** — Use brand-tokens.css everywhere, remove orange
4. **Quiet console** — Add DEBUG check to verbose loggers
5. **Test in incognito** — Verify first-time flow works
6. **Commit and deploy**

---

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `public/index.html` | Delete lines 1-27; update fonts to Newsreader/Inter |
| `public/pricing.html` | Change orange to kelly-blue; add Newsreader headlines |
| `public/about.html` | Ensure Newsreader headlines; consider adding Kelly image |
| `public/js/kelly-lesson-loader.js` | Wrap verbose logs in DEBUG check |
| `public/data/emergency-lessons.js` | Wrap verbose logs in DEBUG check |
| `public/js/kelly-fallback-engine.js` | Wrap verbose logs in DEBUG check |

---

## SUCCESS CRITERIA

- [ ] New visitor sees landing page with Kelly (not instant redirect)
- [ ] All pages use Newsreader for headlines, Inter for body
- [ ] All pages use kelly-blue accent (no orange)
- [ ] Console is quiet in production (no debug spam)
- [ ] Navigation feels cohesive across marketing pages
- [ ] Kelly's face appears on at least one marketing page

---

**This is the path to "feels like a real product, not a prototype."**

