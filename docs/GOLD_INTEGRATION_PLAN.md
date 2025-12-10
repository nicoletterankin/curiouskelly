# 🪙 GOLD INTEGRATION PLAN
*Updated: December 10, 2025*
*Status: ✅ PHASES 1-4 COMPLETE — READY FOR PHASE 5*

---

## 📋 EXECUTIVE SUMMARY

The extracted gold from 14 archived pages has been integrated into the live Curious Kelly site. We are using **Kelly Blue (#2563eb)** as the ONE brand color across all pages. No Campus Orange theme — unified design for December 17 launch.

---

## ✅ COMPLETED TASKS

| # | Task | Status | Details |
|---|------|--------|---------|
| 1 | Consolidated CSS tokens | ✅ Done | `/public/css/brand-tokens.css` - Kelly Blue only |
| 2 | Created component library | ✅ Done | `/public/css/components.css` |
| 3 | Moved legal pages | ✅ Done | `/public/legal/privacy.html`, `terms.html`, `diversity.html` |
| 4 | Updated site-health.html | ✅ Done | New file structure reflected |
| 5 | Created EXTRACTED_GOLD.md | ✅ Done | `/public/admin/EXTRACTED_GOLD.md` |
| 6 | Archived 14 legacy pages | ✅ Done | `/_archive/legacy-pages-2025-12-10/` |
| 7 | Added components.css to index.html | ✅ Done | Design system active |
| 8 | Added components.css to learn.html | ✅ Done | Design system active |
| 9 | Verified careers.html is Kelly Blue | ✅ Done | Affiliate program ready |
| 10 | Updated ALL footer links to /legal/ | ✅ Done | 15+ pages updated |

---

## 🎨 BRAND DECISION: KELLY BLUE ONLY

Per user direction (Dec 10):
> "you got the orange wrong - we are moving to production blue because we live in 7 days"

**Result:**
- ❌ Removed Campus Orange theme from `brand-tokens.css`
- ✅ `--accent-primary` now resolves to Kelly Blue (`#2563eb`) everywhere
- ✅ All component glows use Kelly Blue
- ✅ No theme switching — one unified look

---

## 📊 COMPONENTS NOW AVAILABLE

These are ready to use across all pages:

### Buttons
```html
<button class="btn btn-primary">Primary (Kelly Blue)</button>
<button class="btn btn-secondary">Secondary (White)</button>
<button class="btn btn-outline">Outline</button>
<button class="btn btn-ghost">Ghost</button>
```

### Cards
```html
<div class="card">Basic Card</div>
<div class="card featured">Featured Card (blue glow)</div>
<div class="stat-card">
    <div class="stat-number">365</div>
    <div class="stat-label">Daily Lessons</div>
</div>
```

### Badges
```html
<span class="badge badge-primary">Primary</span>
<span class="badge badge-success">Success</span>
<span class="badge badge-warning">Warning</span>
<span class="badge badge-error">Error</span>
```

### Live Indicator
```html
<div class="live-indicator">
    <span class="live-dot"></span>
    LIVE NOW
</div>
```

### Inputs
```html
<input type="text" class="input-field" placeholder="Email">
<input type="range" class="slider" min="0" max="100">
```

### Grids
```html
<div class="stat-grid"><!-- stat cards --></div>
<div class="feature-grid"><!-- feature cards --></div>
<div class="calendar-grid"><!-- day cards --></div>
```

---

## 🔗 UPDATED FOOTER LINKS

All production pages now use:
```html
<a href="/legal/privacy.html">Privacy</a>
<a href="/legal/terms.html">Terms</a>
```

Updated files (15+):
- `index.html` ✅
- `careers.html` ✅
- `about.html` ✅
- `curriculum.html` ✅
- `pricing.html` ✅
- `help.html` ✅
- `contact.html` ✅
- `trust.html` ✅
- `accessibility.html` ✅
- `join.html` ✅
- `enterprise.html` ✅
- `compare-us.html` ✅
- `perspectives.html` ✅
- `ambassador.html` ✅
- `toc.html` ✅
- `earnings.html` ✅
- `app.html` ✅
- `js/age-gate.js` ✅

---

## 📁 NEW FILE STRUCTURE

```
public/
├── css/
│   ├── brand-tokens.css    # Design tokens (Kelly Blue only)
│   ├── components.css      # Reusable UI components
│   ├── brand-colors.css    # Legacy (still used)
│   └── kelly-animations.css
├── legal/
│   ├── privacy.html        # COPPA/GDPR compliant
│   ├── terms.html          # Full ToS
│   └── diversity.html      # Accessibility commitment
├── admin/
│   ├── site-health.html    # Page inventory
│   ├── mission-control.html # Task dashboard
│   └── EXTRACTED_GOLD.md   # Archive reference
└── [all other pages]

_archive/
└── legacy-pages-2025-12-10/
    ├── about.html
    ├── careers.html
    ├── diversity.html
    ├── enterprise.html
    ├── newsroom.html
    ├── privacy.html
    ├── terms.html
    ├── social.html
    ├── player.html
    ├── dashboard.html
    ├── index-production.html
    ├── index-final.html
    ├── learn-v1.html
    └── learn-v2.html
```

---

## 🚀 PHASE 5: REMAINING INTEGRATION (Optional)

These are enhancements that can be done after launch:

### 5.1 Replace Inline Styles with Components
Audit pages for inline styles that could use component classes:
- Search for `style="background:` → use `.card` or `.btn`
- Search for `border-radius:` → use component classes

### 5.2 Add Persona Bar to Learn Page
The persona bar component (age slider + language toggle) from the gold could enhance the lesson player.

### 5.3 Add Collapsible Sections to FAQ
The collapsible pattern could improve UX on help and FAQ pages.

### 5.4 Polish Legal Pages
Add `.highlight-box` for TL;DR sections in privacy and terms.

---

## ✅ LAUNCH READINESS CHECKLIST

| Item | Status |
|------|--------|
| Unified design system | ✅ Kelly Blue everywhere |
| Component library | ✅ Ready |
| Legal pages in /legal/ | ✅ Done |
| Footer links updated | ✅ 15+ pages |
| Legacy pages archived | ✅ 14 pages |
| Careers/affiliate page | ✅ Ready with calculator |
| Site health dashboard | ✅ Updated |

**Launch Date: December 17, 2025** 🚀

---

*All critical integration complete. Optional Phase 5 enhancements can be done post-launch.*
