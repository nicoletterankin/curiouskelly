# Design Migration Evaluation Framework
**Version:** 1.0  
**Created:** December 5, 2025  
**Purpose:** Ensure quality, reliability, and cohesiveness across all migrated pages

---

## 📋 Evaluation Categories

| Category | Weight | Description |
|----------|--------|-------------|
| Design System Compliance | 30% | Does it match the spec exactly? |
| Visual Cohesiveness | 25% | Does it feel like the same brand as other pages? |
| Functionality | 20% | Do all interactive elements work? |
| Accessibility | 15% | Can everyone use it? |
| Performance | 10% | Does it load fast? |

---

## 🎨 EVAL 1: Design System Compliance (30%)

### 1.1 Typography Check

| Element | Required Value | Pass/Fail |
|---------|---------------|-----------|
| Body font | `'Instrument Sans', -apple-system, sans-serif` | ☐ |
| Heading font | `'Newsreader', Georgia, serif` | ☐ |
| Font import | Google Fonts with preconnect | ☐ |
| Base font size | 16px (browser default) | ☐ |
| Line height body | 1.6-1.7 | ☐ |
| Line height headings | 1.1-1.2 | ☐ |
| No Times New Roman | Zero occurrences | ☐ |
| No Inter | Zero occurrences (except app pages) | ☐ |
| No Fraunces | Zero occurrences | ☐ |
| No DM Sans | Zero occurrences | ☐ |

**Automated Check:**
```bash
# Run from project root
grep -r "Times New Roman" public/*.html --include="*.html" | grep -v "test-" | grep -v "mockup"
grep -r "'Inter'" public/*.html --include="*.html" | grep -v "test-" | grep -v "mockup" | grep -v "learn"
grep -r "Fraunces" public/*.html --include="*.html" | grep -v "test-" | grep -v "mockup"
grep -r "DM Sans" public/*.html --include="*.html" | grep -v "test-" | grep -v "mockup"
```

### 1.2 Color Palette Check

| Variable | Required Value | Pass/Fail |
|----------|---------------|-----------|
| `--bg-color` | `#0a0a0b` | ☐ |
| `--bg-secondary` | `#111113` | ☐ |
| `--bg-elevated` | `#18181b` | ☐ |
| `--text-primary` | `#fafafa` | ☐ |
| `--text-secondary` | `#a1a1aa` | ☐ |
| `--text-muted` | `#71717a` | ☐ |
| `--accent-primary` | `#3b82f6` | ☐ |
| `--accent-hover` | `#2563eb` | ☐ |
| `--border-color` | `#27272a` | ☐ |
| No orange as primary | Only in alerts | ☐ |
| No `#0f0f11` | Old bg color | ☐ |
| No `#f4f4f5` | Old text color | ☐ |

**Automated Check:**
```bash
# Should return 0 results for migrated pages
grep -r "#0f0f11" public/*.html --include="*.html" | grep -v "test-" | grep -v "index-"
grep -r "#f4f4f5" public/*.html --include="*.html" | grep -v "test-" | grep -v "index-"
```

### 1.3 Component Consistency

| Component | Required Pattern | Pass/Fail |
|-----------|-----------------|-----------|
| Header height | 64px | ☐ |
| Header background | `rgba(10, 10, 11, 0.8)` with blur | ☐ |
| Logo | Kelly mark + "Curious Kelly" text | ☐ |
| Logo image | `/images/brand/kelly-mark-circle-64.png` | ☐ |
| Button border-radius | 8px | ☐ |
| Card border-radius | 12-16px | ☐ |
| Max content width | 1100px | ☐ |
| Section padding | 80px vertical | ☐ |

---

## 🔗 EVAL 2: Visual Cohesiveness (25%)

### 2.1 Side-by-Side Comparison

For each migrated page, compare against these reference pages:
- `index.astro` (homepage)
- `api.html` (documentation)
- `affiliates.html` (marketing)

| Check | Description | Pass/Fail |
|-------|-------------|-----------|
| Header matches | Same structure, colors, spacing | ☐ |
| Footer matches | Same columns, links, styling | ☐ |
| Section rhythm | Consistent spacing between sections | ☐ |
| Typography hierarchy | H1 > H2 > H3 sizes feel consistent | ☐ |
| Card styling | Same border, shadow, hover effects | ☐ |
| Button styling | Same padding, colors, hover states | ☐ |
| Link styling | Same color, hover behavior | ☐ |
| Icon usage | Consistent emoji/icon style | ☐ |

### 2.2 Brand Voice Check

| Check | Requirement | Pass/Fail |
|-------|-------------|-----------|
| Tone | Warm, curious, intelligent | ☐ |
| No marketing hyperbole | No "revolutionary", "game-changing" | ☐ |
| Humble claims | No unsubstantiated statistics | ☐ |
| Consistent terminology | "Kelly" not "the AI", "lesson" not "course" | ☐ |
| Company name | "Lesson of the Day PBC" in footer | ☐ |
| Email | Only `hello@curiouskelly.com` | ☐ |

### 2.3 Visual Regression Test

**Manual Screenshot Comparison:**
1. Take screenshot of old page (if available)
2. Take screenshot of new page
3. Check for unintended changes:
   - [ ] Content is preserved
   - [ ] Layout is improved or equivalent
   - [ ] No broken images
   - [ ] No text overflow issues

---

## ⚙️ EVAL 3: Functionality (20%)

### 3.1 Link Verification

| Check | Method | Pass/Fail |
|-------|--------|-----------|
| Internal links work | Click each link | ☐ |
| External links work | Click each link (opens new tab) | ☐ |
| Anchor links work | Smooth scroll to section | ☐ |
| Email links work | Opens mail client | ☐ |
| No 404s | All paths resolve | ☐ |

### 3.2 JavaScript Functionality

For pages with JS (commons.html, curriculum.html, etc.):

| Check | Method | Pass/Fail |
|-------|--------|-----------|
| No console errors | Open DevTools > Console | ☐ |
| Supabase connects | Check network tab | ☐ |
| Data loads | Content appears | ☐ |
| Interactions work | Click buttons, forms | ☐ |
| Modals open/close | Test modal triggers | ☐ |
| Forms submit | Test with valid data | ☐ |

### 3.3 Page-Specific Functional Tests

**commons.html:**
- [ ] Proposal list loads
- [ ] Tab switching works
- [ ] Search filters results
- [ ] Vote buttons respond (requires auth)
- [ ] New proposal modal opens
- [ ] Sidebar navigation works

**curriculum.html:**
- [ ] 365 lessons display
- [ ] Day cards are clickable
- [ ] Search/filter works
- [ ] Month navigation works

**gifts.html:**
- [ ] Stripe links are correct
- [ ] All 4 pricing tiers display
- [ ] Links open Stripe checkout

**help.html:**
- [ ] FAQ accordions expand/collapse
- [ ] Search works
- [ ] Contact links work

---

## ♿ EVAL 4: Accessibility (15%)

### 4.1 Automated Checks

Run these tools on each page:
- [ ] **Lighthouse Accessibility** (Chrome DevTools) - Score ≥ 90
- [ ] **axe DevTools** - 0 critical/serious issues
- [ ] **WAVE** - 0 errors

### 4.2 Manual Checks

| Check | Method | Pass/Fail |
|-------|--------|-----------|
| Keyboard navigation | Tab through entire page | ☐ |
| Focus visible | Can see focused element | ☐ |
| Skip link | First tab focuses skip link | ☐ |
| Heading hierarchy | H1 → H2 → H3 (no skips) | ☐ |
| Alt text | All images have alt | ☐ |
| Color contrast | Text readable on backgrounds | ☐ |
| Link purpose | Links describe destination | ☐ |
| Form labels | All inputs have labels | ☐ |

### 4.3 Color Contrast Verification

| Element | Foreground | Background | Ratio Required | Pass/Fail |
|---------|------------|------------|----------------|-----------|
| Body text | #fafafa | #0a0a0b | 4.5:1 | ☐ |
| Secondary text | #a1a1aa | #0a0a0b | 4.5:1 | ☐ |
| Muted text | #71717a | #0a0a0b | 4.5:1 | ☐ |
| Links | #3b82f6 | #0a0a0b | 4.5:1 | ☐ |
| Button text | #0a0a0b | #fafafa | 4.5:1 | ☐ |

---

## ⚡ EVAL 5: Performance (10%)

### 5.1 Lighthouse Performance

Run Lighthouse on each page:
- [ ] Performance score ≥ 90
- [ ] First Contentful Paint < 1.5s
- [ ] Largest Contentful Paint < 2.5s
- [ ] Cumulative Layout Shift < 0.1
- [ ] Total Blocking Time < 200ms

### 5.2 Asset Checks

| Check | Requirement | Pass/Fail |
|-------|-------------|-----------|
| No unused CSS | CSS is minimal | ☐ |
| Images optimized | WebP where possible | ☐ |
| Fonts preloaded | `rel="preconnect"` present | ☐ |
| No render-blocking | Critical CSS inlined | ☐ |
| Page size | < 500KB total | ☐ |

---

## 📊 Evaluation Scorecard Template

### Page: `[PAGE_NAME].html`
**Evaluator:** _______________  
**Date:** _______________

| Category | Weight | Score (0-100) | Weighted |
|----------|--------|---------------|----------|
| Design System Compliance | 30% | ___ | ___ |
| Visual Cohesiveness | 25% | ___ | ___ |
| Functionality | 20% | ___ | ___ |
| Accessibility | 15% | ___ | ___ |
| Performance | 10% | ___ | ___ |
| **TOTAL** | 100% | | **___** |

**Pass Threshold:** 85/100  
**Status:** ☐ PASS / ☐ FAIL / ☐ NEEDS REVISION

### Issues Found:
1. 
2. 
3. 

### Sign-off:
- [ ] Designer approved
- [ ] Developer approved
- [ ] Ready for production

---

## 🔄 Automated Eval Script

Save as `scripts/eval-migration.js`:

```javascript
/**
 * Migration Evaluation Script
 * Run: node scripts/eval-migration.js [page.html]
 */

const fs = require('fs');
const path = require('path');

const REQUIRED_FONTS = ['Instrument Sans', 'Newsreader'];
const BANNED_FONTS = ['Times New Roman', 'Inter', 'Fraunces', 'DM Sans'];
const REQUIRED_COLORS = {
  '--bg-color': '#0a0a0b',
  '--text-primary': '#fafafa',
  '--accent-primary': '#3b82f6'
};
const BANNED_COLORS = ['#0f0f11', '#f4f4f5'];

function evalPage(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const results = {
    file: path.basename(filePath),
    passed: true,
    issues: []
  };

  // Check for banned fonts
  BANNED_FONTS.forEach(font => {
    if (content.includes(font)) {
      results.passed = false;
      results.issues.push(`Found banned font: ${font}`);
    }
  });

  // Check for required fonts
  REQUIRED_FONTS.forEach(font => {
    if (!content.includes(font)) {
      results.passed = false;
      results.issues.push(`Missing required font: ${font}`);
    }
  });

  // Check for banned colors
  BANNED_COLORS.forEach(color => {
    if (content.includes(color)) {
      results.passed = false;
      results.issues.push(`Found old color: ${color}`);
    }
  });

  // Check for required color variables
  Object.entries(REQUIRED_COLORS).forEach(([variable, value]) => {
    if (!content.includes(variable) || !content.includes(value)) {
      results.issues.push(`Warning: May be missing ${variable}: ${value}`);
    }
  });

  // Check for logo
  if (!content.includes('kelly-mark-circle')) {
    results.issues.push(`Warning: May be missing Kelly logo`);
  }

  // Check for correct company name in footer
  if (!content.includes('Lesson of the Day PBC')) {
    results.issues.push(`Warning: Footer should reference "Lesson of the Day PBC"`);
  }

  // Check for correct email
  if (content.includes('@curiouskelly.com') && !content.includes('hello@curiouskelly.com')) {
    results.passed = false;
    results.issues.push(`Found unauthorized email address`);
  }

  return results;
}

// Run evaluation
const targetFile = process.argv[2];
if (!targetFile) {
  console.log('Usage: node eval-migration.js [page.html]');
  console.log('Example: node eval-migration.js public/about.html');
  process.exit(1);
}

const results = evalPage(targetFile);
console.log('\n=== MIGRATION EVAL RESULTS ===\n');
console.log(`File: ${results.file}`);
console.log(`Status: ${results.passed ? '✅ PASSED' : '❌ FAILED'}`);
if (results.issues.length > 0) {
  console.log('\nIssues:');
  results.issues.forEach((issue, i) => {
    console.log(`  ${i + 1}. ${issue}`);
  });
}
console.log('\n==============================\n');
```

---

## 🧪 Pre-Migration Baseline Tests

Before migrating each page, capture:

1. **Screenshot** of current page
2. **All links** on the page (for comparison)
3. **JS functionality** working state
4. **Lighthouse scores** (baseline)

---

## ✅ Post-Migration Checklist

After migrating each page:

- [ ] Run automated eval script
- [ ] Take new screenshot
- [ ] Compare old vs new screenshot
- [ ] Test all links
- [ ] Test all JS functionality
- [ ] Run Lighthouse
- [ ] Run axe accessibility check
- [ ] Mobile responsive check (375px, 768px, 1024px)
- [ ] Cross-browser check (Chrome, Firefox, Safari)
- [ ] Fill out scorecard
- [ ] Get sign-off

---

## 📈 Quality Metrics Dashboard

Track these metrics across all migrations:

| Metric | Target | Current |
|--------|--------|---------|
| Pages migrated | 27 | 5 |
| Eval pass rate | 100% | ___ |
| Avg eval score | ≥90 | ___ |
| Accessibility issues | 0 | ___ |
| Broken links | 0 | ___ |
| Console errors | 0 | ___ |

---

## 🚨 Failure Modes & Remediation

### Critical Failures (Block deployment)
- Wrong fonts used
- Broken Stripe/Supabase integration
- Accessibility score < 70
- Console errors
- 404 links

### Major Failures (Must fix before launch)
- Visual inconsistency with reference pages
- Performance score < 80
- Missing footer/header
- Wrong company name or email

### Minor Failures (Fix when possible)
- Suboptimal image formats
- Minor spacing differences
- Non-critical accessibility warnings

---

## 📝 Notes

- Run evals on EVERY page before marking complete
- Don't skip the screenshot comparison
- When in doubt, compare to `api.html` (gold standard)
- Keep this document updated as we find new issues



