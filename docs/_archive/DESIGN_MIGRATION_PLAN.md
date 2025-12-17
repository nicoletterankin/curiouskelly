# Complete Design Migration Plan
**Last Updated:** December 5, 2025  
**Target Completion:** December 15, 2025 (before Dec 17 launch)

---

## 🎯 New Design System Specification

### Typography
```css
--font-body: 'Instrument Sans', sans-serif;   /* Body text */
--font-heading: 'Newsreader', serif;          /* Headlines, editorial */
--font-mono: 'JetBrains Mono', monospace;     /* Code blocks (API page only) */
```

### Color Palette
```css
:root {
  /* Backgrounds */
  --bg-color: #0a0a0b;
  --bg-secondary: #111113;
  --bg-elevated: #18181b;
  
  /* Text */
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  
  /* Accent - Blue */
  --accent-primary: #3b82f6;
  --accent-hover: #2563eb;
  --accent-glow: rgba(59, 130, 246, 0.15);
  
  /* Functional */
  --success: #22c55e;
  --error: #ef4444;
  --warning: #f59e0b;      /* Reserved for alerts */
  --alert-orange: #f97316; /* Reserved for alerts */
  
  /* Borders */
  --border-color: #27272a;
  --border-hover: #3f3f46;
}
```

### Font Import
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
```

---

## 📊 Complete Page Inventory (68 HTML files)

### ✅ ALREADY MIGRATED TO NEW DESIGN (4 pages)

| File | Status | Notes |
|------|--------|-------|
| `index.astro` | ✅ Complete | Main homepage |
| `api.html` | ✅ Complete | API documentation |
| `affiliates.html` | ✅ Complete | Fixed cookie duration (30 days) |
| `accessibility.html` | ✅ Complete | WCAG compliance |

---

### 🔴 PRIORITY 1: Marketing Pages (12 pages)
**Must complete before Dec 17 launch**

| File | Current Design | Complexity | Est. Time |
|------|----------------|------------|-----------|
| `about.html` | Times New Roman | Medium | 2 hrs |
| `careers.html` | Times New Roman | Medium | 2 hrs |
| `commons.html` | Inter + Fraunces | **HIGH** | 4 hrs |
| `curriculum.html` | Inter + Fraunces | High | 3 hrs |
| `enterprise.html` | Times New Roman | Medium | 2 hrs |
| `gifts.html` | Times New Roman | Medium | 2 hrs |
| `help.html` | Inter (partial) | Low | 1 hr |
| `newsroom.html` | Times New Roman | Medium | 2 hrs |
| `pricing.html` | Times New Roman | Low | 1 hr |
| `privacy.html` | Unknown | Low | 1 hr |
| `terms.html` | Unknown | Low | 1 hr |
| `trust.html` | DM Sans + Fraunces | Medium | 2 hrs |

**Subtotal: ~23 hours**

---

### 🟡 PRIORITY 2: Secondary Marketing Pages (11 pages)
**Complete by Dec 20**

| File | Current Design | Complexity | Est. Time |
|------|----------------|------------|-----------|
| `ambassador.html` | Inter + Fraunces | Medium | 2 hrs |
| `contact.html` | Inter | Low | 1 hr |
| `diversity.html` | Times New Roman | Medium | 1.5 hrs |
| `impact.html` | DM Sans + Cormorant | Medium | 2 hrs |
| `join.html` | Inter + Fraunces | Low | 1 hr |
| `group.html` | Inter + Fraunces | Low | 1 hr |
| `missions.html` | Times New Roman | Medium | 2 hrs |
| `partner.html` | DM Sans + Cormorant | Medium | 2 hrs |
| `perspectives.html` | Unknown | Medium | 1.5 hrs |
| `social.html` | Times New Roman | Low | 1 hr |
| `affiliate-assets.html` | Inter | Low | 1 hr |

**Subtotal: ~16 hours**

---

### 🟢 PRIORITY 3: App Pages (14 pages)
**These have their OWN design system (Kelly OS) - DO NOT migrate**

| File | Purpose | Action |
|------|---------|--------|
| `learn.html` | Main lesson experience | Keep as-is |
| `learn-v1.html` | Legacy version | Keep/Archive |
| `learn-v2.html` | Alternative version | Keep/Archive |
| `app.html` | App shell | Keep as-is |
| `calendar.html` | Calendar view | Keep as-is |
| `dashboard.html` | User dashboard | Keep as-is |
| `hub.html` | Kelly hub | Keep as-is |
| `kelly.html` | Kelly interface | Keep as-is |
| `lesson-detail.html` | Lesson details | Keep as-is |
| `live.html` | Live feature | Keep as-is |
| `me.html` | User profile | Keep as-is |
| `player.html` | Lesson player | Keep as-is |
| `settings.html` | User settings | Keep as-is |
| `welcome.html` | Onboarding | Keep as-is |

---

### 🔵 PRIORITY 4: Utility Pages (8 pages)
**Quick updates needed**

| File | Current State | Action |
|------|---------------|--------|
| `404.html` | Unknown | Update to new design |
| `payment-failed.html` | Unknown | Update to new design |
| `payment-cancelled.html` | Unknown | Update to new design |
| `day/index.html` | Unknown | Review |
| `press-kit/index.html` | Unknown | Update to new design |
| `components/footer.html` | Component | Update component |
| `components/header.html` | Component | Update component |
| `index.html` | Old homepage | Archive |

---

### ⚪ PRIORITY 5: Admin Pages (2 pages)
**Internal - lower priority**

| File | Action |
|------|--------|
| `admin/affiliates.html` | Update when time permits |
| `affiliate-dashboard.html` | Update when time permits |

---

### 🗑️ DO NOT MIGRATE (17 pages)
**Test files, mockups, old versions - Archive or delete**

| File | Reason |
|------|--------|
| `test-video-player.html` | Test file |
| `test-lipsync.html` | Test file |
| `test-tts-debug.html` | Test file |
| `test-chat.html` | Test file |
| `test-dashboard.html` | Test file |
| `test-learn-page.html` | Test file |
| `debug-lessons.html` | Debug file |
| `unity-test.html` | Unity test |
| `kelly-hero-test.html` | Test file |
| `mockups/index.html` | Mockup |
| `mockups/kelly-hub-mockup.html` | Mockup |
| `mockups/kelly-frame-mockup.html` | Mockup |
| `unity/kelly-v1/index.html` | Unity build |
| `unity/kelly-live/index.html` | Unity build |
| `index-unified.html` | Old version |
| `index-production.html` | Old version |
| `index-final.html` | Old version |
| `index-old-backup.html` | Backup |

---

## 📋 Migration Checklist Per Page

### For each page, complete:

- [ ] **Typography**: Change to Instrument Sans + Newsreader
- [ ] **Colors**: Update CSS variables to new palette
- [ ] **Header**: Replace with standard header component
  ```html
  <header class="header">
    <div class="header-inner">
      <a href="/" class="logo">
        <img src="/images/brand/kelly-mark-circle-64.png" alt="">
        <span>Curious Kelly</span>
      </a>
      <nav class="nav-links">
        <a href="/">Home</a>
        <a href="/curriculum.html">Curriculum</a>
        <a href="/enterprise.html">Enterprise</a>
        <a href="mailto:hello@curiouskelly.com">Contact</a>
      </nav>
    </div>
  </header>
  ```
- [ ] **Footer**: Replace with standard footer
- [ ] **Buttons**: Update to standard button styles
- [ ] **Cards/Boxes**: Use glass effects where appropriate
- [ ] **Responsive**: Test mobile breakpoints
- [ ] **Test**: Verify in browser

---

## 🔥 commons.html - Special Migration Notes

**This is a complex page requiring special attention:**

1. **Current state**: Inter + Fraunces, proper colors but wrong fonts
2. **Functionality**: 
   - Supabase integration for proposals
   - Tab system (Proposals, Knowledge Base, Discussions)
   - Sidebar navigation
   - Real-time voting
   - Modal for creating proposals
   - Activity feed
   - Contributor leaderboard

3. **Migration approach**:
   - Keep ALL JavaScript functionality intact
   - Only update CSS/fonts
   - Test voting after migration
   - Verify modal still works

4. **Key changes needed**:
   ```css
   /* FROM */
   font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
   font-family: 'Fraunces', Georgia, serif;
   
   /* TO */
   font-family: 'Instrument Sans', -apple-system, sans-serif;
   font-family: 'Newsreader', Georgia, serif;
   ```

---

## 📅 Migration Schedule

### Week 1: Dec 5-8 (Priority 1)
| Day | Pages | Hours |
|-----|-------|-------|
| Thu 5 | about, careers | 4 |
| Fri 6 | commons | 4 |
| Sat 7 | curriculum, enterprise | 5 |
| Sun 8 | gifts, help, pricing | 4 |

### Week 2: Dec 9-12 (Priority 1 continued)
| Day | Pages | Hours |
|-----|-------|-------|
| Mon 9 | newsroom, privacy, terms | 4 |
| Tue 10 | trust, contact | 3 |
| Wed 11 | Priority 2 batch 1 | 5 |
| Thu 12 | Priority 2 batch 2 | 5 |

### Week 3: Dec 13-15 (Finalization)
| Day | Task | Hours |
|-----|------|-------|
| Fri 13 | Priority 3-4, utility pages | 4 |
| Sat 14 | Testing, bug fixes | 4 |
| Sun 15 | Final QA | 3 |

---

## ✅ Quality Gates

Before marking a page as complete:

1. [ ] Typography matches spec (Instrument Sans body, Newsreader headings)
2. [ ] Colors match spec (no orange except alerts, blue primary)
3. [ ] Header is consistent with other migrated pages
4. [ ] Footer is consistent with other migrated pages
5. [ ] Mobile responsive (test at 375px, 768px, 1024px)
6. [ ] No console errors
7. [ ] All links work
8. [ ] Forms submit correctly (if applicable)
9. [ ] Page loads in < 2 seconds

---

## 🚨 Migration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking Supabase integration on commons.html | HIGH | Test thoroughly before/after |
| Stripe links breaking on gifts.html | HIGH | Verify all payment links |
| SEO impact from design changes | MEDIUM | Keep meta tags, structure |
| Missing pages in inventory | MEDIUM | Re-audit after initial migration |

---

## 📝 Notes

- **DO NOT** change any JavaScript functionality during migration
- **DO NOT** modify Stripe integration code
- **DO NOT** change URL structure
- **DO** keep all meta tags and OG tags
- **DO** preserve all analytics tracking
- **DO** backup files before modifying

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Already migrated | 4 | ✅ |
| Priority 1 (launch critical) | 12 | 🔴 |
| Priority 2 (secondary) | 11 | 🟡 |
| App pages (keep as-is) | 14 | 🟢 |
| Utility pages | 8 | 🔵 |
| Admin pages | 2 | ⚪ |
| Do not migrate | 17 | 🗑️ |
| **Total** | **68** | |

**Estimated total migration time: ~50 hours**




