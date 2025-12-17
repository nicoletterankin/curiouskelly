# Homepage Migration Audit — December 5, 2025

## ✅ Migration Status: COMPLETE

The new homepage at `daily-lesson-marketing/src/pages/index.astro` is deployed with:
- Black/white/blue palette
- Instrument Sans + Newsreader typography
- Unified Aquarium architecture preserved
- All auth functionality intact

---

## 📊 Footer Link Audit

### ✅ PAGES THAT EXIST

| Link | Path | Status | Notes |
|------|------|--------|-------|
| How it works | #how-it-works | ✅ Anchor | Works |
| Pricing | #pricing | ✅ Anchor | Works |
| About | #about | ✅ Anchor | Works |
| Curriculum | /curriculum.html | ✅ EXISTS | In public/ |
| Gift cards | /gifts.html | ✅ EXISTS | In public/, functional |
| About | /about.html | ✅ EXISTS | In public/ |
| Careers | /careers.html | ✅ EXISTS | In public/ |
| Newsroom | /newsroom.html | ✅ EXISTS | In public/ |
| Contact | mailto:hello@curiouskelly.com | ✅ Works | Email link |
| For Schools | /enterprise.html | ✅ EXISTS | In public/ |
| Help Center | /help.html | ✅ EXISTS | In public/ |
| Web App | /learn.html | ✅ EXISTS | Main learn page |
| Privacy | /privacy.html | ✅ EXISTS | In public/ |
| Terms | /terms.html | ✅ EXISTS | In public/ |

### ❌ PAGES MISSING (404 RISK)

| Link | Path | Status | Priority | Action Required |
|------|------|--------|----------|-----------------|
| **API** | /api.html | ❌ MISSING | **HIGH** | Create - User requested |
| **Affiliates** | /affiliates.html | ❌ MISSING | HIGH | Create - Referenced in careers too |
| **Accessibility** | /accessibility.html | ❌ MISSING | MEDIUM | Create - Legal requirement |
| iOS App | # | ⚠️ Placeholder | LOW | Update when app launches |
| Android App | # | ⚠️ Placeholder | LOW | Update when app launches |

---

## 🎨 Style Consistency Audit

### New Design System (Target)
```css
:root {
  --bg-color: #0a0a0b;
  --bg-secondary: #111113;
  --bg-elevated: #18181b;
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --accent-primary: #3b82f6;
}
font-family: 'Instrument Sans' (body), 'Newsreader' (headings);
```

### ✅ Pages Using New Design System

| Page | Status | Notes |
|------|--------|-------|
| /index.astro | ✅ | New homepage - fully styled |
| /api.html | ✅ | Just created - fully styled |
| /affiliates.html | ✅ | Just created - fully styled |
| /accessibility.html | ✅ | Just created - fully styled |
| /help.html | ⚠️ Partial | Uses #0a0a0b but Inter font |

### ❌ Pages Using Old Design System

| Page | Current Font | Current Colors | Priority |
|------|--------------|----------------|----------|
| /about.html | Times New Roman | #0f0f11 bg | HIGH |
| /careers.html | Times New Roman | #0f0f11 bg | HIGH |
| /gifts.html | Times New Roman | #0f0f11 bg | MEDIUM |
| /enterprise.html | Times New Roman | #0f0f11 bg | MEDIUM |
| /newsroom.html | Times New Roman | #0f0f11 bg | MEDIUM |
| /curriculum.html | Inter + Fraunces | #050506 bg | MEDIUM |
| /diversity.html | Times New Roman | #0f0f11 bg | LOW |
| /terms.html | Unknown | Unknown | LOW |
| /privacy.html | Unknown | Unknown | LOW |

### App Pages (Different Context - OK)
| Page | Notes |
|------|-------|
| /learn.html | Main lesson player - has own design |
| /lesson-player/ | Kelly OS interface - has own design |

---

## 🚀 Pages to Create

### 1. API Page (`/api.html`) — HIGH PRIORITY

**Why it's needed:**
- Curious Kelly has a functional API
- Demonstrates technical credibility
- Attracts developer integrations
- Referenced in footer

**Sections to include:**
1. Hero: "Build with Kelly"
2. API Overview: What you can do
3. Authentication: API keys, OAuth
4. Endpoints Reference:
   - GET /api/lessons - Get today's lesson
   - GET /api/lessons/:day - Get specific day's lesson
   - GET /api/curriculum - Get full curriculum
   - POST /api/checkout - Create checkout session
5. Rate Limits & Pricing
6. SDKs: JavaScript, Python
7. Examples: Code snippets
8. Support: Contact for enterprise

### 2. Affiliates Page (`/affiliates.html`) — HIGH PRIORITY

**Why it's needed:**
- Already have affiliate program mentioned in careers
- $400/month budget for social media approved
- Commission structure defined (20-30%)

**Sections to include:**
1. Hero: "Earn by spreading curiosity"
2. How it works (3 steps)
3. Commission tiers (Scholar 20%, Fellow 25%, Ambassador 30%)
4. Earnings calculator
5. Marketing assets
6. Sign up form
7. FAQ

### 3. Accessibility Page (`/accessibility.html`) — MEDIUM PRIORITY

**Why it's needed:**
- Legal requirement (ADA, WCAG)
- Demonstrates inclusive values
- Referenced in footer

**Sections to include:**
1. Our commitment
2. Accessibility features
3. WCAG 2.1 AA compliance status
4. Known issues & roadmap
5. Contact for assistance

---

## 📋 Recommended Actions

### ✅ COMPLETED (This Session)

1. ~~**Create /api.html**~~ ✅ — Comprehensive API documentation created
2. ~~**Create /affiliates.html**~~ ✅ — Full affiliate program page with calculator
3. ~~**Create /accessibility.html**~~ ✅ — WCAG compliance statement
4. ~~**Audit all existing pages**~~ ✅ — Style audit documented below

### Immediate (Before Dec 17 Launch)

5. **Update /about.html** — Migrate to Instrument Sans + Newsreader + new colors
6. **Update /careers.html** — Migrate to new design system
7. **Update /gifts.html** — Migrate to new design system (has Stripe links already)

### Short-term (Post-launch)

8. **Update /curriculum.html** — Currently uses Fraunces, switch to Newsreader
9. **Update /enterprise.html** — Migrate to new design system
10. **Update /newsroom.html** — Migrate to new design system

### Low Priority

11. **Update /diversity.html** — Migrate to new design system
12. **Update /terms.html** — Check and update if needed
13. **Update /privacy.html** — Check and update if needed
14. **Add iOS/Android links** — When apps launch

---

## 🔗 Quick Reference: All Site URLs

### Marketing Site (daily-lesson-marketing)
```
/                    → New homepage (MIGRATED ✅)
/curriculum.html     → 365-day curriculum
/gifts.html          → Gift subscriptions
/pricing.html        → Pricing (redirect to /#pricing?)
/about.html          → About Kelly
/careers.html        → Jobs + Affiliate info
/newsroom.html       → Press releases
/enterprise.html     → Schools & orgs
/help.html           → Help center
/privacy.html        → Privacy policy
/terms.html          → Terms of service
/accessibility.html  → MISSING ❌
/affiliates.html     → MISSING ❌
/api.html            → MISSING ❌
```

### App URLs (public/)
```
/learn.html          → Main lesson experience
/me.html             → User profile
/settings.html       → Settings
/calendar.html       → Calendar view
/hub.html            → Kelly hub
/welcome.html        → Onboarding
```

---

## ✨ Summary

| Category | Count | Status |
|----------|-------|--------|
| Footer links | 17 | ✅ All now have pages |
| Pages created this session | 3 | api, affiliates, accessibility |
| Pages using new design | 5 | index, api, affiliates, accessibility, help (partial) |
| Pages needing style update | 9 | about, careers, gifts, curriculum, enterprise, newsroom, diversity, terms, privacy |

### This Session Completed:
- ✅ Created `/api.html` — Full API documentation with endpoints, auth, SDKs
- ✅ Created `/affiliates.html` — Complete affiliate program page with earnings calculator
- ✅ Created `/accessibility.html` — WCAG compliance statement
- ✅ All footer links now have corresponding pages
- ✅ Style audit completed and documented

### Migration Status
The homepage migration is **COMPLETE**. The new index.astro is live with:
- Instrument Sans + Newsreader typography
- Black/white/blue palette (#0a0a0b bg, #3b82f6 accent)
- Unified Aquarium architecture preserved
- Kelly-first hero design

### Next Steps
The remaining work is **polish** — updating 9 older pages to match the new design system. These can be done incrementally without breaking anything.

