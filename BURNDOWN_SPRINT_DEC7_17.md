# 🔥 BURNDOWN SPRINT: December 7-17, 2025
## 10 Days to Curious Kelly Launch

**Created:** December 7, 2025  
**Launch Date:** December 17, 2025  
**Days Remaining:** 10

---

## 📊 CURRENT STATE ASSESSMENT

### Database (Supabase) ✅ GOOD
| Table | Count | Status |
|-------|-------|--------|
| `core_lessons` | 365 | ✅ Complete |
| `lesson_atoms` | 21,855 | ✅ Complete (365 × ~60 atoms each) |
| `lesson_shards` | 0 | ⚠️ Not needed for MVP |
| `users` | 3 | ✅ Test users |
| `lesson_age_hooks` | 2,196 | ✅ Complete (365 × 6 age buckets) |
| `archetype_dialog_templates` | 72 | ✅ Kelly voice lines |
| `kelly_video_assets` | 1,213 | ⚠️ None validated/published |

### Frontend Components
| Component | Status | Location |
|-----------|--------|----------|
| Landing Page | 🟡 Exists, needs finishing | `curiouskelly-landing-page.html` |
| Lesson Player V2 | 🔴 Skeleton only | `curious-kellly/lesson-player-v2/` |
| Calendar Page | 🟡 Exists | `calendar-page.html` |
| Kelly Assets | 🟡 Some exist | `public/images/kelly/` |

### Backend
| Service | Status |
|---------|--------|
| Supabase Auth | ✅ Configured |
| Stripe | 🔴 Not configured |
| Email (SendGrid) | 🔴 Not configured |
| hello@curiouskelly.com | 🔴 Not setup |

### Social Media
| Platform | Status |
|----------|--------|
| Twitter @CuriousKelly | 🔴 Not created |
| Instagram @CuriousKellyAI | 🔴 Not created |
| YouTube @CuriousKelly | 🔴 Not created |
| TikTok @CuriousKellyAI | 🔴 Not created |
| Discord | 🔴 Not created |
| LinkedIn | 🔴 Not created |

---

## 🎯 10-DAY EXECUTION PLAN

### DAY 1 (Dec 7 - TODAY) - Foundation
**Theme: "Get the basics working"**

**Morning (4 hours):**
- [ ] Audit and finish landing page HTML/CSS
- [ ] Verify Supabase connection from frontend
- [ ] Test that Day 1 lesson loads from database

**Afternoon (4 hours):**
- [ ] Wire Lesson Player V2 to fetch atoms from Supabase
- [ ] Implement basic phase progression (Welcome → Q1 → Q2 → Q3 → Wisdom)
- [ ] Test age adaptation with archetype selector

**Evening:**
- [ ] Document any blockers
- [ ] Update this burndown

---

### DAY 2 (Dec 8) - Payments
**Theme: "Money can flow"**

- [ ] Create Stripe account (if not exists)
- [ ] Create 3 products:
  - Personal Annual: $199/year
  - Family Annual: $299/year
  - Gift Plan: $199 one-time
- [ ] Implement checkout.js endpoint
- [ ] Add Stripe checkout button to landing page
- [ ] Test sandbox purchase flow
- [ ] Configure Stripe webhook for payment confirmation

---

### DAY 3 (Dec 9) - Email System
**Theme: "We can communicate"**

- [ ] Setup hello@curiouskelly.com email
- [ ] Configure SendGrid or Gmail SMTP
- [ ] Create email templates:
  - Welcome email
  - Gift purchase confirmation
  - Gift recipient notification
  - Password reset
- [ ] Test email deliverability
- [ ] Add email to Supabase user on signup

---

### DAY 4 (Dec 10) - Landing Page Polish
**Theme: "Beautiful first impression"**

- [ ] Update hero section with correct Kelly image
- [ ] Add interactive calendar preview section
- [ ] Add pricing section with Stripe buttons
- [ ] Add testimonials/social proof section
- [ ] Mobile responsive testing
- [ ] Performance optimization (image compression, lazy load)
- [ ] Deploy to curiouskelly.com

---

### DAY 5 (Dec 11) - Lesson Player V2 Complete
**Theme: "The product works"**

- [ ] Complete phase state machine
- [ ] Add Kelly dialog templates integration
- [ ] Add progress tracking (mark lesson complete)
- [ ] Add calendar panel navigation
- [ ] Add age slider with live atom switching
- [ ] Test full lesson flow: Day 1 "The Sun"
- [ ] Add loading states and error handling

---

### DAY 6 (Dec 12) - Social Media Blitz
**Theme: "Kelly exists online"**

**Morning:**
- [ ] Create logo assets in Canva (profile pics, headers)
- [ ] Create @CuriousKelly on Twitter/X
- [ ] Create @CuriousKellyAI on Instagram

**Afternoon:**
- [ ] Create @CuriousKelly on YouTube
- [ ] Create @CuriousKellyAI on TikTok
- [ ] Create Curious Kelly Community on Discord
- [ ] Create Lesson of the Day PBC on LinkedIn

**Evening:**
- [ ] Post "Coming December 17!" teaser on all platforms
- [ ] Set up profile bios with links to curiouskelly.com

---

### DAY 7 (Dec 13) - Content Validation
**Theme: "Quality check"**

- [ ] Run automated content validation on Days 1-30
- [ ] Check for slop patterns in lesson atoms
- [ ] Verify all age hooks are appropriate
- [ ] Test 3 lessons end-to-end across all age buckets
- [ ] Fix any critical content issues
- [ ] Verify archetype dialogs match context

---

### DAY 8 (Dec 14) - Integration Testing
**Theme: "Everything connects"**

- [ ] Test full user journey:
  1. Land on curiouskelly.com
  2. Click "Buy Gift"
  3. Complete Stripe checkout
  4. Receive confirmation email
  5. Gift recipient gets email
  6. Redeem gift code
  7. Start Day 1 lesson
- [ ] Test on multiple devices (desktop, tablet, mobile)
- [ ] Test multiple browsers (Chrome, Safari, Firefox)
- [ ] Fix all critical bugs

---

### DAY 9 (Dec 15) - Pre-Launch Prep
**Theme: "Ready for anything"**

- [ ] Final landing page review
- [ ] Final lesson player review
- [ ] Set up error monitoring (Sentry)
- [ ] Set up analytics tracking
- [ ] Prepare launch announcement content
- [ ] Schedule social media posts for Dec 17
- [ ] Email waitlist: "2 days until launch!"
- [ ] Verify all support documentation

---

### DAY 10 (Dec 16) - Launch Eve
**Theme: "Final checks"**

- [ ] Complete smoke test of all features
- [ ] Verify Stripe is in production mode
- [ ] Verify email delivery works
- [ ] Double-check all links
- [ ] Prepare "We're Live!" posts
- [ ] Brief anyone helping with support
- [ ] Get sleep 😴

---

### LAUNCH DAY (Dec 17) 🚀
**Theme: "Kelly meets the world"**

**6 AM:**
- [ ] Deploy any final fixes
- [ ] Verify all systems operational
- [ ] Post "We're LIVE!" on all social media

**9 AM:**
- [ ] Send launch email to waitlist
- [ ] Monitor first purchases
- [ ] Monitor error logs

**12 PM:**
- [ ] Post "Meet Kelly" content

**7 PM:**
- [ ] Post "Give the Gift" content
- [ ] Review day's metrics

**All Day:**
- [ ] Monitor and respond to support requests
- [ ] Engage with social media comments
- [ ] Celebrate wins! 🎉

---

## 📋 CRITICAL PATH (Must Complete)

These items BLOCK launch if not done:

1. **Landing page deployed at curiouskelly.com** ← No website = no sales
2. **Stripe checkout working** ← No payments = no revenue
3. **Lesson Player V2 loads Day 1** ← No product = nothing to sell
4. **Email system sends** ← No confirmation = bad experience
5. **At least 1 social account** ← No announcements = no awareness

---

## ⚠️ RISK MITIGATION

### If Stripe takes too long:
- **Backup:** Use Stripe Payment Links (no-code)
- **Backup:** Launch with "Coming Soon" and email collection

### If email system fails:
- **Backup:** Use manual Gmail for first 100 customers
- **Backup:** Display success message with "check your email" (even if delayed)

### If Lesson Player has bugs:
- **Backup:** Use static Day 1 content page
- **Backup:** Show calendar + "Full player coming Jan 1"

### If domain isn't ready:
- **Backup:** Use Vercel subdomain (curious-kelly.vercel.app)
- **Backup:** Redirect from existing domain

---

## 📊 SUCCESS METRICS (Dec 17-24)

| Metric | Target | Minimum |
|--------|--------|---------|
| Gift purchases | 100 | 25 |
| Landing page visitors | 5,000 | 1,000 |
| Email signups | 500 | 100 |
| Social followers (total) | 500 | 100 |
| Critical bugs | 0 | &lt;5 |

---

## 🗂️ FILES TO CREATE/MODIFY

### New Files Needed:
```
curious-kellly/
├── backend/
│   └── src/
│       └── api/
│           ├── checkout.js          # Stripe integration
│           └── webhooks/
│               └── stripe.js        # Payment webhook
├── lesson-player-v2/
│   └── js/
│       ├── supabase-client.js       # Supabase connection
│       ├── lesson-loader.js         # Fetch atoms
│       ├── phase-machine.js         # State management
│       └── age-adapter.js           # Age variant switching
└── email-templates/
    ├── welcome.html
    ├── gift-purchased.html
    └── gift-received.html
```

### Files to Update:
- `curiouskelly-landing-page.html` - Add Stripe, polish design
- `curious-kellly/lesson-player-v2/index.html` - Complete UI
- `curious-kellly/lesson-player-v2/js/app.js` - Wire to Supabase
- `public/images/kelly/` - Add missing hero images

---

## 🔑 CREDENTIALS NEEDED

Collect these TODAY:

- [ ] **Stripe Secret Key** (sk_live_...)
- [ ] **Stripe Publishable Key** (pk_live_...)
- [ ] **Domain registrar login** (for curiouskelly.com)
- [ ] **SendGrid API Key** (or Gmail app password)
- [ ] **Vercel account** (for deployment)
- [ ] **Twitter login** (for @CuriousKelly)
- [ ] **Instagram login** (for @CuriousKellyAI)

---

## 📞 DAILY CHECK-IN QUESTIONS

Ask yourself at end of each day:

1. Did I complete today's tasks?
2. What blocked me?
3. Is launch still on track for Dec 17?
4. What's my #1 priority tomorrow?
5. Do I need help with anything?

---

## 💪 MOTIVATION

**You have:**
- ✅ 365 lessons ready
- ✅ 21,855 content atoms
- ✅ Database architecture done
- ✅ Landing page started
- ✅ Clear product vision

**You need:**
- 10 focused days
- Stripe integration
- Email system
- Polish and deploy

**Remember:** A shipped product beats a perfect product. Launch something on Dec 17, then iterate.

---

**LET'S GO! 🚀**

---

*Last Updated: December 7, 2025*
*Document: BURNDOWN_SPRINT_DEC7_17.md*


