# 🗂️ Sprint Artifacts Index
## Everything You Need for Dec 17 Launch

**Created:** December 7, 2025  
**Purpose:** Single source of truth for all sprint files and their status

---

## 📊 EXECUTIVE DASHBOARD

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Days to Launch | 10 | - | ⏳ |
| Core Lessons | 365 | 365 | ✅ |
| Lesson Atoms | 21,855 | 21,900 | ✅ 99.8% |
| Landing Page | 90% | 100% | 🟡 |
| Lesson Player V2 | 60% | 100% | 🟡 |
| Stripe Integration | 0% | 100% | 🔴 |
| Email System | 0% | 100% | 🔴 |
| Social Media | 0% | 100% | 🔴 |

---

## 📁 KEY FILES BY PRIORITY

### 🔴 P0: CRITICAL PATH

#### Landing Page
| File | Status | Action |
|------|--------|--------|
| `legacy_marketing_site.html` | ✅ Exists | Add Stripe, rename to `index.html` |
| `curiouskelly-marketing-site/` | ❌ Empty | Use legacy file instead |

#### Lesson Player V2
| File | Status | Action |
|------|--------|--------|
| `curious-kellly/lesson-player-v2/index.html` | ✅ Good structure | Polish UI |
| `curious-kellly/lesson-player-v2/js/app.js` | 🟡 70% done | Wire to Supabase atoms |
| `curious-kellly/lesson-player-v2/css/styles.css` | ⚠️ Missing | Create |
| `curious-kellly/lesson-player-v2/css/ui-kit.css` | ⚠️ Missing | Create |

#### Backend / Stripe
| File | Status | Action |
|------|--------|--------|
| `curious-kellly/backend/src/api/checkout.js` | ⚠️ Missing | Create |
| `curious-kellly/backend/src/api/webhooks/stripe.js` | ⚠️ Missing | Create |
| `curious-kellly/backend/Procfile` | ✅ Exists | Verify |
| `curious-kellly/backend/railway.json` | ✅ Exists | Verify config |

#### Email
| File | Status | Action |
|------|--------|--------|
| `curious-kellly/email-templates/welcome.html` | ⚠️ Missing | Create |
| `curious-kellly/email-templates/gift-purchased.html` | ⚠️ Missing | Create |
| `curious-kellly/email-templates/gift-received.html` | ⚠️ Missing | Create |

---

### 🟠 P1: IMPORTANT

#### Social Media
| File | Status | Action |
|------|--------|--------|
| `docs/social-media/SOCIAL_MEDIA_STRATEGY.md` | ✅ Complete | Reference |
| `docs/social-media/SOCIAL_MEDIA_BRAND_GUIDELINES.md` | ✅ Exists | Reference |
| `docs/social-media/SOCIAL_MEDIA_LAUNCH_CHECKLIST.md` | ✅ Exists | Follow |
| `tools/social-media-automation/` | ✅ Scripts exist | Use |

#### Kelly Assets
| File | Status | Action |
|------|--------|--------|
| `public/images/kelly/` | 🟡 Partial | Generate missing |
| `Ref/Best Character Reference/` | ✅ Complete | Reference for generation |

---

### 🟡 P2: HELPFUL

#### Database / Content
| File | Status | Action |
|------|--------|--------|
| `docs/backend/SUPABASE_SCHEMA.md` | ✅ Exists | Reference |
| `content/curriculum_365.json` | ✅ 365 lessons | Validate |

#### Documentation
| File | Status | Action |
|------|--------|--------|
| `BURNDOWN_SPRINT_DEC7_17.md` | ✅ New | Follow daily |
| `CHRISTMAS_LAUNCH_PLAN.md` | ✅ Exists | Reference |
| `CHRISTMAS_PRODUCT_BACKLOG.md` | ✅ Exists | Reference |
| `DEC_17_LAUNCH_PLAN.md` | ✅ Exists | Reference |
| `MASTER_LAUNCH_CHECKLIST.md` | ✅ Exists | Follow |

---

## 🗄️ DATABASE STATUS (Supabase)

```sql
-- Quick health check query
SELECT 
  (SELECT COUNT(*) FROM core_lessons) as lessons,
  (SELECT COUNT(*) FROM lesson_atoms) as atoms,
  (SELECT COUNT(*) FROM lesson_age_hooks) as age_hooks,
  (SELECT COUNT(*) FROM archetype_dialog_templates) as dialogs;

-- Result: 365, 21855, 2196, 72 ✅
```

### Tables Ready for Launch
| Table | Rows | Purpose |
|-------|------|---------|
| `core_lessons` | 365 | Main lesson metadata |
| `lesson_atoms` | 21,855 | Phase-specific content |
| `lesson_age_hooks` | 2,196 | Age-specific hooks |
| `archetype_dialog_templates` | 72 | Kelly voice lines |
| `users` | 3 | User accounts |
| `lessons` | 365 | Alternate lesson format |

### Tables Need Attention
| Table | Rows | Issue |
|-------|------|-------|
| `lesson_shards` | 0 | Empty - OK for MVP |
| `kelly_video_assets` | 1,213 | None validated/published |
| `lesson_assets` | 0 | No audio/video assets yet |

---

## 🔑 ENVIRONMENT VARIABLES NEEDED

```bash
# Supabase (Already configured)
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...

# Stripe (NEEDS SETUP)
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email (NEEDS SETUP)
SENDGRID_API_KEY=SG...
# OR
SMTP_HOST=smtp.gmail.com
SMTP_USER=hello@curiouskelly.com
SMTP_PASS=...

# Deployment
VERCEL_TOKEN=...
DOMAIN=curiouskelly.com
```

---

## 🛠️ COMMANDS REFERENCE

### Local Development
```bash
# Start lesson player locally
cd curious-kellly/lesson-player-v2
python -m http.server 8000

# Start backend locally
cd curious-kellly/backend
npm run dev

# Run content validation
python tools/validate_content.py
```

### Deployment
```bash
# Deploy to Vercel
vercel --prod

# Deploy backend to Railway
railway up

# Run migrations
npm run migrate
```

### Database Queries
```sql
-- Get Day 1 lesson with atoms
SELECT cl.*, la.phase, la.archetype, la.content 
FROM core_lessons cl 
JOIN lesson_atoms la ON la.core_lesson_id = cl.id 
WHERE cl.day_number = 1;

-- Get all age hooks for Day 1
SELECT * FROM lesson_age_hooks WHERE day_number = 1;

-- Get Kelly dialog templates
SELECT * FROM archetype_dialog_templates WHERE archetype = 'The Explorer';
```

---

## 📅 DAILY CHECKLIST FORMAT

Use this format for daily standups:

```markdown
## Day X Standup (Dec X, 2025)

### Yesterday
- [ ] What I completed

### Today  
- [ ] What I will work on

### Blockers
- [ ] Any blockers

### Metrics
- Landing page: X% complete
- Player: X% complete
- Stripe: X% complete
```

---

## 🔗 QUICK LINKS

### External Services
- [Supabase Dashboard](https://app.supabase.com/project/tvjalxxsyryjphkforjv)
- [Stripe Dashboard](https://dashboard.stripe.com)
- [Vercel Dashboard](https://vercel.com)
- [Railway Dashboard](https://railway.app)

### Documentation
- [Stripe Checkout Docs](https://stripe.com/docs/checkout/quickstart)
- [SendGrid Docs](https://docs.sendgrid.com/for-developers/sending-email)
- [Supabase Auth Docs](https://supabase.com/docs/guides/auth)

---

## ✅ DEFINITION OF DONE

A feature is "done" when:
1. Code is written and tested locally
2. No console errors
3. Works on mobile (responsive)
4. Deployed to production
5. Verified by clicking through manually

---

**Last Updated:** December 7, 2025  
**Next Update:** After Day 1 tasks complete

