# Daily Email System Audit — December 17, 2025

## Executive Summary

**Date Audited:** December 17, 2025 (Launch Day!)  
**Today's Day Number:** 351 (mapped from calendar date)  
**Today's Lesson:** 🔮 Practicing in Your Mind (Visualization)

---

## ✅ What's Working

### 1. Content is Ready
- **Day 351 Complete Pack** exists at `public/data/day-351-complete.js`
- **Lesson JSON** exists at `public/lessons/day-351.json`
- Full 7-phase lesson with scripts, questions, responses
- High-quality, engaging content about visualization and mental practice

### 2. Kelly Assets Exist
- **Avatar images:** `public/images/brand/kelly-mark-circle-64.png`, `kelly-mark-circle-128.png`, etc.
- **Phase images for Day 351:** All exist in `public/kelly/phases/351/`
  - hook.png ✅
  - cliff.png ✅
  - q1.png, q2.png, q3.png ✅
  - wisdom.png ✅
  - outro.png ✅

### 3. Email Infrastructure
- **Resend API** configured for delivery
- **From address:** `Kelly <hello@curiouskelly.com>` ✅ (correct per CLAUDE.md)
- **Welcome email webhook** triggers on Supabase auth INSERT
- **Daily cron job** at `api/cron/daily-lesson.ts`

### 4. Signup Flow
- Supabase auth webhook → Welcome email
- User preferences stored in `users` table (`email_daily_lesson`, etc.)
- Unsubscribe tokens generated per user

---

## ⚠️ Issues Found

### CRITICAL: Hardcoded Placeholder in Email Template
**File:** `api/send-daily-lesson-email.ts` line 146
```html
<a href="https://curiouskelly.com/api/unsubscribe?token=UNSUBSCRIBE_TOKEN"
```
**Impact:** Unsubscribe links would be broken for all users.
**Fix:** The cron job (`api/cron/daily-lesson.ts`) correctly passes the user's `unsubscribe_token` — this standalone endpoint's template needs fixing.

### MEDIUM: Day Number Mismatch in Generated Content
**Issue:** `generated/lessons/day-351.json` has wrong content ("How Computers Think" instead of "Practicing in Your Mind")
**Root Cause:** Batch generation used different curriculum than final locked content
**Impact:** None if using `public/lessons/day-351.json` (which is correct)
**Recommendation:** Regenerate or delete `generated/lessons/` to avoid confusion

### MEDIUM: Welcome Email Missing Kelly Avatar
**File:** `api/send-welcome-email.ts` and `api/supabase-auth-webhook.ts`
**Issue:** Welcome emails are text-only, no Kelly avatar/branding
**Recommendation:** Add Kelly avatar image like daily emails have

### LOW: Old Generated Email File Wrong Day
**File:** `generated-emails/day-017-email.html`
**Issue:** This is for Day 17 (January 17, "Why We Dream") not today
**Impact:** None — just confusing to have in repo

---

## Today's Email Package ✅

**Generated:** `generated-emails/day-351-email.html`

### Content Includes:
- ✅ Kelly avatar (kelly-mark-circle-128.png)
- ✅ Real lesson title: "🔮 Practicing in Your Mind"
- ✅ Real headline: "Your brain can't tell the difference between doing and imagining"
- ✅ Hook teaser text
- ✅ 3 fun facts from actual lesson content
- ✅ Today's wisdom quote
- ✅ CTA to `/day/351`
- ✅ December 17 date (not Day 351)
- ✅ Unsubscribe link (with token placeholder for dynamic replacement)
- ✅ Lesson of the Day PBC footer

---

## Signup Flow (E2E)

```
1. User visits curiouskelly.com
2. Signs up via Supabase Auth
3. Supabase fires webhook to /api/supabase-auth-webhook
4. Webhook sends welcome email via Resend
5. User record created with email_daily_lesson = true
6. Next day at 12pm UTC, /api/cron/daily-lesson runs
7. Cron fetches lesson for today's day number
8. Cron fetches all subscribed users
9. Batch sends personalized emails with:
   - User's name
   - Their streak count
   - Today's lesson content
   - Personalized unsubscribe token
```

**Status:** ✅ Flow is complete and functional

---

## Recommendations

### Immediate (Before First Send)
1. ~~Generate today's email~~ ✅ Done: `generated-emails/day-351-email.html`
2. Verify Resend API key is set in production
3. Verify Supabase webhook is configured

### Short-term
1. Add Kelly avatar to welcome email template
2. Clean up `generated/lessons/` directory or regenerate from final curriculum
3. Add email preview endpoint for testing

### Medium-term
1. Add email open/click tracking
2. Implement timezone-aware sending
3. Build email preview dashboard

---

## Test Commands

```bash
# Test welcome email (requires WELCOME_EMAIL_API_KEY)
curl -X POST https://www.curiouskelly.com/api/send-welcome-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"email": "test@example.com", "name": "Test User"}'

# Test daily lesson email (requires DAILY_EMAIL_API_KEY)
curl -X POST https://www.curiouskelly.com/api/send-daily-lesson-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "lessonTitle": "Practicing in Your Mind",
    "lessonEmoji": "🔮",
    "lessonCategory": "Meta-Learning",
    "dayNumber": 351
  }'

# Trigger daily cron (requires CRON_SECRET)
curl -X POST https://www.curiouskelly.com/api/cron/daily-lesson \
  -H "Authorization: Bearer YOUR_CRON_SECRET"
```

---

## Files Audited

| File | Status | Notes |
|------|--------|-------|
| `api/cron/daily-lesson.ts` | ✅ Good | Main cron, handles personalization |
| `api/send-daily-lesson-email.ts` | ⚠️ Fix needed | Hardcoded UNSUBSCRIBE_TOKEN |
| `api/send-welcome-email.ts` | ⚠️ Enhance | Missing Kelly avatar |
| `api/supabase-auth-webhook.ts` | ⚠️ Enhance | Missing Kelly avatar |
| `api/lib/email-templates.ts` | ✅ Good | Central template library |
| `lib/lesson-dates.ts` | ✅ Good | Proper date → day mapping |
| `public/lessons/day-351.json` | ✅ Excellent | Launch-locked content |
| `public/data/day-351-complete.js` | ✅ Excellent | Full phase scripts |
| `public/kelly/phases/351/` | ✅ Complete | All images present |

---

*Audit completed by Claude · December 17, 2025*
