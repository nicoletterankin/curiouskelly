# Supabase Independence Roadmap

**Created:** December 17, 2025  
**Status:** Active Planning  
**Goal:** Reduce Supabase dependency to auth + payments only, with clear alternatives

---

## Executive Summary

Curious Kelly uses Supabase for multiple purposes, but the **core lesson experience** can (and now does) work without it. This document outlines the path to full independence.

### Current State (December 2025)

| Component | Supabase Dependency | Independence Status |
|-----------|--------------------|--------------------|
| **Lesson Content** | ❌ Not needed | ✅ 365 static files in `/public/data/` |
| **Lesson API** | ⚡ Fallback only | ✅ Static-first, DB fallback |
| **Authentication** | ✅ Required | 🔄 Replaceable (Clerk, Auth.js) |
| **User Profiles** | ✅ Required | 🔄 Can migrate to Stripe metadata |
| **Payment Data** | ⚠️ Mirrors Stripe | 🔄 Stripe is source of truth |
| **Push Notifications** | ✅ Required | 🔄 Can use OneSignal, Firebase |
| **Analytics/Events** | ✅ Currently used | 🔄 Optional, can use Vercel Analytics |

---

## Phase 1: Content Independence ✅ DONE

**Completed December 17, 2025**

### What Changed
- API endpoints now read from static files FIRST
- Supabase is a fallback, not primary source
- Emergency fallback ensures lessons ALWAYS play

### Files Modified
- `api/lessons/[dayNumber].ts` - Static-first priority
- `api/day/[number].ts` - Static-first priority
- `api/lib/static-lessons.ts` - New static file loader

### Benefits
- Zero cold start for lesson content
- Works during Supabase outages
- Lower DB costs (no egress for content)
- Faster response times

---

## Phase 2: Auth Migration (Post-Launch)

### Current Auth Flow
```
User → Supabase GoTrue → OAuth (Google/Apple/GitHub/Facebook) → Session
```

### Recommended Alternative: **Clerk**

**Why Clerk:**
- Drop-in React/Next.js components
- Built-in OAuth for all providers
- User management dashboard
- Generous free tier (5K MAU free)
- Works great with Vercel

**Migration Steps:**
1. Sign up for Clerk (clerk.com)
2. Install `@clerk/nextjs`
3. Migrate existing users via API
4. Update auth.js to use Clerk hooks
5. Remove Supabase Auth dependencies

**Estimated Effort:** 2-3 days

### Alternative: Auth.js (NextAuth)

**Why Auth.js:**
- Open source, free forever
- Self-hosted, no vendor lock-in
- Works with any database
- Supports all OAuth providers

**Migration Steps:**
1. `pnpm add next-auth`
2. Configure providers in `auth.config.ts`
3. Set up session handling
4. Migrate user records
5. Update client-side auth calls

**Estimated Effort:** 3-5 days

---

## Phase 3: User Data Migration

### What's in Supabase Now

| Table | Purpose | Rows | Migration Strategy |
|-------|---------|------|-------------------|
| `users` | Profiles, preferences | ~1K | Stripe Customer metadata |
| `subscriptions` | Payment status | ~500 | Stripe is source of truth |
| `lesson_completions` | Progress tracking | ~10K | localStorage + optional sync |
| `lesson_purchases` | À la carte | ~100 | Stripe metadata |
| `gift_codes` | Gift redemption | ~50 | Stripe Products |
| `push_tokens` | Notifications | ~500 | OneSignal/Firebase |
| `user_events` | Analytics | ~50K | Vercel Analytics or drop |

### Stripe as User Database

Stripe already stores:
- Customer email
- Payment history
- Subscription status
- Customer metadata (can store preferences!)

**Use Stripe Customer Metadata for:**
```json
{
  "age": 35,
  "language": "en",
  "archetype": "The Explorer",
  "completed_lessons": [1, 2, 3, 17, 351],
  "streak": 7,
  "birthday": "03-15"
}
```

### localStorage for Progress

For non-paying users, store progress locally:
```javascript
localStorage.setItem('kelly_completed_days', JSON.stringify([1, 2, 3]));
localStorage.setItem('kelly_streak', '7');
localStorage.setItem('kelly_preferences', JSON.stringify({ age: 35 }));
```

Benefits:
- Works offline
- No auth required for basic use
- Syncs to Stripe when user subscribes

---

## Phase 4: Push Notification Migration

### Current Flow
```
Supabase → push_tokens table → Web Push API
```

### Alternative: OneSignal

**Why OneSignal:**
- Free tier (10K subscribers)
- Drop-in SDK
- Works on web, iOS, Android
- Better delivery rates

**Migration:**
1. Create OneSignal account
2. Import existing tokens
3. Update `/api/notifications/` to use OneSignal API
4. Remove Supabase push_tokens dependency

### Alternative: Firebase Cloud Messaging

**Why FCM:**
- Free, unlimited
- Part of Firebase (if using other services)
- Industry standard

---

## Phase 5: Analytics Independence

### Current: Supabase user_events

This is optional data. Options:

1. **Vercel Analytics** (Already available)
   - Built into Vercel
   - Privacy-friendly
   - No setup needed

2. **Plausible** 
   - Privacy-focused
   - Simple dashboard
   - $9/month

3. **PostHog**
   - Full product analytics
   - Session replay
   - Free tier available

4. **Drop entirely**
   - Focus on Stripe metrics
   - Revenue is the real metric

---

## Implementation Timeline

### Week 1 (December 17-24, 2025)
- [x] ~~Static-first lesson loading~~
- [x] ~~Graceful degradation for outages~~
- [ ] Monitor Supabase usage in production

### Week 2-3 (Post-Launch)
- [ ] Evaluate auth alternatives (Clerk vs Auth.js)
- [ ] Prototype auth migration
- [ ] Test Stripe metadata for user prefs

### Month 2 (January 2026)
- [ ] Complete auth migration
- [ ] Migrate user preferences to Stripe
- [ ] Update progress tracking to localStorage-first

### Month 3 (February 2026)
- [ ] Push notification migration (OneSignal)
- [ ] Analytics consolidation
- [ ] Supabase → minimal usage (maybe just events)

---

## Cost Comparison

### Current (Supabase Pro)
- $25/month base
- + egress costs
- + compute for RLS
- ~$50-100/month estimated

### After Migration
| Service | Cost |
|---------|------|
| Clerk (auth) | $0 (under 5K MAU) |
| Stripe (payments + user data) | Already paying |
| Vercel (hosting + analytics) | Already paying |
| OneSignal (push) | $0 (free tier) |
| **Total additional** | **$0** |

---

## What Supabase Remains Good For

Even after migration, you might keep Supabase for:

1. **Realtime features** (future chat, multiplayer)
2. **Edge Functions** (if needed)
3. **Quick prototyping** (admin tools)

But for the core learning experience: **you don't need it**.

---

## Emergency Contacts & Rollback

### If Supabase Goes Down

1. **Lessons continue** - Static files are primary
2. **Auth fails gracefully** - Guest mode enabled
3. **Payments work** - Stripe is independent

### If Auth Migration Fails

1. Keep Supabase Auth active as fallback
2. Run both systems in parallel
3. Gradual rollout via feature flags

---

## Quick Reference

### Files to Modify for Each Phase

**Phase 1 (Content):** ✅ Done
- `api/lessons/[dayNumber].ts`
- `api/day/[number].ts`
- `api/lib/static-lessons.ts`

**Phase 2 (Auth):**
- `public/js/auth.js`
- `public/js/lib/supabase.js`
- `api/subscription-status.ts`
- All OAuth callbacks

**Phase 3 (User Data):**
- `api/lesson-complete.ts`
- `api/gift-redeem.ts`
- `public/js/lesson-history.js`

**Phase 4 (Push):**
- `api/notifications/*.ts`
- `public/js/push-notifications.js`
- `api/cron/daily-push-notifications.ts`

---

## Success Metrics

| Metric | Goal |
|--------|------|
| Lesson load time | < 200ms (static) vs ~800ms (DB) |
| Outage resilience | 100% lesson availability |
| Monthly DB cost | Reduce by 80% |
| Auth complexity | Single provider, simpler code |

---

*Last updated: December 17, 2025*
