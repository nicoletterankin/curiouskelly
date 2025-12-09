# 🤖 AGENT PROMPT: Complete Earn to Learn Implementation

**Copy this entire prompt to a new AI session to continue the work.**

---

## YOUR MISSION

You are completing the "Earn to Learn" feature for Curious Kelly - an educational platform where every learner becomes an affiliate. The database is ready. The strategy is documented. You need to implement the frontend UI and API endpoints.

**Time budget:** ~20-28 hours of focused work  
**Launch target:** December 17, 2025  
**Priority:** This is launch-critical (P0)

---

## CONTEXT: WHAT IS CURIOUS KELLY?

Curious Kelly is a daily lesson platform (365 lessons, ages 2-102) with an AI teacher avatar. The "Earn to Learn" system means:

1. Every learner gets a referral code at signup (e.g., `kelly.me/sarah`)
2. When they share and someone subscribes, they earn commission
3. Commission rates increase with learning progress (10% → 35%)
4. Attribution is LIFETIME (no cookie expiration)
5. This addresses ageism, loneliness, and creates income for learners

---

## CURRENT STATE (What's Already Done)

### ✅ Database (LIVE in Supabase)

The following tables exist and are populated:

```sql
-- Users table extended with:
- referral_code (unique, auto-generated)
- commission_tier ('new_learner' to 'legendary_learner')
- commission_rate (0.10 to 0.35)
- pending_earnings, available_earnings, lifetime_earnings
- referred_by_user_id, referred_at
- payout_method, payout_details

-- New tables created:
- commission_tiers (6 tiers with rates and perks)
- referral_clicks (lifetime attribution tracking)
- commission_transactions (every earning event)
- payouts (withdrawal requests)
```

### ✅ Documentation (READ THESE FIRST)

1. **`docs/strategy/EARN_TO_LEARN_COMPLETE_VISION.md`** - The philosophy and social impact
2. **`docs/implementation/EARN_TO_LEARN_IMPLEMENTATION_PLAN.md`** - Detailed technical spec
3. **`CURIOUS_KELLY_UNIFIED_VISION.md`** - Master context document
4. **`docs/backend/migrations/20251207_earn_to_learn.sql`** - Database schema

### ✅ Existing Users Have Referral Codes

```
kelly_97d0 → hello@curiouskelly.com (15% rate, Active Learner)
nicolette_4a9d → nicoletterankin@gmail.com (10% rate, New Learner)
```

---

## WHAT YOU NEED TO IMPLEMENT

### Phase 1: Referral Link Capture (3 hours)

**Goal:** When someone visits `curiouskelly.com/?ref=sarah`, store that forever.

**Files to create/modify:**
- Landing page entry point (detect `?ref=` parameter)
- `api/referral/track-click.ts` - Record click in database

**Logic:**
```javascript
// On any page load:
const refCode = new URLSearchParams(location.search).get('ref');
if (refCode) {
  localStorage.setItem('kelly_referrer', refCode);
  // POST to /api/referral/track-click
}
```

### Phase 2: Signup Attribution (3 hours)

**Goal:** When a referred user signs up, link them to their referrer.

**Files to modify:**
- `curious-kellly/lesson-player-v2/js/app.js` - Add referral to signup
- `api/referral/link.ts` - Update user with referred_by_user_id

**Logic:**
```javascript
// During signup:
const referrerCode = localStorage.getItem('kelly_referrer');
// Include in auth.signUp() metadata
// Call /api/referral/link to update database
```

### Phase 3: Stripe Webhook (4 hours)

**Goal:** When payment happens, record commission for referrer.

**Files to modify:**
- `api/webhooks/stripe.ts` - Add commission recording

**Events to handle:**
- `checkout.session.completed` → Initial subscription
- `invoice.paid` → Renewals (also earn commission!)
- `charge.refunded` → Clawback commission

**Use this function (already exists in DB):**
```sql
SELECT record_commission(referrer_id, referred_id, 'initial_subscription', 99.99, 'pi_xxx');
```

### Phase 4: Share & Earn UI (4 hours)

**Goal:** Add earnings display and share buttons to the drawer menu.

**Files to modify:**
- `curious-kellly/lesson-player-v2/index.html` - Add Share & Earn section
- `curious-kellly/lesson-player-v2/css/styles.css` - Style the section
- `curious-kellly/lesson-player-v2/js/app.js` - Load and display earnings

**UI elements needed:**
```
┌─────────────────────────────┐
│ 💰 Share & Earn             │
├─────────────────────────────┤
│ Pending: $0.00  Available: $0.00 │
│ [Active Learner] 15% commission  │
│ ┌─────────────────────────┐ │
│ │ kelly.me/sarah      [📋]│ │
│ └─────────────────────────┘ │
│ [𝕏] [f] [💬] [✉️]           │
│ View Full Earnings →        │
└─────────────────────────────┘
```

### Phase 5: Lesson Complete Share (2 hours)

**Goal:** After finishing a lesson, prompt user to share.

**Files to modify:**
- `curious-kellly/lesson-player-v2/js/app.js` - Modify `advancePhase()` method

**When phase === 'complete':**
```
┌─────────────────────────────┐
│ ✨ Lesson Complete!         │
│ "How Kindness Spreads"      │
│                             │
│ Know someone who'd love this?│
│ [Share on 𝕏] [Share WhatsApp]│
│ You'll earn 15% if they sub!│
│                             │
│ [Continue to Dashboard →]   │
└─────────────────────────────┘
```

### Phase 6: Earnings Dashboard Modal (6 hours)

**Goal:** Full earnings view with history and payout request.

**Files to modify:**
- `curious-kellly/lesson-player-v2/index.html` - Add modal
- `curious-kellly/lesson-player-v2/js/app.js` - Load transaction history

**Shows:**
- Pending / Available / Lifetime earnings
- Current tier and progress to next
- Total referrals and active subscribers
- Recent transactions list
- Payout request button (enabled when available > $50)

---

## KEY CODE LOCATIONS

```
curious-kellly/lesson-player-v2/
├── index.html          # Main UI - add Share section to drawer
├── css/styles.css      # Styles - add .share-earn-section etc.
├── js/app.js           # KellyOS class - add earnings methods

api/
├── referral/
│   ├── track-click.ts  # NEW: Record referral clicks
│   └── link.ts         # NEW: Link user to referrer
├── webhooks/
│   └── stripe.ts       # MODIFY: Add commission logic
```

---

## DATABASE QUERIES YOU'LL NEED

### Get user's earnings data:
```sql
SELECT 
  referral_code,
  commission_tier,
  commission_rate,
  pending_earnings,
  available_earnings,
  lifetime_earnings,
  total_referrals
FROM users
WHERE id = 'user-uuid';
```

### Record a commission:
```sql
INSERT INTO commission_transactions (
  referrer_id, referred_user_id, transaction_type,
  gross_amount, commission_rate, commission_amount,
  stripe_payment_intent_id, status
) VALUES (
  'referrer-uuid', 'referred-uuid', 'initial_subscription',
  99.99, 0.15, 14.99,
  'pi_xxx', 'pending'
);

-- Also update user's earnings:
UPDATE users SET
  pending_earnings = pending_earnings + 14.99,
  lifetime_earnings = lifetime_earnings + 14.99,
  total_referrals = total_referrals + 1
WHERE id = 'referrer-uuid';
```

### Get commission tiers:
```sql
SELECT tier_name, display_name, min_lessons_completed, base_commission_rate, perks
FROM commission_tiers
ORDER BY sort_order;
```

---

## COMMISSION TIERS (Reference)

| Tier | Lessons | Rate | Display Name |
|------|---------|------|--------------|
| new_learner | 0-6 | 10% | New Learner |
| active_learner | 7-29 | 15% | Active Learner |
| committed_learner | 30-99 | 20% | Committed Learner |
| dedicated_learner | 100-364 | 25% | Dedicated Learner |
| complete_learner | 365+ | 30% | Complete Learner |
| legendary_learner | 1000+ | 35% | Legendary Learner |

---

## RULES TO FOLLOW

1. **Read `CLAUDE.md`** - It has operating rules for this repo
2. **Lifetime cookies** - Attribution NEVER expires (set `attribution_expires_at = NULL`)
3. **No self-referral** - Check `referrer_id !== user_id`
4. **Privacy first** - Hash IPs, don't store PII unnecessarily
5. **Test thoroughly** - Check edge cases (refunds, renewals, tier upgrades)
6. **Mobile responsive** - The drawer must work on mobile

---

## TESTING CHECKLIST

Before considering done:

- [ ] Visit `?ref=testcode` → localStorage has `kelly_referrer`
- [ ] `referral_clicks` table has new row
- [ ] Sign up with referral → user has `referred_by_user_id`
- [ ] Stripe checkout → `commission_transactions` has row
- [ ] Referrer's `pending_earnings` increased
- [ ] UI shows correct referral code
- [ ] Copy button works
- [ ] Social share buttons open correct URLs
- [ ] Lesson complete shows share prompt
- [ ] Earnings dashboard loads transaction history
- [ ] Mobile view works

---

## QUICK START COMMANDS

```bash
# Start the lesson player locally
cd curious-kellly/lesson-player-v2
npx http-server . -p 3001 -c-1 --cors

# Check Supabase connection
# Use the MCP Supabase tools to query tables

# Deploy API changes
cd api
npx vercel --prod
```

---

## SUCCESS CRITERIA

The feature is DONE when:

1. ✅ Anyone can click a referral link and be tracked forever
2. ✅ Signups are attributed to referrers
3. ✅ Payments trigger commission recording
4. ✅ Users can see their referral code and earnings in the UI
5. ✅ Users can share via social buttons
6. ✅ Lesson completion prompts sharing
7. ✅ Earnings dashboard shows full history
8. ✅ All tests pass

---

## ESTIMATED TIME

| Phase | Hours |
|-------|-------|
| Phase 1: Link Capture | 3 |
| Phase 2: Signup Attribution | 3 |
| Phase 3: Stripe Webhook | 4 |
| Phase 4: Share UI | 4 |
| Phase 5: Lesson Complete | 2 |
| Phase 6: Earnings Dashboard | 6 |
| Testing & Polish | 4 |
| **Total** | **~26 hours** |

---

## GO BUILD IT! 🚀

Start with Phase 1 (Link Capture) - it's the foundation.

Then Phase 4 (Share UI) - makes progress visible to users.

Then Phase 3 (Stripe Webhook) - makes money flow.

The rest follows naturally.

**Good luck, agent. Make learners into earners.** ✨

---

*This prompt created: December 7, 2025*  
*For: Curious Kelly Earn to Learn Implementation*  
*Contact: hello@curiouskelly.com*


