# 📋 Curious Kelly — Comprehensive Test Plan & Architecture Review

## Learner Journeys, Edge Cases, Content Readiness & Scale Planning

**Version:** 1.0  
**Date:** November 28, 2025  
**Prepared By:** Product Management & Architecture  
**Status:** Pre-Launch Planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Goals Alignment](#2-product-goals-alignment)
3. [Content Readiness Audit](#3-content-readiness-audit)
4. [Learner Journey Maps](#4-learner-journey-maps)
5. [Test Scenarios & Edge Cases](#5-test-scenarios--edge-cases)
6. [Avatar System Testing](#6-avatar-system-testing)
7. [Scale Architecture for 1B Learners](#7-scale-architecture-for-1b-learners)
8. [Launch Checklist](#8-launch-checklist)
9. [Risk Assessment](#9-risk-assessment)

**Related Document:** See `AVATAR_SYSTEM_ARCHITECTURE.md` for complete avatar technical specs

---

## 1. Executive Summary

### Current State

| Metric                 | Value                       | Status            |
| ---------------------- | --------------------------- | ----------------- |
| Total Lessons          | 365                         | ✅ Complete       |
| Lessons with DNA       | 42 (11.5%)                  | ⚠️ Critical Gap   |
| Lessons without DNA    | 323 (88.5%)                 | 🔴 Needs Work     |
| Supabase lesson_atoms  | 21,915                      | ✅ Populated      |
| Supabase lesson_shards | 38,700                      | ✅ Variants Ready |
| Age Variants           | 6 groups                    | ✅ Defined        |
| Languages              | 3 (EN/ES/FR)                | ✅ Defined        |
| Tones                  | 3 (curious/playful/serious) | ✅ Defined        |
| Difficulty Levels      | 2 (standard/challenge)      | ✅ NEW            |

### Scale Target

- **Goal:** 1 billion learners per year
- **Daily Active:** ~2.74 million/day
- **Peak Concurrent:** ~500,000 (assuming 4-hour peak window)
- **Lessons Served/Day:** ~3 million (assuming 1.1 lessons per active user)

---

## 2. Product Goals Alignment

### Core Product Goals

| #   | Goal                          | Implementation                               | Test Coverage                         |
| --- | ----------------------------- | -------------------------------------------- | ------------------------------------- |
| 1   | Daily habit formation         | Streak system, Today Hub, push notifications | ✅ Built, needs notification testing  |
| 2   | Jump into class right now     | One-tap "Start Today's Lesson" CTA           | ✅ Built                              |
| 3   | Find birthday lesson          | Birthday feature in Hub                      | ✅ Built, needs edge case testing     |
| 4   | Same topic every day globally | 365_day_calendar.json as source of truth     | ✅ Built                              |
| 5   | Multi-generational (2-102)    | 6 age variants per lesson                    | ✅ Built, content partially complete  |
| 6   | Multi-lingual                 | 3 languages                                  | ✅ Built, translation coverage varies |
| 7   | Challenge mode                | 2 vs 3 choices                               | ✅ Built                              |
| 8   | Premium subscription          | Stripe integration                           | ✅ Built, needs E2E testing           |

### Golden Invariants (MUST NEVER BREAK)

1. **Same lesson worldwide on same day** — All users see Day 333 on Nov 28
2. **Languages precomputed** — No runtime translation
3. **60+ min training audio per voice** — Kelly voice consistency
4. **No browser TTS** — Only ElevenLabs or precomputed audio
5. **Streak integrity** — Never lose a user's streak incorrectly
6. **Progress persistence** — Never lose lesson completion data

---

## 3. Content Readiness Audit

### Lesson DNA Status

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT READINESS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   LESSONS WITH FULL DNA:        42 / 365  (11.5%)              │
│   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                 │
│                                                                 │
│   LESSONS NEEDING DNA:         323 / 365  (88.5%)              │
│   ████████████████████████████████████████████████████████████ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Supabase lesson_atoms:       21,915 rows                     │
│   Supabase lesson_shards:      38,700 rows                     │
│   Supabase core_lessons:       365 rows                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Lesson Content Requirements

Each fully-complete lesson requires:

| Component               | Count                                            | Status     |
| ----------------------- | ------------------------------------------------ | ---------- |
| Welcome Phase           | 1 × 6 ages × 3 languages × 3 tones = 54 variants | Partial    |
| Q1 Phase                | 54 variants + 2-3 choices each                   | Partial    |
| Q2 Phase                | 54 variants + 2-3 choices each                   | Partial    |
| Q3 Phase                | 54 variants + 2-3 choices each                   | Partial    |
| Wisdom Phase            | 54 variants                                      | Partial    |
| **Total text variants** | **270+ per lesson**                              |            |
| Audio files             | 270+ per lesson (if voiced)                      | 🔴 Limited |
| Kelly images            | 5 expressions × 1 = 5                            | ✅ Ready   |

### Content Gap Analysis

| Category     | Lessons | DNA Ready | Gap     |
| ------------ | ------- | --------- | ------- |
| Science      | ~73     | 15        | 58      |
| History      | ~52     | 8         | 44      |
| Arts         | ~45     | 5         | 40      |
| Social/Civic | ~48     | 6         | 42      |
| Health       | ~36     | 3         | 33      |
| Math         | ~30     | 2         | 28      |
| Language     | ~25     | 1         | 24      |
| Technology   | ~56     | 2         | 54      |
| **TOTAL**    | **365** | **42**    | **323** |

### Content Production Required (MVP)

**Option A: Launch with 42 complete lessons (current state)**

- Risk: Users hit incomplete content after day 42
- Mitigation: Fallback to auto-generated content from existing data

**Option B: Complete 100 lessons by launch**

- Need: 58 additional DNA files
- Effort: ~2-3 hours per lesson × 58 = 116-174 hours
- Timeline: 3 weeks at 40 hrs/week with 2 content creators

**Option C: Complete all 365 for launch**

- Need: 323 additional DNA files
- Effort: ~2-3 hours per lesson × 323 = 646-969 hours
- Timeline: 12-16 weeks at 40 hrs/week with 2 content creators

### RECOMMENDATION

**Phase 1 (Launch):** Prioritize completing:

1. First 31 days (January) — New Year learners
2. All holidays (25 lessons) — Thanksgiving, Christmas, etc.
3. Today + next 7 days — Always complete

**Phase 2 (Post-launch):**

- Complete 10 lessons/week
- 32 weeks to complete all 323

---

## 4. Learner Journey Maps

### Journey 1: First-Time Visitor (Guest)

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: First-Time Guest                                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ARRIVE                                                       │
│     └── Marketing site / Direct link / Social share             │
│                                                                  │
│  2. LAND                                                         │
│     └── index.html (Marketing + Auth)                           │
│     └── See "Start learning now" CTA                            │
│                                                                  │
│  3. ENTER GUEST MODE                                            │
│     └── Click "Continue as Guest"                               │
│     └── Store guest_mode=true in localStorage                   │
│                                                                  │
│  4. VIEW HUB                                                     │
│     └── hub.html shows Today's Lesson                           │
│     └── Stats: 0 streak, 0% complete, 0 lessons                 │
│     └── Premium upsell banner visible                           │
│                                                                  │
│  5. START FIRST LESSON                                          │
│     └── learn.html loads                                        │
│     └── Default variants: 18-35, EN, curious, 2 choices        │
│     └── Kelly explains citizenship                              │
│                                                                  │
│  6. COMPLETE PHASES                                             │
│     └── Welcome → Q1 → Q2 → Q3 → Wisdom                        │
│     └── Progress stored in localStorage                         │
│                                                                  │
│  7. LESSON COMPLETE                                             │
│     └── Celebration modal                                       │
│     └── Streak = 1, Lessons = 1                                 │
│     └── Return to Hub                                           │
│                                                                  │
│  8. NEXT DAY                                                    │
│     └── Guest returns                                           │
│     └── localStorage persists progress                          │
│     └── Can continue streak                                     │
│                                                                  │
│  9. HIT GUEST LIMIT (Day 4)                                     │
│     └── Show paywall: "Sign up to continue"                    │
│     └── Stripe checkout or create account                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Guest mode activates correctly
- [ ] Progress persists across browser sessions
- [ ] Guest limit enforced (3 lessons)
- [ ] Paywall shown at correct point
- [ ] Guest can convert to authenticated user
- [ ] Progress transfers from guest to user

---

### Journey 2: New User Signup

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: New User Signup                                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ARRIVE AT AUTH                                              │
│     └── index.html or /signup                                   │
│                                                                  │
│  2. CHOOSE SIGNUP METHOD                                        │
│     ├── Email/Password                                          │
│     ├── Google OAuth                                            │
│     └── Apple OAuth                                             │
│                                                                  │
│  3. SUPABASE AUTH                                               │
│     └── auth.users row created                                  │
│     └── Trigger creates public.users row                        │
│     └── Email verification (if enabled)                         │
│                                                                  │
│  4. ONBOARDING (Optional)                                       │
│     ├── Select age group (defaults to 18-35)                   │
│     ├── Select language (defaults to EN)                       │
│     ├── Set birthday (optional)                                 │
│     └── Choose tone preference                                  │
│                                                                  │
│  5. REDIRECT TO HUB                                             │
│     └── hub.html with fresh account                            │
│     └── Streak = 0, can start today                            │
│                                                                  │
│  6. MERGE GUEST PROGRESS (if applicable)                        │
│     └── Check localStorage for guest progress                   │
│     └── Import completed days to user_progress                  │
│     └── Clear localStorage                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Email signup works
- [ ] Google OAuth works
- [ ] Apple OAuth works
- [ ] public.users row created automatically
- [ ] Onboarding saves preferences correctly
- [ ] Guest progress merges into authenticated account
- [ ] No duplicate rows created

---

### Journey 3: Premium Subscriber

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Premium Subscription                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TRIGGER                                                     │
│     ├── Click "Upgrade" from paywall                           │
│     ├── Click "Subscribe" from pricing page                    │
│     └── Click "Start Free Trial" from upsell                   │
│                                                                  │
│  2. STRIPE CHECKOUT                                             │
│     └── /api/create-checkout-session called                    │
│     └── Redirect to Stripe hosted checkout                     │
│     └── 7-day free trial starts                                │
│                                                                  │
│  3. PAYMENT SUCCESS                                             │
│     └── Stripe webhook fires: checkout.session.completed       │
│     └── Update users.subscription_status = 'active'            │
│     └── Update users.stripe_customer_id                        │
│     └── Update users.subscription_started_at                   │
│                                                                  │
│  4. RETURN TO APP                                               │
│     └── success_url: /hub.html?success=true                    │
│     └── Show success toast                                     │
│     └── Remove all upsell banners                              │
│                                                                  │
│  5. ONGOING                                                     │
│     └── Stripe charges monthly                                 │
│     └── Webhook: invoice.paid → keep active                    │
│     └── Webhook: invoice.payment_failed → notify user         │
│                                                                  │
│  6. CANCELLATION                                                │
│     └── User cancels in Stripe portal                          │
│     └── Webhook: customer.subscription.deleted                 │
│     └── Set subscription_status = 'cancelled'                  │
│     └── Set subscription_expires_at (end of period)           │
│     └── User keeps access until expiry                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Checkout session created successfully
- [ ] Stripe checkout loads
- [ ] Trial period is 7 days
- [ ] Webhook updates subscription_status
- [ ] Success redirect works
- [ ] Cancelled users keep access until period end
- [ ] Failed payment notifications work
- [ ] Re-subscription works after cancellation

---

### Journey 4: Returning User (Streak Continuation)

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Streak Continuation                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SCENARIO A: Same Day Return                                    │
│  ────────────────────────────                                   │
│  └── User completed lesson, returns same day                   │
│  └── Show "Completed!" on Today card                           │
│  └── Streak unchanged                                          │
│  └── Can browse calendar, review lesson                        │
│                                                                  │
│  SCENARIO B: Next Day Return (Streak +1)                        │
│  ────────────────────────────────────────                       │
│  └── User returns exactly 1 day later                          │
│  └── New lesson available                                      │
│  └── Complete lesson → streak++                                │
│  └── Streak animation plays                                    │
│                                                                  │
│  SCENARIO C: Missed Day (Streak Reset)                          │
│  ─────────────────────────────────────                          │
│  └── User returns 2+ days later                                │
│  └── Streak resets to 0                                        │
│  └── Show "Welcome back!" message                              │
│  └── Can restart streak today                                  │
│                                                                  │
│  SCENARIO D: Catch-Up Mode (Premium Feature?)                   │
│  ────────────────────────────────────────────                   │
│  └── User wants to do missed lessons                           │
│  └── Can access any past day from calendar                     │
│  └── Does NOT restore streak                                   │
│  └── Does count toward total lessons                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Streak calculation is accurate
- [ ] Timezone handling (user in different timezone than server)
- [ ] Streak resets after exactly 48 hours (not 24)
- [ ] Streak cap (365 days max?)
- [ ] Longest streak tracking works
- [ ] Past lessons accessible but don't restore streak

---

### Journey 5: Multi-Device User

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Cross-Device Sync                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. START ON PHONE                                              │
│     └── Complete Q1 and Q2 of today's lesson                   │
│     └── Progress saved: phase=3, choices={Q1:'A', Q2:'B'}      │
│     └── Close app                                               │
│                                                                  │
│  2. CONTINUE ON TABLET                                          │
│     └── Login (same account)                                   │
│     └── Fetch progress from Supabase                           │
│     └── Resume at Q3                                           │
│     └── Complete lesson                                         │
│                                                                  │
│  3. VIEW ON DESKTOP                                             │
│     └── Login                                                   │
│     └── See completed lesson                                   │
│     └── Streak and stats synced                                │
│                                                                  │
│  EDGE CASE: Offline + Sync Conflict                             │
│  ────────────────────────────────────                           │
│  └── Phone goes offline during lesson                          │
│  └── User completes on tablet while phone offline              │
│  └── Phone comes online → conflict                             │
│  └── Resolution: Last-write-wins? Merge choices?               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Progress syncs within 5 seconds
- [ ] Mid-lesson resume works
- [ ] Choices persist across devices
- [ ] Offline completion syncs when online
- [ ] Conflict resolution strategy defined
- [ ] No duplicate completion records

---

### Journey 6: Birthday Lesson Discovery

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Birthday Lesson                                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SET BIRTHDAY                                                │
│     └── Settings → Personal → Birthday                         │
│     └── Select month and day                                   │
│     └── Calculate lesson day number                            │
│     └── Save to users.birthday_month, birthday_day             │
│                                                                  │
│  2. VIEW IN HUB                                                 │
│     └── Birthday card appears below calendar                   │
│     └── Shows: "Your Birthday · March 15"                     │
│     └── Shows: "Creative Writing"                              │
│                                                                  │
│  3. PREVIEW BIRTHDAY LESSON                                     │
│     └── Tap birthday card                                      │
│     └── Lesson preview modal opens                             │
│     └── Show day number, topic, objective                      │
│                                                                  │
│  4. TAKE BIRTHDAY LESSON                                        │
│     └── "Start This Lesson" button                             │
│     └── Can take any time (not just on birthday)              │
│     └── Special celebration if taken ON birthday?              │
│                                                                  │
│  5. CALENDAR HIGHLIGHT                                          │
│     └── Birthday day shows 🎂 emoji                            │
│     └── Different background color                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Birthday saves correctly
- [ ] Leap year handling (Feb 29)
- [ ] Birthday card displays
- [ ] Correct lesson mapped to birthday
- [ ] Calendar shows birthday indicator
- [ ] Birthday lesson accessible any time
- [ ] Special celebration on actual birthday

---

### Journey 7: Variant Switching Mid-Lesson

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Change Variant During Lesson                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. START LESSON                                                │
│     └── Variants: 18-35, EN, curious, 2 choices                │
│                                                                  │
│  2. REACH Q2                                                    │
│     └── User completed Welcome, Q1                             │
│     └── Currently on Q2                                        │
│                                                                  │
│  3. CHANGE AGE TO 6-12                                          │
│     └── Tap Age button                                         │
│     └── Select "6-12 years"                                    │
│     └── Modal closes                                           │
│                                                                  │
│  4. CONTENT RELOADS                                             │
│     └── Current phase (Q2) reloads with new age variant        │
│     └── Vocabulary simplifies                                  │
│     └── Choices may change                                     │
│     └── Previous choices (Q1) are KEPT                         │
│     └── Kelly expression unchanged                             │
│                                                                  │
│  5. COMPLETE LESSON                                             │
│     └── Record final variant used                              │
│     └── Track: started with 18-35, finished with 6-12         │
│                                                                  │
│  EDGE: Change Difficulty from 2→3                               │
│  ────────────────────────────────                               │
│  └── User on Q2, changes from 2 to 3 choices                   │
│  └── Q2 now shows A, B, C                                      │
│  └── Q1 choice already recorded (only A or B)                  │
│  └── Analytics: note mixed difficulty                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points:**

- [ ] Variant change reloads current phase
- [ ] Previous choices preserved
- [ ] Kelly expression persists
- [ ] Difficulty change adds/removes choice C
- [ ] Mixed-variant lesson completion recorded
- [ ] No crash if variant content missing

---

### Journey 8: Family Account (Future)

```
┌──────────────────────────────────────────────────────────────────┐
│ JOURNEY: Family Account (FUTURE FEATURE)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PARENT ACCOUNT                                              │
│     └── Primary subscription holder                            │
│     └── Can add up to 5 family profiles                        │
│                                                                  │
│  2. ADD CHILD PROFILE                                           │
│     └── Name: "Tommy"                                          │
│     └── Age: 8 → Auto-select 6-12 variant                     │
│     └── No separate email                                      │
│     └── Optional PIN protection                                │
│                                                                  │
│  3. PROFILE SWITCHING                                           │
│     └── Profile picker on Hub                                  │
│     └── Each profile has own progress                         │
│     └── Each profile has own preferences                      │
│     └── Streak is per-profile                                  │
│                                                                  │
│  4. PARENT DASHBOARD                                            │
│     └── See all family progress                                │
│     └── Weekly email summary                                   │
│     └── "Tommy learned about Citizenship this week!"          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Test Points (Future):**

- [ ] Profile creation
- [ ] Profile switching
- [ ] Isolated progress per profile
- [ ] Age auto-selection
- [ ] PIN protection
- [ ] Parent dashboard

---

## 5. Test Scenarios & Edge Cases

### 5.1 Authentication Edge Cases

| ID      | Scenario                                 | Expected Behavior                         | Priority |
| ------- | ---------------------------------------- | ----------------------------------------- | -------- |
| AUTH-01 | User signs up with email, never verifies | Can login but show "Verify email" banner  | P1       |
| AUTH-02 | User forgets password                    | Reset flow via email                      | P1       |
| AUTH-03 | Google OAuth popup blocked               | Show "Enable popups" message              | P2       |
| AUTH-04 | Session expires during lesson            | Show login modal, preserve progress       | P1       |
| AUTH-05 | Same email, different auth methods       | Link accounts or error?                   | P2       |
| AUTH-06 | Delete account                           | Remove all user data, cancel subscription | P1       |
| AUTH-07 | Banned user tries to login               | Block with message                        | P3       |
| AUTH-08 | Login from new device                    | Optional 2FA?                             | P3       |

### 5.2 Lesson Playback Edge Cases

| ID      | Scenario                                   | Expected Behavior                 | Priority |
| ------- | ------------------------------------------ | --------------------------------- | -------- |
| PLAY-01 | Lesson DNA file missing                    | Fallback to Supabase lesson_atoms | P1       |
| PLAY-02 | Supabase content missing                   | Generate from universal_truth     | P1       |
| PLAY-03 | Audio file missing                         | Proceed without audio             | P2       |
| PLAY-04 | Kelly image fails to load                  | Show placeholder                  | P1       |
| PLAY-05 | User taps choice before audio finishes     | Allow (don't block)               | P1       |
| PLAY-06 | User rapidly taps same choice              | Debounce, single selection        | P1       |
| PLAY-07 | Network dies mid-lesson                    | Queue progress, sync when online  | P1       |
| PLAY-08 | Browser refresh during phase               | Resume from last phase            | P1       |
| PLAY-09 | User navigates away, returns               | Resume or restart prompt          | P2       |
| PLAY-10 | Lesson takes >30 minutes                   | Log analytics, no timeout         | P3       |
| PLAY-11 | User hits back button                      | Confirm exit, warn about progress | P1       |
| PLAY-12 | Third choice selected but difficulty was 2 | Impossible if UI correct          | P3       |

### 5.3 Streak Edge Cases

| ID      | Scenario                          | Expected Behavior               | Priority |
| ------- | --------------------------------- | ------------------------------- | -------- |
| STRK-01 | User in EST completes at 11:59 PM | Count for current day           | P1       |
| STRK-02 | User in PST, server in UTC        | Use user's timezone             | P1       |
| STRK-03 | Daylight saving time change       | Handle gracefully               | P2       |
| STRK-04 | Complete 2 lessons same day       | Streak +1 only, lessons +2      | P1       |
| STRK-05 | Complete at 12:01 AM              | New day, streak continues       | P1       |
| STRK-06 | Streak reaches 365 days           | Special celebration             | P3       |
| STRK-07 | Server clock incorrect            | Use client time with validation | P2       |
| STRK-08 | User changes device timezone      | Recalculate based on UTC        | P2       |

### 5.4 Payment Edge Cases

| ID     | Scenario                          | Expected Behavior         | Priority |
| ------ | --------------------------------- | ------------------------- | -------- |
| PAY-01 | Card declined                     | Show error, suggest retry | P1       |
| PAY-02 | 3D Secure required                | Handle Stripe redirect    | P1       |
| PAY-03 | Free trial expires, no card       | Downgrade to free tier    | P1       |
| PAY-04 | Subscription renewal fails        | Grace period (3 days)     | P1       |
| PAY-05 | User disputes charge              | Handle chargeback         | P2       |
| PAY-06 | Refund requested                  | Process via Stripe        | P2       |
| PAY-07 | Price change for existing user    | Honor original price      | P2       |
| PAY-08 | Currency conversion               | Stripe handles            | P3       |
| PAY-09 | Multiple subscriptions same email | Prevent                   | P1       |
| PAY-10 | Gift subscription                 | Future feature            | P3       |

### 5.5 Content Edge Cases

| ID      | Scenario                       | Expected Behavior                     | Priority |
| ------- | ------------------------------ | ------------------------------------- | -------- |
| CONT-01 | Lesson has no 2-5 age variant  | Use closest age or fallback           | P1       |
| CONT-02 | Spanish translation missing    | Use English with "translation coming" | P1       |
| CONT-03 | Third choice not defined       | Only show 2 even in challenge mode    | P1       |
| CONT-04 | Choice text is very long       | Truncate with ellipsis, tooltip       | P2       |
| CONT-05 | Wisdom phase empty             | Use universal_truth                   | P1       |
| CONT-06 | Special characters in content  | Proper encoding                       | P2       |
| CONT-07 | Content contains profanity     | Content moderation                    | P1       |
| CONT-08 | Lesson references future event | Time-appropriate wording              | P3       |

### 5.6 UI/UX Edge Cases

| ID    | Scenario                  | Expected Behavior        | Priority |
| ----- | ------------------------- | ------------------------ | -------- |
| UI-01 | iPhone SE (320px width)   | All content visible      | P1       |
| UI-02 | iPad landscape            | Optimal layout           | P2       |
| UI-03 | 4K desktop monitor        | Not stretched            | P2       |
| UI-04 | User zooms to 200%        | No overflow              | P2       |
| UI-05 | Dark mode OS setting      | Respect (already dark)   | P3       |
| UI-06 | Reduced motion preference | Disable animations       | P2       |
| UI-07 | High contrast mode        | Accessible colors        | P2       |
| UI-08 | Screen reader             | All content accessible   | P1       |
| UI-09 | Keyboard-only navigation  | Tab through all controls | P1       |
| UI-10 | Touch vs mouse            | Appropriate targets      | P1       |
| UI-11 | Notch/home indicator      | Safe area respected      | P1       |
| UI-12 | Split screen (iPad)       | Responsive               | P3       |

### 5.7 Performance Edge Cases

| ID      | Scenario                        | Expected Behavior    | Priority |
| ------- | ------------------------------- | -------------------- | -------- |
| PERF-01 | Slow 3G connection              | Progressive loading  | P1       |
| PERF-02 | Offline mode                    | Show cached content  | P2       |
| PERF-03 | 1000 users hit same lesson      | CDN cache serves     | P1       |
| PERF-04 | User with 365 completed lessons | Fast calendar render | P2       |
| PERF-05 | Large localStorage              | Handle storage quota | P2       |
| PERF-06 | Memory leak from animations     | Clean up properly    | P2       |

---

## 6. Avatar System Testing

### 6.1 Kelly Avatar Overview

Kelly exists in two modes that must work seamlessly:

| Mode   | Technology       | Size  | Load Time | Use Case                  |
| ------ | ---------------- | ----- | --------- | ------------------------- |
| **2D** | PNG images + CSS | ~6MB  | <2s       | Default, works everywhere |
| **3D** | Unity WebGL      | ~40MB | 10-30s    | Premium, opt-in           |

### 6.2 2D Avatar Test Scenarios

| ID    | Scenario         | Steps                       | Expected                               | Priority |
| ----- | ---------------- | --------------------------- | -------------------------------------- | -------- |
| 2D-01 | Initial load     | Open learn.html             | Kelly (curious) visible in <2s         | P1       |
| 2D-02 | Welcome phase    | Start lesson                | Kelly curious expression               | P1       |
| 2D-03 | Q1 asking        | Enter Q1 phase              | Kelly curious, slight animation        | P1       |
| 2D-04 | Q1 choice A      | Select choice A             | Kelly → explaining, teaching animation | P1       |
| 2D-05 | Q1 choice B      | Select choice B             | Kelly → celebrating, sparkle effect    | P1       |
| 2D-06 | Q1 choice C      | Select choice C (challenge) | Kelly → wisdom expression              | P1       |
| 2D-07 | Q3 listening     | Enter Q3 phase              | Kelly → listening expression           | P1       |
| 2D-08 | Wisdom phase     | Enter wisdom                | Kelly → wisdom, serene glow            | P1       |
| 2D-09 | Complete         | Finish lesson               | Kelly → celebrating                    | P1       |
| 2D-10 | Expression cycle | Go through all phases       | All 5 expressions work                 | P1       |
| 2D-11 | Crossfade        | Change expression           | Smooth 400ms transition                | P2       |
| 2D-12 | Breathing        | Observe idle Kelly          | 4s breath cycle visible                | P2       |
| 2D-13 | Speaking state   | Play audio                  | Speaking ring appears                  | P1       |
| 2D-14 | Speaking stop    | Audio ends                  | Speaking ring disappears               | P1       |
| 2D-15 | Image load fail  | Block image URL             | Previous expression stays              | P2       |
| 2D-16 | Rapid changes    | Quickly change phases       | No flicker, queues properly            | P2       |
| 2D-17 | Mobile Safari    | Test on iOS Safari          | All animations work                    | P1       |
| 2D-18 | Android Chrome   | Test on Android             | All animations work                    | P1       |
| 2D-19 | Reduced motion   | Set OS preference           | Animations disabled                    | P2       |
| 2D-20 | Memory check     | Monitor DevTools            | No image memory leaks                  | P2       |

### 6.3 3D Avatar Test Scenarios

| ID    | Scenario        | Steps                    | Expected                     | Priority |
| ----- | --------------- | ------------------------ | ---------------------------- | -------- |
| 3D-01 | WebGL check     | Load on non-WebGL device | Shows "3D not available"     | P1       |
| 3D-02 | Memory check    | Load on low-RAM device   | 3D option disabled           | P2       |
| 3D-03 | Toggle to 3D    | Click 2D/3D button       | Confirmation dialog          | P1       |
| 3D-04 | Load progress   | Confirm 3D load          | Progress % shows             | P1       |
| 3D-05 | Load complete   | Wait for Unity           | 3D Kelly appears             | P1       |
| 3D-06 | Crossfade 2D→3D | Complete load            | Smooth crossfade, no flicker | P1       |
| 3D-07 | Load timeout    | Simulate slow network    | Fallback to 2D after 45s     | P1       |
| 3D-08 | Load fail       | Block Unity files        | Error message, stay in 2D    | P1       |
| 3D-09 | Cancel load     | Click "Stay in 2D"       | Load aborts, stay in 2D      | P1       |
| 3D-10 | Expression sync | Set expression in 3D     | Blendshapes animate          | P1       |
| 3D-11 | Lip sync start  | Play audio with 3D       | Mouth moves to speech        | P1       |
| 3D-12 | Lip sync stop   | Audio ends               | Mouth returns to rest        | P1       |
| 3D-13 | FPS check       | Monitor during 3D        | Above 30 FPS                 | P2       |
| 3D-14 | Memory usage    | Monitor during 3D        | Below 500MB                  | P2       |
| 3D-15 | Toggle back 2D  | Click 2D/3D in 3D mode   | Instant switch to 2D         | P1       |
| 3D-16 | State preserve  | Switch modes mid-phase   | Expression matches           | P1       |
| 3D-17 | Unload Unity    | Switch to 2D, wait 5 min | Memory released              | P2       |
| 3D-18 | Page refresh    | Refresh during 3D load   | Clean restart                | P2       |
| 3D-19 | Desktop Chrome  | Test full 3D flow        | Complete success             | P1       |
| 3D-20 | Desktop Firefox | Test full 3D flow        | Complete success             | P1       |

### 6.4 Mode Switch Test Scenarios

| ID    | Scenario              | Steps                  | Expected                 | Priority |
| ----- | --------------------- | ---------------------- | ------------------------ | -------- |
| SW-01 | Default mode          | Fresh visit            | 2D mode active           | P1       |
| SW-02 | Saved preference      | Set 3D, reload page    | Attempts 3D load         | P1       |
| SW-03 | Switch mid-Welcome    | Change mode in Welcome | Seamless transition      | P1       |
| SW-04 | Switch mid-Q2         | Change mode during Q2  | Phase preserved          | P1       |
| SW-05 | Switch while speaking | Toggle during audio    | Speaking state syncs     | P1       |
| SW-06 | Rapid toggle          | Click toggle 5x fast   | Debounced, no crash      | P2       |
| SW-07 | Toggle badge          | After switch           | Badge shows current mode | P1       |
| SW-08 | Error recovery        | 3D fails, shows error  | Can still use 2D         | P1       |

### 6.5 Avatar Performance Benchmarks

**2D Performance Targets:**

| Metric             | Target | Measurement Method |
| ------------------ | ------ | ------------------ |
| Initial image load | <500ms | Performance.mark   |
| Expression switch  | <400ms | Animation duration |
| Animation FPS      | 60 FPS | Chrome DevTools    |
| Idle CPU           | <5%    | Task Manager       |
| Memory             | <20MB  | DevTools Memory    |

**3D Performance Targets:**

| Metric               | Target  | Measurement Method |
| -------------------- | ------- | ------------------ |
| Unity load (fast 4G) | <15s    | Network timing     |
| Unity load (3G)      | <45s    | Network timing     |
| Render FPS           | >30 FPS | Unity Profiler     |
| Peak memory          | <500MB  | Task Manager       |
| Idle GPU             | <30%    | GPU-Z              |

### 6.6 Avatar Content Checklist

**2D Assets (Required):**

| Image       | File                                  | Status |
| ----------- | ------------------------------------- | ------ |
| Curious     | kelly-directors-chair-curious.png     | ✅     |
| Explaining  | kelly-directors-chair-explaining.png  | ✅     |
| Listening   | kelly-directors-chair-listening.png   | ✅     |
| Wisdom      | kelly-directors-chair-wisdom.png      | ✅     |
| Celebrating | kelly-directors-chair-celebrating.png | ✅     |

**3D Assets (Required):**

| Asset           | File                                  | Status |
| --------------- | ------------------------------------- | ------ |
| Unity Loader    | Kelly_Web_Build.loader.js             | ✅     |
| Unity Data      | Kelly_Web_Build.data.unityweb         | ✅     |
| Unity Framework | Kelly_Web_Build.framework.js.unityweb | ✅     |
| Unity WASM      | Kelly_Web_Build.wasm.unityweb         | ✅     |

**Integration Code (Required):**

| Component          | File                       | Status               |
| ------------------ | -------------------------- | -------------------- |
| 2D Avatar Player   | kelly-2d-avatar.js         | ⚠️ Needs integration |
| 3D Loader          | unity-kelly-loader.js      | 🔴 Needs creation    |
| 3D Bridge          | unity-kelly-bridge.js      | ⚠️ Needs update      |
| Unified Controller | kelly-avatar-controller.js | 🔴 Needs creation    |

---

## 7. Scale Architecture for 1B Learners (Renumbered from 6)

### Traffic Projections

```
┌─────────────────────────────────────────────────────────────────┐
│                    1 BILLION LEARNERS / YEAR                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   DAILY ACTIVE USERS                                            │
│   └── 1B / 365 = 2.74M DAU (assuming each user 1x/day)        │
│   └── Realistic: 30% daily active = 3M users do lesson         │
│                                                                 │
│   PEAK HOUR TRAFFIC                                             │
│   └── 2.74M spread over 4 peak hours = 685K / hour            │
│   └── Per minute: ~11,400 concurrent users                     │
│   └── Per second: ~190 requests                                │
│                                                                 │
│   GLOBAL DISTRIBUTION                                           │
│   └── 40% Americas (EST/PST morning peak)                      │
│   └── 30% Europe (GMT/CET morning peak)                        │
│   └── 20% Asia (IST/JST morning peak)                          │
│   └── 10% Rest of world                                        │
│                                                                 │
│   PEAK: 3 overlapping morning peaks = 500K concurrent          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Infrastructure Requirements

#### 6.1 CDN Strategy (Static Assets)

```yaml
Static Assets:
  Kelly Images:
    - 5 expressions × ~500KB = 2.5MB
    - CDN cache: 1 year
    - Edge locations: 50+ global

  CSS/JS:
    - Combined: ~200KB gzipped
    - CDN cache: Version-based invalidation

  Curriculum JSON:
    - 365_day_calendar.json: ~2MB
    - CDN cache: 24 hours
    - Invalidation on content updates

  Audio Files (Future):
    - ~270 files × 365 lessons × ~100KB = 10TB total
    - CDN with geographic routing
    - Lazy loading by phase

Estimated CDN Cost (1B users):
  - Bandwidth: ~500TB/month
  - CloudFlare Pro: ~$200/month (unlimited bandwidth)
  - Or AWS CloudFront: ~$40,000/month

RECOMMENDATION: Cloudflare for CDN
```

#### 6.2 Database Architecture

```yaml
Current (Supabase):
  - Single Postgres instance
  - Max connections: 200 pooled
  - Good for: 10K concurrent users

Scale Path 1 - Supabase Pro:
  - Dedicated instance
  - Read replicas: 3 regions
  - Max: 100K concurrent users
  - Cost: ~$500/month

Scale Path 2 - Distributed Database:
  - Primary: Supabase for auth/users
  - Content: Read-only Postgres replicas
  - Cache: Redis for hot data
  - Max: 1M concurrent users
  - Cost: ~$5,000/month

Scale Path 3 - Full Distributed:
  - User sharding by region
  - Content in CDN-edge databases
  - Redis cluster for sessions
  - Max: Unlimited
  - Cost: ~$50,000/month

RECOMMENDATION: Start with Supabase Pro, plan for Path 2 at 50K DAU
```

#### 6.3 API Architecture

```yaml
Current:
  - Vercel Serverless Functions
  - Cold start: ~500ms
  - Concurrent: 1000 per function

Scale Requirements:
  - /api/create-checkout-session: Low volume (~100/day)
  - /api/stripe-webhook: Medium volume (~500/day)
  - Lesson content: Served from CDN (not API)

Future APIs Needed:
  - /api/progress: Sync progress (high volume)
    - Consider: WebSocket for real-time sync
    - Or: Batch updates every 30 seconds

  - /api/analytics: Event logging
    - Consider: Direct to data warehouse
    - Not through Supabase

  - /api/search: Lesson search
    - Consider: Algolia or Elasticsearch
    - Index: 365 lessons × metadata

RECOMMENDATION: Keep Vercel for low-volume APIs,
               Use CDN for content,
               Add dedicated analytics pipeline
```

#### 6.4 Authentication at Scale

```yaml
Supabase Auth Limits:
  - Free: 50K MAU
  - Pro: 100K MAU included, $0.00325/MAU after

At 1B Users:
  - Active in any month: ~300M MAU
  - Cost: ~$975,000/year for auth alone!

Alternative at Scale:
  - Self-hosted Auth0: ~$100K/year
  - Custom auth with Postgres: $0
  - Firebase Auth: Similar pricing

RECOMMENDATION: Stay on Supabase to 100K MAU,
  Plan migration at 50K MAU,
  Build abstraction layer now
```

#### 6.5 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCALE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   USER LAYER                                                    │
│   ───────────                                                   │
│   Browser/PWA → Cloudflare CDN (edge) → Static Assets          │
│                    │                                            │
│                    ▼                                            │
│   API LAYER                                                     │
│   ─────────                                                     │
│   Vercel Edge Functions (low latency)                          │
│   └── /api/progress → Redis → Postgres (async)                │
│   └── /api/auth → Supabase Auth                               │
│   └── /api/checkout → Stripe                                  │
│                                                                 │
│   DATA LAYER                                                    │
│   ──────────                                                    │
│   Supabase (Primary)                                           │
│   └── users, user_progress, affiliates                        │
│   Redis (Cache)                                                │
│   └── Session data, hot content                               │
│   BigQuery (Analytics)                                         │
│   └── Event stream, usage data                                │
│                                                                 │
│   CONTENT LAYER                                                 │
│   ─────────────                                                 │
│   Cloudflare R2 (Object Storage)                               │
│   └── Curriculum JSON                                          │
│   └── Audio files                                              │
│   └── Kelly images                                             │
│   Cloudflare CDN (Edge Cache)                                  │
│   └── All static assets                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Launch Checklist

### 7.1 Content Readiness

| Task                                  | Status | Owner   | Due    |
| ------------------------------------- | ------ | ------- | ------ |
| Complete DNA for days 1-31 (January)  | 🔴     | Content | Week 1 |
| Complete DNA for holiday lessons (25) | 🔴     | Content | Week 1 |
| Complete DNA for today + 7 days       | 🔴     | Content | Daily  |
| Verify all 365 lesson titles          | ✅     | Content | Done   |
| Verify all learning objectives        | ✅     | Content | Done   |
| Spanish translations for 50 lessons   | 🔴     | Content | Week 2 |
| French translations for 50 lessons    | 🔴     | Content | Week 2 |
| Audio files for 10 lessons            | 🔴     | Audio   | Week 3 |

### 7.2 Technical Readiness

| Task                           | Status | Owner  | Due    |
| ------------------------------ | ------ | ------ | ------ |
| learn.html production ready    | ✅     | Dev    | Done   |
| hub.html production ready      | ✅     | Dev    | Done   |
| Stripe checkout flow           | ✅     | Dev    | Done   |
| Stripe webhook handling        | ✅     | Dev    | Done   |
| Supabase RLS policies          | ⚠️     | Dev    | Week 1 |
| Error logging (Sentry)         | 🔴     | Dev    | Week 1 |
| Analytics (Mixpanel/Amplitude) | 🔴     | Dev    | Week 1 |
| Performance monitoring         | 🔴     | Dev    | Week 1 |
| CDN configuration              | 🔴     | DevOps | Week 1 |
| SSL certificates               | ✅     | DevOps | Done   |
| Domain configuration           | ✅     | DevOps | Done   |

### 7.3 Quality Assurance

| Task                            | Status | Owner | Due    |
| ------------------------------- | ------ | ----- | ------ |
| Unit tests for kelly-data.js    | 🔴     | QA    | Week 1 |
| E2E tests for guest flow        | 🔴     | QA    | Week 1 |
| E2E tests for auth flow         | 🔴     | QA    | Week 1 |
| E2E tests for payment flow      | 🔴     | QA    | Week 1 |
| Mobile testing (iOS Safari)     | 🔴     | QA    | Week 1 |
| Mobile testing (Android Chrome) | 🔴     | QA    | Week 1 |
| Tablet testing                  | 🔴     | QA    | Week 1 |
| Accessibility audit             | 🔴     | QA    | Week 2 |
| Load testing (1K concurrent)    | 🔴     | QA    | Week 2 |

### 7.4 Legal & Compliance

| Task                    | Status | Owner | Due    |
| ----------------------- | ------ | ----- | ------ |
| Privacy Policy          | ⚠️     | Legal | Week 1 |
| Terms of Service        | ⚠️     | Legal | Week 1 |
| COPPA compliance (kids) | 🔴     | Legal | Week 1 |
| GDPR compliance (EU)    | 🔴     | Legal | Week 1 |
| Cookie consent banner   | 🔴     | Dev   | Week 1 |
| Data deletion flow      | 🔴     | Dev   | Week 1 |

---

## 8. Risk Assessment

### Critical Risks

| Risk               | Impact                  | Probability | Mitigation              |
| ------------------ | ----------------------- | ----------- | ----------------------- |
| Content not ready  | Users hit empty lessons | High        | Fallback content system |
| Supabase outage    | App unusable            | Low         | localStorage fallback   |
| Stripe outage      | No payments             | Low         | Show "try again later"  |
| Kelly images fail  | Broken experience       | Low         | Placeholder images      |
| Mobile Safari bugs | iOS users blocked       | Medium      | Extensive testing       |

### Content Velocity Risk

```
Current: 42 lessons ready
Need by launch: 100 lessons minimum
Gap: 58 lessons

If we get 1000 users/day:
- Day 1-42: Content ready
- Day 43+: Users hit incomplete content
- Churn risk: HIGH

MITIGATION:
1. Auto-generate fallback content from existing data
2. Prioritize most-visited lessons
3. Hire 2 content writers immediately
```

### Scale Risk Timeline

```
Users    | When       | Risk           | Action Required
---------|------------|----------------|------------------
1K       | Week 1     | None           | -
10K      | Month 1    | DB connections | Upgrade Supabase
50K      | Month 3    | Auth costs     | Evaluate migration
100K     | Month 6    | CDN bandwidth  | Optimize assets
500K     | Year 1     | All systems    | Full architecture review
1M+      | Year 2     | Scale limits   | Distributed architecture
```

---

## Appendix A: Test Script Templates

### A.1 Guest Flow Test Script

```
TEST: Guest First Visit

PRECONDITIONS:
- Clear localStorage
- Clear cookies
- Use incognito window

STEPS:
1. Navigate to curiouskelly.com
2. Click "Start learning now"
3. Select "Continue as Guest"
4. Verify Hub loads with today's lesson
5. Click "Start Today's Lesson"
6. Complete Welcome phase
7. Select choice A on Q1
8. Select choice B on Q2
9. Select choice A on Q3
10. View Wisdom phase
11. Verify completion modal shows
12. Verify streak = 1
13. Return to Hub
14. Verify Today card shows "Completed"

EXPECTED:
- All phases complete without error
- Progress saved to localStorage
- Streak incremented
- No console errors
```

### A.2 Variant Switch Test Script

```
TEST: Change Age Mid-Lesson

PRECONDITIONS:
- Logged in user
- Default age: 18-35

STEPS:
1. Start today's lesson
2. Complete Welcome phase
3. On Q1, tap Age button
4. Change age to 6-12
5. Verify Q1 content changes
6. Verify vocabulary is simpler
7. Select choice A
8. Proceed to Q2
9. Verify Q2 uses 6-12 variant
10. Complete lesson
11. Check analytics: age_group_used = "6-12"

EXPECTED:
- Content updates immediately
- No data loss
- Final variant recorded correctly
```

---

## Appendix B: Analytics Events to Track

```javascript
const ANALYTICS_EVENTS = {
  // Acquisition
  page_view: { page, referrer, utm_source },
  signup_started: { method },
  signup_completed: { method, time_to_complete },

  // Engagement
  lesson_started: { day_number, variant },
  phase_completed: { day_number, phase, time_spent },
  choice_selected: { day_number, phase, choice },
  lesson_completed: { day_number, variant, time_spent },
  variant_changed: { type, from, to, phase },

  // Retention
  streak_continued: { streak_length },
  streak_broken: { previous_streak },
  return_visit: { days_since_last },

  // Revenue
  paywall_shown: { trigger },
  checkout_started: { plan },
  subscription_started: { plan, trial },
  subscription_cancelled: { reason, tenure },

  // Errors
  error_occurred: { type, message, stack }
};
```

---

_Document Version 1.0 — November 28, 2025_
_Next Review: December 5, 2025_
