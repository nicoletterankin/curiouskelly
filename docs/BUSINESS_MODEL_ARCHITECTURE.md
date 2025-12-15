# Curious Kelly - Business Model Architecture

## The Core Philosophy

> **"Today's lesson is always accessible for everyone, forever."**

This is not charity. This is strategy:
- **Daily touchpoint** creates habit
- **Outreach is cost-neutral** (emails, texts, push) because subscribers fund it
- **Urgency creates value**: "Learn today or pay for it later"
- **The moat is the relationship**, not the paywall

---

## Access Tiers

### 🆓 FREE TIER: "Today's Lesson"
**Who**: Everyone. No login required.
**What**: Today's lesson only
**Why**: This IS the product. The daily habit. The hook.

| Access | Today | Yesterday | Tomorrow | Any Other Day |
|--------|-------|-----------|----------|---------------|
| Free User | ✅ | ❌ | ❌ | ❌ |

**Key Insight**: "Today" is determined by the CALENDAR, not by when you first visited.
- December 15 = Day 349
- Everyone on December 15 sees Day 349 free
- Tomorrow (Dec 16), Day 349 costs money

### 💳 PAY-PER-LESSON
**Who**: Users who want specific past/future lessons
**What**: One-time purchase for permanent access to a specific day's lesson
**Why**: Low friction, low commitment, captures impulse buyers

| Price Point | Market | Notes |
|-------------|--------|-------|
| $0.99 - $2.99 | Premium markets (US, UK, EU) | |
| $0.29 - $0.99 | Emerging markets | |
| Dynamic | Individual | Based on engagement, history |

**UX**: "You missed this lesson. Own it forever for $1.99"

### ⭐ ALL-ACCESS SUBSCRIPTION
**Who**: Committed learners, families, lifelong learners
**What**: 
- All 365 core lessons
- 40+ emergency/bonus lessons
- Calendar navigation (go back, skip ahead)
- Future: On-demand AI lesson generator
- Future: Personalized learning paths

| Plan | US Price | Notes |
|------|----------|-------|
| Monthly | $9.99/mo | Core offering |
| Annual | $79/year | 33% savings |
| Lifetime | $199 once | For believers |
| Family | $14.99/mo | Up to 6 profiles |

**Market-Tailored Pricing**: Different Stripe Price IDs per region
- India: ₹199/mo (~$2.40)
- Brazil: R$19.90/mo (~$4)
- EU: €8.99/mo
- And so on...

---

## The Future State: On-Demand Lessons

The premium tier unlocks something more valuable than a library:
**An AI that generates personalized lessons on any topic, in Kelly's voice.**

This is where we're heading:
1. Today: 365 pre-generated lessons
2. Tomorrow: "Ask Kelly to teach you anything"
3. Future: Kelly knows your learning history and tailors content

The subscription isn't buying 365 lessons. It's buying **Kelly as your personal teacher**.

---

## Current State → Future State Mapping

### Authentication & Identity

| Current | Future |
|---------|--------|
| Anonymous users can access Day 1 | Anonymous users can access TODAY |
| Testing mode bypasses paywall | Production mode with tiered access |
| No user accounts required | Accounts for purchases & subscriptions |

### Access Control

| Current Code | What It Does | Future Need |
|--------------|--------------|-------------|
| `canAccessDay(1)` → true | Day 1 free | `isToday(dayNumber)` → free |
| `isPremium()` | Check subscription | Same, plus check purchases |
| N/A | N/A | `hasLessonPurchase(dayNumber)` |

### Pricing

| Current | Future |
|---------|--------|
| Fixed Stripe prices | Market-aware prices |
| Same price globally | Geo-detected pricing |
| No individual pricing | Engagement-based offers |

### Database Schema

**Current Tables:**
- `users` - basic user info
- `subscriptions` - subscription status (via Stripe)
- `revenue_events` - payment tracking

**New Tables Needed:**
```sql
-- Individual lesson purchases
CREATE TABLE lesson_purchases (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  day_number INTEGER NOT NULL,
  purchase_price DECIMAL(10,2),
  currency VARCHAR(3) DEFAULT 'USD',
  stripe_payment_id VARCHAR(255),
  purchased_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, day_number)
);

-- User pricing tier (for personalized/regional pricing)
CREATE TABLE user_pricing_tiers (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  region VARCHAR(50),
  tier VARCHAR(50) DEFAULT 'standard',
  custom_discount_pct INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Regional price configuration
CREATE TABLE regional_prices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  region VARCHAR(50) NOT NULL,
  product_type VARCHAR(50) NOT NULL, -- 'single_lesson', 'monthly', 'annual', 'lifetime'
  price DECIMAL(10,2) NOT NULL,
  currency VARCHAR(3) NOT NULL,
  stripe_price_id VARCHAR(255),
  UNIQUE(region, product_type)
);
```

---

## Access Logic: The New `canAccessDay()`

```javascript
function canAccessDay(dayNumber) {
  // 1. TODAY is always free for everyone
  if (isToday(dayNumber)) {
    return { access: true, reason: 'today' };
  }
  
  // 2. Subscribers get everything
  if (isPremium()) {
    return { access: true, reason: 'subscription' };
  }
  
  // 3. Check individual purchase
  if (await hasLessonPurchase(dayNumber)) {
    return { access: true, reason: 'purchased' };
  }
  
  // 4. No access - show paywall with options
  return { 
    access: false, 
    options: {
      buyLesson: await getLessonPrice(dayNumber),
      subscribe: await getSubscriptionPrices()
    }
  };
}

function isToday(dayNumber) {
  const today = new Date();
  const dayOfYear = getDayOfYear(today); // 1-365
  return dayNumber === dayOfYear;
}
```

---

## Paywall UX: The Offer

When a non-subscriber tries to access a non-today lesson:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  This lesson is from December 10th                  │
│                                                     │
│  ┌─────────────────┐  ┌─────────────────────────┐   │
│  │                 │  │                         │   │
│  │  OWN THIS       │  │  UNLOCK ALL 365         │   │
│  │  LESSON         │  │                         │   │
│  │                 │  │  $9.99/month            │   │
│  │  $1.99 once     │  │                         │   │
│  │                 │  │  • Every lesson         │   │
│  │  Forever access │  │  • Skip ahead or back   │   │
│  │  to this day    │  │  • Family sharing       │   │
│  │                 │  │  • Cancel anytime       │   │
│  └─────────────────┘  └─────────────────────────┘   │
│                                                     │
│  💡 Today's lesson is always included.              │
│     Come back tomorrow for the next one.            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## The "7-Day Trial" Problem

> "The 7 day free trial is cheap and I don't like the way it feels."

**Why it feels wrong:**
- It's transactional, not relational
- It implies we're hiding something
- It creates a "gotcha" moment at day 8
- It doesn't match "today is always free"

**The better framing:**
- No trial. No countdown. No pressure.
- **Today is always yours.**
- Want more? Here are your options.
- We believe in daily learning so much, we give you one every day.

This is confidence, not desperation.

---

## Implementation Priority

### Phase 1: Fix "Today is Free" (Immediate)
- [ ] Update `canAccessDay()` to check today vs day 1
- [ ] Update `isToday()` helper based on day of year
- [ ] Remove 7-day trial logic
- [ ] Update paywall messaging

### Phase 2: Pay-Per-Lesson (Week 1)
- [ ] Create `lesson_purchases` table
- [ ] Create Stripe product for single lesson
- [ ] Build purchase flow UI
- [ ] Check purchases in access control

### Phase 3: Regional Pricing (Week 2)
- [ ] Create `regional_prices` table
- [ ] Geo-detect user region
- [ ] Show appropriate prices
- [ ] Configure Stripe prices per region

### Phase 4: On-Demand Lessons (Future)
- [ ] AI lesson generation pipeline
- [ ] Topic request interface
- [ ] Kelly voice synthesis for new content
- [ ] Premium-only access

---

## Summary

| What | Current | Target |
|------|---------|--------|
| Free access | Day 1 only | TODAY only |
| Trial | 7-day | None (today is always free) |
| Pay-per-lesson | No | Yes ($0.99-$2.99) |
| Subscription | Monthly/Annual/Lifetime | Same + market pricing |
| Pricing | Fixed global | Market + individual tailored |
| Future value | 365 lessons | On-demand Kelly AI |

The business model isn't about restricting access.
It's about **making daily learning so valuable that people want more.**
