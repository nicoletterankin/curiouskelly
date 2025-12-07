# 💰 CURIOUS KELLY - PRICING STRATEGY BIBLE

## Master Document for Sales, Promotions, Affiliates & Revenue Operations

**Version:** 1.0
**Effective Date:** December 17, 2025 (Launch)
**Owner:** Revenue Operations
**Last Updated:** November 26, 2025

---

## TABLE OF CONTENTS

1. [Core Pricing Structure](#1-core-pricing-structure)
2. [Annual Promotional Calendar](#2-annual-promotional-calendar)
3. [Discount Logic & Rules](#3-discount-logic--rules)
4. [Affiliate Program](#4-affiliate-program)
5. [Gift Program](#5-gift-program)
6. [Enterprise & Education](#6-enterprise--education)
7. [Stripe Implementation Requirements](#7-stripe-implementation-requirements)
8. [Financial Dashboard Requirements](#8-financial-dashboard-requirements)
9. [Legal & Compliance](#9-legal--compliance)

---

## 1. CORE PRICING STRUCTURE

> 🔒 **LOCKED PRICING** — See `PRICING_LOCKED.md` for canonical source

### 1.1 Standard Pricing (Full Price)

| Plan | Price | Billing | Per Day Cost | Stripe Price ID |
|------|-------|---------|--------------|-----------------|
| **Annual** | $49.99/year | Recurring yearly | $0.14/day | `STRIPE_PRICE_ANNUAL` |
| **Monthly** | $7.99/month | Recurring monthly | $0.27/day | `STRIPE_PRICE_MONTHLY` |
| **Family** | $99.99/year | Recurring yearly | $0.27/day | `STRIPE_PRICE_FAMILY` |
| **Lifetime** | $199.99 | One-time forever | - | `STRIPE_PRICE_LIFETIME` |

### 1.2 Gift Pricing

| Gift | Price | Duration | Stripe Price ID |
|------|-------|----------|-----------------|
| **3 Months** | $24.99 | 3 months | `STRIPE_PRICE_GIFT_3MO` |
| **6 Months** | $39.99 | 6 months | `STRIPE_PRICE_GIFT_6MO` |
| **12 Months** | $49.99 | 12 months | `STRIPE_PRICE_GIFT_12MO` |
| **Lifetime** | $149.99 | Forever | `STRIPE_PRICE_GIFT_LIFETIME` |

### 1.3 Value Proposition by Plan

| Plan | Best For | Savings vs Monthly |
|------|----------|-------------------|
| Annual | Committed learners | Save 48% ($4.17/mo equiv) |
| Monthly | Try before commit | Flexibility |
| Family | Households | Up to 6 members |
| Lifetime | Super fans, gift givers | Break-even at 4 years |

### 1.3 Pricing Philosophy

1. **Annual is the hero** - Push annual as default, highest LTV
2. **Monthly is the gateway** - Low friction entry, convert to annual later
3. **Lifetime is premium** - Limited availability, creates urgency
4. **Never discount below 50%** - Protects brand value
5. **Always show original price** - Anchoring psychology

---

## 2. ANNUAL PROMOTIONAL CALENDAR

### 2.1 Major Sales Events (Tier 1 - Maximum Discount)

| Event | Dates | Discount | Code | Rationale |
|-------|-------|----------|------|-----------|
| **Launch Week** | Dec 17-24, 2025 | 30% off Annual | `LAUNCH30` | First mover reward |
| **New Year New You** | Dec 26 - Jan 7 | 25% off All | `NEWYEAR25` | Resolution season |
| **Back to School** | Aug 1-31 | 30% off Annual | `BACKTOSCHOOL` | Education timing |
| **Black Friday** | Nov 24-30 | 40% off Annual | `BLACKFRIDAY` | Biggest sale of year |
| **Cyber Monday** | Dec 1-3 | 40% off Annual | `CYBERMONDAY` | Extended Black Friday |

### 2.2 Seasonal Events (Tier 2 - Moderate Discount)

| Event | Dates | Discount | Code | Rationale |
|-------|-------|----------|------|-----------|
| **Valentine's Day** | Feb 10-14 | 20% off Gift | `LOVEOFLEARING` | Gift for loved ones |
| **Spring Learning** | Mar 20 - Apr 3 | 20% off Annual | `SPRING20` | Spring cleaning = new habits |
| **Mother's Day** | May 1-12 | 25% off Gift | `FORMOM` | Gift positioning |
| **Father's Day** | Jun 1-16 | 25% off Gift | `FORDAD` | Gift positioning |
| **Summer Brain** | Jun 15 - Jul 15 | 20% off Annual | `SUMMERBRAIN` | Prevent summer slide |
| **Grandparents Day** | Sep 1-8 | 25% off Gift | `GRANDPARENTS` | Intergenerational learning |
| **Halloween** | Oct 25-31 | 15% off | `SPOOKYLEARN` | Fun, engagement |
| **Thanksgiving** | Nov 20-23 | 20% off | `THANKFUL` | Gratitude theme |

### 2.3 Monthly Micro-Events (Tier 3 - Light Discount)

| Month | Theme | Discount | Code | Lesson Tie-In |
|-------|-------|----------|------|---------------|
| January | Goal Setting | 15% off | `GOALS` | Day 1-7: Goal lessons |
| February | Love of Learning | 15% off | `CURIOUS` | Heart/emotion lessons |
| March | Women's History | 15% off | `SHEROES` | Women in science |
| April | Earth Day (Apr 22) | 15% off | `EARTHDAY` | Environment lessons |
| May | Mental Health | 15% off | `MINDFUL` | Psychology lessons |
| June | Pride Month | 15% off | `PRIDE` | Diversity lessons |
| July | Independence | 15% off | `FREEDOM` | History lessons |
| August | Back to School | See Tier 1 | - | - |
| September | Literacy Month | 15% off | `READERS` | Reading lessons |
| October | Science Month | 15% off | `SCIENCE` | STEM lessons |
| November | Gratitude | See Tier 1/2 | - | - |
| December | Holiday/Launch | See Tier 1 | - | - |

### 2.4 Calendar Visualization

```
JAN  FEB  MAR  APR  MAY  JUN  JUL  AUG  SEP  OCT  NOV  DEC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
████                                                  ████  ← Tier 1 (25-40%)
     ██        ██   ██                           ██        ← Tier 2 (20-25%)
░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░      ░░░░ ░░░░          ← Tier 3 (15%)
                                   ████████                ← Back to School
```

---

## 3. DISCOUNT LOGIC & RULES

### 3.1 Discount Tiers

| Tier | Discount Range | When to Use | Approval |
|------|----------------|-------------|----------|
| **Tier 1** | 25-40% | Major holidays, launch | CEO |
| **Tier 2** | 20-25% | Seasonal events, gifts | Marketing |
| **Tier 3** | 10-15% | Monthly themes, retention | Automated |
| **Tier 4** | 5-10% | Abandoned cart, win-back | Automated |

### 3.2 Stacking Rules

| Rule | Policy |
|------|--------|
| Multiple promo codes | ❌ NOT allowed - one code per transaction |
| Promo + Affiliate | ❌ NOT allowed - affiliate OR promo, not both |
| Promo + Gift | ✅ ALLOWED - gifts can use promo codes |
| Promo + Enterprise | ❌ NOT allowed - enterprise has custom pricing |
| Student discount + Promo | ❌ NOT allowed - choose one |

### 3.3 Exclusions

These are NEVER discounted:
- Lifetime plan during Black Friday (already premium)
- First month of monthly plan below $4.99
- Enterprise contracts
- Bulk gift purchases (50+) - custom pricing

### 3.4 Discount Code Format

```
Format: [EVENT][DISCOUNT][YEAR]
Examples:
- LAUNCH30 (Launch, 30% off)
- BFCM40 (Black Friday/Cyber Monday, 40% off)
- SPRING2026 (Spring 2026 promo)

Affiliate Format: [CREATOR][DISCOUNT]
Examples:
- KELLY20 (Generic affiliate, 20% off)
- TEACHER15 (Teacher affiliate, 15% off)
```

### 3.5 Expiration Logic

| Code Type | Expiration |
|-----------|------------|
| Event codes | End of event + 48 hours grace |
| Affiliate codes | Ongoing until deactivated |
| Win-back codes | 7 days from send |
| Abandoned cart | 24 hours from send |

---

## 4. AFFILIATE PROGRAM

### 4.1 Program Overview

**Name:** Curious Kelly Ambassadors
**Commission Model:** Revenue share + discount for audience

### 4.2 Affiliate Tiers

| Tier | Requirements | Commission | Audience Discount | Payout |
|------|--------------|------------|-------------------|--------|
| **Starter** | Apply & approved | 15% | 10% off | Monthly, $50 min |
| **Partner** | 25+ conversions/mo | 20% | 15% off | Monthly, $25 min |
| **Ambassador** | 100+ conversions/mo | 25% | 20% off | Weekly, no min |
| **Champion** | 500+ conversions/mo | 30% | 25% off | Weekly, no min |

### 4.3 Affiliate Categories

| Category | Target Affiliates | Special Terms |
|----------|-------------------|---------------|
| **Educators** | Teachers, professors, tutors | 25% commission, verified .edu |
| **Parents** | Mommy bloggers, family channels | 20% commission |
| **Creators** | YouTubers, TikTokers, podcasters | Standard tiers |
| **Institutions** | Schools, libraries, community centers | Custom bulk pricing |
| **Influencers** | 100K+ followers | Negotiate case-by-case |

### 4.4 Affiliate Tracking

| Metric | How Tracked |
|--------|-------------|
| Clicks | UTM parameters + Stripe metadata |
| Conversions | Promo code + cookie (30-day window) |
| Revenue | Stripe subscription data |
| Churn | Monthly cohort analysis |

### 4.5 Affiliate Payouts

| Item | Policy |
|------|--------|
| Payment method | PayPal, direct deposit, or Stripe Connect |
| Payment schedule | Monthly (1st-5th of following month) |
| Minimum payout | Tier-dependent (see 4.2) |
| Chargebacks | Deducted from next payout |
| Fraud | Immediate termination, forfeit pending |

### 4.6 Affiliate Resources Provided

- Unique tracking link
- Custom promo code
- Brand assets (logos, images, Kelly renders)
- Sample social posts
- Email templates
- Performance dashboard access
- Monthly newsletter with tips

---

## 5. GIFT PROGRAM

### 5.1 Gift Products

| Gift Type | Price | Recipient Gets | Giver Gets |
|-----------|-------|----------------|------------|
| **1 Year Gift** | $99.99 | 12 months access | Gift receipt, thank you |
| **6 Month Gift** | $59.99 | 6 months access | Gift receipt |
| **3 Month Gift** | $34.99 | 3 months access | Gift receipt |
| **Lifetime Gift** | $299.99 | Forever access | Gift receipt, thank you |

### 5.2 Gift Flow

```
1. Giver purchases gift on curiouskelly.com/gifts
2. Giver enters recipient email + optional message
3. Giver selects delivery date (immediate or future)
4. Stripe processes payment
5. Gift code generated
6. Email sent to recipient on delivery date
7. Recipient redeems code, creates account
8. Giver notified of redemption
```

### 5.3 Gift Seasonality

| Season | Gift Revenue Target | Marketing Focus |
|--------|---------------------|-----------------|
| **Christmas (Dec)** | 40% of annual gifts | "Gift of learning" |
| **Mother's Day (May)** | 15% of annual gifts | "For the curious mom" |
| **Father's Day (Jun)** | 10% of annual gifts | "Dad's daily lesson" |
| **Back to School (Aug)** | 15% of annual gifts | "Start the year curious" |
| **Birthdays (ongoing)** | 20% of annual gifts | "Age-appropriate learning" |

### 5.4 Gift Promotions

| Event | Gift Discount | Bonus |
|-------|---------------|-------|
| Christmas Week | 20% off | Free gift wrap email |
| Mother's Day | 25% off | Personalized video from Kelly |
| Black Friday | 30% off | 2 bonus months |
| Valentine's Day | 20% off | Couples bundle (2 for $149) |

### 5.5 Gift Card / Credit System

| Feature | Implementation |
|---------|----------------|
| Gift cards | $25, $50, $100, custom amounts |
| Store credit | Issued for refunds, promotions |
| Expiration | Gift codes: 1 year from purchase |
| Transfer | Gift codes transferable until redeemed |

---

## 6. ENTERPRISE & EDUCATION

### 6.1 Enterprise Tiers

| Tier | Seats | Price/Seat/Year | Features |
|------|-------|-----------------|----------|
| **Team** | 5-25 | $79/seat | Admin dashboard, usage reports |
| **Business** | 26-100 | $69/seat | + SSO, API access |
| **Enterprise** | 101-500 | $59/seat | + Custom content, SLA |
| **Unlimited** | 500+ | Custom | Full customization |

### 6.2 Education Pricing

| Segment | Discount | Verification |
|---------|----------|--------------|
| **K-12 Students** | 50% off | .edu email or ID |
| **College Students** | 40% off | .edu email or SheerID |
| **Teachers** | 50% off | .edu email or ID |
| **Schools (bulk)** | 60% off | Purchase order |
| **Libraries** | Free (sponsored) | Library verification |

### 6.3 Education Sales Cycle

```
Q1 (Jan-Mar): Budget planning season → Outreach to districts
Q2 (Apr-Jun): Pilot programs → Free trials for schools
Q3 (Jul-Sep): Back to school → Close deals, onboarding
Q4 (Oct-Dec): Renewals → Retention focus
```

### 6.4 Enterprise Sales Process

| Stage | Action | Owner |
|-------|--------|-------|
| Lead | Inbound form or outbound | Marketing |
| Qualify | Needs assessment call | Sales |
| Demo | Live Kelly demo | Sales |
| Proposal | Custom pricing | Sales |
| Negotiate | Contract terms | Sales + Legal |
| Close | Signed contract | Sales |
| Onboard | Implementation | Customer Success |
| Expand | Upsell more seats | Customer Success |

---

## 7. STRIPE IMPLEMENTATION REQUIREMENTS

### 7.1 Products to Create in Stripe

```
PRODUCTS:
├── curious_kelly_annual
│   ├── price_annual_standard ($99.99/year)
│   ├── price_annual_launch30 ($69.99/year) - Launch promo
│   ├── price_annual_bf40 ($59.99/year) - Black Friday
│   └── price_annual_student ($49.99/year) - Student
│
├── curious_kelly_monthly
│   ├── price_monthly_standard ($9.99/month)
│   └── price_monthly_promo ($7.99/month) - Promotional
│
├── curious_kelly_lifetime
│   └── price_lifetime ($299.99 one-time)
│
├── curious_kelly_gift_12mo
│   └── price_gift_12mo ($99.99 one-time)
│
├── curious_kelly_gift_6mo
│   └── price_gift_6mo ($59.99 one-time)
│
├── curious_kelly_gift_3mo
│   └── price_gift_3mo ($34.99 one-time)
│
└── curious_kelly_enterprise
    └── Custom pricing per contract
```

### 7.2 Coupon Structure in Stripe

```
COUPONS:
├── Percentage Off
│   ├── LAUNCH30 (30% off, expires Dec 24, 2025)
│   ├── NEWYEAR25 (25% off, expires Jan 7, 2026)
│   ├── BLACKFRIDAY (40% off, Nov 24-30 annually)
│   ├── BACKTOSCHOOL (30% off, Aug 1-31 annually)
│   └── [AFFILIATE]_[XX] (variable %, ongoing)
│
├── Fixed Amount Off
│   ├── FIRST5OFF ($5 off first month)
│   └── GIFT20OFF ($20 off gift purchase)
│
└── Free Trial
    ├── TRIAL7 (7-day free trial)
    └── TRIAL14 (14-day free trial, educators only)
```

### 7.3 Metadata Requirements

Every Stripe transaction must include:

```json
{
  "metadata": {
    "source": "web|ios|android|api",
    "campaign": "launch|blackfriday|organic|affiliate",
    "affiliate_code": "CREATOR123|null",
    "promo_code": "LAUNCH30|null",
    "gift_recipient": "email@example.com|null",
    "gift_sender": "email@example.com|null",
    "user_age": "25",
    "utm_source": "google|facebook|twitter|direct",
    "utm_medium": "cpc|organic|social|email",
    "utm_campaign": "launch_2025|holiday_2025"
  }
}
```

### 7.4 Webhook Events to Handle

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Create user, send welcome email |
| `invoice.paid` | Update subscription status |
| `invoice.payment_failed` | Send dunning email, retry |
| `customer.subscription.updated` | Update plan in database |
| `customer.subscription.deleted` | Mark churned, send win-back |
| `charge.refunded` | Update records, revoke access |
| `coupon.created` | Sync to promo code database |

### 7.5 Billing Portal Requirements

Users should be able to:
- View current plan and billing date
- Update payment method
- View invoice history
- Download receipts
- Cancel subscription (with retention flow)
- Upgrade/downgrade plan
- Apply promo code to existing subscription
- Pause subscription (up to 3 months)

---

## 8. FINANCIAL DASHBOARD REQUIREMENTS

### 8.1 Real-Time Metrics (Update Every Minute)

| Metric | Definition | Target |
|--------|------------|--------|
| **MRR** | Monthly Recurring Revenue | Track growth |
| **ARR** | Annual Recurring Revenue | Board reporting |
| **Active Subscribers** | Paid users with active subscription | Track growth |
| **Trial Users** | Users in free trial | Convert 30%+ |
| **Today's Revenue** | Revenue collected today | Daily target |
| **Today's Signups** | New subscriptions today | Daily target |

### 8.2 Daily Metrics

| Metric | Definition | Benchmark |
|--------|------------|-----------|
| New Trials | Trial starts | 50/day |
| Trial Conversions | Trial → Paid | 30% |
| New Paid | Direct to paid | 20/day |
| Churned | Cancellations | <5%/mo |
| Upgrades | Monthly → Annual | 10%/mo |
| Downgrades | Annual → Monthly | <2%/mo |
| Refunds | Refund requests | <2% |

### 8.3 Cohort Analysis

Track by:
- **Acquisition Month** - When did they sign up?
- **Acquisition Channel** - How did they find us?
- **Plan Type** - Annual vs Monthly vs Lifetime
- **Promo Used** - Which discount code?
- **Affiliate** - Which affiliate referred them?

### 8.4 Affiliate Dashboard

| Metric | For Affiliates | For Admin |
|--------|----------------|-----------|
| Clicks | Their link clicks | All affiliate clicks |
| Conversions | Their conversions | All conversions by affiliate |
| Revenue | Their attributed revenue | Total affiliate revenue |
| Commission | Their pending/paid | Total commission liability |
| Conversion Rate | Their CVR | Average CVR |

### 8.5 Promotion Performance

| Metric | Track |
|--------|-------|
| Code Usage | How many times used |
| Revenue Impact | Revenue with code vs without |
| Conversion Lift | CVR with promo vs baseline |
| Discount Cost | Total discount given |
| ROI | (Revenue - Discount) / Marketing Spend |

### 8.6 Dashboard Views

1. **Executive Summary** - MRR, ARR, growth, churn
2. **Daily Operations** - Today's numbers, alerts
3. **Promotions** - Active promos, performance
4. **Affiliates** - Top performers, payouts due
5. **Cohorts** - Retention curves, LTV
6. **Forecasting** - Projections, targets

---

## 9. LEGAL & COMPLIANCE

### 9.1 Pricing Display Requirements

| Requirement | Implementation |
|-------------|----------------|
| Show original price | Always display full price with strikethrough |
| Clear discount terms | "30% off for new subscribers" |
| Expiration visible | "Ends December 24, 2025" |
| Auto-renewal disclosure | "Renews at $99.99/year" |
| Cancel anytime | "Cancel anytime from your account" |

### 9.2 Refund Policy

| Timeframe | Policy |
|-----------|--------|
| Within 7 days | Full refund, no questions |
| 8-30 days | Prorated refund |
| After 30 days | No refund, can cancel future |
| Lifetime plan | 30-day full refund only |
| Gifts | Non-refundable after redemption |

### 9.3 Geographic Pricing

| Region | Currency | Price Adjustment |
|--------|----------|------------------|
| USA | USD | Base price |
| Canada | CAD | +5% |
| UK | GBP | Parity |
| EU | EUR | Parity |
| Australia | AUD | +10% |
| India | INR | -50% (PPP) |
| Brazil | BRL | -40% (PPP) |
| Other | USD | Base price |

### 9.4 Tax Handling

| Region | Tax | Implementation |
|--------|-----|----------------|
| USA | Sales tax by state | Stripe Tax |
| EU | VAT | Stripe Tax |
| UK | VAT 20% | Stripe Tax |
| Canada | GST/HST | Stripe Tax |
| Australia | GST 10% | Stripe Tax |

### 9.5 GDPR / Privacy

- Store only necessary billing data
- Allow data export on request
- Allow data deletion on request
- Clear consent for marketing emails
- Separate consent for affiliate tracking

---

## 10. APPENDICES

### Appendix A: Promo Code Master List

| Code | Discount | Valid Dates | Notes |
|------|----------|-------------|-------|
| LAUNCH30 | 30% | Dec 17-24, 2025 | Launch week |
| NEWYEAR25 | 25% | Dec 26 - Jan 7 | New year |
| CURIOUS15 | 15% | Ongoing | Generic fallback |
| TEACHER50 | 50% | Ongoing | Verified teachers |
| STUDENT40 | 40% | Ongoing | Verified students |
| BLACKFRIDAY | 40% | Nov 24-30 | Annual event |
| CYBERMONDAY | 40% | Dec 1-3 | Annual event |
| BACKTOSCHOOL | 30% | Aug 1-31 | Annual event |

### Appendix B: Affiliate Application Criteria

**Approved if:**
- 1,000+ social followers OR
- Active blog with 5,000+ monthly visitors OR
- Email list of 500+ subscribers OR
- Educator with verifiable credentials OR
- Referred by existing Ambassador

**Rejected if:**
- No verifiable audience
- Content violates brand guidelines
- History of affiliate fraud
- Competitor or conflicting business

### Appendix C: Win-Back Email Sequence

| Day | Email | Offer |
|-----|-------|-------|
| 0 | "We're sorry to see you go" | None |
| 7 | "Kelly misses you" | 20% off to return |
| 14 | "Your learning streak" | 25% off to return |
| 30 | "Last chance" | 30% off to return |
| 90 | "It's been a while" | 40% off to return |

### Appendix D: Abandoned Cart Sequence

| Trigger | Timing | Offer |
|---------|--------|-------|
| Cart abandoned | +1 hour | Reminder, no discount |
| Still abandoned | +24 hours | 10% off |
| Still abandoned | +48 hours | 15% off, urgency |
| Still abandoned | +7 days | 20% off, final |

---

## DOCUMENT CONTROL

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Nov 26, 2025 | CEO | Initial creation |
| | | | |

---

## APPROVAL

| Role | Name | Signature | Date |
|------|------|-----------|------|
| CEO | | | |
| CFO | | | |
| Head of Marketing | | | |
| Head of Engineering | | | |

---

**This document is the single source of truth for all pricing, promotions, and revenue operations at Curious Kelly PBC.**

*Questions? Contact revenue@curiouskelly.com*

