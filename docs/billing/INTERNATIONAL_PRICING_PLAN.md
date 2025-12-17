# 🌍 International Pricing & PPP Implementation Plan

**Created:** December 17, 2025  
**Status:** Approved for Implementation  
**Priority:** High (affects 65% of global internet users)

---

## Executive Summary

Currently, all Curious Kelly pricing is in USD only. This document outlines a phased approach to:
1. **Phase 1:** Add major currency support (EUR, GBP, CAD, AUD)
2. **Phase 2:** Implement Purchasing Power Parity (PPP) for developing markets
3. **Phase 3:** Add local payment methods

Expected impact: **2-3x increase in international conversions**.

---

## 📊 Current State

| Metric | Status |
|--------|--------|
| Currency | USD only |
| Tax | ✅ Stripe Tax enabled |
| PPP | ❌ Not implemented |
| Local payments | ❌ Cards only |
| Geo-detection | ❌ Not implemented |

---

## 🎯 Target Pricing by Region

### Phase 1: Major Currencies (Parity Pricing)

| Region | Currency | Monthly | Annual | Lifetime | Conversion Rate Assumption |
|--------|----------|---------|--------|----------|---------------------------|
| **USA** | USD | $7.99 | $49.99 | $199.99 | Base |
| **Eurozone** | EUR | €7.99 | €49.99 | €199.99 | ~parity |
| **UK** | GBP | £6.99 | £44.99 | £179.99 | ~parity |
| **Canada** | CAD | $10.99 | $69.99 | $269.99 | +5% effective |
| **Australia** | AUD | $12.99 | $79.99 | $319.99 | +10% effective |

### Phase 2: PPP Markets (Adjusted Pricing)

| Region | Currency | Monthly | Annual | Lifetime | Discount |
|--------|----------|---------|--------|----------|----------|
| **India** | INR | ₹299 | ₹1,999 | ₹7,999 | -50% |
| **Brazil** | BRL | R$29.99 | R$199.99 | R$799.99 | -40% |
| **Mexico** | MXN | MX$99 | MX$699 | MX$2,999 | -35% |
| **Poland** | PLN | zł29.99 | zł199.99 | zł799.99 | -30% |
| **Turkey** | TRY | ₺149 | ₺999 | ₺3,999 | -45% |
| **Indonesia** | IDR | Rp79,000 | Rp499,000 | Rp1,999,000 | -55% |
| **Philippines** | PHP | ₱299 | ₱1,999 | ₱7,999 | -45% |
| **South Africa** | ZAR | R99 | R699 | R2,999 | -40% |

### Gift Pricing (PPP Adjusted)

| Gift Duration | USD | INR (50% off) | BRL (40% off) |
|---------------|-----|---------------|---------------|
| 3 Months | $24.99 | ₹999 | R$99.99 |
| 6 Months | $39.99 | ₹1,499 | R$159.99 |
| 12 Months | $49.99 | ₹1,999 | R$199.99 |
| Lifetime | $149.99 | ₹5,999 | R$599.99 |

---

## 🔧 Technical Implementation

### Phase 1: Multi-Currency Support

#### Step 1.1: Create Stripe Products per Currency

```
Stripe Products Structure:
├── curious_kelly_subscription
│   ├── price_monthly_usd ($7.99/mo)
│   ├── price_monthly_eur (€7.99/mo)
│   ├── price_monthly_gbp (£6.99/mo)
│   ├── price_monthly_cad ($10.99/mo)
│   ├── price_monthly_aud ($12.99/mo)
│   ├── price_annual_usd ($49.99/yr)
│   ├── price_annual_eur (€49.99/yr)
│   ├── price_annual_gbp (£44.99/yr)
│   ├── price_annual_cad ($69.99/yr)
│   └── price_annual_aud ($79.99/yr)
│
├── curious_kelly_lifetime
│   ├── price_lifetime_usd ($199.99)
│   ├── price_lifetime_eur (€199.99)
│   ├── price_lifetime_gbp (£179.99)
│   ├── price_lifetime_cad ($269.99)
│   └── price_lifetime_aud ($319.99)
```

#### Step 1.2: Geo-Detection API

Create `/api/geo-pricing.ts`:

```typescript
import type { VercelRequest, VercelResponse } from '@vercel/node';

// Country to currency mapping
const COUNTRY_CURRENCY: Record<string, string> = {
  // Phase 1: Major currencies
  US: 'USD', CA: 'CAD', GB: 'GBP', AU: 'AUD',
  AT: 'EUR', BE: 'EUR', DE: 'EUR', ES: 'EUR', FI: 'EUR', FR: 'EUR',
  GR: 'EUR', IE: 'EUR', IT: 'EUR', LU: 'EUR', NL: 'EUR', PT: 'EUR',
  
  // Phase 2: PPP markets
  IN: 'INR', BR: 'BRL', MX: 'MXN', PL: 'PLN', TR: 'TRY',
  ID: 'IDR', PH: 'PHP', ZA: 'ZAR',
};

// Price IDs per currency and plan
const PRICE_IDS: Record<string, Record<string, string>> = {
  USD: {
    monthly: 'price_xxx_monthly_usd',
    annual: 'price_xxx_annual_usd',
    lifetime: 'price_xxx_lifetime_usd',
  },
  EUR: {
    monthly: 'price_xxx_monthly_eur',
    annual: 'price_xxx_annual_eur',
    lifetime: 'price_xxx_lifetime_eur',
  },
  // ... more currencies
};

// Display prices for UI
const DISPLAY_PRICES: Record<string, Record<string, string>> = {
  USD: { monthly: '$7.99', annual: '$49.99', lifetime: '$199.99', symbol: '$' },
  EUR: { monthly: '€7.99', annual: '€49.99', lifetime: '€199.99', symbol: '€' },
  GBP: { monthly: '£6.99', annual: '£44.99', lifetime: '£179.99', symbol: '£' },
  INR: { monthly: '₹299', annual: '₹1,999', lifetime: '₹7,999', symbol: '₹' },
  BRL: { monthly: 'R$29.99', annual: 'R$199.99', lifetime: 'R$799.99', symbol: 'R$' },
  // ... more currencies
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Get country from Vercel's geo headers
  const country = (req.headers['x-vercel-ip-country'] as string) || 'US';
  const currency = COUNTRY_CURRENCY[country] || 'USD';
  
  res.json({
    country,
    currency,
    prices: DISPLAY_PRICES[currency] || DISPLAY_PRICES.USD,
    priceIds: PRICE_IDS[currency] || PRICE_IDS.USD,
    isPPP: ['INR', 'BRL', 'MXN', 'PLN', 'TRY', 'IDR', 'PHP', 'ZAR'].includes(currency),
  });
}
```

#### Step 1.3: Frontend Integration

Update `pricing.html` and paywall to fetch geo-pricing:

```javascript
async function loadGeoP ricing() {
  try {
    const response = await fetch('/api/geo-pricing');
    const data = await response.json();
    
    // Update displayed prices
    document.querySelectorAll('[data-price="monthly"]').forEach(el => {
      el.textContent = data.prices.monthly;
    });
    document.querySelectorAll('[data-price="annual"]').forEach(el => {
      el.textContent = data.prices.annual;
    });
    document.querySelectorAll('[data-price="lifetime"]').forEach(el => {
      el.textContent = data.prices.lifetime;
    });
    
    // Store for checkout
    window.KELLY_GEO_PRICING = data;
  } catch (e) {
    console.warn('Geo-pricing failed, using USD');
  }
}

// Call on page load
loadGeoPricing();
```

#### Step 1.4: Checkout Integration

Update `/api/create-checkout.ts`:

```typescript
// Accept currency/priceId from frontend
const { plan, email, currency = 'USD' } = body;

// Get correct price ID for currency
const priceId = getPriceIdForCurrency(plan, currency);

const session = await stripe.checkout.sessions.create({
  mode: plan === 'lifetime' ? 'payment' : 'subscription',
  line_items: [{ price: priceId, quantity: 1 }],
  currency: currency.toLowerCase(), // For lifetime (one-time) purchases
  // ... rest of config
});
```

---

### Phase 2: PPP Implementation

#### Step 2.1: PPP Detection Logic

```typescript
// PPP discount rates by country
const PPP_DISCOUNTS: Record<string, number> = {
  IN: 0.50,  // India: 50% off
  BR: 0.40,  // Brazil: 40% off
  MX: 0.35,  // Mexico: 35% off
  PL: 0.30,  // Poland: 30% off
  TR: 0.45,  // Turkey: 45% off
  ID: 0.55,  // Indonesia: 55% off
  PH: 0.45,  // Philippines: 45% off
  ZA: 0.40,  // South Africa: 40% off
};

function getPPPDiscount(countryCode: string): number {
  return PPP_DISCOUNTS[countryCode] || 0;
}
```

#### Step 2.2: Create PPP-Specific Stripe Prices

For each PPP market, create dedicated prices in Stripe:

```
Manual Stripe Dashboard Work:

Product: Curious Kelly Annual
├── price_annual_inr (₹1,999/year) - India
├── price_annual_brl (R$199.99/year) - Brazil
├── price_annual_mxn (MX$699/year) - Mexico
├── price_annual_pln (zł199.99/year) - Poland
├── price_annual_try (₺999/year) - Turkey
├── price_annual_idr (Rp499,000/year) - Indonesia
├── price_annual_php (₱1,999/year) - Philippines
└── price_annual_zar (R699/year) - South Africa
```

#### Step 2.3: VPN/Abuse Prevention

```typescript
// Store user's initial country at signup
// If they VPN to a PPP country later, don't offer PPP pricing

async function shouldOfferPPP(userId: string, detectedCountry: string): Promise<boolean> {
  // Check if user signed up from a non-PPP country
  const { data: user } = await supabase
    .from('users')
    .select('signup_country, stripe_customer_id')
    .eq('id', userId)
    .single();
  
  // If existing customer from non-PPP country, don't offer PPP
  if (user?.signup_country && !isPPPCountry(user.signup_country)) {
    return false;
  }
  
  // If new user from PPP country, offer it
  if (isPPPCountry(detectedCountry)) {
    return true;
  }
  
  return false;
}
```

---

### Phase 3: Local Payment Methods

#### Stripe Payment Method Support

| Region | Payment Methods | Stripe Support |
|--------|-----------------|----------------|
| **EU** | iDEAL, Bancontact, SEPA, Sofort, giropay | ✅ Built-in |
| **UK** | Bacs Direct Debit | ✅ Built-in |
| **Brazil** | Boleto, Pix | ✅ Built-in |
| **India** | UPI, Netbanking | ✅ Built-in |
| **Mexico** | OXXO | ✅ Built-in |
| **Poland** | P24, BLIK | ✅ Built-in |

#### Enable in Checkout

```typescript
const session = await stripe.checkout.sessions.create({
  payment_method_types: getPaymentMethodsForCountry(country),
  // ...
});

function getPaymentMethodsForCountry(country: string): string[] {
  const methods = ['card']; // Always support cards
  
  switch (country) {
    case 'NL': methods.push('ideal'); break;
    case 'BE': methods.push('bancontact'); break;
    case 'DE': methods.push('giropay', 'sofort'); break;
    case 'BR': methods.push('boleto'); break;
    case 'IN': methods.push('upi'); break;
    case 'MX': methods.push('oxxo'); break;
    case 'PL': methods.push('p24', 'blik'); break;
  }
  
  // EU countries: add SEPA
  if (EU_COUNTRIES.includes(country)) {
    methods.push('sepa_debit');
  }
  
  return methods;
}
```

---

## 📅 Implementation Timeline

### Week 1: Phase 1 Foundation

| Day | Task | Owner |
|-----|------|-------|
| 1 | Create EUR, GBP, CAD, AUD prices in Stripe | You (manual) |
| 1 | Build `/api/geo-pricing` endpoint | Claude |
| 2 | Update pricing page with geo-detection | Claude |
| 2 | Update paywall with geo-detection | Claude |
| 3 | Update checkout APIs to accept currency | Claude |
| 3 | Update webhook to record currency | Claude |
| 4-5 | Testing: Simulate users from different countries | You |

### Week 2: Phase 2 PPP

| Day | Task | Owner |
|-----|------|-------|
| 1 | Create INR, BRL, MXN prices in Stripe | You (manual) |
| 2 | Add PPP detection to geo-pricing API | Claude |
| 2 | Add VPN abuse prevention | Claude |
| 3 | Create PPP landing pages (optional) | Claude |
| 4 | Add PPP badge to pricing ("Adjusted for your region") | Claude |
| 5 | Testing with VPN | You |

### Week 3: Phase 3 Local Payments

| Day | Task | Owner |
|-----|------|-------|
| 1 | Enable iDEAL, Bancontact in Stripe Dashboard | You (manual) |
| 2 | Enable Boleto, UPI, OXXO in Stripe Dashboard | You (manual) |
| 3 | Update checkout to use dynamic payment methods | Claude |
| 4-5 | Test each payment method | You |

---

## 💰 Revenue Impact Projection

### Current State (USD Only)
- Estimated international visitor conversion: 0.5%
- 65% of visitors are international

### Projected with Multi-Currency + PPP
| Region | Current CVR | Projected CVR | Lift |
|--------|-------------|---------------|------|
| EU/UK | 0.8% | 1.5% | +87% |
| Canada/Australia | 0.7% | 1.2% | +71% |
| India | 0.2% | 1.0% | +400% |
| Brazil | 0.3% | 0.9% | +200% |
| Other PPP | 0.3% | 0.8% | +167% |

**Conservative estimate: 2x international revenue within 3 months**

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| VPN abuse for PPP pricing | Store signup country, don't offer PPP to VPN users |
| Currency fluctuations | Review prices quarterly, update if >10% drift |
| Chargebacks on local payments | Boleto/OXXO have higher fraud - monitor closely |
| Accounting complexity | Use Stripe for all reporting, single source |
| Tax compliance | Stripe Tax handles VAT/GST automatically |

---

## 📋 Stripe Dashboard Checklist

### Phase 1: Create These Prices

- [ ] **Monthly EUR** - €7.99/month, recurring
- [ ] **Annual EUR** - €49.99/year, recurring
- [ ] **Lifetime EUR** - €199.99, one-time
- [ ] **Monthly GBP** - £6.99/month, recurring
- [ ] **Annual GBP** - £44.99/year, recurring
- [ ] **Lifetime GBP** - £179.99, one-time
- [ ] **Monthly CAD** - $10.99/month, recurring
- [ ] **Annual CAD** - $69.99/year, recurring
- [ ] **Lifetime CAD** - $269.99, one-time
- [ ] **Monthly AUD** - $12.99/month, recurring
- [ ] **Annual AUD** - $79.99/year, recurring
- [ ] **Lifetime AUD** - $319.99, one-time

### Phase 2: Create PPP Prices

- [ ] **Annual INR** - ₹1,999/year
- [ ] **Annual BRL** - R$199.99/year
- [ ] **Annual MXN** - MX$699/year
- [ ] **Annual PLN** - zł199.99/year
- [ ] **Annual TRY** - ₺999/year

### Phase 3: Enable Payment Methods

- [ ] Settings → Payment Methods → iDEAL
- [ ] Settings → Payment Methods → Bancontact
- [ ] Settings → Payment Methods → SEPA Direct Debit
- [ ] Settings → Payment Methods → giropay
- [ ] Settings → Payment Methods → Boleto
- [ ] Settings → Payment Methods → UPI
- [ ] Settings → Payment Methods → OXXO

---

## 🚀 Quick Start: What to Do Now

1. **I've enabled the paywall** in `config.js` ✅

2. **Next: Create EUR/GBP prices in Stripe**
   - Go to Stripe Dashboard → Products
   - Find `curious_kelly_subscription`
   - Click "Add another price"
   - Set currency to EUR, price €7.99, recurring monthly
   - Repeat for annual, lifetime, and other currencies

3. **Then tell me you're ready** and I'll build the geo-pricing API

---

*Plan created: December 17, 2025*
