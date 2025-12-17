# 🔧 STRIPE BATCH WORK - Complete Setup Guide

**Date:** December 17, 2025  
**Estimated Time:** 45-60 minutes  
**Status:** Ready for you to execute

---

## 📋 Overview

This document contains ALL the manual Stripe Dashboard work needed for international pricing support. Complete everything in one session.

**Direct Links:**
- [Stripe Dashboard](https://dashboard.stripe.com)
- [Products](https://dashboard.stripe.com/products)
- [Payment Methods](https://dashboard.stripe.com/settings/payment_methods)
- [Webhooks](https://dashboard.stripe.com/webhooks)

---

## ✅ PART 1: Create Multi-Currency Prices (20 min)

### Step 1: Find Your Subscription Product

1. Go to **Products** → Find "Curious Kelly" subscription product
2. Click on it to open product details

### Step 2: Add EUR Prices

For each plan, click **"+ Add another price"** and create:

| Plan | Currency | Amount | Billing | Lookup Key (optional) |
|------|----------|--------|---------|----------------------|
| Monthly | EUR | €7.99 | Recurring monthly | `price_monthly_eur` |
| Annual | EUR | €49.99 | Recurring yearly | `price_annual_eur` |
| Family | EUR | €99.99 | Recurring yearly | `price_family_eur` |
| Lifetime | EUR | €199.99 | One time | `price_lifetime_eur` |

### Step 3: Add GBP Prices

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | GBP | £6.99 | Recurring monthly | `price_monthly_gbp` |
| Annual | GBP | £44.99 | Recurring yearly | `price_annual_gbp` |
| Family | GBP | £89.99 | Recurring yearly | `price_family_gbp` |
| Lifetime | GBP | £179.99 | One time | `price_lifetime_gbp` |

### Step 4: Add CAD Prices

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | CAD | $10.99 | Recurring monthly | `price_monthly_cad` |
| Annual | CAD | $69.99 | Recurring yearly | `price_annual_cad` |
| Family | CAD | $139.99 | Recurring yearly | `price_family_cad` |
| Lifetime | CAD | $279.99 | One time | `price_lifetime_cad` |

### Step 5: Add AUD Prices

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | AUD | $12.99 | Recurring monthly | `price_monthly_aud` |
| Annual | AUD | $79.99 | Recurring yearly | `price_annual_aud` |
| Family | AUD | $159.99 | Recurring yearly | `price_family_aud` |
| Lifetime | AUD | $319.99 | One time | `price_lifetime_aud` |

---

## ✅ PART 2: Create PPP Prices (15 min)

These are discounted prices for developing markets.

### Step 6: Add INR Prices (India - 50% PPP)

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | INR | ₹299 | Recurring monthly | `price_monthly_inr` |
| Annual | INR | ₹1,999 | Recurring yearly | `price_annual_inr` |
| Lifetime | INR | ₹7,999 | One time | `price_lifetime_inr` |

### Step 7: Add BRL Prices (Brazil - 40% PPP)

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | BRL | R$29.99 | Recurring monthly | `price_monthly_brl` |
| Annual | BRL | R$199.99 | Recurring yearly | `price_annual_brl` |
| Lifetime | BRL | R$799.99 | One time | `price_lifetime_brl` |

### Step 8: Add MXN Prices (Mexico - 35% PPP)

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | MXN | MX$99 | Recurring monthly | `price_monthly_mxn` |
| Annual | MXN | MX$699 | Recurring yearly | `price_annual_mxn` |
| Lifetime | MXN | MX$2,999 | One time | `price_lifetime_mxn` |

### Step 9: Add PLN Prices (Poland - 30% PPP)

| Plan | Currency | Amount | Billing | Lookup Key |
|------|----------|--------|---------|------------|
| Monthly | PLN | zł29.99 | Recurring monthly | `price_monthly_pln` |
| Annual | PLN | zł199.99 | Recurring yearly | `price_annual_pln` |
| Lifetime | PLN | zł799.99 | One time | `price_lifetime_pln` |

---

## ✅ PART 3: Create Gift Prices per Currency (10 min)

### Step 10: Add EUR Gift Prices

Go to your Gift products and add EUR versions:

| Gift | Currency | Amount | Lookup Key |
|------|----------|--------|------------|
| 3 Month Gift | EUR | €24.99 | `price_gift_3mo_eur` |
| 6 Month Gift | EUR | €39.99 | `price_gift_6mo_eur` |
| 12 Month Gift | EUR | €49.99 | `price_gift_12mo_eur` |
| Lifetime Gift | EUR | €149.99 | `price_gift_lifetime_eur` |

### Step 11: Add GBP Gift Prices

| Gift | Currency | Amount | Lookup Key |
|------|----------|--------|------------|
| 3 Month Gift | GBP | £21.99 | `price_gift_3mo_gbp` |
| 6 Month Gift | GBP | £34.99 | `price_gift_6mo_gbp` |
| 12 Month Gift | GBP | £44.99 | `price_gift_12mo_gbp` |
| Lifetime Gift | GBP | £129.99 | `price_gift_lifetime_gbp` |

---

## ✅ PART 4: Enable Payment Methods (10 min)

### Step 12: Go to Payment Methods Settings

1. Navigate to: **Settings** → **Payment methods**
2. Or direct link: https://dashboard.stripe.com/settings/payment_methods

### Step 13: Enable These Payment Methods

Check the box to enable each one:

#### Europe
- [x] **Cards** (already enabled)
- [ ] **iDEAL** - Netherlands (bank redirect)
- [ ] **Bancontact** - Belgium (bank redirect)
- [ ] **SEPA Direct Debit** - EU (bank transfers)
- [ ] **giropay** - Germany (bank redirect)
- [ ] **Sofort** - Germany/Austria (bank redirect)
- [ ] **EPS** - Austria (bank redirect)
- [ ] **P24** - Poland (bank redirect)
- [ ] **BLIK** - Poland (mobile)
- [ ] **Klarna** - EU (buy now pay later)

#### UK
- [ ] **Bacs Direct Debit** - UK (bank transfers)

#### Americas
- [ ] **ACH Direct Debit** - US (bank transfers)
- [ ] **Cash App Pay** - US
- [ ] **Affirm** - US (buy now pay later)
- [ ] **OXXO** - Mexico (cash voucher)
- [ ] **Boleto** - Brazil (bank slip)
- [ ] **PIX** - Brazil (instant transfer)

#### Asia Pacific
- [ ] **UPI** - India (instant payments)
- [ ] **Konbini** - Japan (convenience store)

#### Global
- [ ] **Link** - Stripe's 1-click checkout (HIGHLY RECOMMENDED)
- [ ] **Apple Pay** - (usually auto-enabled with cards)
- [ ] **Google Pay** - (usually auto-enabled with cards)

### Step 14: Configure Each Payment Method

For each enabled method:
1. Click on it to expand settings
2. Set to **"Automatic"** or **"On"**
3. For SEPA/Bacs: Complete any additional verification if required

---

## ✅ PART 5: Collect Price IDs (5 min)

### Step 15: Get All New Price IDs

After creating prices, go to each product and copy the Price IDs.

Format: `price_1ABC...`

Copy these to a text file for the next step.

---

## ✅ PART 6: Update Vercel Environment Variables (10 min)

### Step 16: Go to Vercel Dashboard

1. Open: https://vercel.com/dashboard
2. Select your project: **UI-TARS-desktop** or **curiouskelly**
3. Go to: **Settings** → **Environment Variables**

### Step 17: Add These Environment Variables

Add each price ID from Step 15:

```
# EUR Prices
STRIPE_PRICE_MONTHLY_EUR=price_xxx
STRIPE_PRICE_ANNUAL_EUR=price_xxx
STRIPE_PRICE_FAMILY_EUR=price_xxx
STRIPE_PRICE_LIFETIME_EUR=price_xxx
STRIPE_PRICE_GIFT_3MO_EUR=price_xxx
STRIPE_PRICE_GIFT_6MO_EUR=price_xxx
STRIPE_PRICE_GIFT_12MO_EUR=price_xxx
STRIPE_PRICE_GIFT_LIFETIME_EUR=price_xxx

# GBP Prices
STRIPE_PRICE_MONTHLY_GBP=price_xxx
STRIPE_PRICE_ANNUAL_GBP=price_xxx
STRIPE_PRICE_FAMILY_GBP=price_xxx
STRIPE_PRICE_LIFETIME_GBP=price_xxx
STRIPE_PRICE_GIFT_3MO_GBP=price_xxx
STRIPE_PRICE_GIFT_6MO_GBP=price_xxx
STRIPE_PRICE_GIFT_12MO_GBP=price_xxx
STRIPE_PRICE_GIFT_LIFETIME_GBP=price_xxx

# CAD Prices
STRIPE_PRICE_MONTHLY_CAD=price_xxx
STRIPE_PRICE_ANNUAL_CAD=price_xxx
STRIPE_PRICE_FAMILY_CAD=price_xxx
STRIPE_PRICE_LIFETIME_CAD=price_xxx

# AUD Prices
STRIPE_PRICE_MONTHLY_AUD=price_xxx
STRIPE_PRICE_ANNUAL_AUD=price_xxx
STRIPE_PRICE_FAMILY_AUD=price_xxx
STRIPE_PRICE_LIFETIME_AUD=price_xxx

# INR Prices (India PPP)
STRIPE_PRICE_MONTHLY_INR=price_xxx
STRIPE_PRICE_ANNUAL_INR=price_xxx
STRIPE_PRICE_LIFETIME_INR=price_xxx

# BRL Prices (Brazil PPP)
STRIPE_PRICE_MONTHLY_BRL=price_xxx
STRIPE_PRICE_ANNUAL_BRL=price_xxx
STRIPE_PRICE_LIFETIME_BRL=price_xxx

# MXN Prices (Mexico PPP)
STRIPE_PRICE_MONTHLY_MXN=price_xxx
STRIPE_PRICE_ANNUAL_MXN=price_xxx
STRIPE_PRICE_LIFETIME_MXN=price_xxx

# PLN Prices (Poland PPP)
STRIPE_PRICE_MONTHLY_PLN=price_xxx
STRIPE_PRICE_ANNUAL_PLN=price_xxx
STRIPE_PRICE_LIFETIME_PLN=price_xxx
```

### Step 18: Redeploy

After adding all environment variables:
1. Go to **Deployments** tab
2. Click **"Redeploy"** on the latest deployment
3. Select **"Redeploy with existing Build Cache"**

---

## ✅ PART 7: Test International Checkout (5 min)

### Step 19: Test Each Currency

Use this URL with different country codes to test:

```
https://www.curiouskelly.com/api/geo-pricing?force_country=DE
https://www.curiouskelly.com/api/geo-pricing?force_country=GB
https://www.curiouskelly.com/api/geo-pricing?force_country=IN
https://www.curiouskelly.com/api/geo-pricing?force_country=BR
```

Verify:
- Correct currency is returned
- Correct prices are shown
- PPP badge appears for IN, BR, MX, PL

### Step 20: Test Checkout Flow

1. Use VPN or `?force_country=DE` to simulate Germany
2. Go to pricing page
3. Click "Subscribe"
4. Verify EUR prices and iDEAL option appear

---

## 📊 Summary Checklist

### Prices Created
- [ ] EUR: Monthly, Annual, Family, Lifetime (4 prices)
- [ ] GBP: Monthly, Annual, Family, Lifetime (4 prices)
- [ ] CAD: Monthly, Annual, Family, Lifetime (4 prices)
- [ ] AUD: Monthly, Annual, Family, Lifetime (4 prices)
- [ ] INR: Monthly, Annual, Lifetime (3 prices)
- [ ] BRL: Monthly, Annual, Lifetime (3 prices)
- [ ] MXN: Monthly, Annual, Lifetime (3 prices)
- [ ] PLN: Monthly, Annual, Lifetime (3 prices)
- [ ] EUR Gifts: 3mo, 6mo, 12mo, Lifetime (4 prices)
- [ ] GBP Gifts: 3mo, 6mo, 12mo, Lifetime (4 prices)

**Total: 36 new prices**

### Payment Methods Enabled
- [ ] iDEAL (Netherlands)
- [ ] Bancontact (Belgium)
- [ ] SEPA Direct Debit (EU)
- [ ] giropay (Germany)
- [ ] Sofort (Germany/Austria)
- [ ] EPS (Austria)
- [ ] P24 (Poland)
- [ ] BLIK (Poland)
- [ ] Klarna (EU)
- [ ] Bacs Direct Debit (UK)
- [ ] ACH Direct Debit (US)
- [ ] Cash App Pay (US)
- [ ] Affirm (US)
- [ ] OXXO (Mexico)
- [ ] Boleto (Brazil)
- [ ] PIX (Brazil)
- [ ] UPI (India)
- [ ] Link (Global)

**Total: 18 payment methods**

### Environment Variables Added
- [ ] All EUR price IDs (8 vars)
- [ ] All GBP price IDs (8 vars)
- [ ] All CAD price IDs (4 vars)
- [ ] All AUD price IDs (4 vars)
- [ ] All INR price IDs (3 vars)
- [ ] All BRL price IDs (3 vars)
- [ ] All MXN price IDs (3 vars)
- [ ] All PLN price IDs (3 vars)

**Total: 36 environment variables**

---

## 🚀 When Complete

Tell me "done" and I'll verify the integration is working!

---

*Created: December 17, 2025*
