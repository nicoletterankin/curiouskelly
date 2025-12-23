# 💳 STRIPE I18N BACKLOG — Complete Checklist

**Created:** December 23, 2025  
**Status:** Ready to Execute  
**Reference:** `docs/billing/STRIPE_BATCH_WORK.md` (full guide)

---

## 📊 OVERVIEW

**Total Work:** 45-60 minutes  
**Total Prices:** 36 new prices  
**Total Payment Methods:** 18 methods  
**Total Environment Variables:** 36 vars

---

## ✅ PART 1: CREATE MULTI-CURRENCY PRICES (20 min)

### EUR Prices (4 prices)
- [ ] Monthly: €7.99/mo → `price_monthly_eur`
- [ ] Annual: €49.99/yr → `price_annual_eur`
- [ ] Family: €99.99/yr → `price_family_eur`
- [ ] Lifetime: €199.99 → `price_lifetime_eur`

### GBP Prices (4 prices)
- [ ] Monthly: £6.99/mo → `price_monthly_gbp`
- [ ] Annual: £44.99/yr → `price_annual_gbp`
- [ ] Family: £89.99/yr → `price_family_gbp`
- [ ] Lifetime: £179.99 → `price_lifetime_gbp`

### CAD Prices (4 prices)
- [ ] Monthly: $10.99/mo → `price_monthly_cad`
- [ ] Annual: $69.99/yr → `price_annual_cad`
- [ ] Family: $139.99/yr → `price_family_cad`
- [ ] Lifetime: $279.99 → `price_lifetime_cad`

### AUD Prices (4 prices)
- [ ] Monthly: $12.99/mo → `price_monthly_aud`
- [ ] Annual: $79.99/yr → `price_annual_aud`
- [ ] Family: $159.99/yr → `price_family_aud`
- [ ] Lifetime: $319.99 → `price_lifetime_aud`

### INR Prices — PPP (3 prices)
- [ ] Monthly: ₹299/mo → `price_monthly_inr`
- [ ] Annual: ₹1,999/yr → `price_annual_inr`
- [ ] Lifetime: ₹7,999 → `price_lifetime_inr`

### BRL Prices — PPP (3 prices)
- [ ] Monthly: R$29.99/mo → `price_monthly_brl`
- [ ] Annual: R$199.99/yr → `price_annual_brl`
- [ ] Lifetime: R$799.99 → `price_lifetime_brl`

### MXN Prices — PPP (3 prices)
- [ ] Monthly: MX$99/mo → `price_monthly_mxn`
- [ ] Annual: MX$699/yr → `price_annual_mxn`
- [ ] Lifetime: MX$2,999 → `price_lifetime_mxn`

### PLN Prices — PPP (3 prices)
- [ ] Monthly: zł29.99/mo → `price_monthly_pln`
- [ ] Annual: zł199.99/yr → `price_annual_pln`
- [ ] Lifetime: zł799.99 → `price_lifetime_pln`

### EUR Gift Prices (4 prices)
- [ ] 3 Month: €24.99 → `price_gift_3mo_eur`
- [ ] 6 Month: €39.99 → `price_gift_6mo_eur`
- [ ] 12 Month: €49.99 → `price_gift_12mo_eur`
- [ ] Lifetime: €149.99 → `price_gift_lifetime_eur`

### GBP Gift Prices (4 prices)
- [ ] 3 Month: £21.99 → `price_gift_3mo_gbp`
- [ ] 6 Month: £34.99 → `price_gift_6mo_gbp`
- [ ] 12 Month: £44.99 → `price_gift_12mo_gbp`
- [ ] Lifetime: £129.99 → `price_gift_lifetime_gbp`

**Subtotal: 36 prices**

---

## ✅ PART 2: ENABLE PAYMENT METHODS (10 min)

### Europe
- [ ] iDEAL (Netherlands)
- [ ] Bancontact (Belgium)
- [ ] SEPA Direct Debit (EU)
- [ ] giropay (Germany)
- [ ] Sofort (Germany/Austria)
- [ ] EPS (Austria)
- [ ] P24 (Poland)
- [ ] BLIK (Poland)
- [ ] Klarna (EU)

### UK
- [ ] Bacs Direct Debit

### Americas
- [ ] ACH Direct Debit (US)
- [ ] Cash App Pay (US)
- [ ] Affirm (US)
- [ ] OXXO (Mexico)
- [ ] Boleto (Brazil)
- [ ] PIX (Brazil)

### Asia Pacific
- [ ] UPI (India)
- [ ] Konbini (Japan)

### Global
- [ ] Link (Stripe 1-click)
- [ ] Apple Pay (usually auto-enabled)
- [ ] Google Pay (usually auto-enabled)

**Subtotal: 18 payment methods**

---

## ✅ PART 3: COLLECT PRICE IDs (5 min)

After creating prices, copy all Price IDs (format: `price_1ABC...`) to a text file.

**Template:**
```
EUR:
STRIPE_PRICE_MONTHLY_EUR=price_xxx
STRIPE_PRICE_ANNUAL_EUR=price_xxx
STRIPE_PRICE_FAMILY_EUR=price_xxx
STRIPE_PRICE_LIFETIME_EUR=price_xxx
STRIPE_PRICE_GIFT_3MO_EUR=price_xxx
STRIPE_PRICE_GIFT_6MO_EUR=price_xxx
STRIPE_PRICE_GIFT_12MO_EUR=price_xxx
STRIPE_PRICE_GIFT_LIFETIME_EUR=price_xxx

GBP:
STRIPE_PRICE_MONTHLY_GBP=price_xxx
STRIPE_PRICE_ANNUAL_GBP=price_xxx
STRIPE_PRICE_FAMILY_GBP=price_xxx
STRIPE_PRICE_LIFETIME_GBP=price_xxx
STRIPE_PRICE_GIFT_3MO_GBP=price_xxx
STRIPE_PRICE_GIFT_6MO_GBP=price_xxx
STRIPE_PRICE_GIFT_12MO_GBP=price_xxx
STRIPE_PRICE_GIFT_LIFETIME_GBP=price_xxx

CAD:
STRIPE_PRICE_MONTHLY_CAD=price_xxx
STRIPE_PRICE_ANNUAL_CAD=price_xxx
STRIPE_PRICE_FAMILY_CAD=price_xxx
STRIPE_PRICE_LIFETIME_CAD=price_xxx

AUD:
STRIPE_PRICE_MONTHLY_AUD=price_xxx
STRIPE_PRICE_ANNUAL_AUD=price_xxx
STRIPE_PRICE_FAMILY_AUD=price_xxx
STRIPE_PRICE_LIFETIME_AUD=price_xxx

INR:
STRIPE_PRICE_MONTHLY_INR=price_xxx
STRIPE_PRICE_ANNUAL_INR=price_xxx
STRIPE_PRICE_LIFETIME_INR=price_xxx

BRL:
STRIPE_PRICE_MONTHLY_BRL=price_xxx
STRIPE_PRICE_ANNUAL_BRL=price_xxx
STRIPE_PRICE_LIFETIME_BRL=price_xxx

MXN:
STRIPE_PRICE_MONTHLY_MXN=price_xxx
STRIPE_PRICE_ANNUAL_MXN=price_xxx
STRIPE_PRICE_LIFETIME_MXN=price_xxx

PLN:
STRIPE_PRICE_MONTHLY_PLN=price_xxx
STRIPE_PRICE_ANNUAL_PLN=price_xxx
STRIPE_PRICE_LIFETIME_PLN=price_xxx
```

---

## ✅ PART 4: ADD ENVIRONMENT VARIABLES (10 min)

**Location:** Vercel Dashboard → Settings → Environment Variables

**Add all 36 variables from Part 3.**

**Important:** Set for **Production** environment (and Preview if desired).

---

## ✅ PART 5: TEST (5 min)

### Test Geo-Pricing API
```
https://www.curiouskelly.com/api/geo-pricing?force_country=DE
https://www.curiouskelly.com/api/geo-pricing?force_country=GB
https://www.curiouskelly.com/api/geo-pricing?force_country=IN
https://www.curiouskelly.com/api/geo-pricing?force_country=BR
```

**Verify:**
- ✅ Correct currency returned
- ✅ Correct prices shown
- ✅ PPP badge appears for IN, BR, MX, PL

### Test Checkout Flow
1. Use VPN or `?force_country=DE` to simulate Germany
2. Go to pricing page
3. Click "Subscribe"
4. Verify EUR prices appear
5. Verify iDEAL payment option appears

---

## 📋 SUMMARY CHECKLIST

### Prices Created
- [ ] EUR: 8 prices (4 plans + 4 gifts)
- [ ] GBP: 8 prices (4 plans + 4 gifts)
- [ ] CAD: 4 prices
- [ ] AUD: 4 prices
- [ ] INR: 3 prices (PPP)
- [ ] BRL: 3 prices (PPP)
- [ ] MXN: 3 prices (PPP)
- [ ] PLN: 3 prices (PPP)

**Total: 36 prices**

### Payment Methods Enabled
- [ ] 9 European methods
- [ ] 1 UK method
- [ ] 6 Americas methods
- [ ] 2 Asia Pacific methods
- [ ] 3 Global methods

**Total: 18 payment methods**

### Environment Variables Added
- [ ] All 36 price IDs added to Vercel

---

## 🚀 WHEN COMPLETE

After completing all steps:
1. Redeploy Vercel project
2. Test checkout flow for each currency
3. Verify payment methods appear correctly
4. Update `I18N_UNIVERSAL_SYSTEM_DIRECTIVE.md` status

---

**"Every currency, every payment method, every learner."**

*Created: December 23, 2025*

