# COPPA Compliance Audit Report

**Date:** December 4, 2025  
**Status:** 🔴 NOT COMPLIANT  
**Priority:** CRITICAL - Must fix before launch

---

## Executive Summary

Curious Kelly is an educational platform targeting ages 2-102, which explicitly includes children under 13. **COPPA (Children's Online Privacy Protection Act) applies** and we are currently **NOT compliant**.

### Current State

| Area | Status | Issue |
|------|--------|-------|
| Age Verification | ❌ Missing | Anyone can sign up without age check |
| Parental Consent | ❌ Missing | No VPC mechanism exists |
| Child Account Type | ❌ Missing | No way to identify child accounts |
| Privacy Policy | ⚠️ Partial | Good text, not implemented |
| Data Collection | ⚠️ Risk | OAuth collects email from everyone |
| Third-Party Tracking | ✅ Good | No GA/FB pixel found |
| Cookies | ⚠️ Review | Affiliate cookies set for all users |

---

## What COPPA Requires

### 1. Verifiable Parental Consent (VPC)
Before collecting ANY personal information from a child under 13, you must:
- Obtain verifiable consent from a parent/guardian
- Methods include: credit card verification, video call, signed consent form, government ID

### 2. Direct Notice to Parents
- Clear, complete notice of data practices
- Must be provided BEFORE collecting child data
- Separate from general privacy policy

### 3. Parental Access Rights
- Parents can review data collected about their child
- Parents can request deletion of child's data
- Parents can refuse further collection

### 4. Data Minimization
- Collect ONLY what's necessary for the activity
- Cannot condition participation on excessive data collection

### 5. Data Security
- Maintain reasonable procedures to protect child data
- Retention limits - delete when no longer needed

### 6. Third-Party Disclosure Limits
- Cannot share child data with third parties (except service providers under contract)

---

## Critical Gaps Identified

### Gap 1: No Age Gate (CRITICAL)
**Current:** Anyone can sign up via OAuth without being asked their age.

**Required:** Must ask age BEFORE signup and route users accordingly:
- Age 13+: Normal signup flow
- Age 2-12: Redirect to parent signup flow

**Fix Priority:** 🔴 BLOCKER

---

### Gap 2: No Parental Consent System (CRITICAL)
**Current:** Privacy policy states parents must create accounts for under-13, but there's no mechanism to enforce or verify this.

**Required:** Verifiable Parental Consent (VPC) before ANY data collection from children.

**Options for VPC:**
1. **Credit Card Verification** - Charge $0.50 and refund (proves adult)
2. **Knowledge-Based Authentication** - Answer questions from credit bureau
3. **Video Conference** - Verify parent identity via video
4. **Signed Consent Form** - Mail/fax/email signed form with ID
5. **Government ID Upload** - Verify parent with ID scan

**Recommended:** Credit card micro-charge (easiest to implement with Stripe)

**Fix Priority:** 🔴 BLOCKER

---

### Gap 3: OAuth Collects Email from Everyone (CRITICAL)
**Current:** Google/Apple/GitHub OAuth returns email for all users.

**Privacy Policy Says:** "We do NOT collect from children under 13: Email addresses"

**Problem:** We ARE collecting emails, we just don't know which are children.

**Fix Options:**
1. **Age gate BEFORE OAuth** - Only adult flow uses OAuth
2. **Child accounts without email** - Use parent's email as contact
3. **Delete child email immediately** - Store only display name

**Fix Priority:** 🔴 BLOCKER

---

### Gap 4: No Child Account Type (HIGH)
**Current:** Database has no way to distinguish child vs adult accounts.

**Required:** Need fields for:
- `is_child_account: boolean`
- `parent_user_id: uuid` (link to parent account)
- `parental_consent_date: timestamp`
- `parental_consent_method: string`

**Fix Priority:** 🟠 HIGH

---

### Gap 5: No Parental Dashboard (HIGH)
**Current:** No way for parents to:
- Review data collected about their child
- Delete their child's account/data
- Revoke consent

**Required:** Parent account features:
- View list of linked child accounts
- View each child's learning data
- Delete child account button
- Revoke consent option

**Fix Priority:** 🟠 HIGH

---

### Gap 6: Affiliate Cookies Set for All Users (MEDIUM)
**Current:** `affiliate-tracking.js` sets cookies for all users including potential children.

**COPPA Says:** Essential cookies for site operation are allowed, but tracking cookies may be problematic.

**Analysis:** Affiliate attribution is for payment (essential), but:
- UTM tracking (`utm_source`, `utm_medium`, `utm_campaign`) is marketing
- 30-day cookie persistence is long

**Fix:** Only set affiliate cookies AFTER verifying user is 13+ or has parental consent.

**Fix Priority:** 🟡 MEDIUM

---

## What We're Doing Right

### ✅ No Third-Party Analytics
- No Google Analytics
- No Facebook Pixel
- No Mixpanel/Amplitude
- This is excellent for COPPA compliance

### ✅ Privacy Policy Section 2
The text is good, it just needs to be implemented:
- Section 2.1 - Parental Consent ✓ (text exists)
- Section 2.2 - Limited Data Collection ✓ (text exists)
- Section 2.3 - Parental Rights ✓ (text exists)

### ✅ First-Party Auth Only
- Using Supabase Auth (our own database)
- Not sharing auth data with third parties

### ✅ Educational Safe Harbor Potential
- If we partner with schools, can use "school consent" for FERPA-covered students
- But this doesn't apply to direct-to-consumer

---

## Implementation Plan

### Phase 1: Age Gate (Week 1) - BLOCKER
1. Add age selection before signup
2. Route under-13 to "Parent Signup" flow
3. Route 13+ to normal OAuth flow

### Phase 2: Parental Consent (Week 2) - BLOCKER
1. Create parent account type
2. Implement VPC via Stripe micro-charge
3. Link child accounts to parent accounts
4. Store consent metadata

### Phase 3: Child Account Type (Week 2)
1. Add database fields for child accounts
2. Modify auth to handle child vs adult
3. Limit data collection for child accounts

### Phase 4: Parental Dashboard (Week 3)
1. Parent account management page
2. View/manage child accounts
3. Data review and deletion
4. Consent revocation

### Phase 5: Cookie Compliance (Week 3)
1. Delay affiliate cookies until age verified
2. Essential cookies only for children

---

## FTC Penalties for Non-Compliance

**COPPA violations can result in:**
- Civil penalties up to **$50,120 per violation** (as of 2024)
- Each instance of improperly collecting child data = one violation
- Class action lawsuits from parents

**Recent FTC Cases:**
- Epic Games (Fortnite): $275 million (2022)
- Google (YouTube): $170 million (2019)
- TikTok: $5.7 million (2019)

---

## Recommended Immediate Actions

### Before Launch (Dec 17):

1. **Option A: Age-Restrict to 13+** ⚡ FASTEST
   - Add age gate requiring users to confirm 13+
   - Remove "ages 2-12" from marketing
   - Update privacy policy
   - **Timeline: 1-2 days**

2. **Option B: Full COPPA Compliance** 📋 CORRECT
   - Implement full age gate + VPC system
   - Build parental consent flow
   - Create parent dashboard
   - **Timeline: 2-3 weeks**

### My Recommendation:
**Launch with Option A (13+ only)**, then implement Option B for Q1 2025 to unlock the under-13 market.

---

## Resources

- [FTC COPPA FAQ](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)
- [COPPA Rule Text](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312)
- [FTC Six-Step Compliance Plan](https://www.ftc.gov/business-guidance/resources/childrens-online-privacy-protection-rule-six-step-compliance-plan-your-business)

---

## Sign-Off Required

**This document requires review and decision by:**
- [ ] Product Owner
- [ ] Legal Counsel (recommended)

**Decision Needed:** Option A (13+ only) or Option B (full COPPA)?

---

*Document prepared by: AI Assistant*  
*Next Review: Before Dec 17, 2025 Launch*


