# Kids Account Compliance for Share & Earn

> **LEGAL NOTICE:** This document describes how Curious Kelly complies with COPPA (Children's Online Privacy Protection Act), GDPR-K, and related regulations regarding minors participating in the Share & Earn referral program.

**Version:** 1.0  
**Date:** December 7, 2025  
**Status:** IMPLEMENTED

---

## Executive Summary

The Share & Earn system implements age-gated access to earnings features:

| Age Group | Can See Referral Link | Can Share | Can Earn | Can Request Payout | Earnings Go To |
|-----------|----------------------|-----------|----------|-------------------|----------------|
| Under 13 (no consent) | ❌ | ❌ | ❌ | ❌ | N/A |
| Under 13 (with parental consent) | ✅ | ✅ | ✅ | ❌ | Parent Account |
| 13-17 | ✅ | ✅ | ✅ | ❌ | Held until 18 |
| 18+ | ✅ | ✅ | ✅ | ✅ | Self |

---

## Regulatory Requirements

### COPPA (USA)

The Children's Online Privacy Protection Act requires:

1. **Verifiable Parental Consent** for collecting personal information from children under 13
2. **No financial transactions** directly with children under 13
3. **Parental access** to view and manage child's data

**Our Implementation:**
- Users under 13 cannot see or use referral links without parental consent
- A parent must link their account and explicitly consent to earnings
- All earnings are credited to the parent account, not the child

### GDPR-K (EU)

The General Data Protection Regulation's provisions for children vary by country (13-16), but we apply:

1. **Age 13 as baseline** for limited participation
2. **Parental consent** required for any financial benefits under 18

### FTC Endorsement Guidelines

Children cannot be used as endorsers without parental consent:

1. **Disclosure required** in all share content: "I may earn a commission"
2. **No incentivizing children** to promote without parental oversight

---

## Technical Implementation

### Database Schema

```sql
-- Users table additions
ALTER TABLE users ADD COLUMN parent_account_id UUID;
ALTER TABLE users ADD COLUMN is_family_admin BOOLEAN;
ALTER TABLE users ADD COLUMN parental_consent_for_earnings BOOLEAN;
ALTER TABLE users ADD COLUMN earnings_held_for_minors NUMERIC;

-- Age calculation view
CREATE VIEW users_with_age AS
SELECT *, calculated_age, is_under_13, is_minor FROM users;

-- Minor earnings ledger
CREATE TABLE minor_earnings_ledger (
  minor_user_id UUID,
  parent_user_id UUID,
  amount NUMERIC,
  status TEXT  -- 'held', 'transferred_to_parent', 'transferred_at_18'
);

-- Compliance audit log
CREATE TABLE earnings_compliance_log (
  user_id UUID,
  event_type TEXT,  -- 'minor_earnings_held', 'payout_blocked_minor', etc.
  details JSONB
);
```

### API Endpoints

| Endpoint | Purpose | Age Restrictions |
|----------|---------|------------------|
| `GET /api/referral/eligibility` | Check user's earning eligibility | Returns age-appropriate permissions |
| `POST /api/referral/payout` | Request payout | Blocks users under 18 |
| `POST /api/family/link` | Link child to parent | Only adults can be family admins |
| `POST /api/family/claim-earnings` | Parent claims child's earnings | Only linked parent |
| `GET /api/family/members` | List family members | Only family admins |

### Frontend Behavior

**Under 13 (no consent):**
```javascript
// Shows: "Ask a Parent to Help!"
// Hides: All earnings UI, referral links, share buttons
showUnder13();
```

**Under 13 (with consent) or 13-17:**
```javascript
// Shows: Full referral UI with notice
// Hides: Payout button
// Shows: "Your earnings are being saved!"
showMinorNotice();
```

**18+:**
```javascript
// Shows: Full functionality including payout
showLoggedIn();
```

---

## Edge Cases & Handling

### 1. Child Refers Someone

**Scenario:** 8-year-old Kelly user shares link with friend.

**Without Parental Consent:**
- Child cannot see their referral link
- If they somehow share a link, click is tracked but no earnings

**With Parental Consent:**
- Click is tracked normally
- Commission goes to parent's account
- Logged in `earnings_compliance_log`

### 2. Someone Refers a Child

**Scenario:** Adult refers a 10-year-old.

**Handling:**
- Referral attribution works normally
- Adult referrer gets credit when child (or parent) subscribes
- Child's account is marked with referrer but cannot earn

### 3. User Turns 18

**Scenario:** User's 18th birthday occurs.

**Handling:**
- Daily cron job checks for birthdays
- `check_age_18_transfer()` function runs
- Held earnings automatically transferred to user's available balance
- Logged in `earnings_compliance_log` as `age_18_release`

### 4. Parent Claims Earnings

**Scenario:** Parent wants to withdraw child's accumulated earnings.

**Handling:**
```sql
SELECT * FROM parent_claim_minor_earnings(parent_uuid, child_uuid);
-- Returns: { success: true, amount_claimed: 25.00 }
```

- Earnings move from `minor_earnings_ledger` to parent's `available_earnings`
- Logged for tax purposes
- Parent can then request payout

### 5. Age Correction

**Scenario:** User's birthday is corrected and they're actually younger.

**Handling:**
- Trigger fires on `birthday`, `birth_year`, or `age` column update
- If new age < 13 and old age >= 13, logs `age_correction_review_needed`
- Manual review required for past earnings

### 6. Self-Referral Within Family

**Scenario:** Parent refers themselves, or child refers sibling.

**Handling:**
- Same-user self-referral blocked (IP + user ID check)
- Family member referrals allowed but flagged for abuse detection
- Logged in `earnings_compliance_log`

### 7. Parent Account Deleted

**Scenario:** Parent deletes their account while child has held earnings.

**Handling:**
- `parent_user_id` set to NULL via FK constraint
- Earnings remain in `minor_earnings_ledger`
- Transfer happens at age 18 regardless

### 8. Minor Tries to Request Payout

**Scenario:** 15-year-old clicks "Request Payout"

**Response:**
```json
{
  "success": false,
  "message": "Users under 18 cannot request payouts. Your earnings are being held until you turn 18, or a parent can claim them if your account is linked."
}
```

- Event logged in `earnings_compliance_log` as `payout_blocked_minor`

---

## Tax Implications

### US Tax Law

- Cannot issue 1099 to minors
- Parent receives 1099 for claimed earnings
- Held earnings are not taxable until distributed

### Tax Form Requirements

| User | Form | When Required |
|------|------|---------------|
| US Adult | W-9 | Before first payout |
| US Parent claiming minor earnings | W-9 | Before first payout |
| Non-US Adult | W-8BEN | Before first payout |
| Minor | N/A | Cannot receive payouts |

---

## Audit Trail

Every compliance-related action is logged:

```sql
INSERT INTO earnings_compliance_log (user_id, event_type, details)
VALUES (
  'user-uuid',
  'minor_earnings_held',
  '{"commission_id": "...", "amount": 5.00, "reason": "User under 18"}'
);
```

**Event Types:**
- `minor_referral_blocked` - Under 13 tried to use referral
- `minor_earnings_held` - Commission held for minor
- `payout_blocked_minor` - Minor tried to request payout
- `parent_claimed_earnings` - Parent withdrew minor's earnings
- `age_18_release` - Earnings released when user turned 18
- `parental_consent_given` - Parent gave consent for child
- `family_link_created` - Child linked to parent account
- `age_correction_review_needed` - Age changed, needs manual review

---

## Testing

Run the compliance evaluation:

```bash
npx ts-node evals/kids-compliance-eval.ts
```

This tests:
- Database schema presence
- Age calculation accuracy
- Eligibility rule enforcement
- API endpoint existence
- Edge case handling

---

## Annual Review

This compliance documentation should be reviewed annually or when:

1. COPPA/GDPR regulations change
2. New features affect minors
3. New payment methods are added
4. Audit findings require updates

---

**Document Owner:** Engineering + Legal  
**Contact:** hello@curiouskelly.com  
**Company:** Lesson of the Day PBC

