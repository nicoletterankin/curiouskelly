# COPPA Compliance Documentation

**Document Type:** Internal Compliance Documentation  
**Last Updated:** January 27, 2026  
**Owner:** Lesson of the Day PBC  
**Status:** ACTIVE  

---

## Executive Summary

Curious Kelly is an educational platform designed for learners of all ages, including children under 13. This document outlines our compliance with the Children's Online Privacy Protection Act (COPPA) (16 CFR Part 312) and related children's privacy regulations.

**COPPA applies because:**
- Our service is directed at children under 13
- We have actual knowledge that some users are under 13
- We collect personal information from users

---

## 1. Verifiable Parental Consent (VPC) Process

### 1.1 When Consent Is Required

Parental consent is REQUIRED before:
- Creating a child profile within a Family Account
- Collecting any personal information from a child under 13
- Syncing a child's learning progress to cloud storage
- Sending any communications to or about the child

### 1.2 Consent Methods

We use FTC-approved methods for Verifiable Parental Consent:

#### Method 1: Credit Card Verification (Primary)
1. Parent enters credit card during subscription purchase
2. Small authorization charge (e.g., $0.50) processed via Stripe
3. Charge immediately refunded
4. Transaction confirms adult status and identity
5. Parent confirms they are the child's parent/guardian

**Implementation:** Integrated with Stripe payment flow. Transaction record retained as consent evidence.

#### Method 2: Knowledge-Based Verification (Secondary)
For accounts without payment (e.g., trial periods):
1. Parent creates account with email verification
2. Parent answers knowledge-based questions from public records
3. Parent confirms they are the child's parent/guardian
4. System generates unique consent ID

**Implementation:** Third-party identity verification service (e.g., Lexis-Nexis, Experian) via API.

#### Method 3: Government ID Verification (Alternative)
1. Parent uploads government-issued photo ID
2. ID is verified for authenticity
3. ID is deleted after verification (not retained)
4. Consent record created

**Implementation:** Third-party ID verification service with immediate deletion policy.

### 1.3 Consent Records

For each consent obtained, we retain:
- Timestamp of consent
- Method used (credit card, knowledge-based, or ID)
- Parent's email address
- Unique consent ID
- What was consented to (specific data collection activities)
- Child profile ID(s) associated

**Retention:** Consent records retained for duration of account plus 7 years (legal requirement).

### 1.4 Consent Withdrawal

Parents may withdraw consent at any time by:
- Using the "Delete Child Profile" button in account settings
- Emailing hello@curiouskelly.com with subject "Withdraw COPPA Consent"
- Calling our support line (when available)

**Processing Time:** Within 48 hours of request

---

## 2. Data Minimization Practices

### 2.1 Data We Collect From Children

| Data Type | Collected | Purpose | Storage |
|-----------|-----------|---------|---------|
| Real name | NO | - | - |
| Email | NO | - | - |
| Phone | NO | - | - |
| Address | NO | - | - |
| Photo/Video | NO | - | - |
| Location | NO | - | - |
| Display name | YES (if Family Account) | Personalization | Cloud (encrypted) |
| Age range | YES (if Family Account) | Content difficulty | Cloud (encrypted) |
| Lesson progress | YES | Track learning | Local or Cloud |
| Anonymous votes | YES | Improve content | Cloud (no PII) |
| Device type | YES | Compatibility | Session only |

### 2.2 Local-First Architecture

By default, all child data is stored **locally on the device**:
- No cloud account required for basic functionality
- Progress stored in browser localStorage
- No personal information transmitted

Cloud sync requires:
1. Parent-created Family Account
2. Verifiable parental consent
3. Explicit opt-in by parent

### 2.3 No Conditioning on Data

Per COPPA 312.5(c), we never condition a child's participation on:
- Disclosure of more personal information than necessary
- Agreement to data collection beyond what's needed

Children can use the full educational experience without providing any personal information.

---

## 3. Data Retention and Deletion Policies

### 3.1 Retention Periods

| Data Category | Retention Period | Deletion Process |
|---------------|-----------------|------------------|
| Anonymous usage (no account) | 90 days | Auto-purge then aggregate |
| Local device data | Until user deletes | User-controlled |
| Child profile data | Until parent deletes or 12 months inactive | Manual or auto-delete |
| Parental consent records | Account lifetime + 7 years | Legal retention |
| Anonymous vote data | Indefinite | Not identifiable |
| Payment records | 7 years | Legal requirement |

### 3.2 Automatic Deletion (Inactive Accounts)

For Family Accounts inactive for 12 months:
1. **Day 0:** Account becomes inactive (no logins for 12 months)
2. **Day 1:** Automated email to parent: "Your account is inactive"
3. **Day 30:** Second email: "Data will be deleted in 30 days"
4. **Day 60:** All child data permanently deleted
5. **Day 60:** Anonymized aggregate data retained (no PII)

### 3.3 Parent-Requested Deletion

When a parent requests deletion:
1. Request received via email, settings, or support
2. Identity verified (must match account holder)
3. Deletion executed within 48 hours
4. Confirmation email sent to parent
5. Audit log entry created (no PII in log)

### 3.4 Data We Cannot Delete

Per legal requirements, we retain:
- Consent records (7 years)
- Payment transaction records (7 years)
- Aggregate, anonymized analytics (indefinite)

---

## 4. Child Safety Measures

### 4.1 No Social Features

Curious Kelly does NOT include:
- ❌ Chat or messaging
- ❌ Forums or comments
- ❌ User-to-user communication
- ❌ Social sharing to external platforms
- ❌ Friend lists or connections
- ❌ Public profiles

### 4.2 No User-Generated Content

Children cannot:
- ❌ Upload photos, videos, or audio
- ❌ Post comments or reviews
- ❌ Create public content
- ❌ Share their activity publicly

### 4.3 No External Links

Lessons do NOT include:
- ❌ Links to external websites
- ❌ Embedded third-party content
- ❌ Ads linking elsewhere
- ❌ Social media buttons

### 4.4 Content Moderation

All lesson content is:
- Created by Lesson of the Day PBC staff
- Reviewed for age-appropriateness before publication
- Tested across age ranges
- Subject to internal content guidelines

### 4.5 No Push Notifications to Children

- Push notifications go ONLY to parent's device/email
- Children's devices do not receive marketing
- Lesson reminders are parent-controlled

### 4.6 Parental Controls

Parents can:
- Set daily time limits
- View lesson history
- Pause or lock child profiles
- Delete child data at any time
- Control notification preferences

---

## 5. Technical Safeguards

### 5.1 Encryption

- **In Transit:** TLS 1.3 for all connections
- **At Rest:** AES-256 encryption for stored data
- **Keys:** Managed via secure key management service

### 5.2 Access Controls

- Role-based access (RBAC) for all systems
- No employee access to child PII without documented need
- All access logged and audited
- Quarterly access reviews

### 5.3 Third-Party Vendors

Approved vendors with COPPA-compliant data processing agreements:

| Vendor | Purpose | DPA Status |
|--------|---------|------------|
| Stripe | Payment processing | Signed |
| Cloudflare | CDN & security | Signed |
| Supabase | Database hosting | Signed |
| Vercel | App hosting | Signed |

### 5.4 Incident Response

In case of a data breach involving children's data:
1. Incident identified and contained (within 4 hours)
2. Legal team notified (within 24 hours)
3. FTC notification if required (within 72 hours)
4. Parent notification (within 72 hours)
5. Post-incident review (within 30 days)

---

## 6. Direct Notice Requirements

### 6.1 Notice to Parents

Before collecting personal information from a child, we provide:
1. **Online notice** (privacy policy) with:
   - What information we collect
   - How we use the information
   - Our disclosure practices
   - Parental rights
   - Contact information

2. **Direct notice** (email to parent) with:
   - Specific information to be collected
   - Purpose of collection
   - Link to full privacy policy
   - Instructions for consent or refusal
   - Contact information

### 6.2 Notice Updates

Parents are notified of material changes:
- 30 days before changes take effect
- Via email to registered parent address
- Via in-app notification
- With clear explanation of what's changing

---

## 7. Operator Contact Information

**For COPPA inquiries:**

Lesson of the Day PBC  
Email: hello@curiouskelly.com  
Subject Line: "COPPA Inquiry"  
Response Time: Within 48 hours  

---

## 8. Compliance Monitoring

### 8.1 Internal Audits

- **Monthly:** Data collection review
- **Quarterly:** Consent process audit
- **Annually:** Full COPPA compliance audit

### 8.2 Staff Training

All employees complete:
- COPPA training at onboarding
- Annual COPPA refresher
- Incident response training

### 8.3 Documentation

All COPPA-related activities are documented:
- Consent records
- Deletion requests and fulfillment
- Parent inquiries and responses
- Policy changes and rationale

---

## 9. FTC Safe Harbor (Future)

We are evaluating participation in an FTC-approved COPPA Safe Harbor program. Safe Harbor participation would provide:
- Independent compliance review
- Dispute resolution mechanism
- Additional assurance to parents

**Status:** Under evaluation

---

## 10. Related Documents

- [Privacy Policy](/privacy.html) - Public-facing children's privacy policy
- [Terms of Service](/terms.html) - Service terms including child provisions
- [Trust & Safety](/trust.html#coppa) - COPPA overview for parents

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial COPPA compliance documentation |

---

**Prepared by:** Lesson of the Day PBC Legal & Compliance  
**Approved by:** Executive Team  
**Next Review:** 2026-04-27
