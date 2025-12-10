# Share & Earn Referral System

> **Philosophy:** Every learner is an affiliate from Day 1. LIFETIME attribution means once you refer someone, you're credited forever.

## Overview

The Share & Earn system integrates referral functionality directly into the Curious Kelly learning experience. Unlike traditional affiliate programs that are separate from the product, every learner automatically gets a referral link and can earn commissions based on their learning progress.

## Key Features

### 1. LIFETIME Attribution
- Referral cookies **never expire**
- If someone clicks your link today and subscribes in 2 years, you still get credited
- No 30-day windows or complex attribution rules

### 2. Commission Tiers Based on Learning
| Tier | Lessons | Commission |
|------|---------|------------|
| New Learner | 0+ | 10% |
| Active Learner | 7+ | 15% |
| Committed Learner | 30+ | 20% |
| Dedicated Learner | 100+ | 25% |
| Complete Learner | 365+ | 30% |
| Legendary Learner | 1000+ | 35% |

### 3. Integrated Experience
- Referral link shown in Share & Earn panel (💰 button)
- Share prompt after every lesson completion
- Full earnings dashboard for tracking

## Technical Architecture

### API Endpoints

#### `POST /api/referral/track`
Records a referral click with LIFETIME attribution.

```typescript
// Request
{
  referralCode: string,
  visitorFingerprint?: string,
  sourceUrl?: string,
  landingPage?: string,
  utmSource?: string,
  utmMedium?: string,
  utmCampaign?: string
}

// Response
{
  success: boolean,
  clickId: string,
  referrerInfo: {
    displayName?: string,
    commissionTier?: string
  }
}
```

#### `GET /api/referral/lookup?code=XYZ`
Validates a referral code and returns public referrer info.

#### `POST /api/referral/convert`
Links a referral click to a newly signed-up user.

### Database Schema

The system uses the `integrate_earn_to_learn` migration which adds:

**Users table extensions:**
- `referral_code` - Unique code for sharing
- `referred_by_user_id` - Who referred this user
- `commission_rate` - Current rate (0.10 to 0.35)
- `commission_tier` - Current tier name
- `total_referrals` - Count of all referrals
- `pending_earnings` - Earnings waiting to clear
- `available_earnings` - Ready for payout
- `lifetime_earnings` - Total ever earned

**New tables:**
- `referral_clicks` - Tracks all clicks (LIFETIME attribution)
- `commission_transactions` - Records all commissions
- `payouts` - Withdrawal requests
- `commission_tiers` - Tier definitions

### Frontend Components

#### `earn-to-learn.js`
Main Share & Earn panel component:
- Earnings summary (available, pending, lifetime)
- Tier progress visualization
- Referral link with copy button
- Social share buttons
- Links to full dashboard

#### `lesson-share-prompt.js`
Post-lesson completion share modal:
- Celebratory animation
- Pre-written share messages
- Automatic referral link inclusion
- Analytics tracking

#### `affiliate-tracking.js`
Client-side tracking (updated for LIFETIME):
- Captures `?ref=CODE` from URL
- Stores in localStorage (no expiration)
- Sends to `/api/referral/track`
- Provides `getReferralCode()` for checkout

## User Flows

### 1. New User from Referral Link
```
User clicks curiouskelly.com/?ref=nicolette
  → affiliate-tracking.js captures code
  → POST /api/referral/track creates click record
  → Code stored in localStorage (LIFETIME)
  → On signup: POST /api/referral/convert
  → Nicolette's total_referrals increments
```

### 2. User Shares After Lesson
```
User completes lesson
  → LessonSharePrompt.show() triggered
  → Modal shows with pre-written message
  → User's referral code included in share URL
  → Share tracked in analytics
```

### 3. User Requests Payout
```
User opens earnings.html
  → Sees available_earnings >= $50
  → Clicks "Request Payout"
  → Selects PayPal, enters email
  → Payout record created (status: pending)
  → Admin processes payout
  → User's available_earnings reduced
```

## Configuration

### Environment Variables
No additional environment variables needed - uses existing Supabase connection.

### Vercel Routes
Add to `vercel.json` if not already present:
```json
{
  "rewrites": [
    { "source": "/api/referral/:path*", "destination": "/api/referral/:path*.ts" }
  ]
}
```

## Monitoring

### Key Metrics to Track
- Click-to-signup conversion rate
- Share rate (shares per lesson complete)
- Commission payout volume
- Tier distribution of earners

### Alerts
Set up alerts for:
- Unusual spike in referral clicks (fraud detection)
- High volume of failed payouts
- Commission transactions exceeding daily threshold

## Legal Compliance

### FTC Disclosure
All share messages include automatic disclosure:
> "(I may earn a commission if you subscribe.)"

### Tax Handling
- Users earning $600+/year receive 1099
- W-9/W-8BEN collection required before payout
- `tax_form_status` field tracks compliance

## Future Enhancements

1. **Bonus Programs** - Additional commission for referring teachers, families
2. **Real-time Notifications** - Push when referral converts
3. **Custom Share Links** - kelly.me/yourname instead of ?ref=code
4. **API Access** - For power affiliates to build integrations

---

*Document: SHARE_AND_EARN.md*  
*Last Updated: December 7, 2025*  
*Contact: hello@curiouskelly.com*



