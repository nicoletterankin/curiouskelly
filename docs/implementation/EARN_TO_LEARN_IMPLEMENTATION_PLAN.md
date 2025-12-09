# 🛠️ EARN TO LEARN: Implementation Plan

**Target Launch:** December 17, 2025  
**Days Remaining:** 10  
**Status:** Database Ready ✅ | Frontend Pending | Backend Pending

---

## OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EARN TO LEARN FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. USER A has referral code: kelly.me/sarah                        │
│                    ↓                                                │
│  2. USER A shares link on social media                              │
│                    ↓                                                │
│  3. USER B clicks link → lands on curiouskelly.com?ref=sarah        │
│                    ↓                                                │
│  4. LIFETIME COOKIE stored (localStorage + server)                  │
│                    ↓                                                │
│  5. USER B signs up (days/weeks/years later)                        │
│                    ↓                                                │
│  6. USER B subscribes via Stripe                                    │
│                    ↓                                                │
│  7. WEBHOOK fires → record_commission(sarah, user_b, $99.99)        │
│                    ↓                                                │
│  8. USER A sees earnings in dashboard                               │
│                    ↓                                                │
│  9. USER A requests payout when available_earnings > $50            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: REFERRAL LINK CAPTURE (Day 1-2)

### 1.1 Landing Page URL Detection

**File:** `src/pages/[[...slug]].astro` or landing page entry

**Logic:**
```javascript
// On page load, check for referral parameter
const urlParams = new URLSearchParams(window.location.search);
const refCode = urlParams.get('ref');

if (refCode) {
  // Store in localStorage (persists forever on this device)
  localStorage.setItem('kelly_referrer', refCode);
  localStorage.setItem('kelly_referrer_timestamp', Date.now());
  
  // Also store in cookie for cross-subdomain access
  document.cookie = `kelly_ref=${refCode}; max-age=31536000000; path=/; domain=.curiouskelly.com`;
  
  // Track the click server-side
  fetch('/api/referral/track-click', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      referral_code: refCode,
      source_url: document.referrer,
      landing_page: window.location.pathname,
      utm_source: urlParams.get('utm_source'),
      utm_medium: urlParams.get('utm_medium'),
      utm_campaign: urlParams.get('utm_campaign')
    })
  });
}
```

### 1.2 Track Click API Endpoint

**File:** `api/referral/track-click.ts`

```typescript
import { createClient } from '@supabase/supabase-js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { 
    referral_code, 
    source_url, 
    landing_page,
    utm_source,
    utm_medium,
    utm_campaign 
  } = req.body;

  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_KEY
  );

  // Look up the referrer by code
  const { data: referrer } = await supabase
    .from('users')
    .select('id')
    .eq('referral_code', referral_code)
    .single();

  if (!referrer) {
    return res.status(404).json({ error: 'Invalid referral code' });
  }

  // Create click record
  const { data: click, error } = await supabase
    .from('referral_clicks')
    .insert({
      referrer_id: referrer.id,
      referral_code,
      source_url,
      landing_page,
      utm_source,
      utm_medium,
      utm_campaign,
      visitor_fingerprint: generateFingerprint(req), // Browser fingerprint
      visitor_ip_hash: hashIP(req.headers['x-forwarded-for']),
      // NO EXPIRATION - lifetime attribution
      attribution_expires_at: null
    })
    .select()
    .single();

  if (error) {
    console.error('Error tracking click:', error);
    return res.status(500).json({ error: 'Failed to track click' });
  }

  return res.status(200).json({ 
    success: true, 
    click_id: click.id 
  });
}

function generateFingerprint(req) {
  // Simple fingerprint from headers
  const ua = req.headers['user-agent'] || '';
  const lang = req.headers['accept-language'] || '';
  return Buffer.from(ua + lang).toString('base64').slice(0, 32);
}

function hashIP(ip) {
  if (!ip) return null;
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(ip).digest('hex').slice(0, 16);
}
```

---

## PHASE 2: SIGNUP ATTRIBUTION (Day 2-3)

### 2.1 Modify Signup Flow

**File:** `curious-kellly/lesson-player-v2/js/app.js` (or auth handler)

**When user signs up:**
```javascript
async signUpUser(email, password, metadata = {}) {
  // Get stored referrer
  const referrerCode = localStorage.getItem('kelly_referrer');
  
  // Include in signup metadata
  const { data, error } = await this.supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        ...metadata,
        referred_by_code: referrerCode,
        referral_timestamp: localStorage.getItem('kelly_referrer_timestamp')
      }
    }
  });

  if (data?.user && referrerCode) {
    // Link the referral
    await this.linkReferral(data.user.id, referrerCode);
  }

  return { data, error };
}

async linkReferral(userId, referrerCode) {
  // Call API to link referral
  const response = await fetch('/api/referral/link', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      referral_code: referrerCode
    })
  });
  
  if (response.ok) {
    // Clear localStorage after successful link
    // (but keep in case they need it again)
    console.log('Referral linked successfully');
  }
}
```

### 2.2 Link Referral API Endpoint

**File:** `api/referral/link.ts`

```typescript
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { user_id, referral_code } = req.body;

  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_KEY
  );

  // Find the referrer
  const { data: referrer } = await supabase
    .from('users')
    .select('id')
    .eq('referral_code', referral_code)
    .single();

  if (!referrer) {
    return res.status(404).json({ error: 'Invalid referral code' });
  }

  // Prevent self-referral
  if (referrer.id === user_id) {
    return res.status(400).json({ error: 'Cannot refer yourself' });
  }

  // Update the new user with referral info
  const { error: updateError } = await supabase
    .from('users')
    .update({
      referred_by_user_id: referrer.id,
      referred_at: new Date().toISOString()
    })
    .eq('id', user_id);

  if (updateError) {
    return res.status(500).json({ error: 'Failed to link referral' });
  }

  // Update the referral click record
  await supabase
    .from('referral_clicks')
    .update({
      converted_to_user_id: user_id,
      converted_at: new Date().toISOString(),
      conversion_type: 'signup'
    })
    .eq('referral_code', referral_code)
    .is('converted_to_user_id', null)
    .order('clicked_at', { ascending: false })
    .limit(1);

  // Increment referrer's total_referrals
  await supabase.rpc('increment_referrals', { referrer_id: referrer.id });

  return res.status(200).json({ success: true });
}
```

---

## PHASE 3: STRIPE COMMISSION WEBHOOK (Day 3-4)

### 3.1 Stripe Webhook Handler

**File:** `api/webhooks/stripe.ts`

```typescript
import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const sig = req.headers['stripe-signature'];
  let event;

  try {
    event = stripe.webhooks.constructEvent(req.body, sig, endpointSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return res.status(400).json({ error: 'Invalid signature' });
  }

  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_KEY
  );

  switch (event.type) {
    case 'checkout.session.completed':
      await handleCheckoutComplete(event.data.object, supabase);
      break;
    
    case 'invoice.paid':
      await handleInvoicePaid(event.data.object, supabase);
      break;
    
    case 'customer.subscription.deleted':
      await handleSubscriptionCancelled(event.data.object, supabase);
      break;
    
    case 'charge.refunded':
      await handleRefund(event.data.object, supabase);
      break;
  }

  return res.status(200).json({ received: true });
}

async function handleCheckoutComplete(session, supabase) {
  const customerEmail = session.customer_email;
  const amountPaid = session.amount_total / 100; // Stripe uses cents
  
  // Find the user
  const { data: user } = await supabase
    .from('users')
    .select('id, referred_by_user_id')
    .eq('email', customerEmail)
    .single();

  if (!user || !user.referred_by_user_id) {
    console.log('No referral for this purchase');
    return;
  }

  // Record the commission
  await recordCommission(supabase, {
    referrer_id: user.referred_by_user_id,
    referred_user_id: user.id,
    transaction_type: 'initial_subscription',
    gross_amount: amountPaid,
    stripe_payment_intent_id: session.payment_intent,
    stripe_subscription_id: session.subscription
  });
}

async function handleInvoicePaid(invoice, supabase) {
  // This handles renewals
  if (!invoice.subscription) return;
  
  const customerEmail = invoice.customer_email;
  const amountPaid = invoice.amount_paid / 100;
  
  const { data: user } = await supabase
    .from('users')
    .select('id, referred_by_user_id')
    .eq('email', customerEmail)
    .single();

  if (!user || !user.referred_by_user_id) return;

  // Check if this is a renewal (not first payment)
  const { count } = await supabase
    .from('commission_transactions')
    .select('*', { count: 'exact', head: true })
    .eq('referred_user_id', user.id);

  if (count === 0) return; // First payment handled by checkout.session.completed

  await recordCommission(supabase, {
    referrer_id: user.referred_by_user_id,
    referred_user_id: user.id,
    transaction_type: 'subscription_renewal',
    gross_amount: amountPaid,
    stripe_invoice_id: invoice.id,
    stripe_subscription_id: invoice.subscription
  });
}

async function handleRefund(charge, supabase) {
  // Claw back commission on refund
  const { data: transaction } = await supabase
    .from('commission_transactions')
    .select('*')
    .eq('stripe_payment_intent_id', charge.payment_intent)
    .single();

  if (!transaction) return;

  // Create clawback transaction
  await supabase
    .from('commission_transactions')
    .insert({
      referrer_id: transaction.referrer_id,
      referred_user_id: transaction.referred_user_id,
      transaction_type: 'refund_clawback',
      gross_amount: -transaction.gross_amount,
      commission_rate: transaction.commission_rate,
      commission_amount: -transaction.commission_amount,
      status: 'approved',
      notes: `Clawback for refund on ${charge.id}`
    });

  // Update referrer's earnings
  await supabase
    .from('users')
    .update({
      lifetime_earnings: supabase.raw(`lifetime_earnings - ${transaction.commission_amount}`),
      pending_earnings: supabase.raw(`pending_earnings - ${transaction.commission_amount}`)
    })
    .eq('id', transaction.referrer_id);
}

async function recordCommission(supabase, {
  referrer_id,
  referred_user_id,
  transaction_type,
  gross_amount,
  stripe_payment_intent_id,
  stripe_invoice_id,
  stripe_subscription_id
}) {
  // Get referrer's current commission rate
  const { data: referrer } = await supabase
    .from('users')
    .select('commission_rate')
    .eq('id', referrer_id)
    .single();

  const commission_rate = referrer?.commission_rate || 0.10;
  const commission_amount = gross_amount * commission_rate;

  // Insert commission transaction
  await supabase
    .from('commission_transactions')
    .insert({
      referrer_id,
      referred_user_id,
      transaction_type,
      gross_amount,
      commission_rate,
      commission_amount,
      stripe_payment_intent_id,
      stripe_invoice_id,
      stripe_subscription_id,
      status: 'pending'
    });

  // Update referrer's earnings
  await supabase
    .from('users')
    .update({
      pending_earnings: supabase.raw(`pending_earnings + ${commission_amount}`),
      lifetime_earnings: supabase.raw(`lifetime_earnings + ${commission_amount}`),
      total_referrals: supabase.raw('total_referrals + 1')
    })
    .eq('id', referrer_id);

  // Send notification email to referrer
  await sendCommissionNotification(referrer_id, {
    amount: commission_amount,
    type: transaction_type,
    total_pending: referrer?.pending_earnings + commission_amount
  });

  console.log(`Commission recorded: $${commission_amount} for ${referrer_id}`);
}
```

---

## PHASE 4: SHARE & EARN UI (Day 4-5)

### 4.1 Add to Drawer Menu

**File:** `curious-kellly/lesson-player-v2/index.html`

**Add to drawer-nav section:**
```html
<!-- Add after existing drawer items, before Settings -->
<div class="drawer-divider"></div>
<div class="drawer-section share-earn-section">
    <div class="drawer-section-header">
        <span class="section-icon">💰</span>
        <span class="section-title">Share & Earn</span>
    </div>
    <div class="share-earn-content">
        <div class="earnings-summary">
            <div class="earnings-stat">
                <span class="stat-value" id="pending-earnings">$0.00</span>
                <span class="stat-label">Pending</span>
            </div>
            <div class="earnings-stat">
                <span class="stat-value" id="available-earnings">$0.00</span>
                <span class="stat-label">Available</span>
            </div>
        </div>
        <div class="commission-tier">
            <span class="tier-badge" id="commission-tier-badge">New Learner</span>
            <span class="tier-rate" id="commission-rate">10% commission</span>
        </div>
        <div class="referral-link-box">
            <input type="text" id="referral-link" readonly value="kelly.me/..." class="referral-input">
            <button class="btn-copy" id="btn-copy-link" title="Copy link">📋</button>
        </div>
        <div class="share-buttons">
            <button class="share-btn twitter" id="share-twitter" title="Share on Twitter">𝕏</button>
            <button class="share-btn facebook" id="share-facebook" title="Share on Facebook">f</button>
            <button class="share-btn whatsapp" id="share-whatsapp" title="Share on WhatsApp">💬</button>
            <button class="share-btn email" id="share-email" title="Share via Email">✉️</button>
        </div>
        <a href="#" class="view-dashboard-link" id="btn-view-earnings">View Full Earnings →</a>
    </div>
</div>
```

### 4.2 Add Styles

**File:** `curious-kellly/lesson-player-v2/css/styles.css`

```css
/* Share & Earn Section */
.share-earn-section {
    padding: 20px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    margin: 10px 0;
}

.drawer-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
}

.section-icon {
    font-size: 1.2rem;
}

.section-title {
    font-weight: 600;
    color: #fff;
}

.earnings-summary {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
}

.earnings-stat {
    text-align: center;
}

.stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: 700;
    color: #22c55e; /* Green for money */
}

.stat-label {
    font-size: 0.75rem;
    color: #71717a;
    text-transform: uppercase;
}

.commission-tier {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 15px;
    padding: 10px;
    background: rgba(217, 119, 87, 0.1); /* Kelly orange */
    border-radius: 8px;
}

.tier-badge {
    background: #d97757;
    color: #fff;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.tier-rate {
    font-size: 0.85rem;
    color: #d97757;
}

.referral-link-box {
    display: flex;
    gap: 8px;
    margin-bottom: 15px;
}

.referral-input {
    flex: 1;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px 12px;
    color: #fff;
    font-family: monospace;
    font-size: 0.85rem;
}

.btn-copy {
    background: rgba(255, 255, 255, 0.1);
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    cursor: pointer;
    transition: background 0.2s;
}

.btn-copy:hover {
    background: rgba(255, 255, 255, 0.2);
}

.btn-copy.copied {
    background: #22c55e;
}

.share-buttons {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}

.share-btn {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    transition: transform 0.2s, opacity 0.2s;
}

.share-btn:hover {
    transform: scale(1.05);
}

.share-btn.twitter { background: #000; color: #fff; }
.share-btn.facebook { background: #1877f2; color: #fff; }
.share-btn.whatsapp { background: #25d366; color: #fff; }
.share-btn.email { background: #6366f1; color: #fff; }

.view-dashboard-link {
    display: block;
    text-align: center;
    color: #d97757;
    text-decoration: none;
    font-size: 0.85rem;
}

.view-dashboard-link:hover {
    text-decoration: underline;
}
```

### 4.3 Add JavaScript Logic

**File:** `curious-kellly/lesson-player-v2/js/app.js`

**Add to KellyOS class:**
```javascript
// Add to init() method
this.loadEarningsData();
this.setupShareListeners();

// Add these methods to the class:

async loadEarningsData() {
    if (!this.state.user) return;
    
    const { data: user, error } = await this.supabase
        .from('users')
        .select(`
            referral_code,
            commission_tier,
            commission_rate,
            pending_earnings,
            available_earnings,
            lifetime_earnings,
            total_referrals
        `)
        .eq('id', this.state.user.id)
        .single();
    
    if (error || !user) return;
    
    this.state.earnings = user;
    this.updateEarningsUI(user);
}

updateEarningsUI(earnings) {
    // Update referral link
    const linkInput = document.getElementById('referral-link');
    if (linkInput) {
        linkInput.value = `kelly.me/${earnings.referral_code}`;
    }
    
    // Update earnings display
    document.getElementById('pending-earnings')?.textContent = 
        `$${earnings.pending_earnings?.toFixed(2) || '0.00'}`;
    document.getElementById('available-earnings')?.textContent = 
        `$${earnings.available_earnings?.toFixed(2) || '0.00'}`;
    
    // Update tier badge
    const tierNames = {
        'new_learner': 'New Learner',
        'active_learner': 'Active Learner',
        'committed_learner': 'Committed Learner',
        'dedicated_learner': 'Dedicated Learner',
        'complete_learner': 'Complete Learner',
        'legendary_learner': 'Legendary Learner'
    };
    
    document.getElementById('commission-tier-badge')?.textContent = 
        tierNames[earnings.commission_tier] || 'New Learner';
    document.getElementById('commission-rate')?.textContent = 
        `${(earnings.commission_rate * 100).toFixed(0)}% commission`;
}

setupShareListeners() {
    // Copy link button
    document.getElementById('btn-copy-link')?.addEventListener('click', () => {
        const linkInput = document.getElementById('referral-link');
        navigator.clipboard.writeText(linkInput.value);
        
        const btn = document.getElementById('btn-copy-link');
        btn.classList.add('copied');
        btn.textContent = '✓';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.textContent = '📋';
        }, 2000);
    });
    
    // Social share buttons
    const referralCode = this.state.earnings?.referral_code || '';
    const shareUrl = `https://curiouskelly.com/?ref=${referralCode}`;
    const shareText = "I'm learning something new every day with Curious Kelly! Join me:";
    
    document.getElementById('share-twitter')?.addEventListener('click', () => {
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`, '_blank');
    });
    
    document.getElementById('share-facebook')?.addEventListener('click', () => {
        window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`, '_blank');
    });
    
    document.getElementById('share-whatsapp')?.addEventListener('click', () => {
        window.open(`https://wa.me/?text=${encodeURIComponent(shareText + ' ' + shareUrl)}`, '_blank');
    });
    
    document.getElementById('share-email')?.addEventListener('click', () => {
        window.open(`mailto:?subject=${encodeURIComponent('Join me on Curious Kelly!')}&body=${encodeURIComponent(shareText + '\n\n' + shareUrl)}`, '_blank');
    });
}
```

---

## PHASE 5: LESSON COMPLETE SHARE PROMPT (Day 5-6)

### 5.1 Modify advancePhase() Method

**File:** `curious-kellly/lesson-player-v2/js/app.js`

**Update the 'complete' phase handling:**
```javascript
advancePhase() {
    const phases = ['welcome', 'Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'complete'];
    const currentIndex = phases.indexOf(this.state.lessonPhase);
    
    if (currentIndex < phases.length - 1) {
        this.state.lessonPhase = phases[currentIndex + 1];
        
        if (this.state.lessonPhase === 'complete') {
            this.showLessonCompleteWithShare();
        } else {
            this.renderPhase();
        }
    }
}

showLessonCompleteWithShare() {
    const lesson = this.state.currentLesson;
    const referralCode = this.state.earnings?.referral_code || '';
    const shareUrl = `https://curiouskelly.com/?ref=${referralCode}&day=${lesson.dayNumber}`;
    
    if (this.dom.questionText) {
        this.dom.questionText.innerHTML = `
            <div class="lesson-complete-message">
                <span class="complete-icon">✨</span>
                <h3>Lesson Complete!</h3>
                <p>You just learned about "${lesson.topic}"</p>
            </div>
        `;
    }
    
    this.dom.choiceContainer.innerHTML = `
        <div class="share-prompt glass-panel-medium">
            <p class="share-prompt-text">Know someone who'd love this lesson?</p>
            <div class="share-prompt-buttons">
                <button class="share-btn-large" id="share-lesson-twitter">
                    Share on 𝕏
                </button>
                <button class="share-btn-large" id="share-lesson-whatsapp">
                    Share on WhatsApp
                </button>
            </div>
            <p class="share-prompt-earnings">
                You'll earn ${(this.state.earnings?.commission_rate * 100 || 10).toFixed(0)}% 
                if they subscribe!
            </p>
        </div>
        <button class="btn-secondary-glass" id="btn-finish-lesson">
            Continue to Dashboard →
        </button>
    `;
    
    // Add listeners
    document.getElementById('share-lesson-twitter')?.addEventListener('click', () => {
        const text = `I just learned "${lesson.topic}" with @CuriousKelly! 🎓 Join me:`;
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`, '_blank');
    });
    
    document.getElementById('share-lesson-whatsapp')?.addEventListener('click', () => {
        const text = `I just learned about "${lesson.topic}" with Curious Kelly! You should try it:`;
        window.open(`https://wa.me/?text=${encodeURIComponent(text + ' ' + shareUrl)}`, '_blank');
    });
    
    document.getElementById('btn-finish-lesson')?.addEventListener('click', () => {
        this.switchMode('dashboard');
    });
}
```

### 5.2 Add Complete Screen Styles

**File:** `curious-kellly/lesson-player-v2/css/styles.css`

```css
/* Lesson Complete Screen */
.lesson-complete-message {
    text-align: center;
    padding: 20px;
}

.complete-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 10px;
}

.lesson-complete-message h3 {
    font-size: 1.5rem;
    margin-bottom: 10px;
}

.share-prompt {
    padding: 25px;
    text-align: center;
    margin-bottom: 15px;
}

.share-prompt-text {
    font-size: 1.1rem;
    margin-bottom: 20px;
    color: #e5e5e5;
}

.share-prompt-buttons {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin-bottom: 20px;
}

.share-btn-large {
    padding: 15px 30px;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s;
}

.share-btn-large:hover {
    transform: scale(1.05);
}

.share-btn-large:nth-child(1) { background: #000; color: #fff; }
.share-btn-large:nth-child(2) { background: #25d366; color: #fff; }

.share-prompt-earnings {
    font-size: 0.85rem;
    color: #22c55e;
    background: rgba(34, 197, 94, 0.1);
    padding: 8px 15px;
    border-radius: 20px;
    display: inline-block;
}
```

---

## PHASE 6: EARNINGS DASHBOARD (Day 6-8)

### 6.1 Full Earnings Modal

**File:** `curious-kellly/lesson-player-v2/index.html`

**Add new modal:**
```html
<!-- Earnings Dashboard Modal -->
<div id="modal-earnings" class="os-modal glass-panel-heavy">
    <div class="modal-header">
        <h2>Your Earnings</h2>
        <button class="btn-close-modal">Close</button>
    </div>
    <div class="modal-content-scroll">
        <div class="earnings-dashboard">
            <!-- Summary Cards -->
            <div class="earnings-cards">
                <div class="earnings-card">
                    <span class="card-label">Pending</span>
                    <span class="card-value" id="dash-pending">$0.00</span>
                    <span class="card-note">Clears in 7 days</span>
                </div>
                <div class="earnings-card">
                    <span class="card-label">Available</span>
                    <span class="card-value highlight" id="dash-available">$0.00</span>
                    <span class="card-note">Ready to withdraw</span>
                </div>
                <div class="earnings-card">
                    <span class="card-label">Lifetime</span>
                    <span class="card-value" id="dash-lifetime">$0.00</span>
                    <span class="card-note">Total earned</span>
                </div>
            </div>
            
            <!-- Tier Progress -->
            <div class="tier-progress-section">
                <h3>Your Commission Tier</h3>
                <div class="tier-current" id="dash-tier-current">
                    <span class="tier-name">New Learner</span>
                    <span class="tier-rate">10%</span>
                </div>
                <div class="tier-progress-bar">
                    <div class="tier-progress-fill" id="dash-tier-progress" style="width: 0%"></div>
                </div>
                <p class="tier-next" id="dash-tier-next">
                    Complete 7 more lessons to unlock 15% commission!
                </p>
            </div>
            
            <!-- Referral Stats -->
            <div class="referral-stats-section">
                <h3>Your Network</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number" id="dash-total-referrals">0</span>
                        <span class="stat-desc">Total Referrals</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="dash-active-referrals">0</span>
                        <span class="stat-desc">Active Subscribers</span>
                    </div>
                </div>
            </div>
            
            <!-- Recent Transactions -->
            <div class="transactions-section">
                <h3>Recent Earnings</h3>
                <div class="transactions-list" id="dash-transactions">
                    <p class="empty-state">No earnings yet. Share your link to start earning!</p>
                </div>
            </div>
            
            <!-- Payout Button -->
            <div class="payout-section">
                <button class="btn-primary-glass btn-payout" id="btn-request-payout" disabled>
                    Request Payout ($50 minimum)
                </button>
            </div>
        </div>
    </div>
</div>
```

---

## PHASE 7: TESTING CHECKLIST (Day 8-9)

### 7.1 Manual Test Cases

| Test | Steps | Expected Result |
|------|-------|-----------------|
| Referral link capture | Visit `?ref=kelly_97d0` | localStorage has `kelly_referrer` |
| Click tracking | Check Supabase `referral_clicks` | New row with referrer_id |
| Signup attribution | Sign up after clicking ref link | User has `referred_by_user_id` |
| Commission on purchase | Complete Stripe checkout | Commission transaction created |
| Tier upgrade | Complete 7 lessons | Commission rate updates to 15% |
| UI display | Open drawer menu | Earnings and link shown |
| Copy link | Click copy button | Link in clipboard |
| Social share | Click Twitter share | Twitter intent opens |
| Lesson complete share | Finish a lesson | Share prompt appears |

### 7.2 Edge Cases to Test

- Self-referral prevention
- Multiple clicks from same user
- Referral after account exists
- Subscription renewal commission
- Refund clawback
- Commission tier boundary (6→7 lessons)

---

## PHASE 8: LAUNCH READINESS (Day 9-10)

### 8.1 Pre-Launch Checklist

- [ ] All existing users have referral codes
- [ ] Landing page captures `?ref=` parameter
- [ ] Signup flow links referrals
- [ ] Stripe webhook deployed and tested
- [ ] Share UI in drawer menu
- [ ] Share prompt on lesson complete
- [ ] Commission tiers calculating correctly
- [ ] Earnings display accurate
- [ ] Copy link works
- [ ] Social shares work
- [ ] Mobile responsive

### 8.2 Monitoring Setup

```javascript
// Add to webhook handler
console.log(`[EARN_TO_LEARN] Commission recorded:`, {
  referrer_id,
  amount: commission_amount,
  type: transaction_type,
  timestamp: new Date().toISOString()
});

// Track in analytics
analytics.track('commission_earned', {
  referrer_id,
  amount: commission_amount,
  type: transaction_type
});
```

---

## FILE SUMMARY

| File | Changes | Priority |
|------|---------|----------|
| `api/referral/track-click.ts` | New file | P0 |
| `api/referral/link.ts` | New file | P0 |
| `api/webhooks/stripe.ts` | Add commission logic | P0 |
| `lesson-player-v2/index.html` | Add Share UI | P0 |
| `lesson-player-v2/css/styles.css` | Add Share styles | P0 |
| `lesson-player-v2/js/app.js` | Add earnings logic | P0 |
| Landing page entry | Add ref capture | P0 |

---

## ESTIMATED HOURS

| Phase | Hours | Cumulative |
|-------|-------|------------|
| Phase 1: Link Capture | 3 | 3 |
| Phase 2: Signup Attribution | 3 | 6 |
| Phase 3: Stripe Webhook | 4 | 10 |
| Phase 4: Share UI | 4 | 14 |
| Phase 5: Lesson Complete | 2 | 16 |
| Phase 6: Earnings Dashboard | 6 | 22 |
| Phase 7: Testing | 4 | 26 |
| Phase 8: Launch Prep | 2 | 28 |

**Total: ~28 hours of implementation**  
**Timeline: 7-8 working days**  
**Buffer for issues: 2-3 days**

---

## READY TO BUILD?

This plan is approved and ready for implementation. Start with Phase 1 (Link Capture) - it's the foundation everything else depends on.

**Command to begin:**
```
"Start Phase 1" - I'll create the track-click endpoint
"Start Phase 4" - I'll add the Share UI (visible immediately)
"Do both" - I'll work on frontend and backend in parallel
```

---

*Document: EARN_TO_LEARN_IMPLEMENTATION_PLAN.md*  
*Created: December 7, 2025*  
*Status: APPROVED - Ready for Implementation*


