import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Embedded Checkout Session Creator (in-app)
 * POST /api/create-checkout
 *
 * Security model:
 * - Client sends a planType (NOT a Stripe price id).
 * - Server maps planType -> env-configured price ids (allow-list).
 * - Returns a client_secret for Stripe Embedded Checkout.
 */

type PlanType = 'monthly' | 'annual' | 'lifetime';

interface CreateCheckoutRequest {
  planType: PlanType;
  customerEmail?: string;
  // Optional attribution metadata
  referralCode?: string;
  affiliateCode?: string;
  promoCode?: string;
  // Optional: used for linking after signup/login
  userId?: string;
  // Optional override; defaults to /learn.html
  returnUrl?: string;
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const stripeKey = process.env.STRIPE_SECRET_KEY;
  if (!stripeKey) {
    return res.status(503).json({
      error: 'stripe_not_configured',
      message: 'Missing STRIPE_SECRET_KEY',
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const Stripe = require('stripe');
  const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });

  const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';
  const body = (req.body || {}) as CreateCheckoutRequest;

  const planType = body.planType;
  if (!planType || !(['monthly', 'annual', 'lifetime'] as const).includes(planType)) {
    return res.status(422).json({ error: 'invalid_plan_type' });
  }

  // Map planType -> configured price ids
  const priceIds: Record<PlanType, string | undefined> = {
    monthly: process.env.STRIPE_PRICE_MONTHLY,
    annual: process.env.STRIPE_PRICE_ANNUAL,
    lifetime: process.env.STRIPE_PRICE_LIFETIME,
  };
  const priceId = priceIds[planType];
  if (!priceId) {
    return res.status(503).json({
      error: 'price_not_configured',
      message: `Missing STRIPE_PRICE_${planType.toUpperCase()} in environment`,
    });
  }

  const customerEmail = body.customerEmail?.trim();
  if (customerEmail && !isValidEmail(customerEmail)) {
    return res.status(422).json({ error: 'invalid_email' });
  }

  // Keep return_url inside Kelly (no domain escape)
  const defaultReturnUrl = `${siteUrl}/learn.html?checkout=success`;
  const returnUrl = body.returnUrl && body.returnUrl.startsWith(siteUrl) ? body.returnUrl : defaultReturnUrl;

  const metadata: Record<string, string> = {
    source: 'kelly_in_app',
    plan_type: planType,
    user_id: body.userId || '',
    referral_code: body.referralCode || '',
    affiliate_code: body.affiliateCode || '',
    promo_code: body.promoCode || '',
  };

  try {
    const mode = planType === 'lifetime' ? 'payment' : 'subscription';

    const session = await stripe.checkout.sessions.create({
      mode,
      ui_mode: 'embedded',
      return_url: returnUrl,

      // Core purchase
      line_items: [{ price: priceId, quantity: 1 }],

      // Email helps attribution + linking even before userId is known
      ...(customerEmail ? { customer_email: customerEmail } : {}),

      // Promo codes
      allow_promotion_codes: true,

      // Keep metadata on both session and subscription (for webhooks / analytics)
      metadata,
      ...(mode === 'subscription'
        ? {
            subscription_data: {
              metadata,
              trial_period_days: 7,
            },
          }
        : {}),

      billing_address_collection: 'auto',

      // If Stripe decides taxes apply, calculate automatically
      automatic_tax: { enabled: true },
    });

    return res.status(200).json({
      sessionId: session.id,
      clientSecret: session.client_secret,
    });
  } catch (error) {
    console.error('Embedded checkout error:', error);
    return res.status(500).json({
      error: 'checkout_failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

