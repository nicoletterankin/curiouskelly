import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Embedded Checkout Session Creator (in-app)
 * POST /api/create-checkout
 * 
 * Simplified version without rate limiting for debugging.
 */

type PlanType = 'monthly' | 'annual' | 'family' | 'lifetime';

interface CreateCheckoutRequest {
  planType: PlanType;
  customerEmail?: string;
  userId?: string;
  returnUrl?: string;
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') return res.status(200).end();
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const body = (req.body || {}) as CreateCheckoutRequest;

    const stripeKey = process.env.STRIPE_SECRET_KEY;
    if (!stripeKey) {
      return res.status(503).json({
        error: 'stripe_not_configured',
        message: 'Missing STRIPE_SECRET_KEY',
      });
    }

    const Stripe = (await import('stripe')).default;
    // Keep this pinned to the Stripe SDK's supported type union for our installed version.
    const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });

    const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

    const planType = body.planType;
    if (!planType || !(['monthly', 'annual', 'family', 'lifetime'] as const).includes(planType)) {
      return res.status(422).json({ error: 'invalid_plan_type' });
    }

    // Map planType -> configured price ids
    const priceIds: Record<PlanType, string | undefined> = {
      monthly: process.env.STRIPE_PRICE_MONTHLY,
      annual: process.env.STRIPE_PRICE_ANNUAL,
      family: process.env.STRIPE_PRICE_FAMILY,
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

    const defaultReturnUrl = `${siteUrl}/learn.html?checkout=success`;
    const returnUrl = body.returnUrl && body.returnUrl.startsWith(siteUrl) ? body.returnUrl : defaultReturnUrl;

    const metadata: Record<string, string> = {
      source: 'kelly_in_app',
      plan_type: planType,
      user_id: body.userId || '',
    };

    const mode = planType === 'lifetime' ? 'payment' : 'subscription';

    const session = await stripe.checkout.sessions.create({
      mode,
      ui_mode: 'embedded',
      locale:
        (Array.isArray(req.headers['accept-language'])
          ? req.headers['accept-language'][0]
          : req.headers['accept-language'])
          ?.split(',')[0]
          ?.split('-')[0] || 'auto',
      return_url: returnUrl,
      line_items: [{ price: priceId, quantity: 1 }],
      ...(customerEmail ? { customer_email: customerEmail } : {}),
      allow_promotion_codes: true,
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
      automatic_tax: { enabled: true },
    });

    return res.status(200).json({
      sessionId: session.id,
      clientSecret: session.client_secret,
    });
  } catch (error) {
    console.error('Checkout error:', error);
    return res.status(500).json({
      error: 'checkout_failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
