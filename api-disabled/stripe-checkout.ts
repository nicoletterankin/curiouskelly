import type { VercelRequest, VercelResponse } from '@vercel/node';

interface CheckoutRequest {
  planType: 'monthly' | 'annual' | 'lifetime' | 'family' | 'gift';
  customerEmail: string;
  promoCode?: string;
  affiliateCode?: string;
  giftData?: {
    recipientEmail: string;
    gifterName?: string;
    message?: string;
    deliveryDate?: string;
  };
  utmSource?: string;
  utmMedium?: string;
  utmCampaign?: string;
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const stripeKey = process.env.STRIPE_SECRET_KEY;
  if (!stripeKey) {
    return res.status(503).json({ 
      error: 'stripe_not_configured',
      message: 'Stripe is not configured. Please add STRIPE_SECRET_KEY to environment variables.'
    });
  }

  // Use require for CommonJS compatibility with Vercel
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const Stripe = require('stripe');
  const stripe = new Stripe(stripeKey, {
    apiVersion: '2023-10-16' // Compatible with stripe v14
  });

  const body = req.body as CheckoutRequest;

  // Validate email
  if (!body.customerEmail || !isValidEmail(body.customerEmail)) {
    return res.status(422).json({ error: 'invalid_email' });
  }

  // Validate plan type
  const validPlanTypes = ['monthly', 'annual', 'lifetime', 'family', 'gift'];
  if (!validPlanTypes.includes(body.planType)) {
    return res.status(422).json({ error: 'invalid_plan_type' });
  }

  // Get price IDs from environment (support multiple naming conventions)
  const priceIds: Record<string, string | undefined> = {
    monthly: process.env.STRIPE_PRICE_MONTHLY,
    annual: process.env.STRIPE_PRICE_ANNUAL,
    lifetime: process.env.STRIPE_PRICE_LIFETIME || process.env.STRIPE_PRICE_FAMILY,
    family: process.env.STRIPE_PRICE_FAMILY || process.env.STRIPE_PRICE_LIFETIME,
    gift: process.env.STRIPE_PRICE_GIFT
  };

  const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';
  const isGiftPlan = body.planType === 'gift';

  // Build metadata
  const commonMetadata = {
    source: 'web',
    utm_source: body.utmSource || 'direct',
    utm_medium: body.utmMedium || 'none',
    utm_campaign: body.utmCampaign || 'none',
    affiliate_code: body.affiliateCode || '',
    promo_code: body.promoCode || ''
  };

  try {
    let sessionConfig: any; // Stripe.Checkout.SessionCreateParams

    if (isGiftPlan) {
      const giftPriceId = priceIds.gift;
      if (!giftPriceId) {
        return res.status(503).json({ 
          error: `price_not_configured`,
          message: `Gift price ID not configured. Add STRIPE_PRICE_GIFT to environment.`
        });
      }

      const giftMeta = body.giftData || {};
      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [{ price: giftPriceId, quantity: 1 }],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/gift-success.html?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        allow_promotion_codes: true,
        metadata: {
          ...commonMetadata,
          type: 'gift',
          recipient_email: giftMeta.recipientEmail || '',
          gift_message: giftMeta.message || '',
          gifter_name: giftMeta.gifterName || '',
          delivery_date: giftMeta.deliveryDate || new Date().toISOString()
        }
      };
    } else if (body.planType === 'lifetime' || body.planType === 'family') {
      const oneTimePriceId = priceIds[body.planType];
      if (!oneTimePriceId) {
        return res.status(503).json({ 
          error: 'price_not_configured',
          message: `${body.planType} price ID not configured. Add STRIPE_PRICE_${body.planType.toUpperCase()} to environment.`
        });
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [{ price: oneTimePriceId, quantity: 1 }],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/welcome.html?session_id={CHECKOUT_SESSION_ID}&plan=${body.planType}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        allow_promotion_codes: true,
        metadata: { ...commonMetadata, type: body.planType }
      };
    } else {
      // Subscription plans (monthly, annual)
      const priceId = priceIds[body.planType];
      if (!priceId) {
        return res.status(503).json({ 
          error: 'price_not_configured',
          message: `Price ID for ${body.planType} not configured. Add STRIPE_PRICE_${body.planType.toUpperCase()} to environment.`
        });
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [{ price: priceId, quantity: 1 }],
        mode: 'subscription',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/welcome.html?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        metadata: { ...commonMetadata, type: body.planType },
        subscription_data: {
          metadata: { ...commonMetadata, plan_type: body.planType },
          trial_period_days: 7
        },
        allow_promotion_codes: true,
        billing_address_collection: 'auto'
      };
    }

    const session = await stripe.checkout.sessions.create(sessionConfig);

    return res.status(200).json({
      sessionId: session.id,
      url: session.url
    });
  } catch (error) {
    console.error('Stripe checkout error:', error);
    return res.status(500).json({
      error: 'checkout_failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
