import type { VercelRequest, VercelResponse } from '@vercel/node';
import { checkRateLimit, getRateLimitHeaders, RATE_LIMITS, getClientIdentifier } from './lib/rate-limit';

/**
 * Stripe Checkout API Handler
 * 
 * 🔒 LOCKED PRICING (See PRICING_LOCKED.md):
 * - Monthly: $7.99/month
 * - Annual: $49.99/year (DEFAULT)
 * - Family: $99.99/year
 * - Lifetime: $199.99 one-time
 * - Gifts: $24.99 (3mo), $39.99 (6mo), $49.99 (12mo), $149.99 (lifetime)
 */

interface CheckoutRequest {
  planType: 'monthly' | 'annual' | 'lifetime' | 'family' | 'gift_3mo' | 'gift_6mo' | 'gift_12mo' | 'gift_lifetime';
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
  // Wrap entire handler in try-catch to capture any errors
  try {
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

    const body = req.body as CheckoutRequest;

    // Rate limiting: use email if provided, otherwise IP
    const identifier = getClientIdentifier(
      req as unknown as { headers: Record<string, string | string[] | undefined> },
      body.customerEmail
    );
    const isGift = body.planType?.startsWith('gift_');
    const rateLimitConfig = isGift ? RATE_LIMITS.giftCheckout : RATE_LIMITS.checkout;
    const rateLimitResult = checkRateLimit(identifier, rateLimitConfig);
    
    // Set rate limit headers
    const rateLimitHeaders = getRateLimitHeaders(rateLimitResult, rateLimitConfig.limit);
    Object.entries(rateLimitHeaders).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
    
    if (!rateLimitResult.allowed) {
      return res.status(429).json({
        error: 'rate_limited',
        message: `Too many checkout attempts. Please try again in ${rateLimitResult.retryAfterSecs} seconds.`,
        retryAfter: rateLimitResult.retryAfterSecs,
      });
    }

    const stripeKey = process.env.STRIPE_SECRET_KEY;
    if (!stripeKey) {
      return res.status(503).json({ 
        error: 'stripe_not_configured',
        message: 'Stripe is not configured. Please add STRIPE_SECRET_KEY to environment variables.'
      });
    }

    // Dynamic import to ensure proper module resolution
    const Stripe = (await import('stripe')).default;
    const stripe = new Stripe(stripeKey, {
      apiVersion: '2024-11-20.acacia' as const, // Latest stable API version
    });

    // Validate email
    if (!body.customerEmail || !isValidEmail(body.customerEmail)) {
      return res.status(422).json({ error: 'invalid_email' });
    }

    // Validate plan type - LOCKED PRICING
    const validPlanTypes = ['monthly', 'annual', 'lifetime', 'family', 'gift_3mo', 'gift_6mo', 'gift_12mo', 'gift_lifetime'];
    if (!validPlanTypes.includes(body.planType)) {
      return res.status(422).json({ error: 'invalid_plan_type' });
    }

    // Get price IDs from environment
    // 🔒 LOCKED PRICING - See PRICING_LOCKED.md
    // Monthly: $7.99, Annual: $49.99, Family: $99.99, Lifetime: $199.99
    // Gifts: $24.99 (3mo), $39.99 (6mo), $49.99 (12mo), $149.99 (lifetime)
    const priceIds: Record<string, string | undefined> = {
      monthly: process.env.STRIPE_PRICE_MONTHLY,
      annual: process.env.STRIPE_PRICE_ANNUAL,
      lifetime: process.env.STRIPE_PRICE_LIFETIME,
      family: process.env.STRIPE_PRICE_FAMILY,
      gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO,
      gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO,
      gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO,
      gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME,
    };

    const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';
    const isGiftPlan = body.planType.startsWith('gift_');

    // Build metadata
    const commonMetadata = {
      source: 'web',
      utm_source: body.utmSource || 'direct',
      utm_medium: body.utmMedium || 'none',
      utm_campaign: body.utmCampaign || 'none',
      affiliate_code: body.affiliateCode || '',
      promo_code: body.promoCode || ''
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let sessionConfig: any; // Stripe.Checkout.SessionCreateParams

    if (isGiftPlan) {
      // Gift plans: gift_3mo, gift_6mo, gift_12mo, gift_lifetime
      const giftPriceId = priceIds[body.planType];
      if (!giftPriceId) {
        return res.status(503).json({ 
          error: `price_not_configured`,
          message: `Gift price ID not configured. Add STRIPE_PRICE_${body.planType.toUpperCase()} to environment.`
        });
      }

      const giftMeta = body.giftData || {
        recipientEmail: '',
        gifterName: '',
        message: '',
        deliveryDate: ''
      };
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
          type: body.planType,
          recipient_email: giftMeta.recipientEmail || '',
          gift_message: giftMeta.message || '',
          gifter_name: giftMeta.gifterName || '',
          delivery_date: giftMeta.deliveryDate || new Date().toISOString()
        }
      };
    } else if (body.planType === 'lifetime') {
      // Lifetime is a one-time payment
      const lifetimePriceId = priceIds[body.planType];
      if (!lifetimePriceId) {
        return res.status(503).json({ 
          error: 'price_not_configured',
          message: `Lifetime price ID not configured. Add STRIPE_PRICE_LIFETIME to environment.`
        });
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [{ price: lifetimePriceId, quantity: 1 }],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/welcome.html?session_id={CHECKOUT_SESSION_ID}&plan=${body.planType}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        allow_promotion_codes: true,
        metadata: { ...commonMetadata, type: body.planType }
      };
    } else {
      // Subscription plans: monthly, annual, family
      // Family is a recurring annual subscription (up to 6 members)
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

    // Generate idempotency key to prevent duplicate charges on retries
    // Uses email + plan + 5-minute window to allow retries within same session
    const timeWindow = Math.floor(Date.now() / (5 * 60 * 1000)); // 5-minute buckets
    const idempotencyKey = `checkout_${body.customerEmail}_${body.planType}_${timeWindow}`;

    const session = await stripe.checkout.sessions.create(sessionConfig, {
      idempotencyKey,
    });

    return res.status(200).json({
      sessionId: session.id,
      url: session.url
    });
  } catch (error) {
    console.error('Stripe checkout error:', error);
    // Never expose stack traces in production
    return res.status(500).json({
      error: 'checkout_failed',
      message: process.env.NODE_ENV === 'development' && error instanceof Error 
        ? error.message 
        : 'An error occurred during checkout. Please try again.'
    });
  }
}
