import type { VercelRequest, VercelResponse } from '@vercel/node';
import { checkRateLimit, getRateLimitHeaders, RATE_LIMITS, getClientIdentifier } from './lib/rate-limit';
import {
  getCurrencyForCountry,
  getPriceIdForPlan,
  getPaymentMethodsForCountry,
  getEffectiveCurrency,
  PRICE_IDS,
} from './lib/pricing-config';

/**
 * Embedded Checkout Session Creator (in-app)
 * POST /api/create-checkout
 *
 * Security model:
 * - Client sends a planType (NOT a Stripe price id).
 * - Server maps planType -> env-configured price ids (allow-list).
 * - Returns a client_secret for Stripe Embedded Checkout.
 * 
 * Multi-currency support:
 * - Client sends optional `currency` param
 * - Server detects country from Vercel geo headers
 * - Uses appropriate price ID and payment methods
 */

type PlanType = 'monthly' | 'annual' | 'family' | 'lifetime';

interface CreateCheckoutRequest {
  planType: PlanType;
  customerEmail?: string;
  // Multi-currency support
  currency?: string; // e.g., 'EUR', 'GBP', 'INR'
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
  // Wrap entire handler in try-catch to capture any errors
  try {
    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') return res.status(200).end();
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const body = (req.body || {}) as CreateCheckoutRequest;

    // Rate limiting: use email or userId if provided, otherwise IP
    const identifier = getClientIdentifier(
      req as unknown as { headers: Record<string, string | string[] | undefined> },
      body.customerEmail,
      body.userId
    );
    const rateLimitResult = checkRateLimit(identifier, RATE_LIMITS.checkout);
    
    // Set rate limit headers
    const rateLimitHeaders = getRateLimitHeaders(rateLimitResult, RATE_LIMITS.checkout.limit);
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
        message: 'Missing STRIPE_SECRET_KEY',
      });
    }

    // Dynamic import to ensure proper module resolution
    const Stripe = (await import('stripe')).default;
    const stripe = new Stripe(stripeKey, { apiVersion: '2024-11-20.acacia' as const });

    const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

    const planType = body.planType;
    if (!planType || !(['monthly', 'annual', 'family', 'lifetime'] as const).includes(planType)) {
      return res.status(422).json({ error: 'invalid_plan_type' });
    }

    // Multi-currency: detect country and get appropriate currency
    const detectedCountry = (req.headers['x-vercel-ip-country'] as string) || 'US';
    const requestedCurrency = body.currency?.toUpperCase() || getCurrencyForCountry(detectedCountry);
    
    // Get effective currency (falls back to USD if price not available)
    const currency = getEffectiveCurrency(planType, requestedCurrency);
    
    // Get price ID for this plan and currency
    const currencyPrices = PRICE_IDS[currency] || PRICE_IDS.USD;
    const priceId = currencyPrices[planType];
    
    if (!priceId) {
      // Fall back to USD pricing
      const fallbackPrices = PRICE_IDS.USD;
      const fallbackPriceId = fallbackPrices[planType];
      if (!fallbackPriceId) {
        return res.status(503).json({
          error: 'price_not_configured',
          message: `Missing STRIPE_PRICE_${planType.toUpperCase()} in environment`,
        });
      }
      // Use fallback
      console.log(`[checkout] No ${currency} price for ${planType}, using USD fallback`);
    }
    
    const effectivePriceId = priceId || PRICE_IDS.USD[planType];
    
    // Get payment methods for this country
    const paymentMethods = getPaymentMethodsForCountry(detectedCountry);

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
      currency: currency,
      detected_country: detectedCountry,
      user_id: body.userId || '',
      referral_code: body.referralCode || '',
      affiliate_code: body.affiliateCode || '',
      promo_code: body.promoCode || '',
    };

    const mode = planType === 'lifetime' ? 'payment' : 'subscription';

    // Generate idempotency key to prevent duplicate charges on retries
    // Uses email/userId + plan + 5-minute window
    const timeWindow = Math.floor(Date.now() / (5 * 60 * 1000)); // 5-minute buckets
    const userKey = body.userId || customerEmail || 'anonymous';
    const idempotencyKey = `embedded_${userKey}_${planType}_${timeWindow}`;

    const session = await stripe.checkout.sessions.create({
      mode,
      ui_mode: 'embedded',
      return_url: returnUrl,

      // Core purchase
      line_items: [{ price: effectivePriceId, quantity: 1 }],

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
      
      // Dynamic payment methods based on country
      // Note: Embedded checkout handles payment methods automatically,
      // but we can hint at preferred methods
      payment_method_configuration: undefined, // Let Stripe auto-detect
    }, {
      idempotencyKey,
    });

    return res.status(200).json({
      sessionId: session.id,
      clientSecret: session.client_secret,
    });
  } catch (error) {
    console.error('Embedded checkout error:', error);
    // Never expose stack traces in production
    return res.status(500).json({
      error: 'checkout_failed',
      message: process.env.NODE_ENV === 'development' && error instanceof Error 
        ? error.message 
        : 'An error occurred during checkout. Please try again.'
    });
  }
}

