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
  country?: string; // Country code (US, DE, IN, etc.)
  currency?: string; // Currency code (USD, EUR, INR, etc.)
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

/**
 * Get Stripe locale from currency/country
 * Stripe supports: auto, bg, cs, da, de, el, en, es, et, fi, fr, hu, id, it, ja, lt, lv, ms, mt, nb, nl, pl, pt, ro, ru, sk, sl, sv, tr, zh
 */
function getStripeLocale(currency: string, country: string): string {
  const localeMap: Record<string, string> = {
    'EUR': 'de', // Default EU to German
    'GBP': 'en',
    'CAD': 'en',
    'AUD': 'en',
    'INR': 'en', // Stripe doesn't support Hindi UI, use English
    'BRL': 'pt',
    'MXN': 'es',
    'PLN': 'pl',
  };
  
  const countryLocaleMap: Record<string, string> = {
    'DE': 'de',
    'FR': 'fr',
    'ES': 'es',
    'IT': 'it',
    'NL': 'nl',
    'PL': 'pl',
    'BR': 'pt',
    'MX': 'es',
  };
  
  return countryLocaleMap[country] || localeMap[currency] || 'auto';
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

    // Get currency from request or default to USD
    const currency = (body.currency || 'USD').toUpperCase();
    const country = body.country || 'US';
    
    // Map planType + currency -> configured price ids
    // TODO: STRIPE_I18N_BACKLOG - After Stripe batch work is complete, these env vars will exist
    // For now, fallback to USD prices if currency-specific price not found
    const getPriceId = (plan: PlanType, curr: string): string | undefined => {
      // Try currency-specific price first (e.g., STRIPE_PRICE_MONTHLY_EUR)
      const currencySpecific = process.env[`STRIPE_PRICE_${plan.toUpperCase()}_${curr}`];
      if (currencySpecific) return currencySpecific;
      
      // Fallback to USD prices
      const usdPrices: Record<PlanType, string | undefined> = {
        monthly: process.env.STRIPE_PRICE_MONTHLY,
        annual: process.env.STRIPE_PRICE_ANNUAL,
        family: process.env.STRIPE_PRICE_FAMILY,
        lifetime: process.env.STRIPE_PRICE_LIFETIME,
      };
      return usdPrices[plan];
    };
    
    const priceId = getPriceId(planType, currency);
    if (!priceId) {
      return res.status(503).json({
        error: 'price_not_configured',
        message: `Missing STRIPE_PRICE_${planType.toUpperCase()}_${currency} or STRIPE_PRICE_${planType.toUpperCase()} in environment`,
        // Flag for backlog: This will work once Stripe batch work is complete
        _backlog_flag: 'STRIPE_I18N_BACKLOG',
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

    // Determine Stripe locale from currency/country
    const stripeLocale = getStripeLocale(currency, country);
    
    const session = await stripe.checkout.sessions.create({
      mode,
      ui_mode: 'embedded',
      locale: stripeLocale,
      return_url: returnUrl,
      line_items: [{ price: priceId, quantity: 1 }],
      ...(customerEmail ? { customer_email: customerEmail } : {}),
      allow_promotion_codes: true,
      metadata: {
        ...metadata,
        country,
        currency,
      },
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
