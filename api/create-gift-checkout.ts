import type { VercelRequest, VercelResponse } from '@vercel/node';
import { checkRateLimit, getRateLimitHeaders, RATE_LIMITS, getClientIdentifier } from './lib/rate-limit';

/**
 * Gift Checkout Session Creator
 * POST /api/create-gift-checkout
 *
 * Creates a Stripe checkout session for gift purchases.
 * Gift purchaser stays on Kelly's site - embedded checkout.
 */

type GiftDuration = '3-month' | '6-month' | '12-month' | 'lifetime';

interface CreateGiftCheckoutRequest {
  giftDuration: GiftDuration;
  senderEmail: string;
  recipientEmail: string;
  recipientName?: string;
  giftMessage?: string;
  // Optional attribution
  referralCode?: string;
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

    const body = (req.body || {}) as CreateGiftCheckoutRequest;

    // Rate limiting: use sender email
    const identifier = getClientIdentifier(
      req as unknown as { headers: Record<string, string | string[] | undefined> },
      body.senderEmail
    );
    const rateLimitResult = checkRateLimit(identifier, RATE_LIMITS.giftCheckout);
    
    // Set rate limit headers
    const rateLimitHeaders = getRateLimitHeaders(rateLimitResult, RATE_LIMITS.giftCheckout.limit);
    Object.entries(rateLimitHeaders).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
    
    if (!rateLimitResult.allowed) {
      return res.status(429).json({
        error: 'rate_limited',
        message: `Too many gift checkout attempts. Please try again in ${rateLimitResult.retryAfterSecs} seconds.`,
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

    const Stripe = (await import('stripe')).default;
    const stripe = new Stripe(stripeKey, { apiVersion: '2024-11-20.acacia' as const });

    const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

    // Validate gift duration
    const validDurations: GiftDuration[] = ['3-month', '6-month', '12-month', 'lifetime'];
    if (!body.giftDuration || !validDurations.includes(body.giftDuration)) {
      return res.status(422).json({ error: 'invalid_gift_duration' });
    }

    // Validate emails
    if (!body.senderEmail || !isValidEmail(body.senderEmail)) {
      return res.status(422).json({ error: 'invalid_sender_email' });
    }
    if (!body.recipientEmail || !isValidEmail(body.recipientEmail)) {
      return res.status(422).json({ error: 'invalid_recipient_email' });
    }

    // Map gift duration -> price id
    const giftPriceIds: Record<GiftDuration, string | undefined> = {
      '3-month': process.env.STRIPE_PRICE_GIFT_3MO,
      '6-month': process.env.STRIPE_PRICE_GIFT_6MO,
      '12-month': process.env.STRIPE_PRICE_GIFT_12MO,
      'lifetime': process.env.STRIPE_PRICE_GIFT_LIFETIME,
    };

    const priceId = giftPriceIds[body.giftDuration];
    if (!priceId) {
      return res.status(503).json({
        error: 'gift_price_not_configured',
        message: `Missing STRIPE_PRICE_GIFT_${body.giftDuration.toUpperCase().replace('-', '_')} in environment`,
      });
    }

    const metadata: Record<string, string> = {
      source: 'kelly_gift',
      gift_duration: body.giftDuration,
      sender_email: body.senderEmail.trim().toLowerCase(),
      recipient_email: body.recipientEmail.trim().toLowerCase(),
      recipient_name: body.recipientName || '',
      gift_message: (body.giftMessage || '').slice(0, 500), // Limit message length
      referral_code: body.referralCode || '',
    };

    // Redirect to gift success page with session ID for confirmation display
    const returnUrl = `${siteUrl}/gift-success.html?session_id={CHECKOUT_SESSION_ID}`;

    // Generate idempotency key to prevent duplicate charges on retries
    // Uses sender email + recipient + duration + 5-minute window
    const timeWindow = Math.floor(Date.now() / (5 * 60 * 1000)); // 5-minute buckets
    const idempotencyKey = `gift_${body.senderEmail}_${body.recipientEmail}_${body.giftDuration}_${timeWindow}`;

    const session = await stripe.checkout.sessions.create({
      mode: 'payment',
      ui_mode: 'embedded',
      locale:
        (Array.isArray(req.headers['accept-language'])
          ? req.headers['accept-language'][0]
          : req.headers['accept-language'])
          ?.split(',')[0]
          ?.split('-')[0] || 'auto',
      return_url: returnUrl,

      line_items: [{ price: priceId, quantity: 1 }],

      // Use sender's email for the purchase
      customer_email: body.senderEmail.trim().toLowerCase(),

      // Allow promo codes for gifts too
      allow_promotion_codes: true,

      metadata,

      billing_address_collection: 'auto',
      automatic_tax: { enabled: true },
    }, {
      idempotencyKey,
    });

    return res.status(200).json({
      sessionId: session.id,
      clientSecret: session.client_secret,
    });
  } catch (error) {
    console.error('Gift checkout error:', error);
    return res.status(500).json({
      error: 'gift_checkout_failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

