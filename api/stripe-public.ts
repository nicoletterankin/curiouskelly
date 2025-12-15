import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Stripe Public Config
 * GET /api/stripe-public
 *
 * Returns ONLY safe-to-expose values needed by Stripe.js (publishable key).
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS (learn.html is same-origin, but keep it simple for local testing)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const publishableKey = process.env.STRIPE_PUBLISHABLE_KEY;
  if (!publishableKey) {
    return res.status(503).json({
      error: 'stripe_not_configured',
      message: 'Missing STRIPE_PUBLISHABLE_KEY',
    });
  }

  return res.status(200).json({
    publishableKey,
  });
}

