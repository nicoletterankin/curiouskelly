import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ 
    status: 'ok',
    timestamp: new Date().toISOString(),
    node: process.version,
    env: {
      hasStripeKey: !!process.env.STRIPE_SECRET_KEY,
      hasStripePrice: !!process.env.STRIPE_PRICE_MONTHLY
    }
  });
}


