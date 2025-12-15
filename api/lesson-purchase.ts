/**
 * Lesson Purchase API
 * 
 * POST /api/lesson-purchase
 * Creates a Stripe checkout session for a single lesson purchase ($1.99).
 * 
 * GET /api/lesson-purchase?day=N
 * Check if a lesson has been purchased.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const DEFAULT_LESSON_PRICE = 199; // $1.99 in cents

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Content-Type', 'application/json');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  
  // GET: Check if a lesson is purchased
  if (req.method === 'GET') {
    const dayNumber = parseInt(req.query.day as string);
    
    if (!dayNumber || dayNumber < 1 || dayNumber > 366) {
      return res.status(400).json({ error: 'Invalid day number' });
    }
    
    // For now, return not purchased (database table pending migration)
    // TODO: Query lesson_purchases table when migrations are run
    return res.status(200).json({ 
      purchased: false, 
      day_number: dayNumber
    });
  }
  
  // POST: Create checkout session
  if (req.method === 'POST') {
    const stripeKey = process.env.STRIPE_SECRET_KEY;
    
    if (!stripeKey) {
      return res.status(503).json({ 
        error: 'stripe_not_configured',
        message: 'Stripe is not configured'
      });
    }
    
    try {
      // Parse body
      const body = req.body || {};
      const dayNumber = parseInt(body.day_number) || 0;
      
      if (dayNumber < 1 || dayNumber > 366) {
        return res.status(400).json({ error: 'Invalid day number' });
      }
      
      // Dynamic import Stripe
      const Stripe = (await import('stripe')).default;
      const stripe = new Stripe(stripeKey, {
        apiVersion: '2023-10-16'
      });
      
      const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';
      const successUrl = `${siteUrl}/learn.html?day=${dayNumber}&purchase=success`;
      const cancelUrl = `${siteUrl}/learn.html?day=${dayNumber}&purchase=cancelled`;
      
      // Create checkout session
      const session = await stripe.checkout.sessions.create({
        payment_method_types: ['card'],
        line_items: [{
          price_data: {
            currency: 'usd',
            product_data: {
              name: `Daily Lesson: Day ${dayNumber}`,
              description: `Permanent access to Day ${dayNumber} lesson`,
            },
            unit_amount: DEFAULT_LESSON_PRICE
          },
          quantity: 1
        }],
        mode: 'payment',
        success_url: successUrl,
        cancel_url: cancelUrl,
        metadata: {
          type: 'single_lesson',
          day_number: dayNumber.toString(),
          source: 'web'
        }
      });
      
      return res.status(200).json({
        sessionId: session.id,
        url: session.url,
        price: {
          amount: DEFAULT_LESSON_PRICE / 100,
          currency: 'USD',
          formatted: '$1.99'
        }
      });
      
    } catch (error) {
      console.error('Error in lesson-purchase:', error);
      return res.status(500).json({
        error: 'checkout_failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}
