/**
 * Lesson Purchase API
 * 
 * POST /api/lesson-purchase
 * Creates a Stripe checkout session for a single lesson purchase ($1.99).
 * 
 * GET /api/lesson-purchase?day=N
 * Check if a lesson has been purchased.
 * 
 * Updated: Dec 15, 2025 - Simplified for initial deployment
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Content-Type', 'application/json');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).send('');
  }
  
  // GET: Check if a lesson is purchased
  if (req.method === 'GET') {
    const dayNumber = parseInt(req.query.day as string);
    
    if (!dayNumber || dayNumber < 1 || dayNumber > 366) {
      return res.status(400).send(JSON.stringify({ error: 'Invalid day number' }));
    }
    
    // For now, return not purchased (database table pending migration)
    return res.status(200).send(JSON.stringify({ 
      purchased: false, 
      day_number: dayNumber,
      note: 'Purchase check pending database migration' 
    }));
  }
  
  // POST: Create checkout session
  if (req.method === 'POST') {
    const stripeKey = process.env.STRIPE_SECRET_KEY;
    
    if (!stripeKey) {
      return res.status(503).send(JSON.stringify({ 
        error: 'stripe_not_configured',
        message: 'Stripe is not configured'
      }));
    }
    
    // For now, return a placeholder (full Stripe integration pending)
    const body = req.body || {};
    const dayNumber = body.day_number;
    
    if (!dayNumber || dayNumber < 1 || dayNumber > 366) {
      return res.status(400).send(JSON.stringify({ error: 'Invalid day number' }));
    }
    
    // Return info about how to purchase (Stripe checkout will be added)
    return res.status(200).send(JSON.stringify({
      message: 'Single lesson purchase endpoint ready',
      day_number: dayNumber,
      price: '$1.99',
      note: 'Full Stripe integration coming soon'
    }));
  }
  
  return res.status(405).send(JSON.stringify({ error: 'Method not allowed' }));
}
