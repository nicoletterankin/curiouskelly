/**
 * Lesson Purchase API
 * 
 * POST /api/lesson-purchase
 * Creates a Stripe checkout session for a single lesson purchase ($1.99).
 * 
 * GET /api/lesson-purchase?day=N
 * Check if a lesson has been purchased by the current user.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

const DEFAULT_LESSON_PRICE = 199; // $1.99 in cents

async function getUserFromRequest(req: VercelRequest): Promise<string | null> {
  if (!isSupabaseConfigured()) return null;
  
  const authHeader = req.headers['authorization'];
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
  
  try {
    const token = authHeader.substring(7);
    const supabase = getSupabaseAdmin();
    const { data: { user } } = await supabase.auth.getUser(token);
    return user?.id || null;
  } catch {
    return null;
  }
}

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
    
    // Check database for purchase
    let purchased = false;
    if (isSupabaseConfigured()) {
      try {
        const userId = await getUserFromRequest(req);
        if (userId) {
          const supabase = getSupabaseAdmin();
          const { data, error } = await supabase
            .from('lesson_purchases')
            .select('id')
            .eq('user_id', userId)
            .eq('day_number', dayNumber)
            .eq('status', 'completed')
            .single();
          
          if (!error && data) {
            purchased = true;
          } else if (error && error.code === '42P01') {
            // Table doesn't exist yet
            console.log('[Purchase] lesson_purchases table not yet created');
          }
        }
      } catch (dbError) {
        console.error('[Purchase] Database error:', dbError);
      }
    }
    
    return res.status(200).json({ 
      purchased, 
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
      
      // Get user ID if authenticated
      const userId = await getUserFromRequest(req);
      
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
          user_id: userId || 'anonymous',
          source: 'web'
        }
      });
      
      // Pre-create pending purchase record if database is ready
      if (isSupabaseConfigured() && userId) {
        try {
          const supabase = getSupabaseAdmin();
          await supabase.from('lesson_purchases').insert({
            user_id: userId,
            day_number: dayNumber,
            purchase_price: DEFAULT_LESSON_PRICE / 100,
            currency: 'USD',
            stripe_checkout_session_id: session.id,
            status: 'pending'
          }).catch(() => {
            // Table might not exist yet
          });
        } catch {
          // Non-critical, continue
        }
      }
      
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
