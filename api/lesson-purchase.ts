/**
 * Lesson Purchase API
 * 
 * POST /api/lesson-purchase
 * 
 * Creates a Stripe checkout session for a single lesson purchase ($1.99).
 * After successful payment, the lesson is permanently unlocked for the user.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

interface LessonPurchaseRequest {
  day_number: number;
  customer_email?: string;
  success_url?: string;
  cancel_url?: string;
}

// Default price for a single lesson
const DEFAULT_LESSON_PRICE = 199; // $1.99 in cents
const DEFAULT_CURRENCY = 'usd';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // GET: Check if a lesson is purchased
  if (req.method === 'GET') {
    return handleCheckPurchase(req, res);
  }
  
  // POST: Create checkout session
  if (req.method === 'POST') {
    return handleCreateCheckout(req, res);
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}

/**
 * Check if a user has purchased a specific lesson
 */
async function handleCheckPurchase(req: VercelRequest, res: VercelResponse) {
  const dayNumber = parseInt(req.query.day as string);
  
  if (!dayNumber || dayNumber < 1 || dayNumber > 366) {
    return res.status(400).json({ error: 'Invalid day number' });
  }
  
  // Get user from auth token
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(200).json({ purchased: false, reason: 'not_authenticated' });
  }
  
  if (!isSupabaseConfigured()) {
    return res.status(503).json({ error: 'Database not configured' });
  }
  
  const supabase = getSupabaseAdmin();
  const token = authHeader.substring(7);
  
  try {
    const { data: { user } } = await supabase.auth.getUser(token);
    if (!user) {
      return res.status(200).json({ purchased: false, reason: 'invalid_token' });
    }
    
    // Check for purchase (table might not exist yet)
    try {
      const { data: purchase, error: purchaseError } = await supabase
        .from('lesson_purchases')
        .select('id, purchased_at')
        .eq('user_id', user.id)
        .eq('day_number', dayNumber)
        .eq('status', 'completed')
        .single();
      
      if (purchaseError && purchaseError.code !== 'PGRST116') {
        // PGRST116 = no rows found, which is fine
        // Other errors might mean table doesn't exist
        console.warn('Purchase check error:', purchaseError.message);
      }
      
      if (purchase) {
        return res.status(200).json({ 
          purchased: true, 
          purchased_at: purchase.purchased_at 
        });
      }
    } catch (e) {
      // Table might not exist yet
      console.warn('Purchase check failed (table may not exist)');
    }
    
    return res.status(200).json({ purchased: false });
    
  } catch (error) {
    console.error('Error checking purchase:', error);
    return res.status(200).json({ purchased: false, reason: 'error' });
  }
}

/**
 * Create a Stripe checkout session for a single lesson
 */
async function handleCreateCheckout(req: VercelRequest, res: VercelResponse) {
  const stripeKey = process.env.STRIPE_SECRET_KEY;
  
  if (!stripeKey) {
    return res.status(503).json({ 
      error: 'stripe_not_configured',
      message: 'Stripe is not configured'
    });
  }
  
  if (!isSupabaseConfigured()) {
    return res.status(503).json({ error: 'Database not configured' });
  }
  
  const supabase = getSupabaseAdmin();
  
  try {
    const body = req.body as LessonPurchaseRequest;
    
    // Validate day number
    if (!body.day_number || body.day_number < 1 || body.day_number > 366) {
      return res.status(400).json({ error: 'Invalid day number' });
    }
    
    // Get user from auth token if provided
    let userId: string | null = null;
    let userEmail: string | null = body.customer_email || null;
    
    const authHeader = req.headers.authorization;
    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      const { data: { user } } = await supabase.auth.getUser(token);
      if (user) {
        userId = user.id;
        userEmail = user.email || userEmail;
        
        // Check if already purchased (table might not exist yet - that's OK)
        try {
          const { data: existingPurchase } = await supabase
            .from('lesson_purchases')
            .select('id')
            .eq('user_id', userId)
            .eq('day_number', body.day_number)
            .eq('status', 'completed')
            .single();
          
          if (existingPurchase) {
            return res.status(409).json({ 
              error: 'already_purchased',
              message: 'You already own this lesson'
            });
          }
        } catch (e) {
          // Table might not exist yet - proceed with purchase
        }
      }
    }
    
    // Get lesson info for the checkout
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('topic, universal_truth')
      .eq('day_number', body.day_number)
      .single();
    
    const lessonTitle = lesson?.topic || `Day ${body.day_number}`;
    
    // Get regional price if available
    let priceAmount = DEFAULT_LESSON_PRICE;
    let currency = DEFAULT_CURRENCY;
    
    // Try to get region from user's pricing tier
    if (userId) {
      const { data: pricingTier } = await supabase
        .from('user_pricing_tiers')
        .select('region')
        .eq('user_id', userId)
        .single();
      
      if (pricingTier?.region) {
        const { data: regionalPrice } = await supabase
          .from('regional_prices')
          .select('price, currency')
          .eq('region', pricingTier.region)
          .eq('product_type', 'single_lesson')
          .eq('is_active', true)
          .single();
        
        if (regionalPrice) {
          priceAmount = Math.round(regionalPrice.price * 100);
          currency = regionalPrice.currency.toLowerCase();
        }
      }
    }
    
    // Dynamic import Stripe
    const Stripe = (await import('stripe')).default;
    const stripe = new Stripe(stripeKey, {
      apiVersion: '2023-10-16'
    });
    
    const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';
    const successUrl = body.success_url || `${siteUrl}/learn.html?day=${body.day_number}&purchase=success`;
    const cancelUrl = body.cancel_url || `${siteUrl}/learn.html?day=${body.day_number}&purchase=cancelled`;
    
    // Create checkout session with dynamic price
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [{
        price_data: {
          currency: currency,
          product_data: {
            name: `Daily Lesson: ${lessonTitle}`,
            description: `Permanent access to Day ${body.day_number} lesson`,
            metadata: {
              day_number: body.day_number.toString()
            }
          },
          unit_amount: priceAmount
        },
        quantity: 1
      }],
      mode: 'payment',
      customer_email: userEmail || undefined,
      success_url: successUrl,
      cancel_url: cancelUrl,
      metadata: {
        type: 'single_lesson',
        day_number: body.day_number.toString(),
        user_id: userId || 'anonymous',
        source: 'web'
      },
      payment_intent_data: {
        metadata: {
          type: 'single_lesson',
          day_number: body.day_number.toString(),
          user_id: userId || 'anonymous'
        }
      }
    });
    
    // Create pending purchase record
    if (userId) {
      await supabase.from('lesson_purchases').insert({
        user_id: userId,
        day_number: body.day_number,
        purchase_price: priceAmount / 100,
        currency: currency.toUpperCase(),
        stripe_checkout_session_id: session.id,
        status: 'pending'
      });
    }
    
    // Log event
    await supabase.from('user_events').insert({
      user_id: userId,
      event_type: 'purchase.initiated',
      event_category: 'learner_action',
      payload: {
        product_type: 'single_lesson',
        day_number: body.day_number,
        price: priceAmount / 100,
        currency: currency.toUpperCase(),
        stripe_session_id: session.id
      },
      day_number: body.day_number
    }).catch(() => {}); // Don't fail on event logging
    
    return res.status(200).json({
      sessionId: session.id,
      url: session.url,
      price: {
        amount: priceAmount / 100,
        currency: currency.toUpperCase(),
        formatted: `$${(priceAmount / 100).toFixed(2)}`
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
