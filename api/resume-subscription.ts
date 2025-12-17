import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';

/**
 * Resume Subscription (unpause a paused subscription)
 * POST /api/resume-subscription
 *
 * Auth: Supabase access token (Authorization: Bearer <token>)
 * 
 * Removes the pause_collection setting to resume billing.
 */

interface ApiResponse {
  ok: boolean;
  error?: string;
  message?: string;
  subscription?: {
    id: string;
    status: string;
    currentPeriodEnd?: string;
  };
}

function getBearerToken(req: VercelRequest): string | null {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
  return authHeader.slice('Bearer '.length).trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'method_not_allowed' } satisfies ApiResponse);
  }

  const token = getBearerToken(req);
  if (!token) return res.status(401).json({ ok: false, error: 'unauthorized' } satisfies ApiResponse);

  const supabaseUrl =
    process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const supabaseAnonKey =
    process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;
  const supabaseServiceKey =
    process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE;

  if (!supabaseUrl || !supabaseAnonKey || !supabaseServiceKey) {
    return res.status(503).json({ ok: false, error: 'server_not_configured' } satisfies ApiResponse);
  }

  const stripeKey = process.env.STRIPE_SECRET_KEY;
  if (!stripeKey) return res.status(503).json({ ok: false, error: 'stripe_not_configured' } satisfies ApiResponse);

  const supabaseAuthClient = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);
  const stripe = new Stripe(stripeKey, { apiVersion: '2024-11-20.acacia' as const });

  try {
    const { data, error } = await supabaseAuthClient.auth.getUser(token);
    if (error || !data?.user) {
      return res.status(401).json({ ok: false, error: 'invalid_token' } satisfies ApiResponse);
    }
    const userId = data.user.id;

    const { data: userRow } = await supabaseAdmin
      .from('users')
      .select('stripe_customer_id,stripe_subscription_id')
      .eq('id', userId)
      .maybeSingle();

    const stripeCustomerId = (userRow?.stripe_customer_id as string | null) || null;
    let stripeSubscriptionId = (userRow?.stripe_subscription_id as string | null) || null;

    if (!stripeCustomerId && !stripeSubscriptionId) {
      return res.status(404).json({
        ok: false,
        error: 'subscription_not_found',
        message: 'No Stripe customer/subscription found for this user',
      } satisfies ApiResponse);
    }

    // Find subscription if not stored
    if (!stripeSubscriptionId && stripeCustomerId) {
      const subs = await stripe.subscriptions.list({ customer: stripeCustomerId, status: 'all', limit: 10 });
      // Look for paused subscription
      const pausedSub = subs.data.find((s) => s.pause_collection);
      stripeSubscriptionId = pausedSub?.id || null;
    }

    if (!stripeSubscriptionId) {
      return res.status(404).json({ ok: false, error: 'subscription_not_found' } satisfies ApiResponse);
    }

    // Get current subscription
    const currentSub = await stripe.subscriptions.retrieve(stripeSubscriptionId);
    
    // Check if actually paused
    if (!currentSub.pause_collection) {
      return res.status(400).json({
        ok: false,
        error: 'not_paused',
        message: 'Subscription is not currently paused',
      } satisfies ApiResponse);
    }

    // Resume the subscription by removing pause_collection
    const updated = await stripe.subscriptions.update(stripeSubscriptionId, {
      pause_collection: null as unknown as Stripe.SubscriptionUpdateParams.PauseCollection, // Remove pause
    });

    // Record event (best effort)
    try {
      const supabaseForEvents = createClient(supabaseUrl, supabaseServiceKey);
      await supabaseForEvents.from('revenue_events').insert({
        event_type: 'subscription_resumed',
        user_id: userId,
        stripe_customer_id: stripeCustomerId,
        stripe_subscription_id: stripeSubscriptionId,
        amount_cents: 0,
        mrr_impact_cents: 0,
        metadata: {
          resumed_at: new Date().toISOString(),
        },
      });
    } catch (_) {
      // Don't fail on event logging
    }

    // Update user record (best effort)
    try {
      await supabaseAdmin
        .from('users')
        .update({
          subscription_status: updated.status,
          paused_until: null,
        })
        .eq('id', userId);
    } catch (_) {
      // Ignore
    }

    return res.status(200).json({
      ok: true,
      message: 'Subscription resumed successfully',
      subscription: {
        id: updated.id,
        status: updated.status,
        currentPeriodEnd: new Date(updated.current_period_end * 1000).toISOString(),
      },
    } satisfies ApiResponse);

  } catch (e) {
    console.error('resume-subscription error:', e);
    return res.status(500).json({
      ok: false,
      error: 'internal_error',
      message: process.env.NODE_ENV === 'development' && e instanceof Error 
        ? e.message 
        : 'An error occurred. Please try again.',
    } satisfies ApiResponse);
  }
}
