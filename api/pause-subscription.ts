import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';

/**
 * Pause Subscription (schedule pause at period end)
 * POST /api/pause-subscription
 *
 * Auth: Supabase access token (Authorization: Bearer <token>)
 * 
 * Pauses subscription for up to 3 months using Stripe's pause_collection.
 * User retains access until current period ends, then paused.
 */

interface PauseRequest {
  pauseMonths: 1 | 2 | 3; // Max 3 months pause
}

interface ApiResponse {
  ok: boolean;
  error?: string;
  message?: string;
  subscription?: {
    id: string;
    status: string;
    pausedUntil?: string;
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

  const body = (req.body || {}) as PauseRequest;
  const pauseMonths = body.pauseMonths;

  // Validate pause duration
  if (!pauseMonths || ![1, 2, 3].includes(pauseMonths)) {
    return res.status(422).json({
      ok: false,
      error: 'invalid_pause_duration',
      message: 'pauseMonths must be 1, 2, or 3',
    } satisfies ApiResponse);
  }

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

    // Find active subscription if not stored
    if (!stripeSubscriptionId && stripeCustomerId) {
      const subs = await stripe.subscriptions.list({ customer: stripeCustomerId, status: 'all', limit: 10 });
      const activeLike = subs.data.find((s) => s.status === 'active' || s.status === 'trialing');
      stripeSubscriptionId = activeLike?.id || null;
    }

    if (!stripeSubscriptionId) {
      return res.status(404).json({ ok: false, error: 'subscription_not_found' } satisfies ApiResponse);
    }

    // Get current subscription to check status
    const currentSub = await stripe.subscriptions.retrieve(stripeSubscriptionId);
    
    if (currentSub.status !== 'active' && currentSub.status !== 'trialing') {
      return res.status(400).json({
        ok: false,
        error: 'cannot_pause',
        message: `Cannot pause subscription with status: ${currentSub.status}`,
      } satisfies ApiResponse);
    }

    // Check if already paused
    if (currentSub.pause_collection) {
      return res.status(400).json({
        ok: false,
        error: 'already_paused',
        message: 'Subscription is already paused',
      } satisfies ApiResponse);
    }

    // Calculate resume date (pause for X months from current period end)
    const currentPeriodEnd = new Date(currentSub.current_period_end * 1000);
    const resumeDate = new Date(currentPeriodEnd);
    resumeDate.setMonth(resumeDate.getMonth() + pauseMonths);

    // Pause the subscription
    const updated = await stripe.subscriptions.update(stripeSubscriptionId, {
      pause_collection: {
        behavior: 'void', // Don't charge during pause
        resumes_at: Math.floor(resumeDate.getTime() / 1000),
      },
    });

    // Record event (best effort)
    try {
      const supabaseForEvents = createClient(supabaseUrl, supabaseServiceKey);
      await supabaseForEvents.from('revenue_events').insert({
        event_type: 'subscription_paused',
        user_id: userId,
        stripe_customer_id: stripeCustomerId,
        stripe_subscription_id: stripeSubscriptionId,
        amount_cents: 0,
        mrr_impact_cents: 0, // Pause doesn't immediately affect MRR
        metadata: {
          pause_months: pauseMonths,
          resumes_at: resumeDate.toISOString(),
          current_period_end: currentPeriodEnd.toISOString(),
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
          subscription_status: 'paused',
          paused_until: resumeDate.toISOString(),
        })
        .eq('id', userId);
    } catch (_) {
      // Ignore
    }

    return res.status(200).json({
      ok: true,
      message: `Subscription paused for ${pauseMonths} month${pauseMonths > 1 ? 's' : ''}`,
      subscription: {
        id: updated.id,
        status: updated.status,
        pausedUntil: resumeDate.toISOString(),
        currentPeriodEnd: currentPeriodEnd.toISOString(),
      },
    } satisfies ApiResponse);

  } catch (e) {
    console.error('pause-subscription error:', e);
    return res.status(500).json({
      ok: false,
      error: 'internal_error',
      message: process.env.NODE_ENV === 'development' && e instanceof Error 
        ? e.message 
        : 'An error occurred. Please try again.',
    } satisfies ApiResponse);
  }
}
