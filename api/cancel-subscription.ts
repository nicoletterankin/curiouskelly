import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';

/**
 * Cancel Subscription (schedule at period end)
 * POST /api/cancel-subscription
 *
 * Auth: Supabase access token (Authorization: Bearer <token>)
 */

interface ApiResponse {
  ok: boolean;
  error?: string;
  message?: string;
  subscription?: {
    id: string;
    status: string;
    cancelAtPeriodEnd: boolean;
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
  const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });

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

    if (!stripeSubscriptionId && stripeCustomerId) {
      const subs = await stripe.subscriptions.list({ customer: stripeCustomerId, status: 'all', limit: 10 });
      const activeLike = subs.data.find((s) => s.status === 'active' || s.status === 'trialing' || s.status === 'past_due');
      stripeSubscriptionId = (activeLike || subs.data[0])?.id || null;
    }

    if (!stripeSubscriptionId) {
      return res.status(404).json({ ok: false, error: 'subscription_not_found' } satisfies ApiResponse);
    }

    const updated = await stripe.subscriptions.update(stripeSubscriptionId, { cancel_at_period_end: true });

    // Best-effort persist
    try {
      await supabaseAdmin
        .from('users')
        .update({
          subscription_status: updated.status,
          cancel_at_period_end: Boolean(updated.cancel_at_period_end),
          stripe_subscription_id: updated.id,
          current_period_end: new Date(updated.current_period_end * 1000).toISOString(),
        })
        .eq('id', userId);
    } catch (_) {
      // ignore
    }

    return res.status(200).json({
      ok: true,
      subscription: {
        id: updated.id,
        status: updated.status,
        cancelAtPeriodEnd: Boolean(updated.cancel_at_period_end),
        currentPeriodEnd: new Date(updated.current_period_end * 1000).toISOString(),
      },
    } satisfies ApiResponse);
  } catch (e) {
    console.error('cancel-subscription error:', e);
    return res.status(500).json({
      ok: false,
      error: 'internal_error',
      message: e instanceof Error ? e.message : 'Unknown error',
    } satisfies ApiResponse);
  }
}

