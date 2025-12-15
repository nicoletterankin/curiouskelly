import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';

/**
 * Subscription Status (server-verified)
 * GET /api/subscription-status
 *
 * - Requires Supabase access token in `Authorization: Bearer <token>`
 * - Verifies identity via Supabase Auth
 * - Uses Stripe as source of truth when possible
 * - Best-effort writes back subscription fields to `public.users` (service role)
 */

type SubscriptionTier = 'included' | 'monthly' | 'annual' | 'lifetime';
type SubscriptionStatus = 'inactive' | 'trialing' | 'active' | 'past_due' | 'cancelled' | 'expired';

interface ApiResponse {
  ok: boolean;
  subscription?: {
    tier: SubscriptionTier;
    status: SubscriptionStatus;
    expiresAt: string | null;
    cancelAtPeriodEnd: boolean;
    stripeCustomerId: string | null;
    stripeSubscriptionId: string | null;
    checkedAt: string;
    source: 'stripe' | 'db';
  };
  error?: string;
  message?: string;
}

function getBearerToken(req: VercelRequest): string | null {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
  return authHeader.slice('Bearer '.length).trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') {
    return res.status(405).json({ ok: false, error: 'method_not_allowed' } satisfies ApiResponse);
  }

  const token = getBearerToken(req);
  if (!token) {
    return res.status(401).json({ ok: false, error: 'unauthorized' } satisfies ApiResponse);
  }

  const supabaseUrl =
    process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const supabaseAnonKey =
    process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;
  const supabaseServiceKey =
    process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE;

  if (!supabaseUrl || !supabaseAnonKey || !supabaseServiceKey) {
    return res.status(503).json({
      ok: false,
      error: 'server_not_configured',
      message: 'Missing Supabase environment variables',
    } satisfies ApiResponse);
  }

  const stripeKey = process.env.STRIPE_SECRET_KEY;
  if (!stripeKey) {
    return res.status(503).json({
      ok: false,
      error: 'stripe_not_configured',
      message: 'Missing STRIPE_SECRET_KEY',
    } satisfies ApiResponse);
  }

  const supabaseAuthClient = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);
  const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });

  try {
    const { data, error } = await supabaseAuthClient.auth.getUser(token);
    if (error || !data?.user) {
      return res.status(401).json({
        ok: false,
        error: 'invalid_token',
        message: error?.message || 'Invalid token',
      } satisfies ApiResponse);
    }

    const userId = data.user.id;

    // Load current DB view (best-effort; schema may have more cols in prod)
    const { data: userRow } = await supabaseAdmin
      .from('users')
      .select('id,email,subscription_tier,subscription_status,stripe_customer_id,stripe_subscription_id,current_period_end,cancel_at_period_end')
      .eq('id', userId)
      .maybeSingle();

    const checkedAt = new Date().toISOString();

    const dbTier = (userRow?.subscription_tier as SubscriptionTier | null) || 'included';
    const dbStatus = (userRow?.subscription_status as SubscriptionStatus | null) || 'inactive';
    const dbExpiresAt = (userRow?.current_period_end as string | null) || null;
    const dbCancelAtPeriodEnd = Boolean(userRow?.cancel_at_period_end);
    const stripeCustomerId = (userRow?.stripe_customer_id as string | null) || null;
    const stripeSubscriptionId = (userRow?.stripe_subscription_id as string | null) || null;

    // If we have a Stripe customer, ask Stripe for truth.
    if (stripeCustomerId) {
      const subs = await stripe.subscriptions.list({
        customer: stripeCustomerId,
        status: 'all',
        limit: 10,
      });

      const activeLike = subs.data.find((s) => s.status === 'active' || s.status === 'trialing' || s.status === 'past_due');
      const latest = activeLike || subs.data[0];

      if (latest) {
        const tierFromMeta = (latest.metadata?.plan_type as SubscriptionTier | undefined) || dbTier;
        const statusFromStripe = (latest.status as SubscriptionStatus) || dbStatus;
        const expiresAt = latest.current_period_end
          ? new Date(latest.current_period_end * 1000).toISOString()
          : dbExpiresAt;

        const cancelAtPeriodEnd = Boolean(latest.cancel_at_period_end);

        // Best-effort persist (don’t fail the endpoint if DB write fails)
        try {
          await supabaseAdmin
            .from('users')
            .update({
              subscription_tier: tierFromMeta,
              subscription_status: statusFromStripe,
              stripe_subscription_id: latest.id,
              current_period_end: expiresAt,
              cancel_at_period_end: cancelAtPeriodEnd,
            })
            .eq('id', userId);
        } catch (_) {
          // ignore
        }

        return res.status(200).json({
          ok: true,
          subscription: {
            tier: tierFromMeta,
            status: statusFromStripe,
            expiresAt,
            cancelAtPeriodEnd,
            stripeCustomerId,
            stripeSubscriptionId: latest.id,
            checkedAt,
            source: 'stripe',
          },
        } satisfies ApiResponse);
      }
    }

    // Fallback: DB-only (still server-side, but may be stale)
    return res.status(200).json({
      ok: true,
      subscription: {
        tier: dbTier,
        status: dbStatus,
        expiresAt: dbExpiresAt,
        cancelAtPeriodEnd: dbCancelAtPeriodEnd,
        stripeCustomerId,
        stripeSubscriptionId,
        checkedAt,
        source: 'db',
      },
    } satisfies ApiResponse);
  } catch (e) {
    console.error('subscription-status error:', e);
    return res.status(500).json({
      ok: false,
      error: 'internal_error',
      message: e instanceof Error ? e.message : 'Unknown error',
    } satisfies ApiResponse);
  }
}

