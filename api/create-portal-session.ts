import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';

/**
 * Stripe Billing Portal Session
 * POST /api/create-portal-session
 *
 * Auth: Supabase access token (Authorization: Bearer <token>)
 */

interface ApiResponse {
  ok: boolean;
  url?: string;
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
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ ok: false, error: 'method_not_allowed' } satisfies ApiResponse);

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

  const siteUrl = process.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

  const supabaseAuthClient = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);
  // Keep this pinned to the Stripe SDK's supported type union for our installed version.
  const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });

  try {
    const { data, error } = await supabaseAuthClient.auth.getUser(token);
    if (error || !data?.user) {
      return res.status(401).json({ ok: false, error: 'invalid_token' } satisfies ApiResponse);
    }
    const userId = data.user.id;

    const { data: userRow } = await supabaseAdmin
      .from('users')
      .select('stripe_customer_id')
      .eq('id', userId)
      .maybeSingle();

    const stripeCustomerId = (userRow?.stripe_customer_id as string | null) || null;
    if (!stripeCustomerId) {
      return res.status(404).json({
        ok: false,
        error: 'customer_not_found',
        message: 'No Stripe customer found for this user',
      } satisfies ApiResponse);
    }

    const session = await stripe.billingPortal.sessions.create({
      customer: stripeCustomerId,
      return_url: `${siteUrl}/learn.html`,
    });

    return res.status(200).json({ ok: true, url: session.url } satisfies ApiResponse);
  } catch (e) {
    console.error('create-portal-session error:', e);
    return res.status(500).json({
      ok: false,
      error: 'internal_error',
      message: e instanceof Error ? e.message : 'Unknown error',
    } satisfies ApiResponse);
  }
}

