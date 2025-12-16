import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

/**
 * GET /api/subscription-status
 * Returns the current user's subscription status
 * Requires Authorization header with Supabase access token
 */

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // No auth = not subscribed
    return res.status(200).json({
      tier: 'free',
      status: 'inactive',
      expiresAt: null,
      source: 'anonymous',
    });
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    return res.status(503).json({ error: 'Database not configured' });
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    const token = authHeader.replace('Bearer ', '');

    // Get user from token
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return res.status(200).json({
        tier: 'free',
        status: 'inactive',
        expiresAt: null,
        source: 'invalid_token',
      });
    }

    // Check subscriptions table
    const { data: subscription, error: subError } = await supabase
      .from('subscriptions')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (subError || !subscription) {
      return res.status(200).json({
        tier: 'free',
        status: 'inactive',
        expiresAt: null,
        source: 'no_subscription',
      });
    }

    // Determine tier from subscription
    const tier = subscription.plan_type || 'free';
    const status = subscription.status || 'inactive';
    const expiresAt = subscription.current_period_end || subscription.expires_at || null;

    return res.status(200).json({
      tier,
      status,
      expiresAt,
      cancelAtPeriodEnd: subscription.cancel_at_period_end || false,
      source: 'database',
    });
  } catch (error) {
    console.error('Subscription status error:', error);
    return res.status(500).json({ error: 'Failed to check subscription status' });
  }
}

