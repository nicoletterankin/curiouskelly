/**
 * CFO Daily Snapshot API
 * POST /api/cfo/daily-snapshot
 * 
 * Triggers daily financial snapshot calculation
 * Should be called by a cron job at midnight
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify cron secret or admin auth
  const authHeader = req.headers.authorization;
  const cronSecret = process.env.CRON_SECRET;
  
  if (authHeader !== `Bearer ${cronSecret}`) {
    // Also allow from Vercel cron
    if (req.headers['x-vercel-cron'] !== '1') {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey);
    const today = new Date().toISOString().split('T')[0];

    // Get user metrics
    const { data: users, error: usersError } = await supabase
      .from('users')
      .select('subscription_tier, subscription_status, created_at');

    if (usersError) throw usersError;

    const userList = users || [];
    const total_users = userList.length;
    const free_users = userList.filter(u => !u.subscription_tier || u.subscription_tier === 'free').length;
    const trial_users = userList.filter(u => u.subscription_status === 'trialing').length;
    const paid_users = userList.filter(u => u.subscription_status === 'active').length;
    const monthly_subscribers = userList.filter(u => u.subscription_tier === 'monthly' && u.subscription_status === 'active').length;
    const annual_subscribers = userList.filter(u => u.subscription_tier === 'annual' && u.subscription_status === 'active').length;
    const lifetime_members = userList.filter(u => u.subscription_tier === 'lifetime').length;
    const gift_recipients = userList.filter(u => u.subscription_tier === 'gift').length;
    const new_users_today = userList.filter(u => u.created_at?.startsWith(today)).length;

    // Calculate MRR (cents)
    const mrr_cents = (monthly_subscribers * 499) + (annual_subscribers * 417);
    const arr_cents = mrr_cents * 12;

    // Get revenue data for today
    const { data: revenueToday } = await supabase
      .from('revenue_events')
      .select('amount_cents')
      .gte('created_at', `${today}T00:00:00`)
      .lt('created_at', `${today}T23:59:59`);

    const revenue_today_cents = (revenueToday || []).reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    // Get MTD revenue
    const monthStart = new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0];
    const { data: revenueMtd } = await supabase
      .from('revenue_events')
      .select('amount_cents, event_type')
      .gte('created_at', monthStart);

    const revenue_mtd_cents = (revenueMtd || [])
      .filter(r => r.event_type !== 'refund_issued')
      .reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    const refunds_mtd_cents = (revenueMtd || [])
      .filter(r => r.event_type === 'refund_issued')
      .reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    // Calculate churn rate
    const cancelled = userList.filter(u => u.subscription_status === 'cancelled').length;
    const churn_rate_monthly = paid_users + cancelled > 0
      ? Number(((cancelled / (paid_users + cancelled)) * 100).toFixed(2))
      : 0;

    // Calculate ARPU (Average Revenue Per User)
    const arpu_cents = paid_users > 0 ? Math.round(mrr_cents / paid_users) : 0;

    // Upsert snapshot
    const { data: snapshot, error: snapshotError } = await supabase
      .from('financial_snapshots')
      .upsert({
        snapshot_date: today,
        total_users,
        free_users,
        trial_users,
        paid_users,
        new_users_today,
        monthly_subscribers,
        annual_subscribers,
        lifetime_members,
        gift_recipients,
        mrr_cents,
        arr_cents,
        revenue_today_cents,
        revenue_mtd_cents,
        refunds_mtd_cents,
        churn_rate_monthly,
        arpu_cents,
      }, { onConflict: 'snapshot_date' })
      .select()
      .single();

    if (snapshotError) throw snapshotError;

    // Run financial health check
    await supabase.rpc('check_financial_health');

    return res.status(200).json({
      success: true,
      snapshot_date: today,
      mrr_dollars: `$${(mrr_cents / 100).toFixed(2)}`,
      total_users,
      paid_users,
      message: 'Daily snapshot calculated successfully'
    });

  } catch (error) {
    console.error('Daily Snapshot Error:', error);
    return res.status(500).json({
      error: 'Failed to calculate daily snapshot',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}


