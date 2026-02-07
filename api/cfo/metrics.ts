/**
 * CFO Financial Metrics API
 * GET /api/cfo/metrics
 * 
 * Returns real-time financial metrics for the CFO dashboard
 * Requires admin authentication
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { adminMiddleware } from '../../lib/admin-auth';
import { cors } from '../../lib/cors';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface FinancialMetrics {
  timestamp: string;
  mrr: {
    current_cents: number;
    current_dollars: string;
    monthly_subscribers: number;
    annual_subscribers: number;
    lifetime_members: number;
    arr_cents: number;
    arr_dollars: string;
  };
  users: {
    total: number;
    free: number;
    trial: number;
    paid: number;
    paid_percentage: string;
    churn_rate: string;
  };
  revenue: {
    today_cents: number;
    today_dollars: string;
    mtd_cents: number;
    mtd_dollars: string;
    refunds_mtd_cents: number;
  };
  health: {
    status: 'healthy' | 'warning' | 'critical';
    alerts: Array<{
      type: string;
      severity: string;
      message: string;
    }>;
  };
  targets: {
    christmas_users: number;
    christmas_subscribers: number;
    christmas_mrr_cents: number;
    current_progress_users: string;
    current_progress_subs: string;
    current_progress_mrr: string;
  };
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // CORS
  if (!cors(req, res)) return;
  
  // Only allow GET
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Admin authentication required
  const authResult = await adminMiddleware(req, res);
  if (!authResult) return; // Response already sent (401 or 403)

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Get MRR data
    const { data: mrrData, error: mrrError } = await supabase
      .from('users')
      .select('subscription_tier, subscription_status');
    
    if (mrrError) throw mrrError;

    // Calculate metrics
    const users = mrrData || [];
    const total = users.length;
    const free = users.filter(u => !u.subscription_tier || u.subscription_tier === 'free').length;
    const trial = users.filter(u => u.subscription_status === 'trialing').length;
    const paid = users.filter(u => u.subscription_status === 'active').length;
    const monthly = users.filter(u => u.subscription_tier === 'monthly' && u.subscription_status === 'active').length;
    const annual = users.filter(u => u.subscription_tier === 'annual' && u.subscription_status === 'active').length;
    const lifetime = users.filter(u => u.subscription_tier === 'lifetime').length;
    const cancelled = users.filter(u => u.subscription_status === 'cancelled').length;

    // MRR calculation (cents)
    // Monthly: $4.99 = 499 cents
    // Annual: $49.99/12 = ~417 cents/month
    const mrr_cents = (monthly * 499) + (annual * 417);
    const arr_cents = mrr_cents * 12;

    // Churn rate
    const churnRate = paid + cancelled > 0 
      ? ((cancelled / (paid + cancelled)) * 100).toFixed(2)
      : '0.00';

    // Get revenue events for today/MTD
    const today = new Date().toISOString().split('T')[0];
    const monthStart = new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0];

    const { data: revenueData } = await supabase
      .from('revenue_events')
      .select('amount_cents, event_type, created_at')
      .gte('created_at', monthStart);

    const todayRevenue = (revenueData || [])
      .filter(r => r.created_at.startsWith(today))
      .reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    const mtdRevenue = (revenueData || [])
      .filter(r => r.event_type !== 'refund_issued')
      .reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    const mtdRefunds = (revenueData || [])
      .filter(r => r.event_type === 'refund_issued')
      .reduce((sum, r) => sum + (r.amount_cents || 0), 0);

    // Get alerts
    const { data: alerts } = await supabase
      .from('financial_alerts')
      .select('alert_type, severity, message')
      .eq('acknowledged', false)
      .order('created_at', { ascending: false })
      .limit(5);

    // Determine health status
    const criticalAlerts = (alerts || []).filter(a => a.severity === 'critical').length;
    const warningAlerts = (alerts || []).filter(a => a.severity === 'warning').length;
    const healthStatus = criticalAlerts > 0 ? 'critical' : warningAlerts > 0 ? 'warning' : 'healthy';

    // Christmas targets
    const christmasUsers = 1000;
    const christmasSubs = 500;
    const christmasMrr = 500000; // $5,000 in cents

    const metrics: FinancialMetrics = {
      timestamp: new Date().toISOString(),
      mrr: {
        current_cents: mrr_cents,
        current_dollars: `$${(mrr_cents / 100).toFixed(2)}`,
        monthly_subscribers: monthly,
        annual_subscribers: annual,
        lifetime_members: lifetime,
        arr_cents: arr_cents,
        arr_dollars: `$${(arr_cents / 100).toFixed(2)}`,
      },
      users: {
        total,
        free,
        trial,
        paid,
        paid_percentage: `${total > 0 ? ((paid / total) * 100).toFixed(1) : 0}%`,
        churn_rate: `${churnRate}%`,
      },
      revenue: {
        today_cents: todayRevenue,
        today_dollars: `$${(todayRevenue / 100).toFixed(2)}`,
        mtd_cents: mtdRevenue,
        mtd_dollars: `$${(mtdRevenue / 100).toFixed(2)}`,
        refunds_mtd_cents: mtdRefunds,
      },
      health: {
        status: healthStatus,
        alerts: (alerts || []).map(a => ({
          type: a.alert_type,
          severity: a.severity,
          message: a.message,
        })),
      },
      targets: {
        christmas_users: christmasUsers,
        christmas_subscribers: christmasSubs,
        christmas_mrr_cents: christmasMrr,
        current_progress_users: `${((total / christmasUsers) * 100).toFixed(1)}%`,
        current_progress_subs: `${((paid / christmasSubs) * 100).toFixed(1)}%`,
        current_progress_mrr: `${((mrr_cents / christmasMrr) * 100).toFixed(1)}%`,
      },
    };

    // Set cache headers (refresh every minute)
    res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate');
    
    return res.status(200).json(metrics);

  } catch (error) {
    console.error('CFO Metrics Error:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch financial metrics',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}



