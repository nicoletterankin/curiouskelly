/**
 * CFO Affiliate Payouts Calculator
 * POST /api/cfo/affiliate-payouts
 * 
 * Calculates affiliate commissions for a given period.
 * Requires admin authentication.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { adminMiddleware } from '../../lib/admin-auth';
import { cors } from '../../lib/cors';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface AffiliateCommission {
  affiliate_id: string;
  affiliate_email: string;
  referral_code: string;
  tier: string;
  commission_rate: number;
  referrals_count: number;
  converted_count: number;
  gross_revenue_cents: number;
  commission_cents: number;
  commission_dollars: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // CORS
  if (!cors(req, res)) return;
  
  if (req.method !== 'POST' && req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Admin authentication required
  const authResult = await adminMiddleware(req, res);
  if (!authResult) return; // Response already sent (401 or 403)

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Get period from query/body
    const periodStart = req.query.start as string || req.body?.start || 
      new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0];
    const periodEnd = req.query.end as string || req.body?.end || 
      new Date().toISOString().split('T')[0];

    // Get all affiliates with their referrals
    const { data: affiliates, error: affError } = await supabase
      .from('affiliates')
      .select(`
        id,
        referral_code,
        tier,
        commission_rate,
        is_founding_100,
        user_id,
        users!affiliates_user_id_fkey(email)
      `)
      .eq('status', 'active');

    if (affError) throw affError;

    const commissions: AffiliateCommission[] = [];

    for (const affiliate of (affiliates || [])) {
      // Get referrals for this affiliate in the period
      const { data: referrals } = await supabase
        .from('referrals')
        .select('id, subscription_value, status')
        .eq('affiliate_id', affiliate.id)
        .gte('created_at', periodStart)
        .lte('created_at', periodEnd);

      const referralList = referrals || [];
      const convertedReferrals = referralList.filter(r => r.status === 'active' || r.status === 'paid');
      const grossRevenue = convertedReferrals.reduce((sum, r) => sum + (Number(r.subscription_value) || 0), 0);
      
      // Determine commission rate (founding 100 get 30% locked)
      const effectiveRate = affiliate.is_founding_100 ? 30 : Number(affiliate.commission_rate);
      const commissionCents = Math.round(grossRevenue * (effectiveRate / 100));

      if (referralList.length > 0 || convertedReferrals.length > 0) {
        commissions.push({
          affiliate_id: affiliate.id,
          affiliate_email: (affiliate.users as any)?.email || 'Unknown',
          referral_code: affiliate.referral_code,
          tier: affiliate.tier,
          commission_rate: effectiveRate,
          referrals_count: referralList.length,
          converted_count: convertedReferrals.length,
          gross_revenue_cents: grossRevenue,
          commission_cents: commissionCents,
          commission_dollars: `$${(commissionCents / 100).toFixed(2)}`,
        });
      }
    }

    // Calculate totals
    const totalGrossRevenue = commissions.reduce((sum, c) => sum + c.gross_revenue_cents, 0);
    const totalCommissions = commissions.reduce((sum, c) => sum + c.commission_cents, 0);
    const totalReferrals = commissions.reduce((sum, c) => sum + c.referrals_count, 0);
    const totalConverted = commissions.reduce((sum, c) => sum + c.converted_count, 0);

    // If POST, create payout records
    if (req.method === 'POST' && req.body?.create_payouts) {
      for (const commission of commissions) {
        if (commission.commission_cents >= 5000) { // Minimum $50 payout
          await supabase.from('affiliate_payouts').insert({
            affiliate_id: commission.affiliate_id,
            payout_period_start: periodStart,
            payout_period_end: periodEnd,
            total_referrals: commission.referrals_count,
            converted_referrals: commission.converted_count,
            gross_revenue_cents: commission.gross_revenue_cents,
            commission_rate: commission.commission_rate,
            commission_cents: commission.commission_cents,
            status: 'pending',
          });
        }
      }
    }

    return res.status(200).json({
      period: { start: periodStart, end: periodEnd },
      summary: {
        total_affiliates: commissions.length,
        total_referrals: totalReferrals,
        total_converted: totalConverted,
        conversion_rate: totalReferrals > 0 ? `${((totalConverted / totalReferrals) * 100).toFixed(1)}%` : '0%',
        gross_revenue_dollars: `$${(totalGrossRevenue / 100).toFixed(2)}`,
        total_commissions_dollars: `$${(totalCommissions / 100).toFixed(2)}`,
        net_revenue_dollars: `$${((totalGrossRevenue - totalCommissions) / 100).toFixed(2)}`,
      },
      affiliates: commissions.sort((a, b) => b.commission_cents - a.commission_cents),
    });

  } catch (error) {
    console.error('Affiliate Payouts Error:', error);
    return res.status(500).json({
      error: 'Failed to calculate affiliate payouts',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}



