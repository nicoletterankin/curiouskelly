import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/affiliate/earnings?userId=xxx
 * Get detailed earnings breakdown for an affiliate
 */
export async function GET(req: NextRequest) {
  try {
    const userId = req.nextUrl.searchParams.get('userId')

    if (!userId) {
      return NextResponse.json(
        { error: 'Missing userId param' },
        { status: 400 }
      )
    }

    // Get affiliate info
    const affiliate = await sql`
      SELECT id, affiliate_code, tier, total_earnings, referral_count
      FROM affiliates
      WHERE user_id = ${userId}
      LIMIT 1
    `

    if (affiliate.length === 0) {
      return NextResponse.json({
        registered: false,
        earnings: 0,
        referrals: [],
      })
    }

    const affiliateId = affiliate[0].id

    // Get all referrals with earnings
    const referrals = await sql`
      SELECT 
        id, referred_user_id, subscription_type, commission_rate, 
        total_commission, status, created_at
      FROM affiliate_referrals
      WHERE affiliate_id = ${affiliateId}
      ORDER BY created_at DESC
      LIMIT 100
    `

    // Get payouts
    const payouts = await sql`
      SELECT id, amount, status, payout_method, paid_at, created_at
      FROM affiliate_payouts
      WHERE affiliate_id = ${affiliateId}
      ORDER BY created_at DESC
      LIMIT 20
    `

    // Calculate totals
    const totalEarned = Number(affiliate[0].total_earnings) || 0
    const pendingReferrals = referrals.filter((r: Record<string, unknown>) => r.status === 'pending').length
    const convertedReferrals = referrals.filter((r: Record<string, unknown>) => r.status === 'converted').length
    const totalPaidOut = payouts
      .filter((p: Record<string, unknown>) => p.status === 'completed')
      .reduce((sum: number, p: Record<string, unknown>) => sum + Number(p.amount || 0), 0)

    // Tier info
    const tierInfo: Record<string, { commission: number, nextTier: string | null, referralsNeeded: number }> = {
      'starter': { commission: 20, nextTier: 'advocate', referralsNeeded: 3 },
      'advocate': { commission: 25, nextTier: 'ambassador', referralsNeeded: 10 },
      'ambassador': { commission: 30, nextTier: 'champion', referralsNeeded: 25 },
      'champion': { commission: 35, nextTier: 'legend', referralsNeeded: 50 },
      'legend': { commission: 40, nextTier: null, referralsNeeded: 0 },
    }

    const currentTier = affiliate[0].tier || 'starter'
    const tierData = tierInfo[currentTier] || tierInfo['starter']

    return NextResponse.json({
      registered: true,
      affiliateCode: affiliate[0].affiliate_code,
      tier: currentTier,
      tierInfo: tierData,
      stats: {
        totalEarned,
        totalPaidOut,
        availableBalance: totalEarned - totalPaidOut,
        totalReferrals: affiliate[0].referral_count || 0,
        pendingReferrals,
        convertedReferrals,
      },
      referrals: referrals.map((r: Record<string, unknown>) => ({
        id: r.id,
        status: r.status,
        subscriptionType: r.subscription_type,
        commissionRate: Number(r.commission_rate),
        commission: Number(r.total_commission) || 0,
        createdAt: r.created_at,
      })),
      payouts: payouts.map((p: Record<string, unknown>) => ({
        id: p.id,
        amount: Number(p.amount),
        status: p.status,
        method: p.payout_method,
        paidAt: p.paid_at,
        createdAt: p.created_at,
      })),
    })

  } catch (error) {
    console.error('[affiliate/earnings] Error:', error)
    return NextResponse.json(
      { error: 'Failed to get earnings' },
      { status: 500 }
    )
  }
}
