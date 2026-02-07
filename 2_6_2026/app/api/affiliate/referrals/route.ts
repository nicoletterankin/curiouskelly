import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/affiliate/referrals?affiliateId=xxx
 * Returns referral history for an affiliate
 */
export async function GET(req: NextRequest) {
  try {
    const affiliateId = req.nextUrl.searchParams.get('affiliateId')

    if (!affiliateId) {
      return NextResponse.json({ error: 'Missing affiliateId' }, { status: 400 })
    }

    const referrals = await sql`
      SELECT 
        id,
        subscription_type,
        commission_rate,
        commission_amount,
        status,
        created_at,
        paid_at
      FROM affiliate_referrals
      WHERE affiliate_id = ${parseInt(affiliateId)}
      ORDER BY created_at DESC
      LIMIT 50
    `

    const formattedReferrals = referrals.map((ref) => ({
      id: ref.id,
      subscriptionType: ref.subscription_type || 'Monthly',
      commissionRate: Number(ref.commission_rate) || 0.20,
      commissionAmount: Number(ref.commission_amount) || 0,
      status: ref.status || 'pending',
      createdAt: ref.created_at,
      paidAt: ref.paid_at,
    }))

    // Get summary stats
    const stats = await sql`
      SELECT 
        COUNT(*) as total_referrals,
        COALESCE(SUM(commission_amount), 0) as total_earned,
        COALESCE(SUM(CASE WHEN status = 'pending' THEN commission_amount ELSE 0 END), 0) as pending
      FROM affiliate_referrals
      WHERE affiliate_id = ${parseInt(affiliateId)}
    `

    return NextResponse.json({
      referrals: formattedReferrals,
      stats: {
        totalReferrals: Number(stats[0]?.total_referrals) || 0,
        totalEarned: Number(stats[0]?.total_earned) || 0,
        pending: Number(stats[0]?.pending) || 0,
      },
    })

  } catch (error) {
    console.error('[affiliate/referrals] Error:', error)
    return NextResponse.json({ 
      referrals: [],
      stats: { totalReferrals: 0, totalEarned: 0, pending: 0 },
    })
  }
}
