import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { nanoid } from 'nanoid'

/**
 * POST /api/affiliate/register
 * Register a new affiliate or get existing affiliate code
 * 
 * Body: { userId: string }
 * Returns: { affiliateCode: string, tier: string, earnings: number }
 * 
 * Actual DB Schema for affiliates table:
 * - id: uuid
 * - user_id: text
 * - affiliate_code: text
 * - tier: text
 * - total_earnings: numeric
 * - referral_count: integer
 * - languages: jsonb
 * - dialects: jsonb
 * - cultural_context: text
 * - created_at, updated_at: timestamp
 */
export async function POST(req: NextRequest) {
  try {
    const { userId } = await req.json()

    if (!userId) {
      return NextResponse.json(
        { error: 'Missing userId' },
        { status: 400 }
      )
    }

    // Check if user already has an affiliate account
    const existing = await sql`
      SELECT id, affiliate_code, tier, total_earnings, referral_count
      FROM affiliates
      WHERE user_id = ${userId}
      LIMIT 1
    `

    if (existing.length > 0) {
      const affiliate = existing[0]
      return NextResponse.json({
        registered: true,
        affiliateId: affiliate.id,
        affiliateCode: affiliate.affiliate_code,
        tier: affiliate.tier || 'starter',
        earnings: Number(affiliate.total_earnings) || 0,
        referralCount: affiliate.referral_count || 0,
      })
    }

    // Generate unique affiliate code (6 char alphanumeric, uppercase)
    const affiliateCode = nanoid(6).toUpperCase()

    // Create new affiliate with actual schema columns
    const newAffiliate = await sql`
      INSERT INTO affiliates (
        user_id, affiliate_code, tier, total_earnings, referral_count,
        created_at, updated_at
      )
      VALUES (
        ${userId}, ${affiliateCode}, 'starter', 0, 0,
        NOW(), NOW()
      )
      RETURNING id, affiliate_code, tier, total_earnings
    `

    return NextResponse.json({
      registered: true,
      affiliateId: newAffiliate[0].id,
      affiliateCode: newAffiliate[0].affiliate_code,
      tier: 'starter',
      earnings: 0,
      referralCount: 0,
      isNew: true,
    })

  } catch (error) {
    console.error('[affiliate/register] Error:', error)
    return NextResponse.json(
      { error: 'Failed to register affiliate' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/affiliate/register?userId=xxx
 * Get affiliate status for a user
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

    const affiliate = await sql`
      SELECT id, affiliate_code, tier, total_earnings, referral_count,
             languages, dialects, cultural_context
      FROM affiliates
      WHERE user_id = ${userId}
      LIMIT 1
    `

    if (affiliate.length === 0) {
      return NextResponse.json({ registered: false })
    }

    const a = affiliate[0]
    return NextResponse.json({
      registered: true,
      affiliateId: a.id,
      affiliateCode: a.affiliate_code,
      tier: a.tier || 'starter',
      earnings: Number(a.total_earnings) || 0,
      referralCount: a.referral_count || 0,
      languages: a.languages,
      dialects: a.dialects,
      culturalContext: a.cultural_context,
    })

  } catch (error) {
    console.error('[affiliate/register] GET Error:', error)
    return NextResponse.json(
      { error: 'Failed to get affiliate status' },
      { status: 500 }
    )
  }
}
