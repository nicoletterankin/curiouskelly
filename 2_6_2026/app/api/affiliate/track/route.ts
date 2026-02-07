import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { cookies } from 'next/headers'

const AFFILIATE_COOKIE = 'ck_ref'
const COOKIE_MAX_AGE = 10 * 365 * 24 * 60 * 60 // 10 years in seconds (lifetime)

/**
 * POST /api/affiliate/track
 * Track a referral when someone lands with ?ref=CODE
 * Sets lifetime cookie and creates DB record
 * 
 * Body: { refCode: string, visitorId: string }
 */
export async function POST(req: NextRequest) {
  try {
    const { refCode, visitorId } = await req.json()

    if (!refCode) {
      return NextResponse.json(
        { error: 'Missing refCode' },
        { status: 400 }
      )
    }

    // Validate affiliate code exists
    const affiliate = await sql`
      SELECT id, user_id, tier
      FROM affiliates
      WHERE affiliate_code = ${refCode.toUpperCase()}
      LIMIT 1
    `

    if (affiliate.length === 0) {
      return NextResponse.json(
        { error: 'Invalid affiliate code' },
        { status: 404 }
      )
    }

    const affiliateId = affiliate[0].id

    // Check if this visitor was already referred (don't overwrite first touch)
    const existingReferral = await sql`
      SELECT id FROM affiliate_referrals
      WHERE referred_user_id = ${visitorId}
      LIMIT 1
    `

    if (existingReferral.length > 0) {
      // Already tracked, just return success
      return NextResponse.json({
        tracked: true,
        existing: true,
        affiliateCode: refCode.toUpperCase(),
      })
    }

    // Get commission rate based on tier
    const commissionRates: Record<string, number> = {
      'starter': 0.20,
      'advocate': 0.25,
      'ambassador': 0.30,
      'champion': 0.35,
      'legend': 0.40,
    }
    const tier = affiliate[0].tier || 'starter'
    const commissionRate = commissionRates[tier] || 0.20

    // Create referral record
    await sql`
      INSERT INTO affiliate_referrals (
        id, affiliate_id, referred_user_id, commission_rate, status, created_at
      )
      VALUES (
        gen_random_uuid(), ${affiliateId}, ${visitorId}, ${commissionRate}, 'pending', NOW()
      )
    `

    // Set lifetime cookie
    const cookieStore = await cookies()
    cookieStore.set(AFFILIATE_COOKIE, refCode.toUpperCase(), {
      maxAge: COOKIE_MAX_AGE,
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
    })

    return NextResponse.json({
      tracked: true,
      existing: false,
      affiliateCode: refCode.toUpperCase(),
      commissionRate,
    })

  } catch (error) {
    console.error('[affiliate/track] Error:', error)
    return NextResponse.json(
      { error: 'Failed to track referral' },
      { status: 500 }
    )
  }
}

/**
 * GET /api/affiliate/track
 * Get the current referral cookie value
 */
export async function GET() {
  try {
    const cookieStore = await cookies()
    const refCookie = cookieStore.get(AFFILIATE_COOKIE)

    if (!refCookie) {
      return NextResponse.json({ hasReferral: false })
    }

    return NextResponse.json({
      hasReferral: true,
      affiliateCode: refCookie.value,
    })

  } catch (error) {
    console.error('[affiliate/track] GET Error:', error)
    return NextResponse.json({ hasReferral: false })
  }
}
