import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

export async function GET(req: NextRequest) {
  const email = req.nextUrl.searchParams.get('email')

  if (!email) {
    return NextResponse.json({ error: 'Email is required' }, { status: 400 })
  }

  try {
    const rows = await sql`
      SELECT
        subscription_tier,
        plan,
        status,
        subscription_status,
        stripe_customer_id,
        stripe_subscription_id,
        current_period_end,
        cancel_at_period_end,
        created_at
      FROM subscribers
      WHERE email = ${email}
      LIMIT 1
    `

    if (rows.length === 0) {
      return NextResponse.json({
        tier: 'free',
        plan: null,
        status: 'none',
        subscriptionStatus: null,
        periodEnd: null,
        cancelAtPeriodEnd: false,
      })
    }

    const sub = rows[0]
    return NextResponse.json({
      tier: sub.subscription_tier || 'free',
      plan: sub.plan,
      status: sub.status,
      subscriptionStatus: sub.subscription_status,
      periodEnd: sub.current_period_end,
      cancelAtPeriodEnd: sub.cancel_at_period_end || false,
      hasStripeCustomer: !!sub.stripe_customer_id,
      createdAt: sub.created_at,
    })
  } catch (error) {
    console.error('[Subscription API] Error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch subscription' },
      { status: 500 }
    )
  }
}
