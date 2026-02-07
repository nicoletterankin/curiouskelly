import { NextRequest, NextResponse } from 'next/server'
import { stripe } from '@/lib/stripe'
import { sql } from '@/lib/db'

export async function POST(req: NextRequest) {
  try {
    const { email } = await req.json()

    if (!email) {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 })
    }

    // Look up Stripe customer ID from our database
    const rows = await sql`
      SELECT stripe_customer_id FROM subscribers
      WHERE email = ${email} AND stripe_customer_id IS NOT NULL
      LIMIT 1
    `

    if (rows.length === 0 || !rows[0].stripe_customer_id) {
      return NextResponse.json(
        { error: 'No subscription found for this email' },
        { status: 404 }
      )
    }

    // Create a Stripe Customer Portal session
    const session = await stripe.billingPortal.sessions.create({
      customer: rows[0].stripe_customer_id,
      return_url: `${req.headers.get('origin') || 'https://thedailylesson.com'}/`,
    })

    return NextResponse.json({ url: session.url })
  } catch (error) {
    console.error('[Stripe Portal] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create portal session' },
      { status: 500 }
    )
  }
}
