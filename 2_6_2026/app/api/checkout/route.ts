import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import Stripe from 'stripe'
import { getStripePriceId, validateBillingCycle, sanitizeRefCode } from '@/lib/stripe-helpers'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-12-18.acacia',
})

const AFFILIATE_COOKIE = 'ck_ref'

export async function POST(req: NextRequest) {
  try {
    const { planId, billingCycle: rawCycle, email, successUrl, cancelUrl } = await req.json()

    // Get affiliate referral from cookie
    const cookieStore = await cookies()
    const refCookie = cookieStore.get(AFFILIATE_COOKIE)
    const affiliateCode = sanitizeRefCode(refCookie?.value)

    if (!planId || !rawCycle || !email) {
      return NextResponse.json(
        { error: 'Missing required fields: planId, billingCycle, email' },
        { status: 400 }
      )
    }

    const billingCycle = validateBillingCycle(rawCycle)
    const priceId = getStripePriceId(planId, billingCycle)

    if (!priceId) {
      return NextResponse.json(
        { error: `No Stripe price configured for ${planId}/${billingCycle}. Check environment variables.` },
        { status: 400 }
      )
    }

    // Create Stripe checkout session
    const session = await stripe.checkout.sessions.create({
      mode: billingCycle === 'lifetime' ? 'payment' : 'subscription',
      payment_method_types: ['card'],
      customer_email: email,
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      success_url: successUrl || `${req.headers.get('origin')}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: cancelUrl || `${req.headers.get('origin')}/pricing`,
      metadata: {
        planId,
        billingCycle,
        affiliateCode: affiliateCode || '',
      },
    })

    return NextResponse.json({
      sessionId: session.id,
      url: session.url,
    })
  } catch (error) {
    console.error('[Checkout] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create checkout session' },
      { status: 500 }
    )
  }
}
