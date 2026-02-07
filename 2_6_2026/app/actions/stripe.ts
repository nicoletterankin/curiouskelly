'use server'

import { headers } from 'next/headers'

import { stripe } from '@/lib/stripe'
import { getPriceIdForTier, sanitizeRefCode } from '@/lib/stripe-helpers'

export async function startCheckoutSession(priceId: string, refCode?: string) {
  if (!priceId) {
    throw new Error('Price ID is required')
  }

  // Sanitize ref code if provided
  const sanitizedRefCode = refCode ? sanitizeRefCode(refCode) : null

  // Create Checkout Session
  const session = await stripe.checkout.sessions.create({
    ui_mode: 'embedded',
    redirect_on_completion: 'never',
    line_items: [
      {
        price: priceId,
        quantity: 1,
      },
    ],
    mode: 'subscription',
    metadata: {
      ref_code: sanitizedRefCode || 'direct',
    },
  })

  if (!session.client_secret) {
    throw new Error('Failed to create checkout session')
  }

  return session.client_secret
}
