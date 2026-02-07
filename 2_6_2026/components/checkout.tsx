'use client'

import { useCallback, useMemo } from 'react'
import { EmbeddedCheckout, EmbeddedCheckoutProvider } from '@stripe/react-stripe-js'
import { loadStripe } from '@stripe/stripe-js'

import { startCheckoutSession } from '@/app/actions/stripe'
import { getPriceIdForTier } from '@/lib/stripe-helpers'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

interface CheckoutProps {
  tierId: string
  interval: 'monthly' | 'yearly' | 'lifetime'
  refCode?: string
}

export default function Checkout({ tierId, interval, refCode }: CheckoutProps) {
  const priceId = useMemo(() => getPriceIdForTier(tierId, interval), [tierId, interval])

  const startCheckoutSessionForTier = useCallback(
    async () => startCheckoutSession(priceId!, refCode),
    [priceId, refCode]
  )

  if (!priceId) {
    return <div className="text-red-500 text-sm">Invalid tier or interval selected</div>
  }

  return (
    <div id="checkout">
      <EmbeddedCheckoutProvider stripe={stripePromise} options={{ clientSecret: startCheckoutSessionForTier }}>
        <EmbeddedCheckout />
      </EmbeddedCheckoutProvider>
    </div>
  )
}
