import useSWR from 'swr'

interface SubscriptionData {
  tier: 'free' | 'individual' | 'family'
  plan: string | null
  status: string
  subscriptionStatus: string | null
  periodEnd: string | null
  cancelAtPeriodEnd: boolean
  hasStripeCustomer: boolean
  createdAt: string | null
}

const fetcher = (url: string) => fetch(url).then(r => r.json())

/**
 * SWR-based hook for subscription status.
 * Pass in the learner's email to check their subscription tier.
 */
export function useSubscription(email: string | null | undefined) {
  const { data, error, isLoading, mutate } = useSWR<SubscriptionData>(
    email ? `/api/subscription?email=${encodeURIComponent(email)}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60_000, // Cache for 1 minute
    }
  )

  const isPremium = data?.tier !== 'free' && data?.status === 'active'
  const isLifetime = data?.plan === 'lifetime'
  const isCanceling = data?.cancelAtPeriodEnd === true

  return {
    subscription: data,
    isPremium,
    isLifetime,
    isCanceling,
    isLoading,
    error,
    refresh: mutate,
  }
}

/**
 * Open the Stripe Customer Portal for billing management.
 */
export async function openCustomerPortal(email: string): Promise<void> {
  const res = await fetch('/api/stripe/portal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  })

  const data = await res.json()

  if (data.url) {
    window.location.href = data.url
  } else {
    throw new Error(data.error || 'Failed to open billing portal')
  }
}

/**
 * Start a Stripe Checkout session and redirect.
 */
export async function startCheckout(
  planId: string,
  billingCycle: string,
  email: string
): Promise<void> {
  const res = await fetch('/api/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      planId,
      billingCycle,
      email,
      successUrl: `${window.location.origin}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancelUrl: window.location.href,
    }),
  })

  const data = await res.json()

  if (data.url) {
    window.location.href = data.url
  } else {
    throw new Error(data.error || 'Failed to start checkout')
  }
}
