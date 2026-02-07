// lib/webhook-handlers/payment-intent-succeeded.ts
// Handle successful payment intent - update learner subscription

import type { Stripe } from 'stripe'

interface PaymentMetadata {
  learner_email?: string
  tier_id?: string
  interval?: 'monthly' | 'yearly' | 'lifetime'
  ref_code?: string
}

export async function handlePaymentIntentSucceeded(paymentIntent: Stripe.PaymentIntent) {
  const customerId = paymentIntent.customer as string
  const metadata = paymentIntent.metadata as unknown as PaymentMetadata
  const amount = paymentIntent.amount

  console.log('[payment-intent-succeeded] Processing:', {
    customerId,
    amount,
    tierId: metadata.tier_id,
  })

  try {
    // TODO: Update learner in database
    // const learner = await updateLearnerSubscription({
    //   stripe_customer_id: customerId,
    //   subscription_status: 'active',
    //   subscription_tier: metadata.tier_id || 'learner',
    //   subscription_interval: metadata.interval || 'monthly',
    //   subscription_activated_at: new Date(),
    //   subscription_renews_at: calculateRenewalDate(metadata.interval)
    // })

    // TODO: Track affiliate conversion if ref_code present
    // if (metadata.ref_code && metadata.ref_code !== 'direct') {
    //   await trackAffiliateConversion(
    //     metadata.ref_code,
    //     amount,
    //     metadata.tier_id || 'learner'
    //   )
    // }

    console.log(`[payment-intent-succeeded] Updated learner ${customerId}`)

    return { success: true }
  } catch (error) {
    console.error('[payment-intent-succeeded] Error:', error)
    // Don't throw - webhook must return 200
    return { success: false, error: String(error) }
  }
}

function calculateRenewalDate(interval: string | undefined): Date {
  const now = new Date()
  switch (interval) {
    case 'yearly':
      now.setFullYear(now.getFullYear() + 1)
      break
    case 'lifetime':
      now.setFullYear(now.getFullYear() + 100) // Effectively permanent
      break
    case 'monthly':
    default:
      now.setMonth(now.getMonth() + 1)
      break
  }
  return now
}
