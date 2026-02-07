import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import { stripe } from '@/lib/stripe'
import { sql } from '@/lib/db'

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!

// Commission amounts in cents by plan type
const COMMISSION_AMOUNTS: Record<string, number> = {
  monthly: 499,    // $4.99 monthly - 20% = $1.00
  annual: 2999,    // $29.99 annual - 20% = $6.00
  lifetime: 9999,  // $99.99 lifetime - 20% = $20.00
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.text()
    const signature = req.headers.get('stripe-signature')!

    let event: Stripe.Event

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret)
    } catch (err) {
      console.error('[Stripe Webhook] Signature verification failed:', err)
      return NextResponse.json({ error: 'Invalid signature' }, { status: 400 })
    }

    console.log('[Stripe Webhook] Event:', event.type)

    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session
        await handleCheckoutCompleted(session)
        break
      }
      case 'customer.subscription.created':
      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription
        await handleSubscriptionUpdated(subscription)
        break
      }
      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription
        await handleSubscriptionDeleted(subscription)
        break
      }
      case 'invoice.paid': {
        const invoice = event.data.object as Stripe.Invoice
        await handleInvoicePaid(invoice)
        break
      }
    }

    return NextResponse.json({ received: true })

  } catch (error) {
    console.error('[Stripe Webhook] Error:', error)
    return NextResponse.json({ error: 'Webhook handler failed' }, { status: 500 })
  }
}

async function handleCheckoutCompleted(session: Stripe.Checkout.Session) {
  const { customer_email, metadata, amount_total } = session
  const affiliateCode = metadata?.affiliateCode
  const planId = metadata?.planId || 'individual'
  const billingCycle = metadata?.billingCycle || 'monthly'

  console.log('[Stripe Webhook] Checkout completed:', {
    email: customer_email,
    affiliateCode,
    planId,
    billingCycle,
    amount: amount_total,
  })

  // Store subscription in database
  if (customer_email) {
    const customerId = typeof session.customer === 'string' ? session.customer : session.customer?.id || null
    const subscriptionId = typeof session.subscription === 'string' ? session.subscription : null

    await sql`
      INSERT INTO subscribers (email, stripe_customer_id, subscription_tier, plan, status, subscription_status, stripe_subscription_id, created_at, updated_at)
      VALUES (${customer_email}, ${customerId}, ${planId}, ${billingCycle}, 'active', 'active', ${subscriptionId}, NOW(), NOW())
      ON CONFLICT (email) DO UPDATE SET
        stripe_customer_id = COALESCE(${customerId}, subscribers.stripe_customer_id),
        subscription_tier = ${planId},
        plan = ${billingCycle},
        status = 'active',
        subscription_status = 'active',
        stripe_subscription_id = COALESCE(${subscriptionId}, subscribers.stripe_subscription_id),
        updated_at = NOW()
    `.catch(e => console.error('[Stripe Webhook] Failed to upsert subscriber:', e))
  }

  // Process affiliate commission if referral exists
  if (affiliateCode && affiliateCode.length > 0) {
    await processAffiliateCommission(affiliateCode, customer_email || '', billingCycle, amount_total || 0)
  }
}

async function handleSubscriptionUpdated(subscription: Stripe.Subscription) {
  console.log('[Stripe Webhook] Subscription updated:', subscription.id, subscription.status)

  // Get customer email from Stripe
  let customerEmail: string | null = null
  if (typeof subscription.customer === 'string') {
    try {
      const customer = await stripe.customers.retrieve(subscription.customer)
      if (!customer.deleted && customer.email) {
        customerEmail = customer.email
      }
    } catch (e) {
      console.error('[Stripe Webhook] Failed to retrieve customer:', e)
    }
  }

  const periodEnd = subscription.current_period_end
    ? new Date(subscription.current_period_end * 1000).toISOString()
    : null
  const cancelAtEnd = subscription.cancel_at_period_end ?? false
  const status = subscription.status // active, past_due, canceled, etc.

  if (customerEmail) {
    await sql`
      UPDATE subscribers
      SET subscription_status = ${status},
          stripe_subscription_id = ${subscription.id},
          current_period_end = ${periodEnd},
          cancel_at_period_end = ${cancelAtEnd},
          updated_at = NOW()
      WHERE email = ${customerEmail}
    `.catch(e => console.error('[Stripe Webhook] Failed to update subscription:', e))
  }
}

async function handleSubscriptionDeleted(subscription: Stripe.Subscription) {
  console.log('[Stripe Webhook] Subscription deleted:', subscription.id)

  // Mark subscriber as canceled
  await sql`
    UPDATE subscribers
    SET subscription_status = 'canceled',
        status = 'canceled',
        cancel_at_period_end = false,
        updated_at = NOW()
    WHERE stripe_subscription_id = ${subscription.id}
  `.catch(e => console.error('[Stripe Webhook] Failed to cancel subscription:', e))
}

async function handleInvoicePaid(invoice: Stripe.Invoice) {
  // Handle recurring payments for Legend tier (10% recurring forever)
  const customerEmail = typeof invoice.customer_email === 'string' ? invoice.customer_email : null
  
  if (!customerEmail) return

  console.log('[Stripe Webhook] Invoice paid:', {
    email: customerEmail,
    amount: invoice.amount_paid,
  })

  // Check if this customer was referred and referrer is Legend tier
  // Process recurring commission for Legend affiliates
  try {
    const referral = await sql`
      SELECT ar.affiliate_id, a.tier, ar.commission_rate
      FROM affiliate_referrals ar
      JOIN affiliates a ON ar.affiliate_id = a.id
      WHERE ar.referred_user_id = ${customerEmail}
        AND ar.status = 'converted'
        AND a.tier = 'legend'
      LIMIT 1
    `

    if (referral.length > 0) {
      const legendCommission = Math.floor(invoice.amount_paid * 0.10) // 10% recurring
      
      await sql`
        UPDATE affiliates
        SET total_earnings = total_earnings + ${legendCommission / 100},
            updated_at = NOW()
        WHERE id = ${referral[0].affiliate_id}
      `

      await sql`
        UPDATE affiliate_referrals
        SET total_commission = total_commission + ${legendCommission / 100}
        WHERE affiliate_id = ${referral[0].affiliate_id}
          AND referred_user_id = ${customerEmail}
      `

      console.log('[Stripe Webhook] Legend recurring commission:', {
        affiliateId: referral[0].affiliate_id,
        commission: legendCommission / 100,
      })
    }
  } catch (e) {
    console.error('[Stripe Webhook] Failed to process recurring commission:', e)
  }
}

async function processAffiliateCommission(
  affiliateCode: string,
  customerEmail: string,
  billingCycle: string,
  amountCents: number
) {
  try {
    // Get affiliate by code
    const affiliate = await sql`
      SELECT id, tier, referral_count
      FROM affiliates
      WHERE affiliate_code = ${affiliateCode.toUpperCase()}
      LIMIT 1
    `

    if (affiliate.length === 0) {
      console.log('[Stripe Webhook] Invalid affiliate code:', affiliateCode)
      return
    }

    const affiliateId = affiliate[0].id
    const tier = affiliate[0].tier || 'starter'
    const currentReferrals = affiliate[0].referral_count || 0

    // Commission rates by tier
    const tierRates: Record<string, number> = {
      'starter': 0.20,
      'advocate': 0.25,
      'ambassador': 0.30,
      'champion': 0.35,
      'legend': 0.40,
    }

    const commissionRate = tierRates[tier] || 0.20
    const commissionCents = Math.floor(amountCents * commissionRate)
    const commissionDollars = commissionCents / 100

    // Update or create referral record
    await sql`
      INSERT INTO affiliate_referrals (
        id, affiliate_id, referred_user_id, subscription_type, 
        commission_rate, total_commission, status, created_at
      )
      VALUES (
        gen_random_uuid(), ${affiliateId}, ${customerEmail}, ${billingCycle},
        ${commissionRate}, ${commissionDollars}, 'converted', NOW()
      )
      ON CONFLICT (affiliate_id, referred_user_id) DO UPDATE SET
        subscription_type = ${billingCycle},
        total_commission = affiliate_referrals.total_commission + ${commissionDollars},
        status = 'converted'
    `.catch(async () => {
      // If no unique constraint, just insert
      await sql`
        UPDATE affiliate_referrals
        SET subscription_type = ${billingCycle},
            total_commission = total_commission + ${commissionDollars},
            status = 'converted'
        WHERE affiliate_id = ${affiliateId} AND referred_user_id = ${customerEmail}
      `
    })

    // Update affiliate totals
    const newReferralCount = currentReferrals + 1
    
    // Check for tier upgrade
    let newTier = tier
    if (newReferralCount >= 50 && tier !== 'legend') newTier = 'legend'
    else if (newReferralCount >= 25 && tier !== 'legend' && tier !== 'champion') newTier = 'champion'
    else if (newReferralCount >= 10 && ['starter', 'advocate'].includes(tier)) newTier = 'ambassador'
    else if (newReferralCount >= 3 && tier === 'starter') newTier = 'advocate'

    await sql`
      UPDATE affiliates
      SET total_earnings = total_earnings + ${commissionDollars},
          referral_count = ${newReferralCount},
          tier = ${newTier},
          updated_at = NOW()
      WHERE id = ${affiliateId}
    `

    console.log('[Stripe Webhook] Commission processed:', {
      affiliateId,
      affiliateCode,
      customerEmail,
      commission: commissionDollars,
      newTier,
      newReferralCount,
    })

  } catch (error) {
    console.error('[Stripe Webhook] Failed to process commission:', error)
  }
}
