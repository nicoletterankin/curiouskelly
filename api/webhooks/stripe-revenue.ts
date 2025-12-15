/**
 * Stripe Revenue Events Webhook
 * POST /api/webhooks/stripe-revenue
 * 
 * Captures all Stripe events and records them to the revenue_events table
 * for CFO financial tracking and reporting
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';
import { Buffer } from 'node:buffer';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const stripeKey = process.env.STRIPE_SECRET_KEY!;
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

// Plan prices in cents for MRR calculation
const PLAN_PRICES: Record<string, { price_cents: number; mrr_cents: number }> = {
  'monthly': { price_cents: 999, mrr_cents: 999 },
  'annual': { price_cents: 9999, mrr_cents: 833 }, // $99.99/12 = ~$8.33/mo
  'lifetime': { price_cents: 29999, mrr_cents: 0 }, // One-time, no MRR
  'gift_3mo': { price_cents: 3499, mrr_cents: 0 },
  'gift_6mo': { price_cents: 5999, mrr_cents: 0 },
  'gift_12mo': { price_cents: 9999, mrr_cents: 0 },
};

/**
 * Record commission for referrer when a referred user pays
 * EARN TO LEARN: Commission rates based on referrer's learning progress
 */
async function recordCommissionIfReferred(
  supabase: ReturnType<typeof createClient>,
  customerEmail: string | null,
  amountCents: number,
  transactionType: 'initial_subscription' | 'subscription_renewal' | 'gift_purchase' | 'lifetime_purchase',
  stripePaymentIntentId?: string,
  stripeInvoiceId?: string,
  stripeSubscriptionId?: string
): Promise<void> {
  if (!customerEmail || amountCents <= 0) return;

  try {
    // Find the user by email
    const { data: user, error: userError } = await supabase
      .from('users')
      .select('id, referred_by_user_id')
      .eq('email', customerEmail.toLowerCase())
      .single();

    if (userError || !user || !user.referred_by_user_id) {
      // No referrer - no commission
      return;
    }

    // Get the referrer's commission rate
    const { data: referrer, error: referrerError } = await supabase
      .from('users')
      .select('id, commission_rate, pending_earnings, lifetime_earnings, total_referrals')
      .eq('id', user.referred_by_user_id)
      .single();

    if (referrerError || !referrer) {
      console.warn('[Commission] Referrer not found:', user.referred_by_user_id);
      return;
    }

    // Calculate commission (convert cents to dollars, apply rate)
    const grossAmount = amountCents / 100;
    const commissionRate = parseFloat(referrer.commission_rate) || 0.10;
    const commissionAmount = grossAmount * commissionRate;

    // Insert commission transaction
    const { error: transactionError } = await supabase
      .from('commission_transactions')
      .insert({
        referrer_id: referrer.id,
        referred_user_id: user.id,
        transaction_type: transactionType,
        gross_amount: grossAmount,
        commission_rate: commissionRate,
        commission_amount: commissionAmount,
        stripe_payment_intent_id: stripePaymentIntentId || null,
        stripe_invoice_id: stripeInvoiceId || null,
        stripe_subscription_id: stripeSubscriptionId || null,
        status: 'pending', // Will be approved after 7 days
        currency: 'USD'
      });

    if (transactionError) {
      console.error('[Commission] Failed to record transaction:', transactionError);
      return;
    }

    // Update referrer's earnings
    const newPendingEarnings = (parseFloat(referrer.pending_earnings) || 0) + commissionAmount;
    const newLifetimeEarnings = (parseFloat(referrer.lifetime_earnings) || 0) + commissionAmount;

    const { error: updateError } = await supabase
      .from('users')
      .update({
        pending_earnings: newPendingEarnings,
        lifetime_earnings: newLifetimeEarnings
      })
      .eq('id', referrer.id);

    if (updateError) {
      console.error('[Commission] Failed to update referrer earnings:', updateError);
      return;
    }

    console.log(`[Commission] Recorded: $${commissionAmount.toFixed(2)} for referrer ${referrer.id} (${(commissionRate * 100).toFixed(0)}% of $${grossAmount.toFixed(2)})`);

  } catch (error) {
    console.error('[Commission] Error recording commission:', error);
  }
}

/**
 * Clawback commission when a refund occurs
 */
async function clawbackCommission(
  supabase: ReturnType<typeof createClient>,
  stripePaymentIntentId: string,
  refundAmountCents: number
): Promise<void> {
  try {
    // Find the original commission transaction
    const { data: transaction, error: findError } = await supabase
      .from('commission_transactions')
      .select('*')
      .eq('stripe_payment_intent_id', stripePaymentIntentId)
      .single();

    if (findError || !transaction) {
      // No commission was recorded for this payment
      return;
    }

    // Calculate clawback amount (proportional to refund)
    const refundRatio = (refundAmountCents / 100) / transaction.gross_amount;
    const clawbackAmount = transaction.commission_amount * refundRatio;

    // Insert clawback transaction
    await supabase
      .from('commission_transactions')
      .insert({
        referrer_id: transaction.referrer_id,
        referred_user_id: transaction.referred_user_id,
        transaction_type: 'refund_clawback',
        gross_amount: -(refundAmountCents / 100),
        commission_rate: transaction.commission_rate,
        commission_amount: -clawbackAmount,
        stripe_payment_intent_id: stripePaymentIntentId,
        status: 'approved', // Clawbacks are immediate
        notes: `Clawback for refund of $${(refundAmountCents / 100).toFixed(2)}`,
        currency: 'USD'
      });

    // Update referrer's earnings (reduce)
    const { data: referrer } = await supabase
      .from('users')
      .select('pending_earnings, lifetime_earnings')
      .eq('id', transaction.referrer_id)
      .single();

    if (referrer) {
      const newPending = Math.max(0, (parseFloat(referrer.pending_earnings) || 0) - clawbackAmount);
      const newLifetime = Math.max(0, (parseFloat(referrer.lifetime_earnings) || 0) - clawbackAmount);

      await supabase
        .from('users')
        .update({
          pending_earnings: newPending,
          lifetime_earnings: newLifetime
        })
        .eq('id', transaction.referrer_id);
    }

    console.log(`[Commission] Clawback: $${clawbackAmount.toFixed(2)} from referrer ${transaction.referrer_id}`);

  } catch (error) {
    console.error('[Commission] Error processing clawback:', error);
  }
}

async function recordRevenueEvent(
  supabase: ReturnType<typeof createClient>,
  eventType: string,
  data: {
    user_id?: string;
    stripe_customer_id?: string;
    stripe_subscription_id?: string;
    stripe_invoice_id?: string;
    amount_cents: number;
    plan_type?: string;
    mrr_impact_cents?: number;
    metadata?: Record<string, any>;
  }
) {
  const { error } = await supabase.from('revenue_events').insert({
    event_type: eventType,
    user_id: data.user_id,
    stripe_customer_id: data.stripe_customer_id,
    stripe_subscription_id: data.stripe_subscription_id,
    stripe_invoice_id: data.stripe_invoice_id,
    amount_cents: data.amount_cents,
    plan_type: data.plan_type,
    mrr_impact_cents: data.mrr_impact_cents || 0,
    metadata: data.metadata || {},
  });

  if (error) {
    console.error('Failed to record revenue event:', error);
    throw error;
  }
}

async function bestEffortSyncUserFromSubscription(
  supabase: ReturnType<typeof createClient>,
  subscription: Stripe.Subscription
): Promise<void> {
  const userId = subscription.metadata?.user_id;
  if (!userId) return;

  try {
    const planType = subscription.metadata?.plan_type || null;
    const currentPeriodEndIso = subscription.current_period_end
      ? new Date(subscription.current_period_end * 1000).toISOString()
      : null;

    await supabase
      .from('users')
      .update({
        stripe_customer_id: subscription.customer as string,
        stripe_subscription_id: subscription.id,
        subscription_tier: planType,
        subscription_status: subscription.status,
        current_period_end: currentPeriodEndIso,
        cancel_at_period_end: subscription.cancel_at_period_end,
      })
      .eq('id', userId);
  } catch (e) {
    console.warn('[stripe-revenue] Failed to sync user from subscription', e);
  }
}

async function bestEffortSyncUserFromCheckoutSession(
  supabase: ReturnType<typeof createClient>,
  session: Stripe.Checkout.Session
): Promise<void> {
  const userId = session.metadata?.user_id;
  if (!userId) return;

  try {
    const tier = session.metadata?.plan_type || session.metadata?.type || null;
    const status = tier === 'lifetime' ? 'active' : 'trialing';
    await supabase
      .from('users')
      .update({
        stripe_customer_id: session.customer as string,
        subscription_tier: tier,
        subscription_status: status,
      })
      .eq('id', userId);
  } catch (e) {
    console.warn('[stripe-revenue] Failed to sync user from checkout session', e);
  }
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const stripe = new Stripe(stripeKey, { apiVersion: '2023-10-16' });
  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  // Verify webhook signature
  const sig = req.headers['stripe-signature'] as string;
  let event: Stripe.Event;

  try {
    // Stripe signature verification REQUIRES the exact raw request body.
    // If we stringify a parsed object, verification can succeed/fail nondeterministically.
    if (!sig) {
      return res.status(400).json({ error: 'Missing stripe-signature header' });
    }

    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const rawBody = Buffer.concat(chunks).toString('utf8');
    event = stripe.webhooks.constructEvent(rawBody, sig, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return res.status(400).json({ error: 'Invalid signature' });
  }

  console.log(`Processing Stripe event: ${event.type}`);

  try {
    switch (event.type) {
      // ============================================
      // SUBSCRIPTION EVENTS
      // ============================================
      case 'customer.subscription.created': {
        const subscription = event.data.object as Stripe.Subscription;
        const planType = subscription.metadata?.plan_type || 'monthly';
        const planInfo = PLAN_PRICES[planType] || PLAN_PRICES['monthly'];

        await recordRevenueEvent(supabase, 'subscription_created', {
          stripe_customer_id: subscription.customer as string,
          stripe_subscription_id: subscription.id,
          amount_cents: 0, // No charge yet (trial)
          plan_type: planType,
          mrr_impact_cents: subscription.status === 'trialing' ? 0 : planInfo.mrr_cents,
          metadata: {
            status: subscription.status,
            trial_end: subscription.trial_end,
          },
        });

        // Keep app entitlements in sync (server-side truth)
        await bestEffortSyncUserFromSubscription(supabase, subscription);

        // Also record trial start if applicable
        if (subscription.status === 'trialing') {
          await recordRevenueEvent(supabase, 'trial_started', {
            stripe_customer_id: subscription.customer as string,
            stripe_subscription_id: subscription.id,
            amount_cents: 0,
            plan_type: planType,
            mrr_impact_cents: 0,
            metadata: { trial_end: subscription.trial_end },
          });
        }
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        const previousAttributes = event.data.previous_attributes as any;
        
        // Check if this is a trial conversion
        if (previousAttributes?.status === 'trialing' && subscription.status === 'active') {
          const planType = subscription.metadata?.plan_type || 'monthly';
          const planInfo = PLAN_PRICES[planType] || PLAN_PRICES['monthly'];

          await recordRevenueEvent(supabase, 'trial_converted', {
            stripe_customer_id: subscription.customer as string,
            stripe_subscription_id: subscription.id,
            amount_cents: planInfo.price_cents,
            plan_type: planType,
            mrr_impact_cents: planInfo.mrr_cents,
          });
        }

        // Keep entitlements current (status/cancel_at_period_end/current_period_end)
        await bestEffortSyncUserFromSubscription(supabase, subscription);
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        const planType = subscription.metadata?.plan_type || 'monthly';
        const planInfo = PLAN_PRICES[planType] || PLAN_PRICES['monthly'];

        await recordRevenueEvent(supabase, 'subscription_cancelled', {
          stripe_customer_id: subscription.customer as string,
          stripe_subscription_id: subscription.id,
          amount_cents: 0,
          plan_type: planType,
          mrr_impact_cents: -planInfo.mrr_cents, // Negative impact
          metadata: { cancellation_reason: subscription.cancellation_details?.reason },
        });

        // Keep entitlements current (revoked/cancelled)
        await bestEffortSyncUserFromSubscription(supabase, subscription);
        break;
      }

      // ============================================
      // PAYMENT EVENTS
      // ============================================
      case 'invoice.payment_succeeded': {
        const invoice = event.data.object as Stripe.Invoice;
        
        // Skip $0 invoices (trial starts)
        if (invoice.amount_paid === 0) break;

        const planType = invoice.subscription_details?.metadata?.plan_type || 'monthly';

        await recordRevenueEvent(supabase, 'payment_succeeded', {
          stripe_customer_id: invoice.customer as string,
          stripe_subscription_id: invoice.subscription as string,
          stripe_invoice_id: invoice.id,
          amount_cents: invoice.amount_paid,
          plan_type: planType,
          metadata: {
            invoice_number: invoice.number,
            billing_reason: invoice.billing_reason,
          },
        });

        // EARN TO LEARN: Record commission for referrer
        const isRenewal = invoice.billing_reason === 'subscription_cycle';
        const transactionType = isRenewal ? 'subscription_renewal' : 'initial_subscription';
        
        await recordCommissionIfReferred(
          supabase,
          invoice.customer_email,
          invoice.amount_paid,
          transactionType,
          invoice.payment_intent as string || undefined,
          invoice.id,
          invoice.subscription as string || undefined
        );

        // If this is a renewal, also record that
        if (isRenewal) {
          await recordRevenueEvent(supabase, 'subscription_renewed', {
            stripe_customer_id: invoice.customer as string,
            stripe_subscription_id: invoice.subscription as string,
            amount_cents: invoice.amount_paid,
            plan_type: planType,
          });
        }
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;

        await recordRevenueEvent(supabase, 'payment_failed', {
          stripe_customer_id: invoice.customer as string,
          stripe_subscription_id: invoice.subscription as string,
          stripe_invoice_id: invoice.id,
          amount_cents: invoice.amount_due,
          metadata: {
            attempt_count: invoice.attempt_count,
            next_payment_attempt: invoice.next_payment_attempt,
          },
        });
        break;
      }

      // ============================================
      // ONE-TIME PAYMENT EVENTS (Gifts, Lifetime)
      // ============================================
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;

        // Sync user record early (best-effort). Subscriptions will be finalized by subscription.created.
        await bestEffortSyncUserFromCheckoutSession(supabase, session);
        
        // Only process one-time payments (gifts, lifetime)
        if (session.mode === 'payment') {
          const paymentType = session.metadata?.type || 'gift';
          
          if (paymentType === 'gift') {
            await recordRevenueEvent(supabase, 'gift_purchased', {
              stripe_customer_id: session.customer as string,
              amount_cents: session.amount_total || 0,
              plan_type: `gift_${session.metadata?.duration || '12mo'}`,
              metadata: {
                recipient_email: session.metadata?.recipient_email,
                gifter_name: session.metadata?.gifter_name,
                gift_message: session.metadata?.gift_message,
              },
            });

            // EARN TO LEARN: Commission for gift purchases (credit to recipient's referrer)
            const recipientEmail = session.metadata?.recipient_email;
            if (recipientEmail && session.amount_total) {
              await recordCommissionIfReferred(
                supabase,
                recipientEmail,
                session.amount_total,
                'gift_purchase',
                session.payment_intent as string || undefined
              );
            }
          } else if (paymentType === 'lifetime') {
            await recordRevenueEvent(supabase, 'payment_succeeded', {
              stripe_customer_id: session.customer as string,
              amount_cents: session.amount_total || 0,
              plan_type: 'lifetime',
              metadata: { type: 'lifetime_purchase' },
            });

            // EARN TO LEARN: Commission for lifetime purchases
            if (session.customer_email && session.amount_total) {
              await recordCommissionIfReferred(
                supabase,
                session.customer_email,
                session.amount_total,
                'lifetime_purchase',
                session.payment_intent as string || undefined
              );
            }
          }
        }
        break;
      }

      // ============================================
      // REFUND EVENTS
      // ============================================
      case 'charge.refunded': {
        const charge = event.data.object as Stripe.Charge;
        
        await recordRevenueEvent(supabase, 'refund_issued', {
          stripe_customer_id: charge.customer as string,
          amount_cents: charge.amount_refunded,
          metadata: {
            refund_reason: charge.refunds?.data?.[0]?.reason,
            original_amount: charge.amount,
          },
        });

        // EARN TO LEARN: Clawback commission on refund
        if (charge.payment_intent && charge.amount_refunded > 0) {
          await clawbackCommission(
            supabase,
            charge.payment_intent as string,
            charge.amount_refunded
          );
        }
        break;
      }

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return res.status(200).json({ received: true, event_type: event.type });

  } catch (error) {
    console.error('Webhook processing error:', error);
    return res.status(500).json({
      error: 'Webhook processing failed',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

// Disable body parsing - we need the raw body for Stripe signature verification
export const config = {
  api: {
    bodyParser: false,
  },
};

