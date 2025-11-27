/**
 * Stripe Webhook Handler
 * Receives events from Stripe (payments, subscriptions, failures)
 * 
 * Webhook URL: https://curiouskelly.com/api/stripe-webhook
 */

import type { APIRoute } from 'astro';
import Stripe from 'stripe';
import {
  identifyCustomer,
  trackSubscriptionPurchased,
  trackPaymentFailed,
  trackEvent,
} from '@/lib/customerio';
import {
  upsertUserFromCheckout,
  activateSubscription,
  updateSubscriptionStatus,
} from '@/lib/supabase';

// Initialize Stripe only if key is present
const stripeKey = import.meta.env.STRIPE_SECRET_KEY;
const stripe = stripeKey ? new Stripe(stripeKey, {
  apiVersion: '2024-11-20.acacia',
}) : null;

export const POST: APIRoute = async ({ request }) => {
  const sig = request.headers.get('stripe-signature');
  const webhookSecret = import.meta.env.STRIPE_WEBHOOK_SECRET;

  if (!stripe || !webhookSecret) {
    console.error('Stripe not configured');
    return new Response('Webhook Error: Service unavailable', { status: 503 });
  }

  if (!sig) {
    console.error('Missing stripe-signature header');
    return new Response('Webhook Error: Missing signature', { status: 400 });
  }

  let event: Stripe.Event;

  try {
    const body = await request.text();
    event = stripe.webhooks.constructEvent(body, sig, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return new Response(`Webhook Error: ${err instanceof Error ? err.message : 'Unknown'}`, {
      status: 400,
    });
  }

  // Handle the event
  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;
        console.log('✅ Checkout completed:', session.id);

        // Get customer email and info
        const customerEmail = session.customer_email || session.customer_details?.email;
        const customerName = session.customer_details?.name || session.metadata?.customer_name;
        const customerId = session.client_reference_id || session.customer as string;
        const stripeCustomerId = session.customer as string;
        const plan = (session.metadata?.plan || 'annual') as 'monthly' | 'annual' | 'gift';
        const isGift = session.metadata?.is_gift === 'true';

        if (customerEmail) {
          // 1. Identify customer in Customer.io
          await identifyCustomer({
            id: customerId,
            email: customerEmail,
            first_name: customerName?.split(' ')[0],
            created_at: Math.floor(Date.now() / 1000),
          });

          // 2. Track trial started (or purchase for gifts)
          await trackEvent(customerId, {
            name: isGift ? 'gift_purchased' : 'trial_started',
            data: {
              session_id: session.id,
              payment_status: session.payment_status,
              plan,
            },
          });

          // 3. Create/update user in database
          const trialEndDate = new Date();
          trialEndDate.setDate(trialEndDate.getDate() + 7);

          const dbResult = await upsertUserFromCheckout({
            email: customerEmail,
            name: customerName,
            stripeCustomerId: stripeCustomerId || customerId,
            plan,
            trialEndDate: isGift ? undefined : trialEndDate,
          });

          if (!dbResult.success) {
            console.error('Failed to create user record:', dbResult.error);
            // Don't fail the webhook - user can still use the service
          }
        }

        break;
      }

      case 'invoice.payment_succeeded': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('✅ Payment succeeded:', invoice.id);

        if (invoice.customer && invoice.subscription) {
          const stripeCustomerId = invoice.customer as string;

          // Get subscription details
          const subscription = await stripe.subscriptions.retrieve(invoice.subscription as string);
          const priceId = subscription.items.data[0]?.price.id;

          // Determine plan type
          let planType: 'monthly' | 'annual' | 'gift' = 'monthly';
          if (priceId === import.meta.env.STRIPE_PRICE_ANNUAL) {
            planType = 'annual';
          } else if (priceId === import.meta.env.STRIPE_PRICE_GIFT) {
            planType = 'gift';
          }

          // 1. Track in Customer.io
          await trackSubscriptionPurchased(stripeCustomerId, {
            plan: planType,
            amount: invoice.amount_paid / 100, // Convert cents to dollars
            stripe_customer_id: stripeCustomerId,
            stripe_subscription_id: subscription.id,
          });

          // 2. Activate subscription in database (trial ended, payment successful)
          const dbResult = await activateSubscription({
            stripeCustomerId,
            plan: planType,
            subscriptionId: subscription.id,
          });

          if (!dbResult.success) {
            console.error('Failed to activate subscription in DB:', dbResult.error);
          }
        }

        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('❌ Payment failed:', invoice.id);

        if (invoice.customer) {
          const stripeCustomerId = invoice.customer as string;

          // 1. Track failed payment in Customer.io
          await trackPaymentFailed(stripeCustomerId, {
            amount: invoice.amount_due / 100,
            reason: invoice.last_finalization_error?.message,
            stripe_invoice_id: invoice.id,
          });

          // 2. Mark subscription as inactive in database
          // (Customer.io handles the dunning emails via automated campaigns)
          await updateSubscriptionStatus({
            stripeCustomerId,
            status: 'inactive', // Will reactivate when payment succeeds
          });
        }

        break;
      }

      case 'customer.subscription.created': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('✅ Subscription created:', subscription.id);
        // User record is created in checkout.session.completed
        // No additional action needed here
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        const stripeCustomerId = subscription.customer as string;
        console.log('🔄 Subscription updated:', subscription.id, 'Status:', subscription.status);

        // Map Stripe status to our status
        let dbStatus: 'active' | 'inactive' | 'cancelled' | 'expired' | 'trialing';
        
        switch (subscription.status) {
          case 'active':
            dbStatus = 'active';
            break;
          case 'trialing':
            dbStatus = 'trialing';
            break;
          case 'past_due':
          case 'unpaid':
            dbStatus = 'inactive';
            break;
          case 'canceled':
            dbStatus = 'cancelled';
            break;
          default:
            dbStatus = 'inactive';
        }

        // Update database
        await updateSubscriptionStatus({
          stripeCustomerId,
          status: dbStatus,
        });

        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        const stripeCustomerId = subscription.customer as string;
        console.log('🗑️ Subscription cancelled:', subscription.id);

        // 1. Track cancellation in Customer.io
        await trackEvent(stripeCustomerId, {
          name: 'subscription_cancelled',
          data: {
            subscription_id: subscription.id,
            cancel_at_period_end: subscription.cancel_at_period_end,
          },
        });

        // 2. Update database - mark as cancelled
        await updateSubscriptionStatus({
          stripeCustomerId,
          status: 'cancelled',
        });

        // Customer.io handles cancellation confirmation emails via automated campaigns
        break;
      }

      case 'customer.subscription.trial_will_end': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('⏰ Trial ending soon:', subscription.id);

        // This fires 3 days before trial ends
        // Customer.io should handle trial reminder emails
        // But we track the event for reference
        if (subscription.customer) {
          const customerId = subscription.customer as string;
          await trackEvent(customerId, {
            name: 'trial_ending_soon',
            data: {
              trial_end: subscription.trial_end,
            },
          });
        }

        break;
      }

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return new Response(JSON.stringify({ received: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error processing webhook:', error);
    return new Response(
      JSON.stringify({ error: 'Webhook handler failed', details: error instanceof Error ? error.message : 'Unknown' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
};





