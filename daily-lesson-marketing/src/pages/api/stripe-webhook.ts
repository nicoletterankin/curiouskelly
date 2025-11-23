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

const stripe = new Stripe(import.meta.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2024-11-20.acacia',
});

export const POST: APIRoute = async ({ request }) => {
  const sig = request.headers.get('stripe-signature');
  const webhookSecret = import.meta.env.STRIPE_WEBHOOK_SECRET;

  if (!sig || !webhookSecret) {
    console.error('Missing stripe-signature header or webhook secret');
    return new Response('Webhook Error: Missing signature or secret', { status: 400 });
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

        // Get customer email
        const customerEmail = session.customer_email || session.customer_details?.email;
        const customerId = session.client_reference_id || session.customer as string;

        if (customerEmail && customerId) {
          // Identify customer in Customer.io
          await identifyCustomer({
            id: customerId,
            email: customerEmail,
            created_at: Math.floor(Date.now() / 1000),
          });

          // Track trial started
          await trackEvent(customerId, {
            name: 'trial_started',
            data: {
              session_id: session.id,
              payment_status: session.payment_status,
            },
          });
        }

        // TODO: Create user in database
        // TODO: Grant access to lessons
        break;
      }

      case 'invoice.payment_succeeded': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('✅ Payment succeeded:', invoice.id);

        if (invoice.customer && invoice.subscription) {
          const customerId = invoice.customer as string;

          // Get subscription details
          const subscription = await stripe.subscriptions.retrieve(invoice.subscription as string);
          const plan = subscription.items.data[0]?.price.id;

          // Determine plan type
          let planType: 'monthly' | 'annual' | 'gift' = 'monthly';
          if (plan === import.meta.env.STRIPE_PRICE_ANNUAL) {
            planType = 'annual';
          } else if (plan === import.meta.env.STRIPE_PRICE_GIFT) {
            planType = 'gift';
          }

          // Track in Customer.io
          await trackSubscriptionPurchased(customerId, {
            plan: planType,
            amount: invoice.amount_paid / 100, // Convert cents to dollars
            stripe_customer_id: customerId,
            stripe_subscription_id: subscription.id,
          });
        }

        // TODO: Update database with payment confirmation
        // TODO: Send receipt email
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('❌ Payment failed:', invoice.id);

        if (invoice.customer) {
          const customerId = invoice.customer as string;

          // Track failed payment
          await trackPaymentFailed(customerId, {
            amount: invoice.amount_due / 100,
            reason: invoice.last_finalization_error?.message,
            stripe_invoice_id: invoice.id,
          });
        }

        // TODO: Send dunning email (payment update required)
        // TODO: Mark subscription as past_due in database
        break;
      }

      case 'customer.subscription.created': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('✅ Subscription created:', subscription.id);

        // TODO: Create subscription record in database
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('🔄 Subscription updated:', subscription.id);

        // Check if status changed
        if (subscription.status === 'active') {
          // Subscription activated (trial ended, payment successful)
          // TODO: Ensure access is granted
        } else if (subscription.status === 'past_due') {
          // Payment failed
          // TODO: Send dunning email
        } else if (subscription.status === 'canceled' || subscription.status === 'unpaid') {
          // Subscription cancelled
          // TODO: Revoke access
        }

        // TODO: Update subscription status in database
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('🗑️ Subscription cancelled:', subscription.id);

        if (subscription.customer) {
          const customerId = subscription.customer as string;

          // Track cancellation
          await trackEvent(customerId, {
            name: 'subscription_cancelled',
            data: {
              subscription_id: subscription.id,
              cancel_at_period_end: subscription.cancel_at_period_end,
            },
          });
        }

        // TODO: Revoke access in database
        // TODO: Send cancellation confirmation email
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





