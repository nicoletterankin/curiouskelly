/**
 * Create Stripe Checkout Session
 * Called when user clicks "Start Trial" or "Buy Now"
 */

import type { APIRoute } from 'astro';
import Stripe from 'stripe';
import { trackTrialStarted } from '@/lib/customerio';

// Initialize Stripe only if key is present (prevents build errors during static generation)
const stripeKey = import.meta.env.STRIPE_SECRET_KEY;
const stripe = stripeKey ? new Stripe(stripeKey, {
  apiVersion: '2024-11-20.acacia',
}) : null;

export const POST: APIRoute = async ({ request }) => {
  try {
    if (!stripe) {
      console.error('Stripe secret key not configured');
      return new Response(
        JSON.stringify({ error: 'Payment service unavailable' }),
        { status: 503, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const body = await request.json();
    const { plan, email, name, customerId, mode = 'subscription' } = body;

    // Validate input
    if (!plan || !email) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields: plan, email' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Determine price ID based on plan
    let priceId: string;
    let isGift = false;

    switch (plan) {
      case 'monthly':
        priceId = import.meta.env.STRIPE_PRICE_MONTHLY;
        break;
      case 'annual':
        priceId = import.meta.env.STRIPE_PRICE_ANNUAL;
        break;
      case 'gift':
        priceId = import.meta.env.STRIPE_PRICE_GIFT;
        isGift = true;
        break;
      default:
        return new Response(
          JSON.stringify({ error: 'Invalid plan. Must be: monthly, annual, or gift' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } }
        );
    }

    if (!priceId) {
      return new Response(
        JSON.stringify({ error: `Price ID not configured for plan: ${plan}` }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Calculate trial end date (7 days from now)
    const trialEndDate = new Date();
    trialEndDate.setDate(trialEndDate.getDate() + 7);

    // Create checkout session
    const session = await stripe.checkout.sessions.create({
      mode: isGift ? 'payment' : 'subscription',
      payment_method_types: ['card'],
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      success_url: `${import.meta.env.PUBLIC_SITE_URL}/welcome?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${import.meta.env.PUBLIC_SITE_URL}/checkout?cancelled=true`,
      customer_email: email,
      client_reference_id: customerId || email,
      
      // Free trial for subscriptions (not gifts)
      ...(isGift
        ? {}
        : {
            subscription_data: {
              trial_period_days: 7,
              metadata: {
                plan: plan,
                trial_end_date: trialEndDate.toISOString(),
              },
            },
          }),

      metadata: {
        plan: plan,
        customer_name: name || '',
        is_gift: isGift.toString(),
      },

      // Collect billing address for tax purposes
      billing_address_collection: 'auto',

      // Allow promotion codes
      allow_promotion_codes: true,
    });

    // Track trial started in Customer.io (for non-gift purchases)
    if (!isGift && customerId) {
      try {
        await trackTrialStarted(customerId, {
          email,
          first_name: name?.split(' ')[0] || 'Friend',
          trial_end_date: trialEndDate.toISOString().split('T')[0],
          plan: plan as 'monthly' | 'annual',
        });
      } catch (error) {
        console.error('Failed to track trial in Customer.io:', error);
        // Don't fail the checkout if tracking fails
      }
    }

    // Return the checkout URL
    return new Response(
      JSON.stringify({
        url: session.url,
        sessionId: session.id,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Error creating checkout session:', error);
    return new Response(
      JSON.stringify({
        error: 'Failed to create checkout session',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
};





