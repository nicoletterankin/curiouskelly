const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

/**
 * Create Stripe checkout session for gift purchase
 */
async function createGiftCheckoutSession({ customerEmail, recipientEmail, giftMessage, gifterName }) {
  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price: process.env.PRICE_ID_GIFT,
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${process.env.FRONTEND_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.FRONTEND_URL}/?canceled=true`,
      customer_email: customerEmail,
      metadata: {
        type: 'gift',
        recipient_email: recipientEmail,
        gift_message: giftMessage || '',
        gifter_name: gifterName || '',
      },
    });

    console.log('✓ Checkout session created:', session.id);
    return session;
  } catch (error) {
    console.error('✗ Stripe checkout error:', error.message);
    throw error;
  }
}

/**
 * Create checkout session for personal plan
 */
async function createPersonalCheckoutSession({ customerEmail }) {
  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price: process.env.PRICE_ID_PERSONAL,
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: `${process.env.FRONTEND_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.FRONTEND_URL}/?canceled=true`,
      customer_email: customerEmail,
      metadata: {
        type: 'personal',
      },
    });

    return session;
  } catch (error) {
    console.error('✗ Stripe checkout error:', error.message);
    throw error;
  }
}

/**
 * Create checkout session for family plan
 */
async function createFamilyCheckoutSession({ customerEmail }) {
  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price: process.env.PRICE_ID_FAMILY,
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: `${process.env.FRONTEND_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.FRONTEND_URL}/?canceled=true`,
      customer_email: customerEmail,
      metadata: {
        type: 'family',
      },
    });

    return session;
  } catch (error) {
    console.error('✗ Stripe checkout error:', error.message);
    throw error;
  }
}

/**
 * Retrieve checkout session
 */
async function getCheckoutSession(sessionId) {
  try {
    const session = await stripe.checkout.sessions.retrieve(sessionId);
    return session;
  } catch (error) {
    console.error('✗ Stripe session retrieval error:', error.message);
    throw error;
  }
}

/**
 * Construct webhook event from request
 */
function constructWebhookEvent(payload, signature) {
  try {
    const event = stripe.webhooks.constructEvent(
      payload,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET
    );
    return event;
  } catch (error) {
    console.error('✗ Webhook signature verification failed:', error.message);
    throw error;
  }
}

module.exports = {
  createGiftCheckoutSession,
  createPersonalCheckoutSession,
  createFamilyCheckoutSession,
  getCheckoutSession,
  constructWebhookEvent,
  stripe
};






