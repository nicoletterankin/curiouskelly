const { constructWebhookEvent } = require('../services/stripe');
const { query } = require('../services/database');
const { generateGiftCode } = require('../services/gift-codes');
const { sendGiftRecipientEmail, sendGifterConfirmationEmail } = require('../services/email');

/**
 * POST /webhook
 * Handle Stripe webhook events
 * Note: This route receives raw body, configured in server.js
 */
async function handleWebhook(req, res) {
  const signature = req.headers['stripe-signature'];
  
  let event;
  try {
    event = constructWebhookEvent(req.body, signature);
  } catch (error) {
    console.error('Webhook signature verification failed:', error.message);
    return res.status(400).send(`Webhook Error: ${error.message}`);
  }

  console.log('Webhook event received:', event.type);

  try {
    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutCompleted(event.data.object);
        break;

      case 'customer.subscription.created':
        await handleSubscriptionCreated(event.data.object);
        break;

      case 'customer.subscription.deleted':
        await handleSubscriptionDeleted(event.data.object);
        break;

      case 'invoice.payment_succeeded':
        await handlePaymentSucceeded(event.data.object);
        break;

      case 'invoice.payment_failed':
        await handlePaymentFailed(event.data.object);
        break;

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    res.json({ received: true });
  } catch (error) {
    console.error('Webhook processing error:', error);
    res.status(500).json({ error: 'Webhook processing failed' });
  }
}

/**
 * Handle checkout session completed
 */
async function handleCheckoutCompleted(session) {
  console.log('Processing checkout completion:', session.id);

  const metadata = session.metadata;
  const customerEmail = session.customer_email || session.customer_details?.email;

  if (metadata.type === 'gift') {
    // Gift purchase flow
    await handleGiftPurchase({
      sessionId: session.id,
      gifterEmail: customerEmail,
      gifterName: metadata.gifter_name,
      recipientEmail: metadata.recipient_email,
      giftMessage: metadata.gift_message,
      amount: session.amount_total / 100, // Convert from cents
      currency: session.currency
    });
  } else if (metadata.type === 'personal' || metadata.type === 'family') {
    // Subscription purchase
    await handleSubscriptionPurchase({
      sessionId: session.id,
      customerEmail,
      plan: metadata.type,
      subscriptionId: session.subscription
    });
  }
}

/**
 * Handle gift purchase
 */
async function handleGiftPurchase({ sessionId, gifterEmail, gifterName, recipientEmail, giftMessage, amount, currency }) {
  try {
    // Generate gift code
    const giftCode = generateGiftCode();

    // Save gift to database
    await query(
      `INSERT INTO gifts 
       (code, gifter_email, gifter_name, recipient_email, gift_message, 
        purchase_date, delivery_date, stripe_session_id) 
       VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7)`,
      [
        giftCode,
        gifterEmail,
        gifterName || null,
        recipientEmail,
        giftMessage || null,
        new Date('2025-12-25T06:00:00Z'), // Christmas morning 6am UTC
        sessionId
      ]
    );

    console.log('✓ Gift saved:', giftCode);

    // Schedule Christmas morning email to recipient
    await sendGiftRecipientEmail({
      recipientEmail,
      recipientName: null,
      gifterName,
      giftMessage,
      giftCode,
      calendarUrl: `${process.env.FRONTEND_URL}/calendar`
    });

    console.log('✓ Gift recipient email scheduled for Christmas');

    // Send immediate confirmation to gifter
    await sendGifterConfirmationEmail({
      gifterEmail,
      gifterName,
      recipientEmail,
      orderNumber: sessionId,
      amount: `$${amount} ${currency.toUpperCase()}`
    });

    console.log('✓ Gifter confirmation email sent');
  } catch (error) {
    console.error('Gift purchase handling error:', error);
    throw error;
  }
}

/**
 * Handle subscription purchase
 */
async function handleSubscriptionPurchase({ sessionId, customerEmail, plan, subscriptionId }) {
  try {
    // Check if user already exists
    const existingUser = await query(
      'SELECT * FROM users WHERE email = $1',
      [customerEmail]
    );

    if (existingUser.rows.length === 0) {
      // Create new user
      await query(
        `INSERT INTO users 
         (email, plan, subscription_status, stripe_customer_id, stripe_subscription_id) 
         VALUES ($1, $2, 'active', $3, $4)`,
        [customerEmail, plan, session.customer, subscriptionId]
      );
      console.log('✓ New user created:', customerEmail);
    } else {
      // Update existing user
      await query(
        `UPDATE users 
         SET plan = $1, subscription_status = 'active', 
             stripe_customer_id = $2, stripe_subscription_id = $3 
         WHERE email = $4`,
        [plan, session.customer, subscriptionId, customerEmail]
      );
      console.log('✓ User updated:', customerEmail);
    }
  } catch (error) {
    console.error('Subscription purchase handling error:', error);
    throw error;
  }
}

/**
 * Handle subscription created
 */
async function handleSubscriptionCreated(subscription) {
  console.log('Subscription created:', subscription.id);
  // Additional logic if needed
}

/**
 * Handle subscription deleted
 */
async function handleSubscriptionDeleted(subscription) {
  console.log('Subscription deleted:', subscription.id);
  
  try {
    // Mark subscription as canceled
    await query(
      `UPDATE users 
       SET subscription_status = 'canceled' 
       WHERE stripe_subscription_id = $1`,
      [subscription.id]
    );
    console.log('✓ User subscription status updated to canceled');
  } catch (error) {
    console.error('Subscription deletion handling error:', error);
  }
}

/**
 * Handle successful payment
 */
async function handlePaymentSucceeded(invoice) {
  console.log('Payment succeeded:', invoice.id);
  // Additional logic like sending receipt email
}

/**
 * Handle failed payment
 */
async function handlePaymentFailed(invoice) {
  console.log('Payment failed:', invoice.id);
  
  try {
    // Update subscription status
    await query(
      `UPDATE users 
       SET subscription_status = 'past_due' 
       WHERE stripe_customer_id = $1`,
      [invoice.customer]
    );
    console.log('✓ User subscription status updated to past_due');
    
    // Send payment failed email (TODO: implement)
  } catch (error) {
    console.error('Payment failure handling error:', error);
  }
}

module.exports = handleWebhook;






