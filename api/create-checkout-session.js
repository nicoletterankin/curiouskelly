// Vercel Serverless Function for Stripe Checkout
// POST /api/create-checkout-session

const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

module.exports = async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { priceId, userId, email } = req.body;

    if (!priceId || !userId) {
      return res.status(400).json({ error: 'Missing priceId or userId' });
    }

    // Create Stripe checkout session
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      payment_method_types: ['card'],
      line_items: [
        {
          price: priceId,
          quantity: 1
        }
      ],
      success_url: `${process.env.NEXT_PUBLIC_APP_URL || 'https://curiouskelly.com'}/hub.html?session_id={CHECKOUT_SESSION_ID}&success=true`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || 'https://curiouskelly.com'}/pricing.html?canceled=true`,
      customer_email: email,
      client_reference_id: userId,
      metadata: {
        userId: userId
      },
      subscription_data: {
        metadata: {
          userId: userId
        },
        trial_period_days: 7 // 7-day free trial
      }
    });

    return res.status(200).json({
      url: session.url,
      sessionId: session.id
    });
  } catch (error) {
    console.error('Stripe checkout error:', error);
    return res.status(500).json({
      error: error.message || 'Failed to create checkout session'
    });
  }
};
