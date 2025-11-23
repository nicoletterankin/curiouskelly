const express = require('express');
const router = express.Router();
const { 
  createGiftCheckoutSession, 
  createPersonalCheckoutSession, 
  createFamilyCheckoutSession 
} = require('../services/stripe');

/**
 * POST /api/checkout/create-session
 * Create Stripe checkout session
 */
router.post('/create-session', async (req, res) => {
  try {
    const { plan, customerEmail, recipientEmail, giftMessage, gifterName } = req.body;

    // Validate required fields
    if (!plan || !customerEmail) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    let session;

    switch (plan.toLowerCase()) {
      case 'gift':
        if (!recipientEmail) {
          return res.status(400).json({ error: 'Recipient email required for gift purchase' });
        }
        session = await createGiftCheckoutSession({
          customerEmail,
          recipientEmail,
          giftMessage,
          gifterName
        });
        break;

      case 'personal':
        session = await createPersonalCheckoutSession({ customerEmail });
        break;

      case 'family':
        session = await createFamilyCheckoutSession({ customerEmail });
        break;

      default:
        return res.status(400).json({ error: 'Invalid plan type' });
    }

    res.json({ sessionId: session.id, url: session.url });
  } catch (error) {
    console.error('Checkout error:', error);
    res.status(500).json({ error: 'Failed to create checkout session' });
  }
});

module.exports = router;






