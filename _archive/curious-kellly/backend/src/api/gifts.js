const express = require('express');
const router = express.Router();
const { query } = require('../services/database');
const { generateGiftCode, isValidGiftCodeFormat } = require('../services/gift-codes');
const { sendGiftRecipientEmail } = require('../services/email');

/**
 * POST /api/gifts/create
 * Create gift record (called from webhook after successful payment)
 */
router.post('/create', async (req, res) => {
  try {
    const { 
      gifterEmail, 
      gifterName, 
      recipientEmail, 
      giftMessage, 
      stripeSessionId 
    } = req.body;

    // Validate required fields
    if (!gifterEmail || !recipientEmail || !stripeSessionId) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Generate unique gift code
    const giftCode = generateGiftCode();

    // Insert gift record
    const result = await query(
      `INSERT INTO gifts 
       (code, gifter_email, gifter_name, recipient_email, gift_message, 
        purchase_date, delivery_date, stripe_session_id) 
       VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7) 
       RETURNING *`,
      [
        giftCode,
        gifterEmail,
        gifterName || null,
        recipientEmail,
        giftMessage || null,
        new Date('2025-12-25T06:00:00Z'), // Christmas morning
        stripeSessionId
      ]
    );

    const gift = result.rows[0];

    // Schedule Christmas morning email
    await sendGiftRecipientEmail({
      recipientEmail,
      recipientName: null, // Will extract from email
      gifterName,
      giftMessage,
      giftCode,
      calendarUrl: `${process.env.FRONTEND_URL}/calendar`
    });

    res.json({
      success: true,
      giftCode,
      deliveryDate: gift.delivery_date
    });
  } catch (error) {
    console.error('Gift creation error:', error);
    res.status(500).json({ error: 'Failed to create gift' });
  }
});

/**
 * GET /api/gifts/verify/:code
 * Verify gift code validity
 */
router.get('/verify/:code', async (req, res) => {
  try {
    const { code } = req.params;

    // Validate format
    if (!isValidGiftCodeFormat(code)) {
      return res.status(400).json({ error: 'Invalid gift code format' });
    }

    // Check if gift exists and is not redeemed
    const result = await query(
      'SELECT * FROM gifts WHERE code = $1',
      [code]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Gift code not found' });
    }

    const gift = result.rows[0];

    if (gift.redeemed) {
      return res.status(400).json({ 
        error: 'Gift code already redeemed',
        redeemedAt: gift.redeemed_at
      });
    }

    res.json({
      valid: true,
      gift: {
        code: gift.code,
        gifterName: gift.gifter_name,
        giftMessage: gift.gift_message,
        purchaseDate: gift.purchase_date
      }
    });
  } catch (error) {
    console.error('Gift verification error:', error);
    res.status(500).json({ error: 'Failed to verify gift code' });
  }
});

/**
 * POST /api/gifts/redeem
 * Redeem gift code and create user account
 */
router.post('/redeem', async (req, res) => {
  try {
    const { giftCode, userEmail, userName } = req.body;

    // Validate required fields
    if (!giftCode || !userEmail) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Verify gift code
    const giftResult = await query(
      'SELECT * FROM gifts WHERE code = $1 AND redeemed = FALSE',
      [giftCode]
    );

    if (giftResult.rows.length === 0) {
      return res.status(400).json({ error: 'Invalid or already redeemed gift code' });
    }

    const gift = giftResult.rows[0];

    // Create user account
    const userResult = await query(
      `INSERT INTO users 
       (email, name, plan, subscription_status, gift_code_used) 
       VALUES ($1, $2, 'gift', 'active', $3) 
       RETURNING *`,
      [userEmail, userName || null, giftCode]
    );

    const user = userResult.rows[0];

    // Mark gift as redeemed
    await query(
      'UPDATE gifts SET redeemed = TRUE, redeemed_at = NOW(), redeemed_by_user_id = $1 WHERE code = $2',
      [user.id, giftCode]
    );

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        plan: user.plan
      },
      gift: {
        gifterName: gift.gifter_name,
        giftMessage: gift.gift_message
      }
    });
  } catch (error) {
    console.error('Gift redemption error:', error);
    res.status(500).json({ error: 'Failed to redeem gift' });
  }
});

module.exports = router;






