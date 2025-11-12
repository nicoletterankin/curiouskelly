/**
 * Safety API Routes
 * Content moderation and safety checks
 */

const express = require('express');
const SafetyService = require('../services/safety');

const router = express.Router();

/**
 * Moderate text content
 * POST /api/safety/moderate
 * Body: { text: "content to check" }
 */
router.post('/moderate', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: text'
      });
    }

    const service = new SafetyService();
    const result = await service.moderateContent(text);

    res.json({
      status: 'ok',
      data: result
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Check if content is age-appropriate
 * POST /api/safety/age-check
 * Body: { text: "content", age: 35 }
 */
router.post('/age-check', async (req, res) => {
  try {
    const { text, age } = req.body;

    if (!text || !age) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: text, age'
      });
    }

    if (age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be between 2 and 102'
      });
    }

    const service = new SafetyService();
    const result = service.isAgeAppropriate(text, age);

    res.json({
      status: 'ok',
      data: result
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Safe-completion: Rewrite unsafe content
 * POST /api/safety/safe-rewrite
 * Body: { text: "potentially unsafe content" }
 */
router.post('/safe-rewrite', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: text'
      });
    }

    const service = new SafetyService();
    const result = await service.safeCompletion(text);

    res.json({
      status: 'ok',
      data: result
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Full safety check (moderation + age-check)
 * POST /api/safety/check
 * Body: { text: "content", age: 35 }
 */
router.post('/check', async (req, res) => {
  try {
    const { text, age } = req.body;

    if (!text || !age) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: text, age'
      });
    }

    const service = new SafetyService();

    // Run both checks
    const [moderation, ageCheck] = await Promise.all([
      service.moderateContent(text),
      Promise.resolve(service.isAgeAppropriate(text, age))
    ]);

    const overallSafe = moderation.safe && ageCheck.appropriate;

    res.json({
      status: 'ok',
      data: {
        safe: overallSafe,
        moderation,
        ageCheck,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;














