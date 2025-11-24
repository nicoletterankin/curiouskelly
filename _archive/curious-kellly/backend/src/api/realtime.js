/**
 * Realtime API Routes
 * Endpoints for OpenAI integration
 */

const express = require('express');
const RealtimeService = require('../services/realtime');

const { logVoiceLatency } = require('../utils/voiceMetricsLogger');
const SafetyService = require('../services/safety');

const router = express.Router();

/**
 * Test OpenAI connection
 * GET /api/realtime/test
 */
router.get('/test', async (req, res) => {
  try {
    const service = new RealtimeService();
    const result = await service.testConnection();
    
    if (result.success) {
      res.json({
        status: 'ok',
        message: 'OpenAI connection successful',
        data: result
      });
    } else {
      res.status(500).json({
        status: 'error',
        message: 'OpenAI connection failed',
        error: result.error
      });
    }
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get Kelly's response
 * POST /api/realtime/kelly
 * Body: { age: 35, topic: "leaves", message: "Why do leaves change color?" }
 */
router.post('/kelly', async (req, res) => {
  let learnerAge;
  let topicName;
  let sessionId;
  let requestStartedAt;
  let requestStartedAtIso;

  try {
    const { age, topic, message } = req.body;
    sessionId = req.body.sessionId;
    learnerAge = age;
    topicName = topic;

    // Validation
    if (!learnerAge || !topicName || !message) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: age, topic, message'
      });
    }

    if (learnerAge < 2 || learnerAge > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be between 2 and 102'
      });
    }

    // SAFETY CHECK: Moderate user input first
    const safetyService = new SafetyService();
    const inputModeration = await safetyService.moderateContent(message);
    
    if (!inputModeration.safe) {
      return res.status(400).json({
        status: 'error',
        message: 'Content blocked by safety filter',
        reason: 'Input contains inappropriate content',
        moderation: {
          flagged: inputModeration.flagged,
          categories: inputModeration.categories
        }
      });
    }

    // Check age-appropriateness
    const ageCheck = safetyService.isAgeAppropriate(message, learnerAge);
    if (!ageCheck.appropriate) {
      return res.status(400).json({
        status: 'error',
        message: 'Content not age-appropriate',
        reason: ageCheck.reason,
        recommendedMinAge: ageCheck.recommendedMinAge
      });
    }

    // Get Kelly's response
    const realtimeService = new RealtimeService();
    requestStartedAt = new Date();
    requestStartedAtIso = requestStartedAt.toISOString();
    const result = await realtimeService.getKellyResponse(learnerAge, topicName, message);

    if (result.success) {
      const responseSentAt = new Date();
      const latencyMs = responseSentAt.getTime() - requestStartedAt.getTime();

      await logVoiceLatency({
        sessionId,
        source: 'realtime_kelly',
        topic: topicName,
        learnerAge,
        ageBucket: getAgeBucket(learnerAge),
        kellyAge: result.kellyAge,
        kellyPersona: result.kellyPersona,
        requestStartedAt: requestStartedAtIso,
        responseSentAt: responseSentAt.toISOString(),
        latencyMs,
        status: 'ok'
      });

      // SAFETY CHECK: Moderate Kelly's output
      const outputModeration = await safetyService.moderateContent(result.response);
      
      if (!outputModeration.safe) {
        // Use safe-completion to rewrite
        const safeVersion = await safetyService.safeCompletion(result.response);
        result.response = safeVersion.safeVersion;
        result.safetyRewritten = true;
      }

      res.json({
        status: 'ok',
        data: result,
        safety: {
          inputChecked: true,
          outputChecked: true,
          rewritten: result.safetyRewritten || false
        }
      });
    } else {
      const responseSentAt = new Date();
      await logVoiceLatency({
        sessionId,
        source: 'realtime_kelly',
        topic: topicName,
        learnerAge,
        ageBucket: getAgeBucket(learnerAge),
        kellyAge: result.kellyAge,
        kellyPersona: result.kellyPersona,
        requestStartedAt: requestStartedAtIso,
        responseSentAt: responseSentAt.toISOString(),
        latencyMs: responseSentAt.getTime() - requestStartedAt.getTime(),
        status: 'error'
      });
      res.status(500).json({
        status: 'error',
        message: 'Failed to get Kelly response',
        error: result.error
      });
    }
  } catch (error) {
    if (learnerAge && topicName && requestStartedAt) {
      try {
        const responseSentAt = new Date();
        await logVoiceLatency({
          sessionId,
          source: 'realtime_kelly',
          topic: topicName,
          learnerAge,
          ageBucket: getAgeBucket(learnerAge),
          requestStartedAt: requestStartedAtIso,
          responseSentAt: responseSentAt.toISOString(),
          latencyMs: responseSentAt.getTime() - requestStartedAt.getTime(),
          status: 'error'
        });
      } catch (logError) {
        console.error('[Realtime API] Failed to log latency error', logError);
      }
    }
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get ephemeral API key for Realtime API
 * POST /api/realtime/ephemeral-key
 * Body: { learnerAge: 35, sessionId?: "optional-session-id" }
 */
router.post('/ephemeral-key', async (req, res) => {
  try {
    const { learnerAge, sessionId } = req.body;

    if (!learnerAge || learnerAge < 2 || learnerAge > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Invalid learnerAge (must be 2-102)',
      });
    }

    const realtimeService = new RealtimeService();
    const result = await realtimeService.createEphemeralKey(learnerAge, sessionId);

    if (result.success) {
      res.json({
        status: 'ok',
        data: result.ephemeralKey,
      });
    } else {
      res.status(500).json({
        status: 'error',
        message: 'Failed to create ephemeral key',
        error: result.error,
      });
    }
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message,
    });
  }
});

module.exports = router;

function getAgeBucket(age) {
  if (age >= 2 && age <= 5) return '2-5';
  if (age >= 6 && age <= 12) return '6-12';
  if (age >= 13 && age <= 17) return '13-17';
  if (age >= 18 && age <= 35) return '18-35';
  if (age >= 36 && age <= 60) return '36-60';
  if (age >= 61 && age <= 102) return '61-102';
  return 'unknown';
}

