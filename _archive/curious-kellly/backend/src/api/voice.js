/**
 * Voice API Routes
 * Realtime voice conversation endpoints
 */

const express = require('express');
const VoiceService = require('../services/voice');
const SafetyService = require('../services/safety');
const LessonService = require('../services/lessons');

const router = express.Router();

/**
 * Generate speech via ElevenLabs and return audio binary
 * POST /api/voice/tts
 * Body: { age: 35, text: "Hello" }
 */
router.post('/tts', async (req, res) => {
  try {
    const { age, text } = req.body;

    if (!age || !text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: age, text'
      });
    }

    const lessonService = new LessonService();
    const kellyAge = lessonService.getKellyAge(age);

    const voiceService = new VoiceService();
    const result = await voiceService.generateSpeech(text, kellyAge);

    res.setHeader('Content-Type', result.mimeType);
    res.send(result.audio);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Start a realtime voice session
 * POST /api/voice/session/start
 * Body: { age: 35, topic: "leaves", sessionId: "abc-123" }
 */
router.post('/session/start', async (req, res) => {
  try {
    const { age, topic, sessionId } = req.body;

    if (!age || !topic) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: age, topic'
      });
    }

    if (age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be between 2 and 102'
      });
    }

    // Get Kelly's age and persona
    const lessonService = new LessonService();
    const kellyAge = lessonService.getKellyAge(age);
    const kellyPersona = lessonService.getKellyPersona(age);

    // Start voice session
    const voiceService = new VoiceService();
    const session = await voiceService.startRealtimeSession({
      age,
      topic,
      kellyAge,
      kellyPersona,
      sessionId: sessionId || `voice-${Date.now()}`
    });

    res.json({
      status: 'ok',
      data: session
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Send text message to Kelly (fallback until Realtime API)
 * POST /api/voice/message
 * Body: { age: 35, topic: "leaves", message: "Why do leaves change color?" }
 */
router.post('/message', async (req, res) => {
  try {
    const { age, topic, message, sessionId } = req.body;

    if (!age || !topic || !message) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: age, topic, message'
      });
    }

    // Safety check
    const safetyService = new SafetyService();
    const moderation = await safetyService.moderateContent(message);
    
    if (!moderation.safe) {
      return res.status(400).json({
        status: 'error',
        message: 'Content blocked by safety filter',
        moderation: {
          flagged: moderation.flagged,
          categories: moderation.categories
        }
      });
    }

    // Get Kelly's configuration
    const lessonService = new LessonService();
    const kellyAge = lessonService.getKellyAge(age);
    const kellyPersona = lessonService.getKellyPersona(age);

    // Get response
    const voiceService = new VoiceService();
    const result = await voiceService.getTextResponse({
      age,
      topic,
      kellyAge,
      kellyPersona
    }, message, {
      sessionId,
      source: 'voice_message'
    });

    if (!result.success) {
      return res.status(500).json({
        status: 'error',
        message: 'Failed to get Kelly response',
        error: result.error
      });
    }

    // Safety check Kelly's output
    const outputModeration = await safetyService.moderateContent(result.response);
    
    if (!outputModeration.safe) {
      // Use safe-completion
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
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Measure voice latency
 * GET /api/voice/latency/:age
 */
router.get('/latency/:age', async (req, res) => {
  try {
    const age = parseInt(req.params.age);

    if (isNaN(age) || age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be a number between 2 and 102'
      });
    }

    const lessonService = new LessonService();
    const kellyAge = lessonService.getKellyAge(age);
    const kellyPersona = lessonService.getKellyPersona(age);

    const voiceService = new VoiceService();
    const latency = await voiceService.measureLatency({
      age,
      topic: 'test',
      kellyAge,
      kellyPersona
    });

    res.json({
      status: 'ok',
      data: latency
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get voice configuration for Kelly age
 * GET /api/voice/config/:age
 */
router.get('/config/:age', async (req, res) => {
  try {
    const age = parseInt(req.params.age);

    if (isNaN(age) || age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be a number between 2 and 102'
      });
    }

    const lessonService = new LessonService();
    const kellyAge = lessonService.getKellyAge(age);
    const kellyPersona = lessonService.getKellyPersona(age);

    const voiceService = new VoiceService();
    const voiceConfig = voiceService.getVoiceConfig(kellyAge);

    res.json({
      status: 'ok',
      data: {
        learnerAge: age,
        kellyAge,
        kellyPersona,
        voiceConfig
      }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Handle barge-in (interruption)
 * POST /api/voice/interrupt/:sessionId
 */
router.post('/interrupt/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;

    const voiceService = new VoiceService();
    const result = await voiceService.handleInterruption(sessionId);

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
 * WebRTC connection endpoint (future)
 * POST /api/voice/realtime/connect
 */
router.post('/realtime/connect', async (req, res) => {
  try {
    const { sessionId, offer } = req.body;

    if (!sessionId) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing sessionId'
      });
    }

    const voiceService = new VoiceService();
    const result = await voiceService.handleWebRTCConnection(sessionId, offer);

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

module.exports = router;


