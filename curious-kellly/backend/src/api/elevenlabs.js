/**
 * ElevenLabs Voice API Routes
 * Week 3-4: Real-time voice synthesis endpoints
 */

const express = require('express');
const ElevenLabsVoiceService = require('../services/elevenlabs_voice');
const LessonService = require('../services/lessons');

const router = express.Router();

/**
 * Test ElevenLabs connection
 * GET /api/elevenlabs/test
 */
router.get('/test', async (req, res) => {
  try {
    const voiceService = new ElevenLabsVoiceService();
    const result = await voiceService.testConnection();

    res.json({
      status: result.success ? 'ok' : 'error',
      message: result.message,
      voiceId: voiceService.voiceId,
      modelId: voiceService.modelId
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Generate speech from text
 * POST /api/elevenlabs/speak
 * Body: { text: "Hello", kellyAge: 27, language: "en" }
 */
router.post('/speak', async (req, res) => {
  try {
    const { text, kellyAge = 27, language = 'en' } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: text'
      });
    }

    const voiceService = new ElevenLabsVoiceService();
    const audioBuffer = await voiceService.generateSpeech(text, kellyAge, language);

    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioBuffer.length);
    res.send(audioBuffer);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Generate speech for lesson content
 * POST /api/elevenlabs/lesson-speak
 * Body: { lessonId: "the-sun", age: 8, section: "welcome", language: "en" }
 */
router.post('/lesson-speak', async (req, res) => {
  try {
    const { lessonId, age, section = 'welcome', language = 'en' } = req.body;

    if (!lessonId || !age) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required fields: lessonId, age'
      });
    }

    // Get lesson content
    const lessonService = new LessonService();
    const lesson = await lessonService.getLessonForAge(lessonId, age, language);

    const text = lesson.localized.content[section];
    if (!text) {
      return res.status(404).json({
        status: 'error',
        message: `Section '${section}' not found in lesson`
      });
    }

    const kellyAge = lesson.content.kellyAge;

    // Generate speech
    const voiceService = new ElevenLabsVoiceService();
    const audioBuffer = await voiceService.generateSpeech(text, kellyAge, language);

    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioBuffer.length);
    res.setHeader('X-Kelly-Age', kellyAge);
    res.setHeader('X-Lesson-Id', lessonId);
    res.setHeader('X-Section', section);
    res.send(audioBuffer);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Stream speech (for realtime applications)
 * POST /api/elevenlabs/stream
 * Body: { text: "Hello", kellyAge: 27 }
 */
router.post('/stream', async (req, res) => {
  try {
    const { text, kellyAge = 27 } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: text'
      });
    }

    const voiceService = new ElevenLabsVoiceService();
    const stream = await voiceService.streamSpeech(text, kellyAge);

    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Transfer-Encoding', 'chunked');

    stream.pipe(res);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;
