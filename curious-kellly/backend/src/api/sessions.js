/**
 * Sessions API Routes
 * Manage user lesson sessions
 */

const express = require('express');
const SessionService = require('../services/session');
const LessonService = require('../services/lessons');

const router = express.Router();
const sessionService = new SessionService();

/**
 * Start a new session
 * POST /api/sessions/start
 * Body: { age: 35, lessonId: "leaves-change-color" }
 */
router.post('/start', async (req, res) => {
  try {
    const { age, lessonId, userId } = req.body;

    if (!age) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: age'
      });
    }

    if (age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be between 2 and 102'
      });
    }

    // If no lessonId provided, use today's lesson
    let finalLessonId = lessonId;
    if (!finalLessonId) {
      const lessonService = new LessonService();
      const todaysLesson = await lessonService.getTodaysLesson();
      finalLessonId = todaysLesson.id;
    }

    const session = await sessionService.createSession(age, finalLessonId, userId);

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
 * Get all active sessions (monitoring)
 * GET /api/sessions/active
 * NOTE: Must come before /:sessionId to avoid route conflict
 */
router.get('/active', async (req, res) => {
  try {
    const active = await sessionService.getActiveSessions();

    res.json({
      status: 'ok',
      data: {
        activeSessions: active,
        count: active.length,
        storage: sessionService.useRedis ? 'redis' : 'memory'
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
 * Get session history for a user
 * GET /api/sessions/history/:userId
 */
router.get('/history/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const limit = parseInt(req.query.limit) || 50;
    const history = await sessionService.getSessionHistory(userId, limit);

    res.json({
      status: 'ok',
      data: {
        history,
        count: history.length
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
 * Get session status
 * GET /api/sessions/:sessionId
 */
router.get('/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await sessionService.getSession(sessionId);

    res.json({
      status: 'ok',
      data: session
    });
  } catch (error) {
    const statusCode = error.message.includes('not found') || error.message.includes('expired') ? 404 : 500;
    res.status(statusCode).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Update session progress
 * POST /api/sessions/:sessionId/progress
 * Body: { currentPhase: "teaching", completedPhase: "welcome", ... }
 */
router.post('/:sessionId/progress', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const updates = req.body;

    const session = await sessionService.updateProgress(sessionId, updates);

    res.json({
      status: 'ok',
      data: session
    });
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Complete session
 * POST /api/sessions/:sessionId/complete
 */
router.post('/:sessionId/complete', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await sessionService.completeSession(sessionId);

    res.json({
      status: 'ok',
      data: session,
      message: 'Lesson completed! Great job!'
    });
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Pause/Resume session
 * POST /api/sessions/:sessionId/toggle-pause
 */
router.post('/:sessionId/toggle-pause', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await sessionService.togglePause(sessionId);

    res.json({
      status: 'ok',
      data: session,
      message: session.state.isPaused ? 'Session paused' : 'Session resumed'
    });
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get session statistics
 * GET /api/sessions/:sessionId/stats
 */
router.get('/:sessionId/stats', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const stats = await sessionService.getSessionStats(sessionId);

    res.json({
      status: 'ok',
      data: stats
    });
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Cleanup expired sessions
 * POST /api/sessions/cleanup
 */
router.post('/cleanup', async (req, res) => {
  try {
    const result = await sessionService.cleanupExpiredSessions();

    res.json({
      status: 'ok',
      data: result,
      message: `Cleaned ${result.cleaned} expired session(s)`
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;



