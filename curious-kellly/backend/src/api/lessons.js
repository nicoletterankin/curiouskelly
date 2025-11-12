/**
 * Lessons API Routes
 * Endpoints for daily lessons and content
 */

const express = require('express');
const LessonService = require('../services/lessons');
const SessionService = require('../services/session');

const router = express.Router();

/**
 * Get today's lesson (universal, no age specified)
 * GET /api/lessons/today
 */
router.get('/today', async (req, res) => {
  try {
    const service = new LessonService();
    const lesson = await service.getTodaysLesson();
    const localization = await service.getLocalizationBundle(lesson.id);

    res.json({
      status: 'ok',
      data: {
        id: lesson.id,
        title: lesson.title,
        description: lesson.description,
        metadata: lesson.metadata,
        dayOfYear: service.getDayOfYear(),
        availableAges: ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'],
        supportedLocales: localization.supportedLanguages || []
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
 * Get today's lesson for specific age
 * GET /api/lessons/today/:age
 */
router.get('/today/:age', async (req, res) => {
  try {
    const age = parseInt(req.params.age);
    const locale = (req.query.locale || 'en').toLowerCase().split('-')[0];

    if (isNaN(age) || age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be a number between 2 and 102'
      });
    }

    const service = new LessonService();
    const lesson = await service.getTodaysLessonForAge(age, locale);

    res.json({
      status: 'ok',
      data: lesson
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get specific lesson by ID
 * GET /api/lessons/:id
 */
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const includeLocales = req.query.includeLocales === 'true';
    const service = new LessonService();
    const lesson = await service.loadLesson(id);
    const localization = await service.getLocalizationBundle(id);

    const response = {
      status: 'ok',
      data: {
        lesson,
        supportedLocales: localization ? localization.supportedLanguages : []
      }
    };

    if (includeLocales && localization) {
      response.data.locales = localization.locales;
    }

    res.json(response);
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get specific lesson for specific age
 * GET /api/lessons/:id/age/:age
 */
router.get('/:id/age/:age', async (req, res) => {
  try {
    const { id } = req.params;
    const age = parseInt(req.params.age);
    const locale = (req.query.locale || 'en').toLowerCase().split('-')[0];

    if (isNaN(age) || age < 2 || age > 102) {
      return res.status(400).json({
        status: 'error',
        message: 'Age must be a number between 2 and 102'
      });
    }

    const service = new LessonService();
    const lesson = await service.getLessonForAge(id, age, locale);

    res.json({
      status: 'ok',
      data: lesson
    });
  } catch (error) {
    res.status(404).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get all available lessons
 * GET /api/lessons
 */
router.get('/', async (req, res) => {
  try {
    const service = new LessonService();
    const lessons = await service.getAllLessons();

    res.json({
      status: 'ok',
      data: {
        lessons,
        total: lessons.length,
        target: 365
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
 * Get lesson statistics
 * GET /api/lessons/stats
 */
router.get('/api/stats', async (req, res) => {
  try {
    const service = new LessonService();
    const stats = await service.getStats();

    res.json({
      status: 'ok',
      data: stats
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Validate lesson JSON
 * POST /api/lessons/validate
 * Body: { lesson: {...} }
 */
router.post('/validate', async (req, res) => {
  try {
    const { lesson } = req.body;

    if (!lesson) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing lesson data'
      });
    }

    const service = new LessonService();
    const validation = await service.validateLesson(lesson);

    res.json({
      status: 'ok',
      data: validation
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;











