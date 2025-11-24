/**
 * Curious Kellly Backend - Main Server
 * The Daily Lesson API
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const expressWs = require('express-ws');
require('dotenv').config();

const app = express();

// Enable WebSocket support
expressWs(app);

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Request logging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'curious-kellly-backend',
    timestamp: new Date().toISOString(),
    version: '0.1.0',
    environment: process.env.NODE_ENV
  });
});

// Import routes
const realtimeRoutes = require('./api/realtime');
const realtimeWsRoutes = require('./api/realtime_ws');
const safetyRoutes = require('./api/safety');
const lessonsRoutes = require('./api/lessons');
const sessionsRoutes = require('./api/sessions');
const voiceRoutes = require('./api/voice');
const ragRoutes = require('./api/rag');
const reinmakerManifestRoute = require('./api/reinmaker/manifest.route');
const reinmakerQuestsRoute = require('./api/reinmaker/quests.route');

// Import safety middleware
const { moderateInput, checkAgeAppropriate, moderationRateLimit } = require('./middleware/safety');

// Apply safety middleware to user input endpoints
app.use('/api/realtime/kelly', moderationRateLimit, moderateInput);
app.use('/api/voice', moderationRateLimit, moderateInput);

// Mount routes
app.use('/api/realtime', realtimeRoutes);
app.use('/api/realtime', realtimeWsRoutes); // WebSocket routes
app.use('/api/safety', safetyRoutes);
app.use('/api/lessons', lessonsRoutes);
app.use('/api/sessions', sessionsRoutes);
app.use('/api/voice', voiceRoutes);
app.use('/api/rag', ragRoutes);
app.use('/api/reinmaker/manifest', reinmakerManifestRoute);
app.use('/api/reinmaker/quests', reinmakerQuestsRoute);

// API info endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Curious Kellly API',
    description: 'The Daily Lesson - Backend Service',
    version: '0.1.0',
    endpoints: {
      health: '/health',
      realtime: {
        testOpenAI: '/api/realtime/test',
        getKellyResponse: '/api/realtime/kelly (POST)'
      },
      safety: {
        moderate: '/api/safety/moderate (POST)',
        ageCheck: '/api/safety/age-check (POST)',
        fullCheck: '/api/safety/check (POST)',
        safeRewrite: '/api/safety/safe-rewrite (POST)'
      },
      lessons: {
        today: '/api/lessons/today',
        todayForAge: '/api/lessons/today/:age',
        byId: '/api/lessons/:id',
        byIdForAge: '/api/lessons/:id/age/:age',
        all: '/api/lessons',
        validate: '/api/lessons/validate (POST)'
      },
      sessions: {
        start: '/api/sessions/start (POST)',
        get: '/api/sessions/:sessionId',
        updateProgress: '/api/sessions/:sessionId/progress (POST)',
        complete: '/api/sessions/:sessionId/complete (POST)',
        stats: '/api/sessions/:sessionId/stats'
      },
      voice: {
        startSession: '/api/voice/session/start (POST)',
        message: '/api/voice/message (POST)',
        latency: '/api/voice/latency/:age',
        config: '/api/voice/config/:age',
        interrupt: '/api/voice/interrupt/:sessionId (POST)',
        realtimeConnect: '/api/voice/realtime/connect (POST)'
      },
      reinmaker: {
        manifest: '/api/reinmaker/manifest',
        quest: '/api/reinmaker/quests/:id'
      },
      rag: {
        search: '/api/rag/search (POST)',
        context: '/api/rag/context (POST)',
        populate: '/api/rag/populate (POST)',
        status: '/api/rag/status'
      }
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log('🚀 Curious Kellly Backend Started');
  console.log(`📍 Server running on http://localhost:${PORT}`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV}`);
  console.log(`⏰ Started at: ${new Date().toISOString()}`);
});

