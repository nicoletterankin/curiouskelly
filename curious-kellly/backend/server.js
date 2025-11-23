require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { moderateInput, checkAgeAppropriate } = require('./src/middleware/safety');

// Import routes
const checkoutRoutes = require('./src/api/checkout');
const giftsRoutes = require('./src/api/gifts');
const usersRoutes = require('./src/api/users');
const lessonsRoutes = require('./src/api/lessons');

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());

// CORS configuration
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:8000',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Body parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Global Safety Middleware (P0 Requirement)
app.use('/api', moderateInput);
app.use('/api', checkAgeAppropriate);

// Stripe webhook needs raw body
app.use('/webhook', express.raw({ type: 'application/json' }));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API routes
app.use('/api/checkout', checkoutRoutes);
app.use('/api/gifts', giftsRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/lessons', lessonsRoutes);

// Stripe webhook (separate from other routes due to raw body requirement)
app.post('/webhook', require('./src/api/webhook'));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  
  // Don't leak error details in production
  const isProduction = process.env.NODE_ENV === 'production';
  
  res.status(err.status || 500).json({
    error: isProduction ? 'Internal server error' : err.message,
    ...(isProduction ? {} : { stack: err.stack })
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Curious Kelly Backend running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`Frontend URL: ${process.env.FRONTEND_URL}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

module.exports = app;



