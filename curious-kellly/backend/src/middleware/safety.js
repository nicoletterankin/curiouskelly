/**
 * Safety Middleware
 * Automatically moderates all user inputs and AI outputs
 */

const SafetyService = require('../services/safety');
const rateLimit = require('express-rate-limit');

// Create safety service instance (singleton)
const safetyService = new SafetyService();

// Cache for moderation results (TTL: 5 minutes)
const moderationCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Rate limiter for moderation requests
 */
const moderationRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many moderation requests, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

/**
 * Get cached moderation result
 */
function getCachedModeration(text) {
  const cacheKey = text.toLowerCase().trim();
  const cached = moderationCache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.result;
  }
  
  if (cached) {
    moderationCache.delete(cacheKey);
  }
  
  return null;
}

/**
 * Cache moderation result
 */
function cacheModeration(text, result) {
  const cacheKey = text.toLowerCase().trim();
  moderationCache.set(cacheKey, {
    result,
    timestamp: Date.now()
  });
  
  // Clean up old cache entries periodically (every 10 minutes)
  if (moderationCache.size > 1000) {
    const now = Date.now();
    for (const [key, value] of moderationCache.entries()) {
      if (now - value.timestamp > CACHE_TTL) {
        moderationCache.delete(key);
      }
    }
  }
}

/**
 * Middleware to moderate user input
 */
async function moderateInput(req, res, next) {
  try {
    // Check if there's user-provided text to moderate
    const textToCheck = req.body.text || req.body.message || req.body.input || req.body.query;
    
    if (!textToCheck || typeof textToCheck !== 'string') {
      return next();
    }

    // Check cache first
    const cached = getCachedModeration(textToCheck);
    if (cached) {
      req.safetyCheck = cached;
      req.safetyCheck.cached = true;
      return next();
    }

    // Moderate the content
    const result = await safetyService.moderateContent(textToCheck);
    
    // Cache the result
    cacheModeration(textToCheck, result);
    
    req.safetyCheck = result;
    
    // If content is unsafe, block the request
    if (!result.safe) {
      return res.status(400).json({
        status: 'error',
        message: 'Content violates safety guidelines',
        reason: result.categories.length > 0 
          ? result.categories[0].category 
          : 'Custom rule violation',
        safety: result
      });
    }
    
    next();
  } catch (error) {
    console.error('Moderation middleware error:', error);
    // Fail-safe: block the request if moderation fails
    return res.status(500).json({
      status: 'error',
      message: 'Safety check failed',
      error: error.message
    });
  }
}

/**
 * Middleware to check age-appropriate content
 */
async function checkAgeAppropriate(req, res, next) {
  try {
    const age = req.body.age || req.params.age || req.query.age;
    const text = req.body.text || req.body.message || req.body.input;
    
    if (!age || !text) {
      return next();
    }

    const ageNum = parseInt(age);
    if (isNaN(ageNum) || ageNum < 2 || ageNum > 102) {
      return next();
    }

    const result = safetyService.isAgeAppropriate(text, ageNum);
    
    if (!result.appropriate) {
      return res.status(400).json({
        status: 'error',
        message: 'Content not age-appropriate',
        reason: result.reason,
        recommendedMinAge: result.recommendedMinAge,
        ageCheck: result
      });
    }
    
    req.ageCheck = result;
    next();
  } catch (error) {
    console.error('Age check middleware error:', error);
    next();
  }
}

/**
 * Middleware to moderate AI output before sending to client
 */
async function moderateOutput(text, age) {
  try {
    if (!text || typeof text !== 'string') {
      return { safe: true, text };
    }

    // Check cache
    const cached = getCachedModeration(text);
    if (cached) {
      if (!cached.safe) {
        // Try safe rewrite
        const rewrite = await safetyService.safeCompletion(text);
        return {
          safe: false,
          original: text,
          safeVersion: rewrite.safeVersion,
          rewritten: true
        };
      }
      return { safe: true, text, cached: true };
    }

    // Moderate the output
    const result = await safetyService.moderateContent(text);
    cacheModeration(text, result);

    if (!result.safe) {
      // Try to rewrite to make it safe
      const rewrite = await safetyService.safeCompletion(text);
      return {
        safe: false,
        original: text,
        safeVersion: rewrite.safeVersion,
        rewritten: true,
        moderation: result
      };
    }

    // Check age-appropriateness if age provided
    if (age) {
      const ageCheck = safetyService.isAgeAppropriate(text, age);
      if (!ageCheck.appropriate) {
        const rewrite = await safetyService.safeCompletion(text);
        return {
          safe: false,
          original: text,
          safeVersion: rewrite.safeVersion,
          rewritten: true,
          ageCheck
        };
      }
    }

    return { safe: true, text };
  } catch (error) {
    console.error('Output moderation error:', error);
    // Fail-safe: return safe default message
    return {
      safe: false,
      original: text,
      safeVersion: "I can't help with that topic. Let's talk about something else!",
      rewritten: true,
      error: error.message
    };
  }
}

/**
 * Log safety violations for monitoring
 */
function logViolation(req, result, type = 'input') {
  const logEntry = {
    timestamp: new Date().toISOString(),
    type,
    ip: req.ip || req.connection.remoteAddress,
    userAgent: req.get('user-agent'),
    violation: {
      categories: result.categories,
      customViolations: result.customViolations,
      flagged: result.flagged
    },
    sessionId: req.body.sessionId || req.params.sessionId
  };

  // In production, send to logging service (e.g., Sentry, DataDog)
  console.warn('🚨 Safety Violation:', JSON.stringify(logEntry, null, 2));
  
  // TODO: Send to analytics service
}

module.exports = {
  moderateInput,
  checkAgeAppropriate,
  moderateOutput,
  moderationRateLimit,
  safetyService,
  logViolation
};













