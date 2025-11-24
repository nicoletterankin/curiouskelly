/**
 * Session Service
 * Manages user sessions and lesson progress
 * Supports in-memory (default) and Redis (if configured)
 */

const { v4: uuidv4 } = require('uuid');

class SessionService {
  constructor() {
    // In-memory storage (fallback)
    this.sessions = new Map();
    this.SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
    
    // Initialize Redis if configured
    this.redis = null;
    this.useRedis = false;
    this.initializeRedis();
    
    // Start cleanup interval
    this.startCleanupInterval();
  }

  /**
   * Initialize Redis connection if available
   */
  initializeRedis() {
    if (process.env.REDIS_URL || process.env.REDIS_HOST) {
      try {
        const redis = require('redis');
        
        const redisConfig = process.env.REDIS_URL 
          ? { url: process.env.REDIS_URL }
          : {
              host: process.env.REDIS_HOST || 'localhost',
              port: process.env.REDIS_PORT || 6379,
              password: process.env.REDIS_PASSWORD
            };

        this.redis = redis.createClient(redisConfig);
        
        this.redis.on('error', (err) => {
          console.error('Redis connection error:', err);
          this.useRedis = false;
        });

        this.redis.on('connect', () => {
          console.log('✅ Redis connected');
          this.useRedis = true;
        });

        this.redis.connect().catch(err => {
          console.warn('⚠️  Redis connection failed, using in-memory storage:', err.message);
          this.useRedis = false;
        });
      } catch (error) {
        console.warn('⚠️  Redis not available, using in-memory storage:', error.message);
        this.useRedis = false;
      }
    }
  }

  /**
   * Start periodic cleanup of expired sessions
   */
  startCleanupInterval() {
    // Run cleanup every 5 minutes
    setInterval(() => {
      this.cleanupExpiredSessions();
    }, 5 * 60 * 1000);
  }

  /**
   * Create a new session
   */
  async createSession(age, lessonId, userId = null) {
    const sessionId = uuidv4();
    const session = {
      sessionId,
      userId,
      age,
      lessonId,
      startedAt: new Date().toISOString(),
      lastActivity: Date.now(),
      progress: {
        currentPhase: 'welcome',
        completedPhases: [],
        interactionsCompleted: [],
        teachingMomentsViewed: []
      },
      state: {
        isActive: true,
        isPaused: false,
        isCompleted: false
      }
    };

    if (this.useRedis && this.redis) {
      await this.saveSessionToRedis(sessionId, session);
    } else {
      this.sessions.set(sessionId, session);
    }
    
    return session;
  }

  /**
   * Save session to Redis
   */
  async saveSessionToRedis(sessionId, session) {
    try {
      const key = `session:${sessionId}`;
      await this.redis.setEx(key, this.SESSION_TIMEOUT / 1000, JSON.stringify(session));
    } catch (error) {
      console.error('Redis save error:', error);
      // Fallback to in-memory
      this.sessions.set(sessionId, session);
    }
  }

  /**
   * Get session from Redis
   */
  async getSessionFromRedis(sessionId) {
    try {
      const key = `session:${sessionId}`;
      const data = await this.redis.get(key);
      if (data) {
        return JSON.parse(data);
      }
      return null;
    } catch (error) {
      console.error('Redis get error:', error);
      return null;
    }
  }

  /**
   * Get session by ID
   */
  async getSession(sessionId) {
    let session;

    if (this.useRedis && this.redis) {
      session = await this.getSessionFromRedis(sessionId);
    } else {
      session = this.sessions.get(sessionId);
    }
    
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`);
    }

    // Check if session has expired
    const lastActivity = typeof session.lastActivity === 'number' 
      ? session.lastActivity 
      : new Date(session.lastActivity).getTime();
      
    if (Date.now() - lastActivity > this.SESSION_TIMEOUT) {
      await this.deleteSession(sessionId);
      throw new Error(`Session expired: ${sessionId}`);
    }

    // Update last activity
    session.lastActivity = Date.now();
    await this.saveSession(sessionId, session);
    
    return session;
  }

  /**
   * Save session (Redis or in-memory)
   */
  async saveSession(sessionId, session) {
    if (this.useRedis && this.redis) {
      await this.saveSessionToRedis(sessionId, session);
    } else {
      this.sessions.set(sessionId, session);
    }
  }

  /**
   * Delete session
   */
  async deleteSession(sessionId) {
    if (this.useRedis && this.redis) {
      try {
        const key = `session:${sessionId}`;
        await this.redis.del(key);
      } catch (error) {
        console.error('Redis delete error:', error);
      }
    }
    this.sessions.delete(sessionId);
  }

  /**
   * Update session progress
   */
  async updateProgress(sessionId, updates) {
    const session = await this.getSession(sessionId);

    if (updates.currentPhase) {
      session.progress.currentPhase = updates.currentPhase;
    }

    if (updates.completedPhase) {
      if (!session.progress.completedPhases.includes(updates.completedPhase)) {
        session.progress.completedPhases.push(updates.completedPhase);
      }
    }

    if (updates.interactionCompleted) {
      session.progress.interactionsCompleted.push({
        interactionId: updates.interactionCompleted,
        completedAt: new Date().toISOString()
      });
    }

    if (updates.teachingMomentViewed) {
      session.progress.teachingMomentsViewed.push({
        timestamp: updates.teachingMomentViewed,
        viewedAt: new Date().toISOString()
      });
    }

    session.lastActivity = Date.now();
    await this.saveSession(sessionId, session);
    
    return session;
  }

  /**
   * Mark session as complete
   */
  async completeSession(sessionId) {
    const session = await this.getSession(sessionId);
    session.state.isCompleted = true;
    session.state.isActive = false;
    session.completedAt = new Date().toISOString();
    
    const duration = new Date(session.completedAt) - new Date(session.startedAt);
    session.durationMs = duration;
    session.durationMin = Math.floor(duration / 60000);

    await this.saveSession(sessionId, session);
    return session;
  }

  /**
   * Pause/Resume session
   */
  async togglePause(sessionId) {
    const session = await this.getSession(sessionId);
    session.state.isPaused = !session.state.isPaused;
    session.lastActivity = Date.now();
    await this.saveSession(sessionId, session);
    return session;
  }

  /**
   * Get session statistics
   */
  async getSessionStats(sessionId) {
    const session = await this.getSession(sessionId);
    const duration = session.completedAt 
      ? new Date(session.completedAt) - new Date(session.startedAt)
      : Date.now() - new Date(session.startedAt).getTime();

    return {
      sessionId,
      lessonId: session.lessonId,
      age: session.age,
      durationMin: Math.floor(duration / 60000),
      currentPhase: session.progress.currentPhase,
      phasesCompleted: session.progress.completedPhases.length,
      interactionsCompleted: session.progress.interactionsCompleted.length,
      teachingMomentsViewed: session.progress.teachingMomentsViewed.length,
      isCompleted: session.state.isCompleted,
      completionPercentage: this.calculateCompletion(session)
    };
  }

  /**
   * Calculate completion percentage
   */
  calculateCompletion(session) {
    const totalPhases = 5; // welcome, teaching, practice, wisdom, reflection
    const completed = session.progress.completedPhases.length;
    return Math.floor((completed / totalPhases) * 100);
  }

  /**
   * Clean up expired sessions
   */
  async cleanupExpiredSessions() {
    const now = Date.now();
    let cleaned = 0;

    // Clean in-memory sessions
    for (const [sessionId, session] of this.sessions.entries()) {
      const lastActivity = typeof session.lastActivity === 'number' 
        ? session.lastActivity 
        : new Date(session.lastActivity).getTime();
        
      if (now - lastActivity > this.SESSION_TIMEOUT) {
        this.sessions.delete(sessionId);
        cleaned++;
      }
    }

    // Redis cleanup is handled by TTL, but we can scan for expired keys
    // (Redis automatically expires keys based on TTL)
    
    return {
      cleaned,
      remaining: this.sessions.size,
      storage: this.useRedis ? 'redis' : 'memory'
    };
  }

  /**
   * Get all active sessions (for monitoring)
   */
  async getActiveSessions() {
    const active = [];
    const now = Date.now();

    // Get from in-memory
    for (const session of this.sessions.values()) {
      const lastActivity = typeof session.lastActivity === 'number' 
        ? session.lastActivity 
        : new Date(session.lastActivity).getTime();
        
      if (now - lastActivity <= this.SESSION_TIMEOUT && session.state.isActive) {
        active.push({
          sessionId: session.sessionId,
          age: session.age,
          lessonId: session.lessonId,
          currentPhase: session.progress.currentPhase,
          startedAt: session.startedAt,
          durationMin: Math.floor((now - new Date(session.startedAt).getTime()) / 60000)
        });
      }
    }

    // If using Redis, also scan Redis keys (optional, can be expensive)
    // For now, we'll rely on in-memory cache for active sessions monitoring
    
    return active;
  }

  /**
   * Update session activity (called by WebSocket keepalive)
   */
  async updateSessionActivity(sessionId) {
    try {
      const session = await this.getSession(sessionId);
      session.lastActivity = Date.now();
      await this.saveSession(sessionId, session);
    } catch (error) {
      // Session might not exist, that's okay
      console.warn(`[Session] Failed to update activity: ${error.message}`);
    }
  }

  /**
   * Get session history for a user
   */
  async getSessionHistory(userId, limit = 50) {
    // This would query a database in production
    // For now, return sessions from memory that match userId
    const history = [];
    
    for (const session of this.sessions.values()) {
      if (session.userId === userId && session.state.isCompleted) {
        history.push({
          sessionId: session.sessionId,
          lessonId: session.lessonId,
          age: session.age,
          completedAt: session.completedAt,
          durationMin: session.durationMin,
          completionPercentage: this.calculateCompletion(session)
        });
      }
    }

    // Sort by completion time (newest first)
    history.sort((a, b) => new Date(b.completedAt) - new Date(a.completedAt));
    
    return history.slice(0, limit);
  }
}

// We need to install uuid package
// Will add to package.json

module.exports = SessionService;



