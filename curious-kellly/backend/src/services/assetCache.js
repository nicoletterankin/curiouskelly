/**
 * Asset Cache Service
 * 
 * Provides caching for audio files, lesson DNA, and other assets.
 * Uses Redis for distributed caching and in-memory for local development.
 * 
 * Rules from CLAUDE.md:
 * - Preload next-phase assets
 * - Reuse across variants
 * - Hash and version outputs
 * - Store seeds/config for determinism
 */

const crypto = require('crypto');

class AssetCacheService {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.ttl = options.ttl || 3600; // 1 hour default
    this.memoryCache = new Map();
    this.redisClient = options.redisClient || null;
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0
    };
  }

  /**
   * Generate cache key for asset
   * @param {string} type - Asset type (audio, lesson, video, etc.)
   * @param {Object} params - Parameters that identify the asset
   * @returns {string} Cache key
   */
  generateKey(type, params) {
    const sortedParams = Object.keys(params)
      .sort()
      .reduce((acc, key) => {
        acc[key] = params[key];
        return acc;
      }, {});

    const content = JSON.stringify({ type, ...sortedParams });
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    return `asset:${type}:${hash.substring(0, 16)}`;
  }

  /**
   * Get asset from cache
   * @param {string} type - Asset type
   * @param {Object} params - Asset parameters
   * @returns {Promise<any|null>} Cached asset or null
   */
  async get(type, params) {
    if (!this.enabled) return null;

    const key = this.generateKey(type, params);

    // Try memory cache first
    if (this.memoryCache.has(key)) {
      const cached = this.memoryCache.get(key);
      if (cached.expiresAt > Date.now()) {
        this.stats.hits++;
        return cached.data;
      } else {
        this.memoryCache.delete(key);
      }
    }

    // Try Redis if available
    if (this.redisClient) {
      try {
        const cached = await this.redisClient.get(key);
        if (cached) {
          const data = JSON.parse(cached);
          // Also store in memory cache
          this.memoryCache.set(key, {
            data,
            expiresAt: Date.now() + this.ttl * 1000
          });
          this.stats.hits++;
          return data;
        }
      } catch (error) {
        console.error('Redis cache error:', error);
      }
    }

    this.stats.misses++;
    return null;
  }

  /**
   * Set asset in cache
   * @param {string} type - Asset type
   * @param {Object} params - Asset parameters
   * @param {any} data - Data to cache
   * @param {number} ttl - TTL in seconds (optional)
   * @returns {Promise<void>}
   */
  async set(type, params, data, ttl = null) {
    if (!this.enabled) return;

    const key = this.generateKey(type, params);
    const cacheTTL = ttl || this.ttl;

    // Store in memory cache
    this.memoryCache.set(key, {
      data,
      expiresAt: Date.now() + cacheTTL * 1000
    });

    // Store in Redis if available
    if (this.redisClient) {
      try {
        await this.redisClient.setex(key, cacheTTL, JSON.stringify(data));
      } catch (error) {
        console.error('Redis cache error:', error);
      }
    }

    this.stats.sets++;
  }

  /**
   * Delete asset from cache
   * @param {string} type - Asset type
   * @param {Object} params - Asset parameters
   * @returns {Promise<void>}
   */
  async delete(type, params) {
    const key = this.generateKey(type, params);

    // Delete from memory cache
    this.memoryCache.delete(key);

    // Delete from Redis if available
    if (this.redisClient) {
      try {
        await this.redisClient.del(key);
      } catch (error) {
        console.error('Redis cache error:', error);
      }
    }
  }

  /**
   * Clear all cached assets
   * @returns {Promise<void>}
   */
  async clear() {
    // Clear memory cache
    this.memoryCache.clear();

    // Clear Redis if available
    if (this.redisClient) {
      try {
        const keys = await this.redisClient.keys('asset:*');
        if (keys.length > 0) {
          await this.redisClient.del(...keys);
        }
      } catch (error) {
        console.error('Redis cache error:', error);
      }
    }

    this.stats = { hits: 0, misses: 0, sets: 0 };
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache stats
   */
  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? (this.stats.hits / (this.stats.hits + this.stats.misses) * 100).toFixed(2)
      : 0;

    return {
      ...this.stats,
      hitRate: `${hitRate}%`,
      memorySize: this.memoryCache.size
    };
  }

  /**
   * Preload assets for next phase
   * @param {string} lessonId - Lesson ID
   * @param {string} currentPhase - Current phase
   * @param {string} ageVariant - Age variant
   * @param {string} language - Language
   * @returns {Promise<void>}
   */
  async preloadNextPhase(lessonId, currentPhase, ageVariant, language) {
    // Define phase order
    const phaseOrder = ['welcome', 'main', 'wisdom'];
    const currentIndex = phaseOrder.indexOf(currentPhase);

    if (currentIndex === -1 || currentIndex >= phaseOrder.length - 1) {
      return; // No next phase or unknown phase
    }

    const nextPhase = phaseOrder[currentIndex + 1];

    // Preload audio for next phase
    const audioKey = this.generateKey('audio', {
      lessonId,
      phase: nextPhase,
      ageVariant,
      language
    });

    // Check if already cached
    const cached = await this.get('audio', {
      lessonId,
      phase: nextPhase,
      ageVariant,
      language
    });

    if (!cached) {
      console.log(`🔄 Preloading next phase: ${lessonId}/${nextPhase}/${ageVariant}/${language}`);
      // In real implementation, this would fetch and cache the audio file
      // For now, we just mark it for preloading
    }
  }
}

module.exports = AssetCacheService;


