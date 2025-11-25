/**
 * Cache Manager for Curious Kelly
 * 
 * Multi-tier caching system:
 * - Memory cache (fastest, volatile)
 * - IndexedDB cache (persistent, local)
 * - Supabase cache (persistent, global) - for sharing across users
 * 
 * @module cache-manager
 */

import supabaseService from './supabase-service.js';

// =============================================================================
// CONSTANTS & CONFIGURATION
// =============================================================================

const CACHE_CONFIG = {
  // IndexedDB configuration
  dbName: 'curious-kelly-cache',
  dbVersion: 1,
  
  // Store names
  stores: {
    audio: 'audio-cache',
    expressions: 'expression-cache',
    lessons: 'lesson-cache',
    metadata: 'metadata-cache',
  },
  
  // Cache TTL (time-to-live) in milliseconds
  ttl: {
    audio: 7 * 24 * 60 * 60 * 1000,        // 7 days
    expressions: 30 * 24 * 60 * 60 * 1000,  // 30 days
    lessons: 24 * 60 * 60 * 1000,           // 1 day
    metadata: 60 * 60 * 1000,               // 1 hour
  },
  
  // Maximum cache size per store (entries)
  maxEntries: {
    audio: 100,
    expressions: 500,
    lessons: 365,
    metadata: 1000,
  },
  
  // Memory cache limits
  memoryMaxEntries: 50,
  memoryMaxSizeMB: 100,
};

// =============================================================================
// CACHE MANAGER CLASS
// =============================================================================

export default class CacheManager {
  /**
   * Create a new CacheManager instance
   */
  constructor(options = {}) {
    // Memory cache (Map for fast access)
    this.memoryCache = new Map();
    this.memoryCacheOrder = []; // LRU tracking
    
    // IndexedDB reference
    this.db = null;
    this.dbReady = false;
    
    // Configuration
    this.config = { ...CACHE_CONFIG, ...options };
    
    // Statistics
    this.stats = {
      memoryHits: 0,
      memoryMisses: 0,
      indexedDbHits: 0,
      indexedDbMisses: 0,
      totalSets: 0,
      totalEvictions: 0,
    };
    
    // Initialize IndexedDB
    this.initPromise = this.initIndexedDB();
  }

  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================

  /**
   * Initialize IndexedDB for persistent caching
   */
  async initIndexedDB() {
    if (typeof indexedDB === 'undefined') {
      console.warn('[CacheManager] IndexedDB not available');
      return false;
    }

    return new Promise((resolve) => {
      const request = indexedDB.open(this.config.dbName, this.config.dbVersion);
      
      request.onerror = (event) => {
        console.error('[CacheManager] Failed to open IndexedDB:', event.target.error);
        resolve(false);
      };
      
      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.dbReady = true;
        console.log('[CacheManager] IndexedDB initialized');
        resolve(true);
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create stores for each cache type
        for (const [key, storeName] of Object.entries(this.config.stores)) {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { keyPath: 'cacheKey' });
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('expiresAt', 'expiresAt', { unique: false });
            store.createIndex('category', 'category', { unique: false });
            console.log(`[CacheManager] Created store: ${storeName}`);
          }
        }
      };
    });
  }

  /**
   * Ensure IndexedDB is ready before operations
   */
  async ensureReady() {
    if (!this.dbReady) {
      await this.initPromise;
    }
    return this.dbReady;
  }

  // ===========================================================================
  // MAIN CACHE API
  // ===========================================================================

  /**
   * Get a cached item
   * Checks memory first, then IndexedDB
   * 
   * @param {string} key - Cache key
   * @param {string} category - Cache category ('audio', 'expressions', 'lessons', 'metadata')
   * @returns {Promise<*>} Cached value or null
   */
  async get(key, category = 'lessons') {
    // Check memory cache first
    const memoryResult = this.getFromMemory(key);
    if (memoryResult !== null) {
      this.stats.memoryHits++;
      return memoryResult;
    }
    this.stats.memoryMisses++;
    
    // Check IndexedDB
    const dbResult = await this.getFromIndexedDB(key, category);
    if (dbResult !== null) {
      this.stats.indexedDbHits++;
      // Promote to memory cache
      this.setInMemory(key, dbResult);
      return dbResult;
    }
    this.stats.indexedDbMisses++;
    
    return null;
  }

  /**
   * Set a cached item
   * Stores in both memory and IndexedDB
   * 
   * @param {string} key - Cache key
   * @param {*} value - Value to cache
   * @param {string} category - Cache category
   * @param {Object} options - Additional options
   */
  async set(key, value, category = 'lessons', options = {}) {
    const ttl = options.ttl || this.config.ttl[category] || this.config.ttl.lessons;
    
    // Store in memory
    this.setInMemory(key, value);
    
    // Store in IndexedDB
    await this.setInIndexedDB(key, value, category, ttl);
    
    this.stats.totalSets++;
  }

  /**
   * Delete a cached item
   * 
   * @param {string} key - Cache key
   * @param {string} category - Cache category
   */
  async delete(key, category = 'lessons') {
    // Remove from memory
    this.memoryCache.delete(key);
    this.memoryCacheOrder = this.memoryCacheOrder.filter(k => k !== key);
    
    // Remove from IndexedDB
    await this.deleteFromIndexedDB(key, category);
  }

  /**
   * Clear all caches (or specific category)
   * 
   * @param {string} category - Optional category to clear
   */
  async clear(category = null) {
    // Clear memory
    if (category) {
      // Clear only keys matching category pattern
      for (const key of this.memoryCache.keys()) {
        if (key.startsWith(`${category}-`)) {
          this.memoryCache.delete(key);
        }
      }
    } else {
      this.memoryCache.clear();
      this.memoryCacheOrder = [];
    }
    
    // Clear IndexedDB
    await this.clearIndexedDB(category);
  }

  // ===========================================================================
  // MEMORY CACHE OPERATIONS
  // ===========================================================================

  /**
   * Get from memory cache
   */
  getFromMemory(key) {
    const entry = this.memoryCache.get(key);
    
    if (!entry) {
      return null;
    }
    
    // Check expiration
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.memoryCache.delete(key);
      this.memoryCacheOrder = this.memoryCacheOrder.filter(k => k !== key);
      return null;
    }
    
    // Update LRU order
    this.updateLRU(key);
    
    return entry.value;
  }

  /**
   * Set in memory cache
   */
  setInMemory(key, value, ttl = null) {
    // Enforce memory limit
    this.enforceMemoryLimit();
    
    const entry = {
      value,
      timestamp: Date.now(),
      expiresAt: ttl ? Date.now() + ttl : null,
    };
    
    this.memoryCache.set(key, entry);
    this.updateLRU(key);
  }

  /**
   * Update LRU order for a key
   */
  updateLRU(key) {
    const index = this.memoryCacheOrder.indexOf(key);
    if (index > -1) {
      this.memoryCacheOrder.splice(index, 1);
    }
    this.memoryCacheOrder.push(key);
  }

  /**
   * Enforce memory cache limits using LRU eviction
   */
  enforceMemoryLimit() {
    while (this.memoryCacheOrder.length >= this.config.memoryMaxEntries) {
      const oldestKey = this.memoryCacheOrder.shift();
      if (oldestKey) {
        this.memoryCache.delete(oldestKey);
        this.stats.totalEvictions++;
      }
    }
  }

  // ===========================================================================
  // INDEXEDDB CACHE OPERATIONS
  // ===========================================================================

  /**
   * Get from IndexedDB
   */
  async getFromIndexedDB(key, category) {
    const ready = await this.ensureReady();
    if (!ready) return null;
    
    const storeName = this.config.stores[category] || this.config.stores.lessons;
    
    return new Promise((resolve) => {
      try {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.get(key);
        
        request.onsuccess = () => {
          const entry = request.result;
          
          if (!entry) {
            resolve(null);
            return;
          }
          
          // Check expiration
          if (entry.expiresAt && Date.now() > entry.expiresAt) {
            // Delete expired entry asynchronously
            this.deleteFromIndexedDB(key, category);
            resolve(null);
            return;
          }
          
          resolve(entry.value);
        };
        
        request.onerror = () => {
          console.warn('[CacheManager] IndexedDB read error');
          resolve(null);
        };
      } catch (error) {
        console.warn('[CacheManager] IndexedDB transaction error:', error);
        resolve(null);
      }
    });
  }

  /**
   * Set in IndexedDB
   */
  async setInIndexedDB(key, value, category, ttl) {
    const ready = await this.ensureReady();
    if (!ready) return false;
    
    const storeName = this.config.stores[category] || this.config.stores.lessons;
    
    return new Promise((resolve) => {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        
        const entry = {
          cacheKey: key,
          value,
          category,
          timestamp: Date.now(),
          expiresAt: Date.now() + ttl,
        };
        
        const request = store.put(entry);
        
        request.onsuccess = () => {
          // Enforce store limits
          this.enforceStoreLimit(storeName, category);
          resolve(true);
        };
        
        request.onerror = () => {
          console.warn('[CacheManager] IndexedDB write error');
          resolve(false);
        };
      } catch (error) {
        console.warn('[CacheManager] IndexedDB transaction error:', error);
        resolve(false);
      }
    });
  }

  /**
   * Delete from IndexedDB
   */
  async deleteFromIndexedDB(key, category) {
    const ready = await this.ensureReady();
    if (!ready) return false;
    
    const storeName = this.config.stores[category] || this.config.stores.lessons;
    
    return new Promise((resolve) => {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.delete(key);
        
        request.onsuccess = () => resolve(true);
        request.onerror = () => resolve(false);
      } catch (error) {
        resolve(false);
      }
    });
  }

  /**
   * Clear IndexedDB store(s)
   */
  async clearIndexedDB(category = null) {
    const ready = await this.ensureReady();
    if (!ready) return false;
    
    const storeNames = category 
      ? [this.config.stores[category]].filter(Boolean)
      : Object.values(this.config.stores);
    
    for (const storeName of storeNames) {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        store.clear();
      } catch (error) {
        console.warn(`[CacheManager] Failed to clear store ${storeName}:`, error);
      }
    }
    
    return true;
  }

  /**
   * Enforce store size limits by removing oldest entries
   */
  async enforceStoreLimit(storeName, category) {
    const maxEntries = this.config.maxEntries[category] || 100;
    
    try {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const countRequest = store.count();
      
      countRequest.onsuccess = () => {
        const count = countRequest.result;
        
        if (count > maxEntries) {
          // Delete oldest entries
          const entriesToDelete = count - maxEntries + 10; // Delete extra buffer
          const index = store.index('timestamp');
          const request = index.openCursor();
          let deleted = 0;
          
          request.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor && deleted < entriesToDelete) {
              cursor.delete();
              deleted++;
              this.stats.totalEvictions++;
              cursor.continue();
            }
          };
        }
      };
    } catch (error) {
      console.warn('[CacheManager] Failed to enforce store limit:', error);
    }
  }

  // ===========================================================================
  // SPECIALIZED CACHE METHODS
  // ===========================================================================

  /**
   * Cache audio blob with URL
   */
  async cacheAudio(key, audioBlob, metadata = {}) {
    const value = {
      blob: audioBlob,
      url: URL.createObjectURL(audioBlob),
      metadata: {
        ...metadata,
        size: audioBlob.size,
        type: audioBlob.type,
        cachedAt: new Date().toISOString(),
      },
    };
    
    await this.set(key, value, 'audio');
    return value.url;
  }

  /**
   * Get cached audio
   */
  async getAudio(key) {
    const cached = await this.get(key, 'audio');
    
    if (cached) {
      // Recreate URL if blob exists but URL was revoked
      if (cached.blob && (!cached.url || !await this.isUrlValid(cached.url))) {
        cached.url = URL.createObjectURL(cached.blob);
        // Update cache with new URL
        this.setInMemory(key, cached);
      }
      return cached;
    }
    
    return null;
  }

  /**
   * Cache expression data
   */
  async cacheExpressions(key, expressions) {
    await this.set(key, expressions, 'expressions');
  }

  /**
   * Get cached expressions
   */
  async getExpressions(key) {
    return this.get(key, 'expressions');
  }

  /**
   * Cache lesson content
   */
  async cacheLesson(key, lessonData) {
    await this.set(key, lessonData, 'lessons');
  }

  /**
   * Get cached lesson
   */
  async getLesson(key) {
    return this.get(key, 'lessons');
  }

  /**
   * Cache metadata (short TTL)
   */
  async cacheMetadata(key, metadata) {
    await this.set(key, metadata, 'metadata');
  }

  /**
   * Get cached metadata
   */
  async getMetadata(key) {
    return this.get(key, 'metadata');
  }

  // ===========================================================================
  // SUPABASE CACHE OPERATIONS (GLOBAL PERSISTENCE)
  // ===========================================================================

  /**
   * Check if generated content exists in Supabase global cache
   * This allows sharing pre-generated content across users
   */
  async checkGlobalCache(lessonSlug, ageBucket, language, phase) {
    try {
      const storagePath = `generated/${lessonSlug}/${ageBucket}-${language}-${phase}.mp3`;
      
      const { data, error } = await supabaseService.client.storage
        .from('lesson-audio')
        .getPublicUrl(storagePath);
      
      if (error) {
        return null;
      }
      
      // Verify the URL actually works
      const response = await fetch(data.publicUrl, { method: 'HEAD' });
      if (response.ok) {
        return data.publicUrl;
      }
      
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Upload generated content to Supabase for global caching
   */
  async uploadToGlobalCache(lessonSlug, ageBucket, language, phase, audioBlob) {
    try {
      const storagePath = `generated/${lessonSlug}/${ageBucket}-${language}-${phase}.mp3`;
      
      const { data, error } = await supabaseService.client.storage
        .from('lesson-audio')
        .upload(storagePath, audioBlob, {
          contentType: 'audio/mpeg',
          upsert: true,
          cacheControl: '31536000', // 1 year cache
        });
      
      if (error) {
        console.warn('[CacheManager] Failed to upload to global cache:', error);
        return null;
      }
      
      // Get public URL
      const { data: urlData } = supabaseService.client.storage
        .from('lesson-audio')
        .getPublicUrl(storagePath);
      
      return urlData?.publicUrl || null;
    } catch (error) {
      console.warn('[CacheManager] Global cache upload error:', error);
      return null;
    }
  }

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Check if a blob URL is still valid
   */
  async isUrlValid(url) {
    if (!url || !url.startsWith('blob:')) {
      return false;
    }
    
    try {
      const response = await fetch(url, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Clean up expired entries from all caches
   */
  async cleanup() {
    const now = Date.now();
    
    // Clean memory cache
    for (const [key, entry] of this.memoryCache.entries()) {
      if (entry.expiresAt && now > entry.expiresAt) {
        this.memoryCache.delete(key);
        this.memoryCacheOrder = this.memoryCacheOrder.filter(k => k !== key);
      }
    }
    
    // Clean IndexedDB stores
    if (this.dbReady) {
      for (const storeName of Object.values(this.config.stores)) {
        await this.cleanupStore(storeName);
      }
    }
    
    console.log('[CacheManager] Cleanup complete');
  }

  /**
   * Clean expired entries from a specific store
   */
  async cleanupStore(storeName) {
    try {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const index = store.index('expiresAt');
      const now = Date.now();
      
      const request = index.openCursor(IDBKeyRange.upperBound(now));
      
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    } catch (error) {
      console.warn(`[CacheManager] Cleanup error for ${storeName}:`, error);
    }
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const totalRequests = this.stats.memoryHits + this.stats.memoryMisses;
    const memoryHitRate = totalRequests > 0 
      ? ((this.stats.memoryHits / totalRequests) * 100).toFixed(1) 
      : 0;
    
    const dbRequests = this.stats.indexedDbHits + this.stats.indexedDbMisses;
    const dbHitRate = dbRequests > 0
      ? ((this.stats.indexedDbHits / dbRequests) * 100).toFixed(1)
      : 0;
    
    return {
      ...this.stats,
      memoryCacheSize: this.memoryCache.size,
      memoryHitRate: `${memoryHitRate}%`,
      indexedDbHitRate: `${dbHitRate}%`,
      dbReady: this.dbReady,
    };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      memoryHits: 0,
      memoryMisses: 0,
      indexedDbHits: 0,
      indexedDbMisses: 0,
      totalSets: 0,
      totalEvictions: 0,
    };
  }

  /**
   * Get estimated storage usage
   */
  async getStorageUsage() {
    const usage = {
      memory: {
        entries: this.memoryCache.size,
        estimatedSizeBytes: 0,
      },
      indexedDb: {},
      total: 0,
    };
    
    // Estimate memory size
    for (const entry of this.memoryCache.values()) {
      usage.memory.estimatedSizeBytes += JSON.stringify(entry).length * 2; // UTF-16
    }
    
    // Get IndexedDB usage
    if (this.dbReady) {
      for (const [category, storeName] of Object.entries(this.config.stores)) {
        try {
          const transaction = this.db.transaction([storeName], 'readonly');
          const store = transaction.objectStore(storeName);
          const countRequest = store.count();
          
          await new Promise((resolve) => {
            countRequest.onsuccess = () => {
              usage.indexedDb[category] = {
                entries: countRequest.result,
              };
              resolve();
            };
            countRequest.onerror = () => resolve();
          });
        } catch {
          // Skip on error
        }
      }
    }
    
    return usage;
  }
}

// =============================================================================
// SINGLETON INSTANCE
// =============================================================================

// Export singleton for convenient global access
export const cacheManager = new CacheManager();

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Generate a standard cache key for lesson content
 */
export function generateLessonCacheKey(lessonId, ageBucket, language, phase) {
  return `lesson-${lessonId}-${ageBucket}-${language}-${phase}`;
}

/**
 * Generate a cache key for custom/generated content
 */
export function generateCustomCacheKey(lessonSlug, ageBucket, language, phase) {
  return `custom-${lessonSlug}-${ageBucket}-${language}-${phase}`;
}

/**
 * Generate a cache key for audio
 */
export function generateAudioCacheKey(lessonSlug, ageBucket, language, phase) {
  return `audio-${lessonSlug}-${ageBucket}-${language}-${phase}`;
}

/**
 * Generate a cache key for expressions
 */
export function generateExpressionCacheKey(lessonSlug, ageBucket, language, phase) {
  return `expr-${lessonSlug}-${ageBucket}-${language}-${phase}`;
}


