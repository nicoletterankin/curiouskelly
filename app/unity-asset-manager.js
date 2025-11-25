/**
 * Unity Asset Manager - Handles age-based character model loading
 * 
 * Responsibilities:
 * - Map age slider values (2-102) to character model URLs
 * - Load character models via Unity bridge
 * - Handle model loading errors and fallbacks
 * - Cache loaded models for performance
 */

/* eslint-env browser */
export default class UnityAssetManager {
  constructor(unityBridge) {
    this.unityBridge = unityBridge;
    this.currentModel = null;
    this.loadedModels = new Map(); // Cache: ageBucket -> model config
    this.loadingPromises = new Map(); // Track in-flight loads

    // Age bucket to character model mapping
    this.ageModelMap = {
      '2-5': {
        modelUrl: '/unity/character-models/age-2-5.glb',
        fallbackUrl: '/unity/character-models/age-6-12.glb',
        voicePitch: 1.2,
        animationSpeed: 1.1,
        description: 'Toddler/Preschool Kelly',
      },
      '6-12': {
        modelUrl: '/unity/character-models/age-6-12.glb',
        fallbackUrl: '/unity/character-models/age-13-17.glb',
        voicePitch: 1.1,
        animationSpeed: 1.05,
        description: 'Elementary School Kelly',
      },
      '13-17': {
        modelUrl: '/unity/character-models/age-13-17.glb',
        fallbackUrl: '/unity/character-models/age-18-35.glb',
        voicePitch: 1.0,
        animationSpeed: 1.0,
        description: 'Teen Kelly',
      },
      '18-35': {
        modelUrl: '/unity/character-models/age-18-35.glb',
        fallbackUrl: '/unity/character-models/age-36-60.glb',
        voicePitch: 0.95,
        animationSpeed: 0.98,
        description: 'Young Adult Kelly',
      },
      '36-60': {
        modelUrl: '/unity/character-models/age-36-60.glb',
        fallbackUrl: '/unity/character-models/age-18-35.glb',
        voicePitch: 0.9,
        animationSpeed: 0.95,
        description: 'Adult Kelly',
      },
      '61-102': {
        modelUrl: '/unity/character-models/age-61-102.glb',
        fallbackUrl: '/unity/character-models/age-36-60.glb',
        voicePitch: 0.85,
        animationSpeed: 0.9,
        description: 'Elder Kelly',
      },
    };

    // Default fallback model (always available)
    this.defaultModel = this.ageModelMap['18-35'];

    // Listen for Unity responses
    this.setupListeners();
  }

  /**
   * Setup Unity bridge event listeners
   */
  setupListeners() {
    if (!this.unityBridge) return;

    // Listen for character-loaded confirmation
    window.addEventListener('message', (event) => {
      if (event.data?.type === 'unity-bridge-command') {
        const { event: eventName, payload } = event.data;
        
        if (eventName === 'character-loaded') {
          this.handleCharacterLoaded(payload);
        } else if (eventName === 'error' && payload.context === 'character-load') {
          this.handleCharacterError(payload);
        }
      }
    });
  }

  /**
   * Get age bucket from age value (2-102)
   */
  getAgeBucket(age) {
    if (age >= 2 && age <= 5) return '2-5';
    if (age >= 6 && age <= 12) return '6-12';
    if (age >= 13 && age <= 17) return '13-17';
    if (age >= 18 && age <= 35) return '18-35';
    if (age >= 36 && age <= 60) return '36-60';
    if (age >= 61 && age <= 102) return '61-102';
    return '18-35'; // Default fallback
  }

  /**
   * Get model configuration for age bucket
   */
  getModelConfig(ageBucket) {
    return this.ageModelMap[ageBucket] || this.defaultModel;
  }

  /**
   * Load character model for age bucket
   */
  async loadCharacterModel(ageBucket, sessionId = null) {
    // Validate age bucket
    if (!this.ageModelMap[ageBucket]) {
      console.warn(`[UnityAssetManager] Invalid age bucket: ${ageBucket}, using default`);
      ageBucket = '18-35';
    }

    // Check if already loaded
    if (this.loadedModels.has(ageBucket) && this.currentModel === ageBucket) {
      return this.loadedModels.get(ageBucket);
    }

    // Check if already loading
    if (this.loadingPromises.has(ageBucket)) {
      return this.loadingPromises.get(ageBucket);
    }

    // Start loading
    const config = this.ageModelMap[ageBucket];
    const loadPromise = this.performLoad(config, ageBucket, sessionId);
    this.loadingPromises.set(ageBucket, loadPromise);

    try {
      const result = await loadPromise;
      this.loadedModels.set(ageBucket, config);
      this.currentModel = ageBucket;
      return result;
    } catch (error) {
      console.error(`[UnityAssetManager] Failed to load model for ${ageBucket}:`, error);
      // Try fallback
      return this.loadFallbackModel(ageBucket, sessionId);
    } finally {
      this.loadingPromises.delete(ageBucket);
    }
  }

  /**
   * Perform actual model load via Unity bridge
   */
  async performLoad(config, ageBucket, sessionId) {
    if (!this.unityBridge) {
      throw new Error('Unity bridge not available');
    }

    // Emit load event to Unity
    this.unityBridge.emit('character-load', {
      modelUrl: config.modelUrl,
      fallbackUrl: config.fallbackUrl,
      ageBucket,
      voicePitch: config.voicePitch,
      animationSpeed: config.animationSpeed,
      sessionId,
      timestamp: new Date().toISOString(),
    });

    // Wait for Unity confirmation (with timeout)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Character model load timeout: ${ageBucket}`));
      }, 15000); // 15 second timeout

      const handler = (event) => {
        if (event.data?.type === 'unity-bridge-command') {
          const { event: eventName, payload } = event.data;
          
          if (eventName === 'character-loaded' && payload.ageBucket === ageBucket) {
            clearTimeout(timeout);
            window.removeEventListener('message', handler);
            resolve(config);
          } else if (eventName === 'error' && 
                     payload.context === 'character-load' && 
                     payload.ageBucket === ageBucket) {
            clearTimeout(timeout);
            window.removeEventListener('message', handler);
            reject(new Error(payload.message || 'Character load error'));
          }
        }
      };

      window.addEventListener('message', handler);
    });
  }

  /**
   * Handle character loaded confirmation from Unity
   */
  handleCharacterLoaded(payload) {
    const { ageBucket, modelUrl } = payload;
    console.log(`[UnityAssetManager] Character loaded: ${ageBucket} from ${modelUrl}`);
    
    if (this.loadedModels.has(ageBucket)) {
      this.currentModel = ageBucket;
    }
  }

  /**
   * Handle character load error from Unity
   */
  handleCharacterError(payload) {
    const { ageBucket, message } = payload;
    console.error(`[UnityAssetManager] Character load error for ${ageBucket}:`, message);
    
    // Try fallback if not already using it
    if (payload.ageBucket && !payload.isFallback) {
      const config = this.ageModelMap[ageBucket];
      if (config?.fallbackUrl) {
        console.log(`[UnityAssetManager] Attempting fallback model for ${ageBucket}`);
        this.loadFallbackModel(ageBucket, payload.sessionId);
      }
    }
  }

  /**
   * Load fallback model when primary fails
   */
  async loadFallbackModel(ageBucket, sessionId) {
    const config = this.ageModelMap[ageBucket];
    if (!config?.fallbackUrl) {
      // Use default model
      console.log(`[UnityAssetManager] No fallback for ${ageBucket}, using default (18-35)`);
      return this.loadCharacterModel('18-35', sessionId);
    }

    console.log(`[UnityAssetManager] Loading fallback model for ${ageBucket}: ${config.fallbackUrl}`);

    // Determine fallback age bucket from URL
    const fallbackBucket = this.getBucketFromUrl(config.fallbackUrl) || '18-35';

    // Emit fallback load event
    if (this.unityBridge) {
      this.unityBridge.emit('character-load', {
        modelUrl: config.fallbackUrl,
        ageBucket: fallbackBucket,
        originalBucket: ageBucket,
        sessionId,
        isFallback: true,
        voicePitch: config.voicePitch,
        animationSpeed: config.animationSpeed,
      });
    }

    // Cache fallback config
    const fallbackConfig = { ...config, modelUrl: config.fallbackUrl };
    this.loadedModels.set(fallbackBucket, fallbackConfig);
    this.currentModel = fallbackBucket;

    return fallbackConfig;
  }

  /**
   * Extract age bucket from model URL (helper)
   */
  getBucketFromUrl(url) {
    const match = url.match(/age-(\d+-\d+)\.glb/);
    return match ? match[1] : null;
  }

  /**
   * Get current loaded model
   */
  getCurrentModel() {
    return this.currentModel ? this.loadedModels.get(this.currentModel) : null;
  }

  /**
   * Check if model is loaded for age bucket
   */
  isModelLoaded(ageBucket) {
    return this.loadedModels.has(ageBucket) && this.currentModel === ageBucket;
  }

  /**
   * Preload models for adjacent age buckets (performance optimization)
   */
  async preloadAdjacentModels(currentBucket) {
    const buckets = Object.keys(this.ageModelMap);
    const currentIndex = buckets.indexOf(currentBucket);
    
    if (currentIndex === -1) return;

    // Preload next and previous buckets
    const adjacentBuckets = [
      buckets[currentIndex - 1],
      buckets[currentIndex + 1],
    ].filter(Boolean);

    for (const bucket of adjacentBuckets) {
      if (!this.loadedModels.has(bucket) && !this.loadingPromises.has(bucket)) {
        // Load in background (don't await)
        this.loadCharacterModel(bucket).catch(err => {
          console.debug(`[UnityAssetManager] Preload failed for ${bucket}:`, err);
        });
      }
    }
  }
}

