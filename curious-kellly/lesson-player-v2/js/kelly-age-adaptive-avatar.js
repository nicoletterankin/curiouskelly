/**
 * Kelly Age-Adaptive Avatar System
 * 
 * Adapts Kelly's visual appearance based on learner age (2-102).
 * Uses a hybrid approach:
 * 
 * 1. CSS-BASED SIMULATION (Now): Visual filters and transforms
 * 2. IMAGE VARIANTS (Future): Dedicated age-specific images
 * 3. 3D BLEND SHAPES (Post-launch): Unity WebGL age morphing
 * 
 * The system works with existing Kelly assets while preparing
 * infrastructure for future age-variant images.
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// AGE BUCKET DEFINITIONS
// =============================================================================

/**
 * 6 age buckets matching the voice engine and design docs.
 * Each bucket defines visual characteristics for that age range.
 */
const AGE_BUCKETS = {
  '2-5': {
    name: 'Little Learner',
    minAge: 2,
    maxAge: 5,
    persona: 'Playful Friend',
    // CSS filter values
    filters: {
      brightness: 1.08,    // Brighter, more vibrant
      saturation: 1.15,    // More colorful
      warmth: 0,           // Neutral
      softness: 0.3,       // Slight softening
      hueRotate: -5        // Slightly cooler (youthful)
    },
    // Transform effects
    transform: {
      scale: 1.0,          // Slightly larger feel
      headTilt: 2          // Slight playful tilt
    },
    // Expression intensity multiplier
    expressionMultiplier: 1.4,  // +40% expression intensity
    // Animation speed
    animationSpeed: 1.1,
    // Background mood
    bgMood: 'playful',
    // Future: dedicated image set
    imageSet: 'child',
    // Kelly's teaching style for this age
    teachingStyle: 'Story-based, Fun & Playful'
  },

  '6-12': {
    name: 'Curious Explorer',
    minAge: 6,
    maxAge: 12,
    persona: 'Cool Big Sister',
    filters: {
      brightness: 1.04,
      saturation: 1.08,
      warmth: 2,
      softness: 0.15,
      hueRotate: 0
    },
    transform: {
      scale: 1.0,
      headTilt: 1
    },
    expressionMultiplier: 1.25,
    animationSpeed: 1.05,
    bgMood: 'curious',
    imageSet: 'kid',
    teachingStyle: 'Hands-on, Engaging & Curious'
  },

  '13-17': {
    name: 'Teen Scholar',
    minAge: 13,
    maxAge: 17,
    persona: 'Smart Mentor',
    filters: {
      brightness: 1.0,
      saturation: 1.0,
      warmth: 0,
      softness: 0,
      hueRotate: 0
    },
    transform: {
      scale: 1.0,
      headTilt: 0
    },
    expressionMultiplier: 0.9,  // Slightly restrained (teens appreciate subtlety)
    animationSpeed: 1.0,
    bgMood: 'focused',
    imageSet: 'teen',
    teachingStyle: 'Direct, Relatable, No Fluff'
  },

  '18-35': {
    name: 'Adult Learner',
    minAge: 18,
    maxAge: 35,
    persona: 'Equal Partner',
    filters: {
      brightness: 1.0,       // BASE (Kelly is 27)
      saturation: 1.0,
      warmth: 0,
      softness: 0,
      hueRotate: 0
    },
    transform: {
      scale: 1.0,
      headTilt: 0
    },
    expressionMultiplier: 1.0,  // Baseline
    animationSpeed: 1.0,
    bgMood: 'professional',
    imageSet: 'adult',        // Base Kelly image
    teachingStyle: 'Practical, Clear, Conversational'
  },

  '36-60': {
    name: 'Seasoned Mind',
    minAge: 36,
    maxAge: 60,
    persona: 'Respectful Guide',
    filters: {
      brightness: 0.98,
      saturation: 0.95,
      warmth: 5,            // Warmer tones
      softness: 0.1,
      hueRotate: 3          // Slightly warmer hue
    },
    transform: {
      scale: 1.0,
      headTilt: 0
    },
    expressionMultiplier: 0.85,  // More measured expressions
    animationSpeed: 0.95,
    bgMood: 'confident',
    imageSet: 'mature',
    teachingStyle: 'Efficient, Substantive, Respectful'
  },

  '61-102': {
    name: 'Wisdom Keeper',
    minAge: 61,
    maxAge: 102,
    persona: 'Warm Companion',
    filters: {
      brightness: 0.96,
      saturation: 0.88,
      warmth: 10,           // Warm, golden tones
      softness: 0.25,       // Gentler overall
      hueRotate: 8          // Warmer hue shift
    },
    transform: {
      scale: 1.0,
      headTilt: -1          // Slight gentle lean
    },
    expressionMultiplier: 0.75,  // Gentle, subtle expressions
    animationSpeed: 0.9,
    bgMood: 'warm',
    imageSet: 'elder',
    teachingStyle: 'Warm, Thoughtful, Reflective'
  }
};

// =============================================================================
// IMAGE PATH CONFIGURATION
// =============================================================================

/**
 * Image path templates for age-variant assets.
 * Currently uses fallback to base Kelly image, but structured for future expansion.
 */
const IMAGE_PATHS = {
  // Base path for age-variant images (future)
  ageVariantBase: '/assets/kelly/age-variants/',
  
  // Current fallback paths (existing assets)
  fallback: {
    default: '/images/expressions/curious-main.jpeg',
    expressions: {
      curious: '/images/expressions/curious-main.jpeg',
      explaining: '/images/expressions/explaining-main.jpeg',
      listening: '/images/expressions/listening-main.jpeg',
      celebrating: '/images/expressions/celebrating-main.jpeg',
      wisdom: '/images/expressions/wisdom-main.jpeg'
    },
    poses: {
      up: '/kelly/poses/kelly_choice_left.png',
      down: '/kelly/poses/kelly_choice_right.png',
      welcome: '/kelly/poses/kelly_welcome.png',
      hint: '/kelly/poses/kelly_hint.png',
      listening: '/kelly/poses/kelly_listening.png',
      idle: '/kelly/poses/kelly_idle.png'
    }
  },
  
  // Future age-variant image paths (to be commissioned)
  // Format: kelly-{imageSet}-{expression}.png
  ageVariants: {
    child: '/assets/kelly/age-variants/child/',    // 2-5
    kid: '/assets/kelly/age-variants/kid/',        // 6-12
    teen: '/assets/kelly/age-variants/teen/',      // 13-17
    adult: '/assets/kelly/age-variants/adult/',    // 18-35 (base)
    mature: '/assets/kelly/age-variants/mature/',  // 36-60
    elder: '/assets/kelly/age-variants/elder/'     // 61-102
  }
};

// =============================================================================
// KELLY AGE-ADAPTIVE AVATAR CLASS
// =============================================================================

class KellyAgeAdaptiveAvatar {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    if (!this.container) {
      console.error('[KellyAgeAdaptive] Container not found');
      return;
    }
    
    // Configuration
    this.options = {
      initialAge: options.initialAge || 27,
      transitionDuration: options.transitionDuration || 400, // ms
      enableCSSSimulation: options.enableCSSSimulation !== false,
      enableImageVariants: options.enableImageVariants || false, // Enable when images exist
      onAgeChange: options.onAgeChange || null,
      onBucketChange: options.onBucketChange || null,
      ...options
    };
    
    // State
    this.currentAge = this.options.initialAge;
    this.currentBucket = this.getAgeBucket(this.currentAge);
    this.currentExpression = 'curious';
    this.isTransitioning = false;
    
    // DOM elements
    this.imageElement = null;
    this.filterOverlay = null;
    
    // Initialize
    this.init();
  }
  
  // ---------------------------------------------------------------------------
  // INITIALIZATION
  // ---------------------------------------------------------------------------
  
  init() {
    // Find or create image element
    this.imageElement = this.container.querySelector('img') || 
                        this.container.querySelector('.kelly-image');
    
    if (!this.imageElement) {
      console.warn('[KellyAgeAdaptive] No image element found in container');
      return;
    }
    
    // Add age-adaptive class
    this.imageElement.classList.add('kelly-age-adaptive');
    
    // Create filter overlay for CSS effects
    this.createFilterOverlay();
    
    // Apply initial age settings
    this.applyAgeSettings(this.currentAge, false);
    
    // Inject required CSS
    this.injectStyles();
    
    console.log(`[KellyAgeAdaptive] Initialized for age ${this.currentAge} (${this.currentBucket.name})`);
  }
  
  createFilterOverlay() {
    // Create overlay div for warmth/softness effects
    this.filterOverlay = document.createElement('div');
    this.filterOverlay.className = 'kelly-age-filter-overlay';
    this.filterOverlay.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      pointer-events: none;
      z-index: 1;
      opacity: 0;
      transition: opacity ${this.options.transitionDuration}ms ease;
    `;
    
    // Insert after image
    if (this.imageElement.parentElement) {
      this.imageElement.parentElement.style.position = 'relative';
      this.imageElement.parentElement.appendChild(this.filterOverlay);
    }
  }
  
  injectStyles() {
    if (document.getElementById('kelly-age-adaptive-styles')) return;
    
    const styleSheet = document.createElement('style');
    styleSheet.id = 'kelly-age-adaptive-styles';
    styleSheet.textContent = `
      .kelly-age-adaptive {
        transition: 
          filter ${this.options.transitionDuration}ms ease,
          transform ${this.options.transitionDuration}ms ease;
      }
      
      .kelly-age-filter-overlay {
        mix-blend-mode: soft-light;
      }
      
      /* Age bucket visual themes */
      .kelly-age-adaptive.age-bucket-child {
        --kelly-mood-color: rgba(147, 197, 253, 0.1); /* Soft blue */
      }
      
      .kelly-age-adaptive.age-bucket-kid {
        --kelly-mood-color: rgba(134, 239, 172, 0.1); /* Fresh green */
      }
      
      .kelly-age-adaptive.age-bucket-teen {
        --kelly-mood-color: rgba(165, 180, 252, 0.1); /* Cool purple */
      }
      
      .kelly-age-adaptive.age-bucket-adult {
        --kelly-mood-color: transparent; /* Base/neutral */
      }
      
      .kelly-age-adaptive.age-bucket-mature {
        --kelly-mood-color: rgba(251, 191, 36, 0.05); /* Subtle gold */
      }
      
      .kelly-age-adaptive.age-bucket-elder {
        --kelly-mood-color: rgba(251, 146, 60, 0.08); /* Warm amber */
      }
      
      /* Smooth transitions between ages */
      .kelly-age-transitioning {
        transition: 
          filter ${this.options.transitionDuration * 0.5}ms ease-out,
          transform ${this.options.transitionDuration * 0.5}ms ease-out,
          opacity ${this.options.transitionDuration * 0.5}ms ease-out;
      }
      
      /* Expression intensity animations */
      @keyframes kelly-expression-pop {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
      }
      
      .kelly-expression-change {
        animation: kelly-expression-pop 300ms ease;
      }
    `;
    document.head.appendChild(styleSheet);
  }
  
  // ---------------------------------------------------------------------------
  // AGE BUCKET METHODS
  // ---------------------------------------------------------------------------
  
  /**
   * Get the age bucket for a given age
   * @param {number} age - Age in years (2-102)
   * @returns {object} Age bucket configuration
   */
  getAgeBucket(age) {
    age = Math.max(2, Math.min(102, age));
    
    for (const [key, bucket] of Object.entries(AGE_BUCKETS)) {
      if (age >= bucket.minAge && age <= bucket.maxAge) {
        return { key, ...bucket };
      }
    }
    
    // Default fallback to adult
    return { key: '18-35', ...AGE_BUCKETS['18-35'] };
  }
  
  /**
   * Get age bucket key (string) for a given age
   */
  getAgeBucketKey(age) {
    const bucket = this.getAgeBucket(age);
    return bucket.key;
  }
  
  /**
   * Get all age buckets
   */
  static getAgeBuckets() {
    return AGE_BUCKETS;
  }
  
  // ---------------------------------------------------------------------------
  // AGE SETTING METHODS
  // ---------------------------------------------------------------------------
  
  /**
   * Set Kelly's age and apply visual changes
   * @param {number} age - Target age (2-102)
   * @param {boolean} animate - Whether to animate the transition
   */
  setAge(age, animate = true) {
    age = Math.max(2, Math.min(102, age));
    
    const previousAge = this.currentAge;
    const previousBucket = this.currentBucket;
    this.currentAge = age;
    this.currentBucket = this.getAgeBucket(age);
    
    // Check if bucket changed
    const bucketChanged = previousBucket.key !== this.currentBucket.key;
    
    // Apply visual settings
    this.applyAgeSettings(age, animate);
    
    // Fire callbacks
    if (this.options.onAgeChange) {
      this.options.onAgeChange(age, previousAge, this.currentBucket);
    }
    
    if (bucketChanged && this.options.onBucketChange) {
      this.options.onBucketChange(this.currentBucket, previousBucket);
      console.log(`[KellyAgeAdaptive] Bucket change: ${previousBucket.name} → ${this.currentBucket.name}`);
    }
    
    return this.currentBucket;
  }
  
  /**
   * Apply visual settings for the current age
   * @param {number} age - Target age
   * @param {boolean} animate - Whether to animate
   */
  applyAgeSettings(age, animate = true) {
    const bucket = this.getAgeBucket(age);
    
    if (!this.imageElement) return;
    
    // Add transitioning class if animating
    if (animate) {
      this.imageElement.classList.add('kelly-age-transitioning');
      this.isTransitioning = true;
    }
    
    // Apply CSS filters (if enabled)
    if (this.options.enableCSSSimulation) {
      this.applyCSSFilters(bucket);
    }
    
    // Update age bucket class
    this.updateBucketClass(bucket);
    
    // Try to load age-variant image (if enabled and available)
    if (this.options.enableImageVariants) {
      this.loadAgeVariantImage(bucket);
    }
    
    // Remove transitioning class after animation
    if (animate) {
      setTimeout(() => {
        this.imageElement.classList.remove('kelly-age-transitioning');
        this.isTransitioning = false;
      }, this.options.transitionDuration);
    }
  }
  
  /**
   * Apply CSS filter effects for age simulation
   */
  applyCSSFilters(bucket) {
    const { filters, transform } = bucket;
    
    // Build filter string
    const filterParts = [];
    
    // Brightness
    if (filters.brightness !== 1.0) {
      filterParts.push(`brightness(${filters.brightness})`);
    }
    
    // Saturation
    if (filters.saturation !== 1.0) {
      filterParts.push(`saturate(${filters.saturation})`);
    }
    
    // Hue rotation for warmth
    if (filters.hueRotate !== 0) {
      filterParts.push(`hue-rotate(${filters.hueRotate}deg)`);
    }
    
    // Softness (blur)
    if (filters.softness > 0) {
      filterParts.push(`blur(${filters.softness}px)`);
    }
    
    // Apply filters
    this.imageElement.style.filter = filterParts.length > 0 
      ? filterParts.join(' ') 
      : 'none';
    
    // Apply transforms
    const transformParts = [];
    
    if (transform.scale !== 1.0) {
      transformParts.push(`scale(${transform.scale})`);
    }
    
    if (transform.headTilt !== 0) {
      transformParts.push(`rotate(${transform.headTilt}deg)`);
    }
    
    this.imageElement.style.transform = transformParts.length > 0
      ? transformParts.join(' ')
      : 'none';
    
    // Apply warmth overlay
    if (filters.warmth > 0 && this.filterOverlay) {
      const warmthIntensity = filters.warmth / 100;
      this.filterOverlay.style.background = `rgba(255, 180, 100, ${warmthIntensity})`;
      this.filterOverlay.style.opacity = '1';
    } else if (this.filterOverlay) {
      this.filterOverlay.style.opacity = '0';
    }
  }
  
  /**
   * Update the CSS class for the current age bucket
   */
  updateBucketClass(bucket) {
    // Remove all bucket classes
    Object.values(AGE_BUCKETS).forEach(b => {
      this.imageElement.classList.remove(`age-bucket-${b.imageSet}`);
    });
    
    // Add current bucket class
    this.imageElement.classList.add(`age-bucket-${bucket.imageSet}`);
    
    // Update data attribute
    this.imageElement.dataset.ageBucket = bucket.key;
    this.imageElement.dataset.agePersona = bucket.persona;
  }
  
  /**
   * Attempt to load an age-variant image
   * Falls back to current image if variant doesn't exist
   */
  async loadAgeVariantImage(bucket) {
    const basePath = IMAGE_PATHS.ageVariants[bucket.imageSet];
    if (!basePath) return;
    
    const imagePath = `${basePath}kelly-${bucket.imageSet}-${this.currentExpression}.png`;
    
    // Check if image exists
    try {
      const response = await fetch(imagePath, { method: 'HEAD' });
      if (response.ok) {
        // Image exists, load it
        const previousSrc = this.imageElement.src;
        this.imageElement.src = imagePath;
        console.log(`[KellyAgeAdaptive] Loaded age variant: ${imagePath}`);
      }
      // If not found, keep current image (CSS filters will handle visual adaptation)
    } catch (e) {
      // Image not found, CSS filters will handle it
      console.log(`[KellyAgeAdaptive] Age variant not found, using CSS simulation`);
    }
  }
  
  // ---------------------------------------------------------------------------
  // EXPRESSION METHODS
  // ---------------------------------------------------------------------------
  
  /**
   * Set Kelly's expression (with age-adjusted intensity)
   * @param {string} expression - Expression name
   */
  setExpression(expression) {
    this.currentExpression = expression;
    
    // Trigger expression animation
    this.imageElement.classList.add('kelly-expression-change');
    setTimeout(() => {
      this.imageElement.classList.remove('kelly-expression-change');
    }, 300);
    
    // If image variants enabled, try to load expression-specific image
    if (this.options.enableImageVariants) {
      this.loadAgeVariantImage(this.currentBucket);
    }
  }
  
  /**
   * Get the expression intensity multiplier for the current age
   */
  getExpressionMultiplier() {
    return this.currentBucket.expressionMultiplier;
  }
  
  /**
   * Get animation speed for the current age
   */
  getAnimationSpeed() {
    return this.currentBucket.animationSpeed;
  }
  
  // ---------------------------------------------------------------------------
  // UTILITY METHODS
  // ---------------------------------------------------------------------------
  
  /**
   * Get current age bucket info
   */
  getCurrentBucket() {
    return this.currentBucket;
  }
  
  /**
   * Get teaching style for current age
   */
  getTeachingStyle() {
    return this.currentBucket.teachingStyle;
  }
  
  /**
   * Get persona name for current age
   */
  getPersona() {
    return this.currentBucket.persona;
  }
  
  /**
   * Interpolate between two ages (for smooth slider dragging)
   * @param {number} fromAge - Starting age
   * @param {number} toAge - Target age
   * @param {number} progress - Progress 0-1
   */
  interpolateAge(fromAge, toAge, progress) {
    const interpolatedAge = Math.round(fromAge + (toAge - fromAge) * progress);
    this.setAge(interpolatedAge, false);
    return interpolatedAge;
  }
  
  /**
   * Reset to default (27 years old)
   */
  reset() {
    this.setAge(27, true);
  }
  
  /**
   * Cleanup
   */
  destroy() {
    if (this.filterOverlay && this.filterOverlay.parentElement) {
      this.filterOverlay.parentElement.removeChild(this.filterOverlay);
    }
    
    if (this.imageElement) {
      this.imageElement.style.filter = 'none';
      this.imageElement.style.transform = 'none';
      this.imageElement.classList.remove('kelly-age-adaptive');
      Object.values(AGE_BUCKETS).forEach(b => {
        this.imageElement.classList.remove(`age-bucket-${b.imageSet}`);
      });
    }
    
    console.log('[KellyAgeAdaptive] Destroyed');
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

window.KellyAgeAdaptiveAvatar = KellyAgeAdaptiveAvatar;
window.KELLY_AGE_BUCKETS = AGE_BUCKETS;

console.log('[KellyAgeAdaptiveAvatar] Module loaded');







