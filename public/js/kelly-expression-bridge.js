/**
 * Kelly Expression Bridge v1.0
 * 
 * Bridges the expression generator system with the lip-sync system.
 * Combines emotional expressions (eyes, brows, gestures) with
 * real-time lip-sync for a complete facial animation system.
 * 
 * Usage:
 *   KellyExpressionBridge.init();
 *   KellyExpressionBridge.setArchetype('The Explorer');
 *   KellyExpressionBridge.setExpression('curious');
 *   // Lip-sync handles mouth automatically
 */

// =============================================================================
// ARCHETYPE EXPRESSION PRESETS (Subset for browser)
// =============================================================================

const EXPRESSION_PRESETS = {
  // Universal expressions
  neutral: { smile: 25, eyebrowRaise: 10, eyesWide: 0 },
  thinking: { eyebrowRaise: 40, eyeSquint: 20, smile: 10 },
  curious: { eyebrowRaise: 55, eyesWide: 35, headTilt: 20, smile: 30 },
  explaining: { eyebrowRaise: 35, smile: 35, eyesWide: 15 },
  excited: { smile: 80, eyebrowRaise: 60, eyesWide: 50 },
  surprised: { eyesWide: 70, eyebrowRaise: 80, mouthOpen: 20 },
  listening: { eyebrowRaise: 30, eyesWide: 20, smile: 40, headTilt: 10 },
  celebrating: { smile: 95, eyesClosed: 20, cheekRaise: 60 },
  wisdom: { smile: 45, eyesClosed: 15, eyebrowRaise: 25, nod: 20 },
  
  // Archetype-specific expressions
  scientist_analyzing: { eyebrowRaise: 45, eyeSquint: 30, lipsPursed: 25 },
  explorer_discovery: { smile: 90, eyebrowRaise: 70, eyesWide: 80 },
  storyteller_dramatic: { eyebrowRaise: 75, eyesWide: 55, smile: 40 },
  empath_caring: { smile: 60, eyebrowRaise: 20, warmGaze: 70 },
  coach_encouraging: { smile: 70, eyebrowRaise: 45, nod: 40 },
  artist_creative: { smile: 50, headTilt: 35, eyebrowRaise: 55 },
};

// Blendshape mapping from expression values to ARKit-compatible names
const EXPRESSION_TO_BLENDSHAPE = {
  smile: ['mouthSmileLeft', 'mouthSmileRight'],
  eyebrowRaise: ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'],
  eyeSquint: ['eyeSquintLeft', 'eyeSquintRight'],
  eyesWide: ['eyeWideLeft', 'eyeWideRight'],
  eyesClosed: ['eyeBlinkLeft', 'eyeBlinkRight'],
  cheekRaise: ['cheekSquintLeft', 'cheekSquintRight'],
  lipsPursed: ['mouthPucker'],
  headTilt: ['headTilt'],  // Custom - handled separately
  nod: ['headNod'],        // Custom - handled separately
  warmGaze: ['eyeLookInLeft', 'eyeLookInRight'],
  mouthOpen: ['jawOpen'],
};

// =============================================================================
// KELLY EXPRESSION BRIDGE
// =============================================================================

const KellyExpressionBridge = {
  // State
  isInitialized: false,
  currentArchetype: 'The Explorer',
  currentExpression: 'neutral',
  currentExpressionBlendshapes: {},
  
  // Transition settings
  transitionDuration: 300, // ms
  transitionInProgress: false,
  transitionStartTime: 0,
  fromBlendshapes: {},
  toBlendshapes: {},
  
  // Animation frame
  animationFrameId: null,
  
  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================
  
  /**
   * Initialize the expression bridge
   */
  init() {
    if (this.isInitialized) return this;
    
    // Initialize lip-sync if available
    if (window.KellyLipSync && !window.KellyLipSync.isInitialized) {
      window.KellyLipSync.init();
    }
    
    // Set default expression
    this.currentExpressionBlendshapes = this.expressionToBlendshapes(
      EXPRESSION_PRESETS.neutral
    );
    
    // Start update loop
    this.startUpdateLoop();
    
    this.isInitialized = true;
    console.log('[KellyExpressionBridge] Initialized');
    return this;
  },
  
  // ===========================================================================
  // ARCHETYPE & EXPRESSION CONTROL
  // ===========================================================================
  
  /**
   * Set the current archetype
   * @param {string} archetype - Archetype name (e.g., 'The Explorer')
   */
  setArchetype(archetype) {
    this.currentArchetype = archetype;
    console.log('[KellyExpressionBridge] Archetype set to:', archetype);
    return this;
  },
  
  /**
   * Set expression with smooth transition
   * @param {string} expressionName - Expression name or archetype-specific expression
   * @param {Object} options - Additional options
   */
  setExpression(expressionName, options = {}) {
    const duration = options.duration || this.transitionDuration;
    
    // Get expression preset
    let expression;
    
    // Check for archetype-specific expression
    const archetypeKey = this.currentArchetype.toLowerCase().replace('the ', '') + '_' + expressionName;
    if (EXPRESSION_PRESETS[archetypeKey]) {
      expression = EXPRESSION_PRESETS[archetypeKey];
    } else if (EXPRESSION_PRESETS[expressionName]) {
      expression = EXPRESSION_PRESETS[expressionName];
    } else {
      expression = EXPRESSION_PRESETS.neutral;
    }
    
    // Convert to blendshapes
    const targetBlendshapes = this.expressionToBlendshapes(expression);
    
    // Start transition
    this.fromBlendshapes = { ...this.currentExpressionBlendshapes };
    this.toBlendshapes = targetBlendshapes;
    this.transitionStartTime = performance.now();
    this.transitionDuration = duration;
    this.transitionInProgress = true;
    this.currentExpression = expressionName;
    
    console.log('[KellyExpressionBridge] Transitioning to:', expressionName);
    return this;
  },
  
  /**
   * Set custom expression blendshapes directly
   * @param {Object} blendshapes - Blendshape values
   * @param {number} duration - Transition duration in ms
   */
  setCustomExpression(blendshapes, duration = 300) {
    this.fromBlendshapes = { ...this.currentExpressionBlendshapes };
    this.toBlendshapes = blendshapes;
    this.transitionStartTime = performance.now();
    this.transitionDuration = duration;
    this.transitionInProgress = true;
    this.currentExpression = 'custom';
    return this;
  },
  
  // ===========================================================================
  // EXPRESSION CONVERSION
  // ===========================================================================
  
  /**
   * Convert expression values to ARKit blendshapes
   * @param {Object} expression - Expression preset values
   * @returns {Object} ARKit-compatible blendshapes
   */
  expressionToBlendshapes(expression) {
    const blendshapes = {};
    
    for (const [key, value] of Object.entries(expression)) {
      if (EXPRESSION_TO_BLENDSHAPE[key]) {
        const shapes = EXPRESSION_TO_BLENDSHAPE[key];
        for (const shape of shapes) {
          blendshapes[shape] = value;
        }
      }
    }
    
    return blendshapes;
  },
  
  // ===========================================================================
  // UPDATE LOOP
  // ===========================================================================
  
  /**
   * Start the update loop
   */
  startUpdateLoop() {
    const update = () => {
      this.update();
      this.animationFrameId = requestAnimationFrame(update);
    };
    update();
  },
  
  /**
   * Main update function - combines expressions with lip-sync
   */
  update() {
    // Update expression transition if in progress
    if (this.transitionInProgress) {
      const elapsed = performance.now() - this.transitionStartTime;
      const progress = Math.min(1, elapsed / this.transitionDuration);
      
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      
      // Interpolate blendshapes
      this.currentExpressionBlendshapes = this.interpolateBlendshapes(
        this.fromBlendshapes,
        this.toBlendshapes,
        eased
      );
      
      if (progress >= 1) {
        this.transitionInProgress = false;
      }
    }
    
    // Merge expression blendshapes with lip-sync
    const finalBlendshapes = this.mergeWithLipSync();
    
    // Send to Unity
    this.sendToUnity(finalBlendshapes);
    
    // Send to 2D avatar
    this.sendTo2D(finalBlendshapes);
  },
  
  /**
   * Interpolate between two blendshape states
   */
  interpolateBlendshapes(from, to, t) {
    const result = {};
    const allKeys = new Set([...Object.keys(from || {}), ...Object.keys(to || {})]);
    
    for (const key of allKeys) {
      const fromValue = (from && from[key]) || 0;
      const toValue = (to && to[key]) || 0;
      result[key] = fromValue + (toValue - fromValue) * t;
    }
    
    return result;
  },
  
  /**
   * Merge expression blendshapes with lip-sync blendshapes
   * Lip-sync takes priority for mouth shapes
   */
  mergeWithLipSync() {
    const merged = { ...this.currentExpressionBlendshapes };
    
    // Get lip-sync blendshapes
    if (window.KellyLipSync && window.KellyLipSync.isActive) {
      const lipSyncShapes = window.KellyLipSync.getBlendshapes();
      
      // Mouth-related shapes from lip-sync (full override)
      const mouthShapes = [
        'jawOpen', 'mouthOpen', 'mouthFunnel', 'mouthPucker',
        'mouthStretchLeft', 'mouthStretchRight',
        'mouthSmileLeft', 'mouthSmileRight',
        'mouthPressLeft', 'mouthPressRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight',
        'mouthLowerDownLeft', 'mouthLowerDownRight',
        'mouthClose',
      ];
      
      for (const shape of mouthShapes) {
        if (lipSyncShapes[shape] !== undefined) {
          // Blend: prioritize lip-sync but keep some expression influence on smiles
          if (shape.includes('Smile')) {
            // For smiles, blend lip-sync (70%) with expression (30%)
            merged[shape] = (lipSyncShapes[shape] * 0.7) + 
                           ((merged[shape] || 0) * 0.3);
          } else {
            merged[shape] = lipSyncShapes[shape];
          }
        }
      }
    }
    
    return merged;
  },
  
  // ===========================================================================
  // OUTPUT
  // ===========================================================================
  
  /**
   * Send blendshapes to Unity
   */
  sendToUnity(blendshapes) {
    if (!window.unityInstance) return;
    
    try {
      const json = JSON.stringify(blendshapes);
      window.unityInstance.SendMessage('kelly_fbx_v4', 'SetBlendshapes', json);
    } catch (e) {
      // Unity not loaded
    }
  },
  
  /**
   * Send to 2D avatar systems
   */
  sendTo2D(blendshapes) {
    // Update KellyPoseManager expression
    if (window.KellyPoseManager) {
      // Map expression to pose
      const poseMap = {
        'curious': 'curious',
        'thinking': 'curious',
        'explaining': 'explaining',
        'excited': 'celebrating',
        'celebrating': 'celebrating',
        'listening': 'listening',
        'wisdom': 'wisdom',
        'neutral': 'hello',
      };
      
      const pose = poseMap[this.currentExpression] || 'hello';
      if (window.KellyPoseManager.currentPose !== pose) {
        window.KellyPoseManager.setPose?.(pose);
      }
    }
    
    // Update kelly 2D avatar if available
    if (window.kelly2DAvatar) {
      // Set eye expression
      const eyeValue = (blendshapes.eyeWideLeft || 0) / 100;
      window.kelly2DAvatar.setEyeOpenness?.(eyeValue);
      
      // Set brow position
      const browValue = (blendshapes.browInnerUp || 0) / 100;
      window.kelly2DAvatar.setBrowPosition?.(browValue);
    }
  },
  
  // ===========================================================================
  // PHASE INTEGRATION
  // ===========================================================================
  
  /**
   * Set expression based on lesson phase
   * @param {string} phase - Phase name (welcome, q1, q2, q3, wisdom)
   */
  setPhaseExpression(phase) {
    const phaseExpressions = {
      'welcome': 'excited',
      'question': 'curious',
      'q1': 'curious',
      'q2': 'explaining',
      'q3': 'thinking',
      'wisdom': 'wisdom',
      'celebrating': 'celebrating',
    };
    
    const expression = phaseExpressions[phase] || 'neutral';
    this.setExpression(expression, { duration: 400 });
    return this;
  },
  
  /**
   * React to user action
   * @param {string} action - Action type (correct, incorrect, thinking, etc.)
   */
  reactToUser(action) {
    const reactions = {
      'correct': { expression: 'celebrating', duration: 500 },
      'incorrect': { expression: 'curious', duration: 400 },  // Encouraging
      'thinking': { expression: 'thinking', duration: 300 },
      'asking': { expression: 'listening', duration: 300 },
      'excited': { expression: 'excited', duration: 400 },
    };
    
    const reaction = reactions[action] || { expression: 'neutral', duration: 300 };
    this.setExpression(reaction.expression, { duration: reaction.duration });
    return this;
  },
  
  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================
  
  /**
   * Stop the expression bridge
   */
  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Reset to neutral
    this.currentExpressionBlendshapes = this.expressionToBlendshapes(
      EXPRESSION_PRESETS.neutral
    );
    
    console.log('[KellyExpressionBridge] Stopped');
    return this;
  },
  
  /**
   * Get current expression name
   */
  getCurrentExpression() {
    return this.currentExpression;
  },
  
  /**
   * Get current blendshapes
   */
  getBlendshapes() {
    return this.mergeWithLipSync();
  },
};

// =============================================================================
// EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyExpressionBridge = KellyExpressionBridge;
}

// Auto-init after DOM ready
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyExpressionBridge.init());
  } else {
    KellyExpressionBridge.init();
  }
}

