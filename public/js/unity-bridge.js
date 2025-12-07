// Unity Bridge - Communicates with Kelly 3D Avatar
// GameObject: kelly_fbx_v4
// Script: KellyWebGLBridge.cs
// Updated with blendshape support for lip-sync integration
// Enhanced with Intelligent Director integration

class UnityBridge {
  constructor(unityInstance) {
    this.unity = unityInstance;
    this.ready = false;
    this.gameObjectName = 'kelly_fbx_v4';
    
    // Blendshape throttling (prevent overwhelming Unity with updates)
    this.lastBlendshapeUpdate = 0;
    this.blendshapeUpdateInterval = 33; // ~30fps
    
    // Expression state
    this.currentExpression = 'neutral';
    this.currentPhase = null;
    
    // Director integration
    this.useDirector = true;
    
    // Listen for Unity ready signal
    window.UnityReady = () => {
      this.ready = true;
      console.log('✅ Unity bridge connected');
      this.notifyDirectorOfReady();
    };
    
    // Try to detect if Unity is ready
    setTimeout(() => {
      if (this.unity && this.unity.SendMessage) {
        this.ready = true;
        console.log('✅ Unity bridge ready');
        this.notifyDirectorOfReady();
      }
    }, 1000);
    
    // Expose globally for other systems
    window.unityBridge = this;
  }
  
  // Notify director that Unity is ready
  notifyDirectorOfReady() {
    if (window.KellyDirector) {
      window.KellyDirector.unityBridge = this;
      console.log('✅ Unity bridge connected to Director');
    }
  }
  
  // Set Kelly's facial expression
  // Valid expressions: happy, curious, explaining, listening, wisdom, celebrating, neutral
  setExpression(expression) {
    if (!this.ready || !this.unity) return;
    
    try {
      // C# expects just the expression name as a string
      this.unity.SendMessage(this.gameObjectName, 'SetExpression', expression);
      console.log(`🎭 Unity SetExpression: ${expression}`);
    } catch (error) {
      console.warn('Unity SetExpression failed:', error);
    }
  }
  
  // Set phase context (maps to appropriate expression)
  // Valid phases: welcome, question, q1, q2, q3, wisdom, celebrating
  setPhase(phase) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'SetPhase', phase);
      console.log(`📖 Unity SetPhase: ${phase}`);
    } catch (error) {
      console.warn('Unity SetPhase failed:', error);
    }
  }
  
  // Trigger animation by name
  playAnimation(animationName) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'PlayAnimation', animationName);
      console.log(`🎬 Unity PlayAnimation: ${animationName}`);
    } catch (error) {
      console.warn('Unity PlayAnimation failed:', error);
    }
  }
  
  // Start lip sync with text (for mouth animation)
  // Note: Web handles actual audio via ElevenLabs
  startLipSync(text) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'StartLipSync', text);
      console.log(`👄 Unity StartLipSync: "${text.substring(0, 30)}..."`);
    } catch (error) {
      console.warn('Unity StartLipSync failed:', error);
    }
  }
  
  // Stop lip sync and close mouth
  stopLipSync() {
    if (!this.ready || !this.unity) return;
    
    try {
      // StopLipSync doesn't need a parameter but SendMessage requires one
      this.unity.SendMessage(this.gameObjectName, 'StopLipSync', '');
      console.log('🔇 Unity StopLipSync');
    } catch (error) {
      console.warn('Unity StopLipSync failed:', error);
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════
  // BLENDSHAPE LIP-SYNC METHODS
  // High-fidelity facial animation via blendshape values
  // ═══════════════════════════════════════════════════════════════════
  
  /**
   * Set facial blendshapes directly (for advanced lip-sync)
   * @param {Object} blendshapes - Object with blendshape names and values (0-100)
   * @example setBlendshapes({ jawOpen: 50, mouthSmileLeft: 30, mouthSmileRight: 30 })
   */
  setBlendshapes(blendshapes) {
    if (!this.ready || !this.unity) return;
    
    // Throttle updates
    const now = performance.now();
    if (now - this.lastBlendshapeUpdate < this.blendshapeUpdateInterval) {
      return;
    }
    this.lastBlendshapeUpdate = now;
    
    try {
      // Convert object to JSON for Unity
      const json = typeof blendshapes === 'string' ? blendshapes : JSON.stringify(blendshapes);
      this.unity.SendMessage(this.gameObjectName, 'SetBlendshapes', json);
    } catch (error) {
      // Fail silently - blendshapes are called frequently
    }
  }
  
  /**
   * Set a single blendshape value
   * @param {string} name - Blendshape name (e.g., 'jawOpen', 'mouthSmileLeft')
   * @param {number} value - Value from 0-100
   */
  setSingleBlendshape(name, value) {
    if (!this.ready || !this.unity) return;
    
    try {
      const data = JSON.stringify({ name, value: Math.max(0, Math.min(100, value)) });
      this.unity.SendMessage(this.gameObjectName, 'SetSingleBlendshape', data);
    } catch (error) {
      console.warn('Unity SetSingleBlendshape failed:', error);
    }
  }
  
  /**
   * Reset all blendshapes to default (neutral face)
   */
  resetBlendshapes() {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'ResetBlendshapes', '');
      console.log('🔄 Unity ResetBlendshapes');
    } catch (error) {
      console.warn('Unity ResetBlendshapes failed:', error);
    }
  }
  
  /**
   * Enable/disable real-time blendshape updates
   * @param {boolean} enabled - Whether to enable blendshape updates
   */
  setBlendshapesEnabled(enabled) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'SetBlendshapesEnabled', enabled ? 'true' : 'false');
      console.log(`🎭 Unity blendshapes ${enabled ? 'enabled' : 'disabled'}`);
    } catch (error) {
      console.warn('Unity SetBlendshapesEnabled failed:', error);
    }
  }
  
  // Set speaking state
  setSpeaking(speaking) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage(this.gameObjectName, 'SetSpeaking', speaking ? 'true' : 'false');
      console.log(`🗣️ Unity SetSpeaking: ${speaking}`);
    } catch (error) {
      console.warn('Unity SetSpeaking failed:', error);
    }
  }
  
  // Convenience: Play gesture (alias for playAnimation)
  playGesture(gesture) {
    this.playAnimation(gesture);
  }
  
  // Convenience: Stop animation (stop lip sync)
  stopAnimation() {
    this.stopLipSync();
  }
  
  // Play full phase (expression + animation + lip sync)
  async playPhase(phaseData) {
    if (!this.ready) {
      console.warn('Unity not ready, skipping 3D animation');
      return;
    }
    
    // Set phase context (this will set the appropriate expression)
    if (phaseData.phase) {
      this.setPhase(phaseData.phase);
    } else if (phaseData.emotion || phaseData.expression) {
      // Direct expression if no phase provided
      this.setExpression(phaseData.emotion || phaseData.expression);
    }
    
    // Play animation if specified
    if (phaseData.gesture || phaseData.animation) {
      setTimeout(() => this.playAnimation(phaseData.gesture || phaseData.animation), 500);
    }
    
    // Start lip sync if text provided
    if (phaseData.text || phaseData.audioUrl) {
      this.startLipSync(phaseData.text || phaseData.audioUrl);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
  // INTELLIGENT DIRECTOR INTEGRATION
  // ═══════════════════════════════════════════════════════════════════
  
  /**
   * Intelligently set expression based on text content
   * Uses the Intelligent Director if available
   * @param {string} text - Text to analyze for expression
   */
  intelligentExpression(text) {
    if (!this.ready || !this.unity) return;
    
    if (window.KellyDirector && this.useDirector) {
      // Let the Director analyze and decide
      const analysis = window.KellyDirector.analyzeAndDirect(text);
      console.log(`🧠 Intelligent expression: ${analysis.dominantExpression} (${(analysis.confidence * 100).toFixed(0)}%)`);
      return analysis;
    }
    
    // Fallback: simple expression based on punctuation
    if (text.includes('!')) {
      this.setExpression('excited');
    } else if (text.includes('?')) {
      this.setExpression('curious');
    } else {
      this.setExpression('explaining');
    }
  }
  
  /**
   * Perform text with intelligent direction
   * @param {string} text - Text to perform
   * @param {Object} options - Options for the performance
   */
  async perform(text, options = {}) {
    if (window.KellyPerformance) {
      return window.KellyPerformance.perform(text, options);
    }
    
    // Fallback: basic expression + lip sync
    this.intelligentExpression(text);
    this.startLipSync(text);
    
    // Wait for approximate speaking duration
    const words = text.split(/\s+/).length;
    const duration = (words / 2.5) * 1000;
    await new Promise(resolve => setTimeout(resolve, duration));
    
    this.stopLipSync();
  }
  
  /**
   * React to user action with appropriate expression
   * @param {string} action - Action type (correct, incorrect, timeout, etc.)
   */
  reactToUser(action) {
    if (window.KellyDirector) {
      window.KellyDirector.reactToUser(action);
      return;
    }
    
    // Fallback reactions
    const reactions = {
      correct: 'celebrating',
      incorrect: 'encouraging',
      timeout: 'curious',
      skip: 'thinking',
      start: 'excited',
      complete: 'celebrating',
    };
    
    this.setExpression(reactions[action] || 'neutral');
  }
  
  /**
   * Enable/disable director integration
   * @param {boolean} enabled - Whether to use the director
   */
  setDirectorEnabled(enabled) {
    this.useDirector = enabled;
    console.log(`🎬 Director integration: ${enabled ? 'enabled' : 'disabled'}`);
  }
}

// Export for use
window.UnityBridge = UnityBridge;
