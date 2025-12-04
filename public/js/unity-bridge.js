// Unity Bridge - Communicates with Kelly 3D Avatar
// GameObject: kelly_fbx_v4
// Script: KellyWebGLBridge.cs

class UnityBridge {
  constructor(unityInstance) {
    this.unity = unityInstance;
    this.ready = false;
    this.gameObjectName = 'kelly_fbx_v4';
    
    // Listen for Unity ready signal
    window.UnityReady = () => {
      this.ready = true;
      console.log('✅ Unity bridge connected');
    };
    
    // Try to detect if Unity is ready
    setTimeout(() => {
      if (this.unity && this.unity.SendMessage) {
        this.ready = true;
        console.log('✅ Unity bridge ready');
      }
    }, 1000);
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

// Export for use
window.UnityBridge = UnityBridge;
