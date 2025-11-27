// Unity Bridge - Communicates with Kelly 3D Avatar

class UnityBridge {
  constructor(unityInstance) {
    this.unity = unityInstance;
    this.ready = false;
    
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
  setExpression(emotion, intensity = 0.7) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage('Kelly', 'SetExpression', JSON.stringify({
        emotion: emotion,
        intensity: intensity,
        duration: 2.0
      }));
    } catch (error) {
      console.warn('Unity SetExpression failed:', error);
    }
  }
  
  // Trigger gesture animation
  playGesture(gesture) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage('Kelly', 'PlayGesture', gesture);
    } catch (error) {
      console.warn('Unity PlayGesture failed:', error);
    }
  }
  
  // Start lip sync with audio
  startLipSync(audioUrl) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage('Kelly', 'StartLipSync', audioUrl);
    } catch (error) {
      console.warn('Unity StartLipSync failed:', error);
    }
  }
  
  // Stop current animation
  stopAnimation() {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage('Kelly', 'StopAnimation', '');
    } catch (error) {
      console.warn('Unity StopAnimation failed:', error);
    }
  }
  
  // Set Kelly's age appearance
  setAge(age) {
    if (!this.ready || !this.unity) return;
    
    try {
      this.unity.SendMessage('Kelly', 'SetAge', age.toString());
    } catch (error) {
      console.warn('Unity SetAge failed:', error);
    }
  }
  
  // Play full phase (expression + gesture + audio)
  async playPhase(phaseData) {
    if (!this.ready) {
      console.warn('Unity not ready, skipping 3D animation');
      return;
    }
    
    // Set expression
    this.setExpression(phaseData.emotion, phaseData.intensity);
    
    // Play gesture if specified
    if (phaseData.gesture) {
      setTimeout(() => this.playGesture(phaseData.gesture), 500);
    }
    
    // Start lip sync if audio URL provided
    if (phaseData.audioUrl) {
      this.startLipSync(phaseData.audioUrl);
    }
  }
}

// Export for use
window.UnityBridge = UnityBridge;

