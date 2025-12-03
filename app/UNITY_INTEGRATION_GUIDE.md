# Unity Integration Implementation Guide

**Quick Start:** How to integrate the new Unity components into your app.

---

## 1. SETUP (One-Time)

### Import New Modules

Add to `app/script.js` (at the top with other imports):

```javascript
import UnityLoader from './unity-loader.js';
import UnityAssetManager from './unity-asset-manager.js';
import UnityAudioCoordinator from './unity-audio-coordinator.js';
```

### Initialize in UnifiedLessonApp Constructor

```javascript
class UnifiedLessonApp {
  constructor() {
    // ... existing code ...
    
    this.unityBridge = new UnityBridge();
    
    // NEW: Initialize Unity components
    this.unityLoader = new UnityLoader({
      buildUrl: '/unity/kelly-live/Build',
      useIframe: true, // Use iframe for production
      canvasId: 'unity-canvas',
      iframeId: 'unity-iframe',
      onLoad: (instance) => {
        console.log('Unity loaded:', instance);
        this.setUnityStatus('Unity ready');
      },
      onError: (type, error) => {
        console.error('Unity load error:', type, error);
      },
      onProgress: (progress) => {
        console.log('Unity loading:', Math.round(progress * 100) + '%');
      },
    });
    
    this.assetManager = new UnityAssetManager(this.unityBridge);
    this.audioCoordinator = new UnityAudioCoordinator(this.unityBridge);
  }
}
```

---

## 2. LOAD UNITY ON APP INIT

### In `init()` method:

```javascript
async init() {
  this.cacheElements();
  this.bindEvents();
  this.setupStateSubscriptions();
  // ... existing code ...
  
  // NEW: Load Unity
  try {
    await this.unityLoader.load();
    this.setUnityStatus('Unity connected');
  } catch (error) {
    console.error('Unity initialization failed:', error);
    // Error UI is handled by UnityLoader
  }
}
```

---

## 3. INTEGRATE AGE SLIDER

### Update age slider handler (around line 163):

```javascript
this.elements.ageSlider?.addEventListener('input', async (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  this.stateManager.setState({ age: value, ageBucket: bucket });
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  
  // NEW: Load character model for new age
  const state = this.stateManager.getState();
  if (state.sessionId && this.unityBridge.hasActiveTransport()) {
    // Emit age change event
    this.unityBridge.emit('age-changed', {
      age: value,
      ageBucket: bucket,
      sessionId: state.sessionId,
    });
    
    // Load character model
    try {
      await this.assetManager.loadCharacterModel(bucket, state.sessionId);
      this.setUnityStatus(`Kelly updated to age ${value}`);
      
      // Preload adjacent models for smooth transitions
      this.assetManager.preloadAdjacentModels(bucket);
    } catch (error) {
      console.error('Character model load failed:', error);
      this.setUnityStatus('Character model unavailable - using default');
    }
    
    // Update audio for new age
    if (state.currentPhase) {
      await this.audioCoordinator.updateAudioForStateChange(state, 'age');
    }
  }
});
```

---

## 4. INTEGRATE LANGUAGE SELECTOR

### Update language selector handler (around line 184):

```javascript
this.elements.languageSelector?.addEventListener('change', async (event) => {
  const language = event.target.value;
  this.stateManager.setState({ language });
  this.elements.sessionStatus.textContent = `Language set to ${language.toUpperCase()}`;
  
  // NEW: Update Unity audio
  const state = this.stateManager.getState();
  if (state.sessionId && this.unityBridge.hasActiveTransport()) {
    // Emit language change event
    this.unityBridge.emit('language-changed', {
      language,
      sessionId: state.sessionId,
      currentPhase: state.currentPhase,
    });
    
    // Reload audio with new language
    try {
      await this.audioCoordinator.updateAudioForStateChange(state, 'language');
      this.setUnityStatus(`Audio updated to ${language.toUpperCase()}`);
    } catch (error) {
      console.error('Audio reload failed:', error);
    }
  }
});
```

---

## 5. INTEGRATE PHASE RENDERING

### Update `renderPhase()` method (around line 496):

```javascript
renderPhase(state) {
  const normalized = this.normalizePhase(state.currentPhase);
  this.elements.phasePill.textContent = this.formatText(normalized);
  
  // Emit phase progress (existing)
  this.unityBridge.emit('phase-progress', {
    phase: normalized,
    sessionId: state.sessionId,
    lessonId: state.sessionLessonId,
  });

  // NEW: Load audio for phase
  if (state.sessionId && state.selectedLesson && this.unityBridge.hasActiveTransport()) {
    this.audioCoordinator.loadAudio(state, normalized)
      .then((audioData) => {
        if (audioData) {
          console.log(`Audio loaded for phase ${normalized}:`, audioData.url);
          // Preload next phase audio
          this.audioCoordinator.preloadNextPhase(state, normalized);
        }
      })
      .catch((error) => {
        console.error('Audio load failed:', error);
      });
  }

  // ... rest of existing render logic ...
}
```

---

## 6. INTEGRATE ARCHETYPE CHANGE

### Update vibe tuner handler (around line 282):

```javascript
// In setupStateSubscriptions(), update the vibe change handler:

if (state.vibeCoords !== prev.vibeCoords) {
  const archetype = this.getArchetype(state.vibeCoords.x, state.vibeCoords.y);
  
  // Update UI Labels
  if (this.elements.archetypeName) this.elements.archetypeName.textContent = archetype.name;
  if (this.elements.archetypeTraits) this.elements.archetypeTraits.textContent = archetype.traits;
  
  // If archetype changed, reload DNA if lesson selected
  if (archetype.name !== state.currentArchetype) {
    this.stateManager.setState({ currentArchetype: archetype.name });
    
    // NEW: Emit archetype change to Unity
    if (state.sessionId && this.unityBridge.hasActiveTransport()) {
      this.unityBridge.emit('archetype-changed', {
        archetype: archetype.name,
        traits: archetype.traits,
        sessionId: state.sessionId,
      });
    }
    
    if (state.selectedLesson) {
      this.loadLessonDNA(state.selectedLesson);
    }
  }
}
```

---

## 7. HANDLE AUDIO PLAYBACK EVENTS

### Add event listeners for Unity audio events:

```javascript
// In init() or constructor:

// Listen for audio completion
window.addEventListener('unity-audio-complete', (event) => {
  const { phase } = event.detail;
  console.log(`Audio playback complete for phase: ${phase}`);
  
  // Optionally auto-advance phase or show next button
  // This depends on your UX design
});

// Listen for audio unavailable
window.addEventListener('unity-audio-unavailable', (event) => {
  const { phase } = event.detail;
  console.warn(`Audio unavailable for phase: ${phase}`);
  
  // Show user-friendly message
  this.elements.sessionStatus.textContent = 
    'Audio unavailable for this phase. Text content still available.';
});
```

---

## 8. ERROR HANDLING INTEGRATION

### Unity disabled event handler:

```javascript
// In init():

window.addEventListener('unity-disabled', () => {
  console.log('Unity disabled - continuing without avatar');
  this.elements.sessionStatus.textContent = 
    'Learning mode: Continuing without avatar. All content available.';
  
  // Optionally hide Unity-related UI elements
  const unityContainer = document.getElementById('unity-container');
  if (unityContainer) {
    unityContainer.style.display = 'none';
  }
});
```

---

## 9. COMPLETE INTEGRATION EXAMPLE

### Minimal working example:

```javascript
import StateManager from './state-manager.js';
import SessionClient from './session-client.js';
import UnityBridge from './unity-bridge.js';
import UnityLoader from './unity-loader.js';
import UnityAssetManager from './unity-asset-manager.js';
import UnityAudioCoordinator from './unity-audio-coordinator.js';
import SupabaseService from './supabase-service.js';

class UnifiedLessonApp {
  constructor() {
    this.sessionClient = new SessionClient();
    this.unityBridge = new UnityBridge();
    
    // Initialize Unity components
    this.unityLoader = new UnityLoader({
      useIframe: true,
      onLoad: () => this.setUnityStatus('Unity ready'),
    });
    
    this.assetManager = new UnityAssetManager(this.unityBridge);
    this.audioCoordinator = new UnityAudioCoordinator(this.unityBridge);
    
    this.stateManager = new StateManager({
      age: 25,
      ageBucket: '18-35',
      language: 'en',
      // ... rest of state
    });
  }

  async init() {
    // ... existing init code ...
    
    // Load Unity
    try {
      await this.unityLoader.load();
    } catch (error) {
      console.error('Unity load failed:', error);
    }
  }

  // Age slider handler with Unity integration
  bindEvents() {
    this.elements.ageSlider?.addEventListener('input', async (event) => {
      const value = Number(event.target.value);
      const bucket = this.getBucketForAge(value);
      this.stateManager.setState({ age: value, ageBucket: bucket });
      
      const state = this.stateManager.getState();
      if (state.sessionId) {
        this.unityBridge.emit('age-changed', { age: value, ageBucket: bucket, sessionId: state.sessionId });
        await this.assetManager.loadCharacterModel(bucket, state.sessionId);
        await this.audioCoordinator.updateAudioForStateChange(state, 'age');
      }
    });
    
    // ... other event handlers ...
  }

  // Phase rendering with audio
  renderPhase(state) {
    const normalized = this.normalizePhase(state.currentPhase);
    this.unityBridge.emit('phase-progress', { phase: normalized, sessionId: state.sessionId });
    
    if (state.sessionId) {
      this.audioCoordinator.loadAudio(state, normalized);
    }
    
    // ... rest of render logic ...
  }
}
```

---

## 10. TESTING CHECKLIST

- [ ] Unity loads successfully (iframe or canvas)
- [ ] Age slider changes character model
- [ ] Language selector changes audio
- [ ] Phase changes load correct audio
- [ ] Archetype changes emit events
- [ ] Audio fallbacks work when files missing
- [ ] Character model fallbacks work
- [ ] Error UI shows when Unity fails
- [ ] Real-time age/language changes work mid-lesson
- [ ] Audio preloading works for next phase

---

## 11. TROUBLESHOOTING

### Unity not loading?
- Check browser console for errors
- Verify Unity build files exist in `/public/unity/kelly-live/Build/`
- Check iframe `src` attribute is set correctly
- Verify CORS settings if loading from different origin

### Character models not switching?
- Check `unity-asset-manager.js` age bucket mapping
- Verify GLB files exist in `/public/unity/character-models/`
- Check Unity console for model load errors
- Verify `character-load` event is being emitted

### Audio not loading?
- Check audio file paths match URL pattern
- Verify files exist: `/lessons/audio/{lesson-slug}/{ageBucket}-{language}-{phase}.mp3`
- Check browser network tab for 404 errors
- Verify `audio-load` event is being emitted

### Events not reaching Unity?
- Check Unity bridge connection status
- Verify `postMessage` is working (check browser console)
- Check Unity-side message listeners are set up
- Verify event names match Unity expectations

---

## END OF GUIDE

For detailed technical specifications, see `UNITY_INTEGRATION_PLAN.md`.











