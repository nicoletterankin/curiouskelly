# Unity Integration Technical Implementation Plan

**Status:** Design Phase  
**Last Updated:** 2025-01-XX  
**Based On:** Current codebase analysis (`app/unity-bridge.js`, `app/script.js`, `public/app.html`)

---

## EXECUTIVE SUMMARY

This plan provides a complete technical implementation for integrating Unity WebGL avatar system with the Curious Kelly lesson player, supporting:
- ✅ Age-based character model switching (2-102 range)
- ✅ Lesson phase content playback (welcome, q1, q2, q3, wisdom)
- ✅ Real-time age/language changes mid-lesson
- ✅ Coordinated audio playback with lip-sync

---

## 1. UNITY LOADING APPROACH

### **RECOMMENDATION: Hybrid Approach**

**Primary:** Iframe-based loading (for isolation and security)  
**Fallback:** Direct canvas embed (for development/testing)

### Rationale

**Iframe Approach** (Recommended):
- ✅ **Isolation:** Unity WebGL runs in separate context, preventing conflicts
- ✅ **Security:** postMessage with origin validation
- ✅ **Performance:** Can be lazy-loaded, doesn't block main thread
- ✅ **Error Recovery:** Iframe failure doesn't crash main app
- ✅ **Already Implemented:** `app/index.html` uses iframe pattern

**Direct Canvas** (Fallback):
- ✅ **Lower Latency:** Direct communication, no postMessage overhead
- ✅ **Simpler Debugging:** Same-origin, easier to inspect
- ⚠️ **Risk:** Can conflict with main app JavaScript
- ⚠️ **Current State:** `public/app.html` uses this (needs migration)

### Implementation Decision

**Use Iframe for Production:**
- File: `app/index.html` (already configured)
- File: `public/app.html` (needs migration to iframe)

**Keep Direct Canvas for Development:**
- Use for local testing/debugging
- Migrate to iframe before production deploy

---

## 2. FILE STRUCTURE

### Recommended Directory Structure

```
UI-TARS-desktop/
├── app/
│   ├── unity-bridge.js              ✅ EXISTS - Core bridge class
│   ├── unity-loader.js              🆕 NEW - Unity loading & initialization
│   ├── unity-asset-manager.js       🆕 NEW - Age → character model mapping
│   ├── unity-audio-coordinator.js   🆕 NEW - Audio URL calculation & sync
│   └── script.js                    ✅ EXISTS - Main app (needs integration)
│
├── public/
│   ├── unity/
│   │   ├── kelly-live/              ✅ EXISTS - Primary Unity build
│   │   │   ├── Build/              ✅ Complete WebGL build
│   │   │   └── index.html          ✅ Unity loader page
│   │   │
│   │   ├── kelly-v1/                ✅ EXISTS - Legacy build (keep for fallback)
│   │   │
│   │   └── character-models/        🆕 NEW - Age-based character assets
│   │       ├── age-2-5.glb          🆕 Map to age bucket 2-5
│   │       ├── age-6-12.glb         🆕 Map to age bucket 6-12
│   │       ├── age-13-17.glb         🆕 Map to age bucket 13-17
│   │       ├── age-18-35.glb         🆕 Map to age bucket 18-35
│   │       ├── age-36-60.glb         🆕 Map to age bucket 36-60
│   │       └── age-61-102.glb        🆕 Map to age bucket 61-102
│   │
│   └── lessons/
│       └── audio/                   ✅ EXISTS - Audio files
│           └── {lesson-slug}/
│               └── {ageBucket}-{language}-{phase}.mp3
│
└── docs/
    └── unity/
        ├── INTEGRATION_GUIDE.md     🆕 NEW - Developer guide
        ├── ASSET_MANAGEMENT.md      🆕 NEW - Character model specs
        └── AUDIO_PIPELINE.md        🆕 NEW - Audio sync documentation
```

### File Responsibilities

| File | Purpose | Status |
|------|---------|--------|
| `app/unity-bridge.js` | Core communication bridge (postMessage/WebSocket) | ✅ Exists |
| `app/unity-loader.js` | Unity WebGL initialization, iframe management | 🆕 New |
| `app/unity-asset-manager.js` | Age → character model mapping, asset loading | 🆕 New |
| `app/unity-audio-coordinator.js` | Audio URL calculation, playback coordination | 🆕 New |
| `app/script.js` | Main app logic (needs Unity integration hooks) | ✅ Exists (needs updates) |

---

## 3. COMMUNICATION PROTOCOL

### Transport Layer

**Primary:** `window.postMessage` (iframe communication)  
**Fallback:** WebSocket (`ws://localhost:7777` for native Unity clients)

### Message Envelope Structure

```typescript
interface UnityMessage {
  type: 'unity-bridge-event' | 'unity-bridge-command' | 'unity-bridge-handshake';
  event: string;
  payload: Record<string, any>;
  timestamp: string; // ISO 8601
}
```

### Event Catalog

#### **Outbound Events (Web → Unity)**

| Event | Trigger | Payload | Priority |
|-------|---------|---------|----------|
| `bridge-handshake` | Unity connects | `{ status: "acknowledged", transport: "postMessage" }` | ✅ Exists |
| `session-start` | Lesson selected | `{ mode, sessionId, lessonId, phase }` | ✅ Exists |
| `phase-progress` | Phase changes | `{ phase, sessionId, completedPhase? }` | ✅ Exists |
| `choice-selected` | User picks choice | `{ choiceId, currentPhase, nextPhase, sessionId }` | ✅ Exists |
| `session-complete` | Lesson finished | `{ lessonId, durationMin }` | ✅ Exists |
| `age-changed` | Age slider moves | `{ age, ageBucket, sessionId }` | 🆕 **NEW** |
| `language-changed` | Language selector changes | `{ language, sessionId, currentPhase }` | 🆕 **NEW** |
| `archetype-changed` | Vibe tuner changes | `{ archetype, traits, sessionId }` | 🆕 **NEW** |
| `audio-load` | Phase/age/language changes | `{ url, phase, ageBucket, language, sessionId }` | 🆕 **NEW** |
| `character-load` | Age changes | `{ modelUrl, ageBucket, sessionId }` | 🆕 **NEW** |

#### **Inbound Events (Unity → Web)**

| Event | Purpose | Payload | Status |
|-------|---------|---------|--------|
| `unity-bridge-handshake` | Unity ready | `{ version, transport }` | ✅ Exists |
| `ping` | Health check | `{}` | ✅ Exists |
| `state-update` | Telemetry | `{ fps?, pose?, latency? }` | ✅ Exists |
| `audio-ready` | Audio loaded | `{ url, duration }` | 🆕 **NEW** |
| `character-loaded` | Model loaded | `{ modelUrl, ageBucket }` | 🆕 **NEW** |
| `lip-sync-ready` | Lip-sync data ready | `{ phase, url }` | 🆕 **NEW** |
| `playback-started` | Audio playback started | `{ url, phase }` | 🆕 **NEW** |
| `playback-complete` | Audio finished | `{ url, phase }` | 🆕 **NEW** |
| `error` | Unity error | `{ message, code, context }` | ✅ Exists |

### Event Flow Examples

#### **Age Change Flow**
```
User moves age slider (25 → 35)
  ↓
script.js: ageSlider.addEventListener('input')
  ↓
StateManager.setState({ age: 35, ageBucket: '18-35' })
  ↓
unity-asset-manager.js: calculateCharacterModel(35)
  ↓
unityBridge.emit('age-changed', { age: 35, ageBucket: '18-35', sessionId })
  ↓
unityBridge.emit('character-load', { modelUrl: '/unity/character-models/age-18-35.glb', ... })
  ↓
Unity receives → loads new model → emits 'character-loaded'
```

#### **Phase Change Flow**
```
User selects choice → phase changes (practice → wisdom)
  ↓
script.js: handleChoiceSelection() → stateManager.setState({ currentPhase: 'wisdom' })
  ↓
unity-audio-coordinator.js: calculateAudioUrl(state, 'wisdom')
  ↓
unityBridge.emit('phase-progress', { phase: 'wisdom', sessionId })
  ↓
unityBridge.emit('audio-load', { url: '/lessons/audio/the-sun/18-35-en-wisdomMoment.mp3', ... })
  ↓
Unity receives → loads audio → emits 'audio-ready' → starts playback → emits 'playback-started'
```

---

## 4. ASSET MANAGEMENT

### Age → Character Model Mapping

**Location:** `app/unity-asset-manager.js` (new file)

### Mapping Strategy

```javascript
const AGE_TO_MODEL_MAP = {
  '2-5': {
    modelUrl: '/unity/character-models/age-2-5.glb',
    fallbackUrl: '/unity/character-models/age-6-12.glb', // If missing
    voicePitch: 1.2,  // Higher pitch for younger
    animationSpeed: 1.1
  },
  '6-12': {
    modelUrl: '/unity/character-models/age-6-12.glb',
    fallbackUrl: '/unity/character-models/age-13-17.glb',
    voicePitch: 1.1,
    animationSpeed: 1.05
  },
  '13-17': {
    modelUrl: '/unity/character-models/age-13-17.glb',
    fallbackUrl: '/unity/character-models/age-18-35.glb',
    voicePitch: 1.0,
    animationSpeed: 1.0
  },
  '18-35': {
    modelUrl: '/unity/character-models/age-18-35.glb',
    fallbackUrl: '/unity/character-models/age-36-60.glb',
    voicePitch: 0.95,
    animationSpeed: 0.98
  },
  '36-60': {
    modelUrl: '/unity/character-models/age-36-60.glb',
    fallbackUrl: '/unity/character-models/age-18-35.glb',
    voicePitch: 0.9,
    animationSpeed: 0.95
  },
  '61-102': {
    modelUrl: '/unity/character-models/age-61-102.glb',
    fallbackUrl: '/unity/character-models/age-36-60.glb',
    voicePitch: 0.85,
    animationSpeed: 0.9
  }
};
```

### Character Model Loading Logic

```javascript
class UnityAssetManager {
  constructor(unityBridge) {
    this.unityBridge = unityBridge;
    this.currentModel = null;
    this.loadedModels = new Map(); // Cache loaded models
  }

  getModelForAge(age) {
    const bucket = this.getAgeBucket(age);
    return AGE_TO_MODEL_MAP[bucket];
  }

  async loadCharacterModel(ageBucket, sessionId) {
    const config = AGE_TO_MODEL_MAP[ageBucket];
    if (!config) {
      console.error(`No model config for age bucket: ${ageBucket}`);
      return null;
    }

    // Check cache
    if (this.loadedModels.has(ageBucket)) {
      return this.loadedModels.get(ageBucket);
    }

    // Emit load event to Unity
    this.unityBridge.emit('character-load', {
      modelUrl: config.modelUrl,
      fallbackUrl: config.fallbackUrl,
      ageBucket,
      voicePitch: config.voicePitch,
      animationSpeed: config.animationSpeed,
      sessionId
    });

    // Wait for Unity confirmation (via event listener)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Character model load timeout: ${ageBucket}`));
      }, 10000);

      const handler = (event) => {
        if (event.event === 'character-loaded' && 
            event.payload.ageBucket === ageBucket) {
          clearTimeout(timeout);
          this.unityBridge.removeListener(handler);
          this.loadedModels.set(ageBucket, config);
          this.currentModel = ageBucket;
          resolve(config);
        }
      };

      this.unityBridge.on('character-loaded', handler);
    });
  }

  getAgeBucket(age) {
    if (age >= 2 && age <= 5) return '2-5';
    if (age >= 6 && age <= 12) return '6-12';
    if (age >= 13 && age <= 17) return '13-17';
    if (age >= 18 && age <= 35) return '18-35';
    if (age >= 36 && age <= 60) return '36-60';
    if (age >= 61 && age <= 102) return '61-102';
    return '18-35'; // Default
  }
}
```

### Integration Points

**File:** `app/script.js`

```javascript
// In UnifiedLessonApp constructor:
this.assetManager = new UnityAssetManager(this.unityBridge);

// In age slider handler (line ~163):
this.elements.ageSlider?.addEventListener('input', async (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  this.stateManager.setState({ age: value, ageBucket: bucket });
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  
  // NEW: Load character model
  const state = this.stateManager.getState();
  if (state.sessionId) {
    this.unityBridge.emit('age-changed', {
      age: value,
      ageBucket: bucket,
      sessionId: state.sessionId
    });
    
    await this.assetManager.loadCharacterModel(bucket, state.sessionId);
  }
});
```

---

## 5. AUDIO PIPELINE

### Audio URL Calculation

**Location:** `app/unity-audio-coordinator.js` (new file)

### Phase → Audio File Mapping

```javascript
const PHASE_TO_AUDIO_PHASE = {
  'welcome': 'welcome',
  'teaching': 'mainContent',  // q1 maps to mainContent
  'practice': 'mainContent',   // q2, q3 map to mainContent
  'wisdom': 'wisdomMoment'
};
```

### Audio URL Structure

```
/lessons/audio/{lesson-slug}/{ageBucket}-{language}-{audioPhase}.mp3

Examples:
/lessons/audio/the-sun/18-35-en-welcome.mp3
/lessons/audio/the-sun/18-35-en-mainContent.mp3
/lessons/audio/the-sun/18-35-en-wisdomMoment.mp3
/lessons/audio/the-sun/18-35-es-welcome.mp3
/lessons/audio/the-sun/18-35-fr-welcome.mp3
```

### Audio Coordinator Implementation

```javascript
class UnityAudioCoordinator {
  constructor(unityBridge) {
    this.unityBridge = unityBridge;
    this.currentAudio = null;
    this.audioCache = new Map(); // Cache loaded audio URLs
  }

  calculateAudioUrl(state, phase) {
    if (!state.selectedLesson) return null;

    // Get lesson slug
    const lessonSlug = state.selectedLesson.slug || 
                       this.slugify(state.selectedLesson.topic || state.selectedLesson.title);
    
    // Map phase to audio phase name
    const audioPhase = PHASE_TO_AUDIO_PHASE[phase] || 'mainContent';
    
    // Build URL
    const url = `/lessons/audio/${lessonSlug}/${state.ageBucket}-${state.language}-${audioPhase}.mp3`;
    
    return url;
  }

  async loadAudio(state, phase) {
    const url = this.calculateAudioUrl(state, phase);
    if (!url) {
      console.warn('Cannot calculate audio URL', { state, phase });
      return null;
    }

    // Check cache
    if (this.audioCache.has(url)) {
      return this.audioCache.get(url);
    }

    // Emit load event to Unity
    this.unityBridge.emit('audio-load', {
      url,
      phase,
      ageBucket: state.ageBucket,
      language: state.language,
      sessionId: state.sessionId
    });

    // Wait for Unity confirmation
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Audio load timeout: ${url}`));
      }, 10000);

      const handler = (event) => {
        if (event.event === 'audio-ready' && 
            event.payload.url === url) {
          clearTimeout(timeout);
          this.unityBridge.removeListener(handler);
          this.audioCache.set(url, event.payload);
          this.currentAudio = url;
          resolve(event.payload);
        }
      };

      this.unityBridge.on('audio-ready', handler);
    });
  }

  slugify(text) {
    return text.toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .trim();
  }

  // Handle real-time changes
  async updateAudioForStateChange(state, changeType) {
    const currentPhase = state.currentPhase;
    const url = this.calculateAudioUrl(state, currentPhase);
    
    if (changeType === 'language' || changeType === 'age') {
      // Reload audio with new language/age
      await this.loadAudio(state, currentPhase);
    }
  }
}
```

### Integration Points

**File:** `app/script.js`

```javascript
// In UnifiedLessonApp constructor:
this.audioCoordinator = new UnityAudioCoordinator(this.unityBridge);

// In renderPhase() method (line ~496):
renderPhase(state) {
  const normalized = this.normalizePhase(state.currentPhase);
  this.elements.phasePill.textContent = this.formatText(normalized);
  
  // Emit phase progress
  this.unityBridge.emit('phase-progress', {
    phase: normalized,
    sessionId: state.sessionId,
    lessonId: state.sessionLessonId,
  });

  // NEW: Load audio for phase
  if (state.sessionId && state.selectedLesson) {
    this.audioCoordinator.loadAudio(state, normalized);
  }

  // ... rest of render logic
}

// In language selector handler (line ~184):
this.elements.languageSelector?.addEventListener('change', async (event) => {
  const language = event.target.value;
  this.stateManager.setState({ language });
  this.elements.sessionStatus.textContent = `Language set to ${language.toUpperCase()}`;
  
  // NEW: Update audio
  const state = this.stateManager.getState();
  if (state.sessionId) {
    this.unityBridge.emit('language-changed', {
      language,
      sessionId: state.sessionId,
      currentPhase: state.currentPhase
    });
    
    await this.audioCoordinator.updateAudioForStateChange(state, 'language');
  }
});
```

### Lip-Sync Coordination

**Strategy:** Unity receives audio URL and loads corresponding lip-sync data file

```
Audio URL: /lessons/audio/the-sun/18-35-en-welcome.mp3
Lip-Sync:  /lessons/audio/the-sun/18-35-en-welcome.json (or .dat)
```

Unity should:
1. Receive `audio-load` event
2. Load audio file
3. Load corresponding lip-sync data (if exists)
4. Emit `lip-sync-ready` when ready
5. Start synchronized playback
6. Emit `playback-started` when audio begins
7. Emit `playback-complete` when audio ends

---

## 6. ERROR HANDLING

### Error Scenarios & Responses

#### **A. Unity Build Fails to Load**

**Scenario:** Unity WebGL build fails to initialize  
**Detection:** `script.onerror` or `createUnityInstance` rejection  
**Response:**

```javascript
// File: app/unity-loader.js
class UnityLoader {
  async loadUnityBuild(config) {
    try {
      const script = document.createElement('script');
      script.src = config.loaderUrl;
      
      return new Promise((resolve, reject) => {
        script.onerror = () => {
          this.handleLoadError('script_load_failed', config);
          reject(new Error('Unity script failed to load'));
        };
        
        script.onload = () => {
          if (typeof createUnityInstance === 'undefined') {
            this.handleLoadError('unity_instance_undefined', config);
            reject(new Error('Unity instance creator not found'));
            return;
          }
          
          createUnityInstance(config.canvas, config.unityConfig, (progress) => {
            // Progress callback
          })
          .then((instance) => {
            resolve(instance);
          })
          .catch((error) => {
            this.handleLoadError('unity_instance_failed', config, error);
            reject(error);
          });
        };
        
        document.body.appendChild(script);
      });
    } catch (error) {
      this.handleLoadError('unity_load_exception', config, error);
      throw error;
    }
  }

  handleLoadError(type, config, error = null) {
    console.error(`[UnityLoader] ${type}:`, error);
    
    // Show user-friendly error UI
    this.showErrorUI({
      title: 'Avatar System Unavailable',
      message: 'Kelly\'s avatar is temporarily offline. You can still learn with text and audio.',
      actions: [
        { label: 'Retry', action: () => this.retryLoad() },
        { label: 'Continue Without Avatar', action: () => this.disableUnity() }
      ]
    });
    
    // Emit error event for analytics
    if (window.gtag) {
      gtag('event', 'unity_load_error', {
        error_type: type,
        error_message: error?.message || 'unknown'
      });
    }
  }

  showErrorUI(errorInfo) {
    const overlay = document.getElementById('unity-overlay');
    if (overlay) {
      overlay.innerHTML = `
        <div class="unity-error">
          <div class="error-icon">⚠️</div>
          <h3>${errorInfo.title}</h3>
          <p>${errorInfo.message}</p>
          <div class="error-actions">
            ${errorInfo.actions.map(action => 
              `<button class="error-action-btn" onclick="${action.action}">${action.label}</button>`
            ).join('')}
          </div>
        </div>
      `;
      overlay.classList.remove('hidden');
    }
  }

  disableUnity() {
    this.stateManager?.setState({ unityEnabled: false });
    // Hide Unity container, show fallback UI
  }

  async retryLoad() {
    await this.loadUnityBuild(this.config);
  }
}
```

#### **B. Character Model Fails to Load**

**Scenario:** GLB file missing or corrupted  
**Detection:** Unity emits `error` event with `character-load` context  
**Response:**

```javascript
// File: app/unity-asset-manager.js
async loadCharacterModel(ageBucket, sessionId) {
  try {
    const config = AGE_TO_MODEL_MAP[ageBucket];
    
    // Emit load event
    this.unityBridge.emit('character-load', { ... });
    
    // Set timeout
    const timeout = setTimeout(() => {
      // Fallback to default model
      console.warn(`Character model timeout, using fallback for ${ageBucket}`);
      this.loadFallbackModel(ageBucket, sessionId);
    }, 10000);
    
    // Wait for confirmation
    const handler = (event) => {
      if (event.event === 'character-loaded') {
        clearTimeout(timeout);
        // Success
      } else if (event.event === 'error' && 
                 event.payload.context === 'character-load') {
        clearTimeout(timeout);
        // Use fallback
        this.loadFallbackModel(ageBucket, sessionId);
      }
    };
    
  } catch (error) {
    console.error('Character model load error:', error);
    this.loadFallbackModel(ageBucket, sessionId);
  }
}

loadFallbackModel(ageBucket, sessionId) {
  const config = AGE_TO_MODEL_MAP[ageBucket];
  if (config.fallbackUrl) {
    console.log(`Loading fallback model: ${config.fallbackUrl}`);
    this.unityBridge.emit('character-load', {
      modelUrl: config.fallbackUrl,
      ageBucket,
      sessionId,
      isFallback: true
    });
  } else {
    // Use default 18-35 model
    this.unityBridge.emit('character-load', {
      modelUrl: AGE_TO_MODEL_MAP['18-35'].modelUrl,
      ageBucket: '18-35',
      sessionId,
      isFallback: true,
      originalBucket: ageBucket
    });
  }
}
```

#### **C. Audio File Missing**

**Scenario:** Audio file doesn't exist for current phase/age/language  
**Detection:** Unity emits `error` event with `audio-load` context  
**Response:**

```javascript
// File: app/unity-audio-coordinator.js
async loadAudio(state, phase) {
  try {
    const url = this.calculateAudioUrl(state, phase);
    
    // Emit load event
    this.unityBridge.emit('audio-load', { url, ... });
    
    // Set timeout
    const timeout = setTimeout(() => {
      this.handleAudioError('timeout', url, state, phase);
    }, 10000);
    
    const handler = (event) => {
      if (event.event === 'audio-ready') {
        clearTimeout(timeout);
        // Success
      } else if (event.event === 'error' && 
                 event.payload.context === 'audio-load') {
        clearTimeout(timeout);
        this.handleAudioError('load_failed', url, state, phase);
      }
    };
    
  } catch (error) {
    this.handleAudioError('exception', null, state, phase, error);
  }
}

handleAudioError(type, url, state, phase, error = null) {
  console.warn(`Audio load error (${type}):`, url, error);
  
  // Try fallback strategies
  const fallbacks = [
    // 1. Try same age, default language (en)
    { ...state, language: 'en' },
    // 2. Try default age (18-35), same language
    { ...state, ageBucket: '18-35' },
    // 3. Try default age and language
    { ...state, ageBucket: '18-35', language: 'en' }
  ];
  
  for (const fallbackState of fallbacks) {
    const fallbackUrl = this.calculateAudioUrl(fallbackState, phase);
    if (fallbackUrl !== url) {
      console.log(`Trying fallback audio: ${fallbackUrl}`);
      return this.loadAudio(fallbackState, phase);
    }
  }
  
  // If all fallbacks fail, show warning but continue
  this.elements.sessionStatus.textContent = 
    'Audio unavailable for this phase. Text content still available.';
}
```

#### **D. Bridge Connection Lost**

**Scenario:** Unity iframe crashes or connection drops  
**Detection:** No response to ping, WebSocket closes  
**Response:**

```javascript
// File: app/unity-bridge.js
// Add to existing class:

startHealthCheck() {
  this.healthCheckInterval = setInterval(() => {
    if (!this.hasActiveTransport()) {
      this.setStatus('Unity connection lost – reconnecting…');
      this.attemptReconnect();
      return;
    }
    
    // Ping Unity
    this.emit('ping', {});
    
    // Set timeout for pong
    this.pongTimeout = setTimeout(() => {
      console.warn('[UnityBridge] No pong received, connection may be lost');
      this.setStatus('Unity not responding – checking connection…');
      this.attemptReconnect();
    }, 3000);
  }, 10000); // Check every 10 seconds
}

handleMessage(event) {
  // ... existing code ...
  
  if (data.type === 'unity-bridge-command' && data.event === 'pong') {
    // Clear pong timeout
    if (this.pongTimeout) {
      clearTimeout(this.pongTimeout);
      this.pongTimeout = null;
    }
  }
}

attemptReconnect() {
  // If iframe, reload it
  if (this.postTarget?.targetWindow) {
    const iframe = document.querySelector('#unity-iframe');
    if (iframe) {
      const src = iframe.src;
      iframe.src = ''; // Clear
      setTimeout(() => {
        iframe.src = src; // Reload
      }, 1000);
    }
  }
  
  // If WebSocket, reconnect
  if (this.wsUrl) {
    this.openWebSocket();
  }
}
```

---

## 7. IMPLEMENTATION CHECKLIST

### Phase 1: Core Infrastructure (Week 1)

- [ ] Create `app/unity-loader.js` - Unity initialization
- [ ] Create `app/unity-asset-manager.js` - Character model mapping
- [ ] Create `app/unity-audio-coordinator.js` - Audio URL calculation
- [ ] Extend `app/unity-bridge.js` - Add new events (age-changed, language-changed, etc.)
- [ ] Update `app/script.js` - Integrate new managers
- [ ] Add error handling to all components

### Phase 2: Age-Based Character Loading (Week 2)

- [ ] Create character model GLB files for each age bucket (or verify existing)
- [ ] Implement age → model mapping in `unity-asset-manager.js`
- [ ] Add age change handler in `script.js`
- [ ] Test character model switching
- [ ] Add fallback logic for missing models

### Phase 3: Audio Integration (Week 2-3)

- [ ] Implement audio URL calculation in `unity-audio-coordinator.js`
- [ ] Add phase → audio phase mapping
- [ ] Integrate audio loading in phase change handlers
- [ ] Add language change handler
- [ ] Test audio playback coordination
- [ ] Add fallback logic for missing audio

### Phase 4: Real-Time Updates (Week 3)

- [ ] Implement mid-lesson age change
- [ ] Implement mid-lesson language change
- [ ] Implement mid-lesson archetype change
- [ ] Add state synchronization
- [ ] Test all real-time change scenarios

### Phase 5: Error Handling & Polish (Week 4)

- [ ] Implement all error scenarios
- [ ] Add user-friendly error messages
- [ ] Add retry logic
- [ ] Add fallback UI for Unity failures
- [ ] Performance optimization
- [ ] Testing & bug fixes

---

## 8. UNITY SIDE REQUIREMENTS

### Unity Script Requirements

Unity must implement listeners for:

1. **Character Loading:**
   - Listen for `character-load` event
   - Load GLB model from `modelUrl`
   - Apply `voicePitch` and `animationSpeed`
   - Emit `character-loaded` when complete

2. **Audio Loading:**
   - Listen for `audio-load` event
   - Load audio file from `url`
   - Load corresponding lip-sync data (if exists)
   - Emit `audio-ready` when loaded
   - Start playback → emit `playback-started`
   - On completion → emit `playback-complete`

3. **Phase Handling:**
   - Listen for `phase-progress` event
   - Update avatar animations based on phase
   - Coordinate with audio playback

4. **Real-Time Updates:**
   - Handle `age-changed` → switch character model
   - Handle `language-changed` → reload audio with new language
   - Handle `archetype-changed` → update personality/expressions

### Unity Message Handler Example (C#)

```csharp
// Unity C# script example
public class KellyBridge : MonoBehaviour
{
    void Start()
    {
        // Send handshake
        SendHandshake();
        
        // Listen for messages
        Application.ExternalCall("window.addEventListener", "message", "OnMessage");
    }
    
    void OnMessage(string messageJson)
    {
        var message = JsonUtility.FromJson<UnityMessage>(messageJson);
        
        if (message.type == "unity-bridge-event")
        {
            HandleEvent(message.event, message.payload);
        }
    }
    
    void HandleEvent(string eventName, Dictionary<string, object> payload)
    {
        switch (eventName)
        {
            case "character-load":
                LoadCharacter(payload["modelUrl"].ToString());
                break;
            case "audio-load":
                LoadAudio(payload["url"].ToString());
                break;
            case "age-changed":
                HandleAgeChange((int)payload["age"]);
                break;
            // ... etc
        }
    }
    
    void SendHandshake()
    {
        var handshake = new {
            type = "unity-bridge-handshake",
            event = "ready",
            version = "1.0.0",
            transport = "postMessage"
        };
        Application.ExternalCall("window.postMessage", JsonUtility.ToJson(handshake), "*");
    }
}
```

---

## 9. TESTING STRATEGY

### Unit Tests

- [ ] Test age bucket calculation
- [ ] Test audio URL generation
- [ ] Test character model mapping
- [ ] Test error fallback logic

### Integration Tests

- [ ] Test age slider → character model change
- [ ] Test language selector → audio reload
- [ ] Test phase change → audio load
- [ ] Test mid-lesson age change
- [ ] Test mid-lesson language change

### Error Scenario Tests

- [ ] Test Unity build load failure
- [ ] Test character model missing
- [ ] Test audio file missing
- [ ] Test bridge connection loss
- [ ] Test timeout scenarios

---

## 10. PERFORMANCE CONSIDERATIONS

### Optimization Strategies

1. **Model Caching:** Cache loaded character models in memory
2. **Audio Preloading:** Preload next phase audio during current phase
3. **Lazy Loading:** Load Unity only when needed
4. **Asset Bundling:** Bundle character models efficiently
5. **Compression:** Use compressed audio formats (MP3, OGG)

### Performance Targets

- Character model switch: < 2 seconds
- Audio load: < 1 second
- Phase transition: < 500ms
- Age change: < 3 seconds (including model load)

---

## END OF PLAN

**Next Steps:**
1. Review this plan with Unity team
2. Create new files (`unity-loader.js`, `unity-asset-manager.js`, `unity-audio-coordinator.js`)
3. Extend `unity-bridge.js` with new events
4. Update `script.js` with integration hooks
5. Implement Unity-side message handlers
6. Test end-to-end flow

**Questions?** See `docs/unity/INTEGRATION_GUIDE.md` (to be created)



