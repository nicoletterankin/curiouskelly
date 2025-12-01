# 🎭 Kelly Avatar System — Complete Architecture

## 2D & 3D Avatar Integration for Millions of Daily Learners

**Version:** 2.0  
**Date:** November 28, 2025  
**Status:** Architecture Specification + Implementation Plan  
**Impact:** Millions of students daily curriculum experience

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [2D Avatar Player Architecture](#2-2d-avatar-player-architecture)
3. [3D Unity Avatar Architecture](#3-3d-unity-avatar-architecture)
4. [2D/3D Switch System](#4-2d3d-switch-system)
5. [Integration with learn.html](#5-integration-with-learnhtml)
6. [Performance Requirements](#6-performance-requirements)
7. [Test Scenarios](#7-test-scenarios)
8. [Content Production Requirements](#8-content-production-requirements)
9. [Scale Considerations](#9-scale-considerations)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. System Overview

### The Kelly Avatar Experience

Kelly is the heart of the learning experience. She must:

- **Feel alive** — Breathe, blink, react instantly
- **Match the learner** — Age-appropriate appearance
- **Speak naturally** — Lip-sync for 3D, speaking indicators for 2D
- **Work everywhere** — From low-end phones to high-end desktops

### Two Avatar Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                    KELLY AVATAR SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   MODE 1: 2D AVATAR (DEFAULT)                                  │
│   ───────────────────────────                                   │
│   • PNG images with CSS animations                             │
│   • 5 expressions × 6 ages = 30 images                         │
│   • ~6MB total asset size                                      │
│   • Works on ALL devices                                       │
│   • No GPU required                                            │
│   • Instant load (<2 seconds on 3G)                           │
│                                                                 │
│   MODE 2: 3D AVATAR (PREMIUM/OPTIONAL)                         │
│   ─────────────────────────────────────                        │
│   • Unity WebGL build                                          │
│   • Full lip-sync via visemes                                  │
│   • Real-time 3D rendering                                     │
│   • ~40MB build size                                           │
│   • Requires WebGL-capable device                             │
│   • 5-30 second load time                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   USER PREFERENCE STORAGE                                       │
│   └── localStorage: 'kelly_mode' = '2D' | '3D'                │
│   └── Default: '2D' (progressive enhancement)                  │
│   └── Premium users: Can unlock 3D                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### File Inventory

```
EXISTING ASSETS:
├── 2D Avatar Images
│   └── public/images/kelly/
│       ├── kelly-directors-chair-curious.png
│       ├── kelly-directors-chair-explaining.png
│       ├── kelly-directors-chair-listening.png
│       ├── kelly-directors-chair-wisdom.png
│       ├── kelly-directors-chair-celebrating.png
│       ├── kelly-chair-curious.png (alternate)
│       ├── kelly-chair-explaining.png (alternate)
│       ├── kelly-chair-listening.png (alternate)
│       ├── kelly-chair-wisdom.png (alternate)
│       └── kelly-chair-celebrating.png (alternate)
│
├── 2D Avatar Code
│   └── daily-lesson-marketing/public/lesson-player/
│       ├── js/kelly-avatar-system.js (comprehensive)
│       ├── js/kelly-2d-avatar.js (simple)
│       └── css/kelly-avatar-animations.css
│
├── 3D Avatar Build
│   └── digital-kelly/engines/Kelly_Engine_V2/onlykelly/
│       └── Kelly_Web_Build/
│           ├── Build/
│           │   ├── Kelly_Web_Build.loader.js
│           │   ├── Kelly_Web_Build.data.unityweb (~30MB)
│           │   ├── Kelly_Web_Build.framework.js.unityweb
│           │   └── Kelly_Web_Build.wasm.unityweb
│           └── index.html (standalone test)
│
└── Unity Integration Code
    └── app/
        ├── unity-loader.js
        ├── unity-bridge.js
        ├── unity-asset-manager.js
        └── unity-audio-coordinator.js
```

---

## 2. 2D Avatar Player Architecture

### 2.1 Expression State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                  2D AVATAR STATE MACHINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PHASE              EXPRESSION        IMAGE FILE               │
│   ─────              ──────────        ──────────               │
│   Welcome      →     curious      →    kelly-*-curious.png     │
│                                                                 │
│   Q1 (asking)  →     curious      →    kelly-*-curious.png     │
│   Q1 (choice A)→     explaining   →    kelly-*-explaining.png  │
│   Q1 (choice B)→     celebrating  →    kelly-*-celebrating.png │
│   Q1 (choice C)→     wisdom       →    kelly-*-wisdom.png      │
│                                                                 │
│   Q2 (asking)  →     curious      →    kelly-*-curious.png     │
│   Q2 (choice A)→     explaining   →    kelly-*-explaining.png  │
│   Q2 (choice B)→     celebrating  →    kelly-*-celebrating.png │
│   Q2 (choice C)→     wisdom       →    kelly-*-wisdom.png      │
│                                                                 │
│   Q3 (asking)  →     listening    →    kelly-*-listening.png   │
│   Q3 (choice A)→     explaining   →    kelly-*-explaining.png  │
│   Q3 (choice B)→     celebrating  →    kelly-*-celebrating.png │
│   Q3 (choice C)→     wisdom       →    kelly-*-wisdom.png      │
│                                                                 │
│   Wisdom       →     wisdom       →    kelly-*-wisdom.png      │
│   Complete     →     celebrating  →    kelly-*-celebrating.png │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

* = "directors-chair" or "chair" variant
```

### 2.2 CSS Animation System

```css
/* Core animations applied to .kelly-avatar */

/* Breathing - Constant subtle movement */
@keyframes kelly-breathe {
  0%,
  100% {
    transform: scale(1) translateY(0);
  }
  50% {
    transform: scale(1.008) translateY(-3px);
  }
}
.kelly-avatar {
  animation: kelly-breathe 4s ease-in-out infinite;
}

/* Speaking indicator - When audio plays */
@keyframes kelly-speaking {
  0%,
  100% {
    filter: brightness(1);
  }
  50% {
    filter: brightness(1.05);
  }
}
.kelly-avatar.speaking {
  animation: kelly-speaking 0.5s ease-in-out infinite;
}

/* Expression transition - Crossfade between images */
.kelly-avatar {
  transition: opacity 0.4s ease;
}
.kelly-avatar.transitioning {
  opacity: 0.7;
  transform: scale(0.98);
}

/* Celebration effect - When choice is made */
@keyframes kelly-celebrate {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.02) translateY(-5px);
  }
  100% {
    transform: scale(1);
  }
}
.kelly-avatar.celebrating {
  animation: kelly-celebrate 0.6s ease;
}

/* Mobile optimizations */
@media (prefers-reduced-motion: reduce) {
  .kelly-avatar {
    animation: none;
  }
}
```

### 2.3 Kelly 2D Avatar Class

```javascript
/**
 * Kelly2DAvatarPlayer
 * Manages 2D Kelly avatar with expression switching and animations
 */
class Kelly2DAvatarPlayer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      imageSet: options.imageSet || 'directors-chair', // 'directors-chair' | 'chair'
      basePath: options.basePath || '/images/kelly/',
      preload: options.preload !== false,
      enableBreathing: options.enableBreathing !== false,
      enableSpeakingIndicator: options.enableSpeakingIndicator !== false,
      ...options
    };

    this.state = {
      expression: 'curious',
      isSpeaking: false,
      isTransitioning: false,
      age: '18-35' // For future age-variant support
    };

    this.expressions = ['curious', 'explaining', 'listening', 'wisdom', 'celebrating'];
    this.imageCache = new Map();

    this.init();
  }

  init() {
    this.createDOM();
    if (this.options.preload) {
      this.preloadImages();
    }
  }

  createDOM() {
    this.container.innerHTML = `
            <div class="kelly-2d-container">
                <img 
                    class="kelly-avatar ${this.options.enableBreathing ? 'breathing' : ''}"
                    id="kelly-avatar"
                    src="${this.getImagePath('curious')}"
                    alt="Kelly"
                />
                <div class="kelly-speaking-ring" id="kelly-speaking-ring"></div>
            </div>
        `;

    this.avatarImg = this.container.querySelector('#kelly-avatar');
    this.speakingRing = this.container.querySelector('#kelly-speaking-ring');
  }

  getImagePath(expression) {
    return `${this.options.basePath}kelly-${this.options.imageSet}-${expression}.png`;
  }

  async preloadImages() {
    const promises = this.expressions.map((expr) => {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          this.imageCache.set(expr, img);
          resolve();
        };
        img.onerror = reject;
        img.src = this.getImagePath(expr);
      });
    });

    try {
      await Promise.all(promises);
      console.log('[Kelly2D] All images preloaded');
    } catch (e) {
      console.warn('[Kelly2D] Some images failed to preload');
    }
  }

  /**
   * Set Kelly's expression
   */
  async setExpression(expression) {
    if (!this.expressions.includes(expression)) {
      console.warn(`[Kelly2D] Unknown expression: ${expression}`);
      return;
    }

    if (this.state.expression === expression) return;
    if (this.state.isTransitioning) return;

    this.state.isTransitioning = true;
    this.avatarImg.classList.add('transitioning');

    // Wait for fade out
    await this.wait(200);

    // Switch image
    this.avatarImg.src = this.getImagePath(expression);
    this.state.expression = expression;

    // Wait for image load
    await this.waitForImageLoad(this.avatarImg);

    // Fade in
    this.avatarImg.classList.remove('transitioning');

    await this.wait(200);
    this.state.isTransitioning = false;

    // Dispatch event
    this.dispatchEvent('expression-changed', { expression });
  }

  /**
   * Set speaking state (for audio sync)
   */
  setSpeaking(speaking) {
    this.state.isSpeaking = speaking;

    if (speaking) {
      this.avatarImg.classList.add('speaking');
      this.speakingRing?.classList.add('active');
    } else {
      this.avatarImg.classList.remove('speaking');
      this.speakingRing?.classList.remove('active');
    }

    this.dispatchEvent('speaking-changed', { speaking });
  }

  /**
   * Play celebration animation
   */
  celebrate() {
    this.avatarImg.classList.add('celebrating');
    setTimeout(() => {
      this.avatarImg.classList.remove('celebrating');
    }, 600);
  }

  /**
   * Set expression based on lesson phase
   */
  setPhase(phase, choice = null) {
    const expressionMap = {
      welcome: 'curious',
      q1: choice
        ? choice === 'a'
          ? 'explaining'
          : choice === 'b'
            ? 'celebrating'
            : 'wisdom'
        : 'curious',
      q2: choice
        ? choice === 'a'
          ? 'explaining'
          : choice === 'b'
            ? 'celebrating'
            : 'wisdom'
        : 'curious',
      q3: choice
        ? choice === 'a'
          ? 'explaining'
          : choice === 'b'
            ? 'celebrating'
            : 'wisdom'
        : 'listening',
      wisdom: 'wisdom',
      complete: 'celebrating'
    };

    const expression = expressionMap[phase] || 'curious';
    this.setExpression(expression);

    if (choice && (choice === 'b' || phase === 'complete')) {
      this.celebrate();
    }
  }

  // Utility methods
  wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  waitForImageLoad(img) {
    return new Promise((resolve) => {
      if (img.complete) resolve();
      else {
        img.onload = resolve;
        img.onerror = resolve;
      }
    });
  }

  dispatchEvent(name, detail) {
    document.dispatchEvent(new CustomEvent(`kelly-${name}`, { detail }));
  }

  destroy() {
    this.container.innerHTML = '';
    this.imageCache.clear();
  }
}
```

### 2.4 Age Variant Support (Future)

```
PLANNED: Kelly Age Variants
────────────────────────────

Current: Single adult Kelly (27-year-old appearance)

Future: 6 age variants matching learner
├── kelly-age-3.png   → 2-5 year learners (toddler Kelly)
├── kelly-age-9.png   → 6-12 year learners (kid Kelly)
├── kelly-age-15.png  → 13-17 year learners (teen Kelly)
├── kelly-age-27.png  → 18-35 year learners (adult Kelly) ← CURRENT
├── kelly-age-48.png  → 36-60 year learners (mature Kelly)
└── kelly-age-82.png  → 61+ year learners (elder Kelly)

Production Required:
- 6 age variants × 5 expressions = 30 images
- 30 images × 2 sets (directors-chair, chair) = 60 images
- Currently have: 10 images (2 sets × 5 expressions, adult only)
- Gap: 50 images needed for full age support
```

---

## 3. 3D Unity Avatar Architecture

### 3.1 Unity Build Structure

```
Kelly_Web_Build/
├── Build/
│   ├── Kelly_Web_Build.loader.js      ← Entry point (loads WASM)
│   ├── Kelly_Web_Build.data.unityweb  ← Assets (~30MB compressed)
│   ├── Kelly_Web_Build.framework.js.unityweb ← Unity runtime
│   └── Kelly_Web_Build.wasm.unityweb  ← WebAssembly binary
├── StreamingAssets/
│   └── aa/ ← Addressables (dynamic assets)
└── TemplateData/
    └── style.css ← Loading screen styles
```

### 3.2 Unity C# Controller

```csharp
// KellyAvatarController.cs - Existing implementation

public class KellyAvatarController : MonoBehaviour
{
    [Header("Face Configuration")]
    public SkinnedMeshRenderer faceMesh;
    public int visemeMultiplier = 100;

    // Viseme mapping for CC4/iClone blendshapes
    private Dictionary<string, string> visemeMap = new Dictionary<string, string>
    {
        {"sil", "V_Explosive"},    // Silence/B/P/M
        {"PP", "V_Explosive"},     // P sound
        {"FF", "V_Dental_Lip"},    // F/V sounds
        {"TH", "V_Tight_O"},       // TH sound
        {"DD", "V_Dental_Lip"},    // D/T sounds
        {"kk", "V_Tight_O"},       // K/G sounds
        {"CH", "V_Tight_O"},       // CH/J sounds
        {"SS", "V_Dental_Lip"},    // S/Z sounds
        {"nn", "V_Dental_Lip"},    // N sound
        {"RR", "V_Tight_O"},       // R sound
        {"aa", "V_Wide"},          // AH sound
        {"E", "V_Dental_Lip"},     // EE sound
        {"ih", "V_Wide"},          // IH sound
        {"oh", "V_Tight_O"},       // OH sound
        {"ou", "V_Tight_O"}        // OO sound
    };

    // Called by JavaScript via SendMessage
    public void ProcessViseme(string json)
    {
        string[] parts = json.Split(':');
        if (parts.Length == 2 && float.TryParse(parts[1], out float weight))
        {
            SetViseme(parts[0], weight);
        }
    }

    public void SetViseme(string visemeName, float weight)
    {
        if (faceMesh == null) return;
        if (!visemeMap.ContainsKey(visemeName)) return;

        string blendShapeName = visemeMap[visemeName];
        int index = faceMesh.sharedMesh.GetBlendShapeIndex(blendShapeName);

        if (index != -1)
        {
            faceMesh.SetBlendShapeWeight(index, weight * visemeMultiplier);
        }
    }

    // Expression control
    public void SetExpression(string expressionName)
    {
        // TODO: Implement expression blendshapes
        // Maps: curious, explaining, celebrating, listening, wisdom
    }

    // Animation control
    public void PlayAnimation(string animationName)
    {
        // TODO: Trigger Animator states
        // Maps: idle, talking, celebrating, thinking
    }
}
```

### 3.3 JavaScript Unity Bridge

```javascript
/**
 * UnityKellyBridge
 * Handles all JavaScript ↔ Unity communication
 */
class UnityKellyBridge {
  constructor() {
    this.unityInstance = null;
    this.ready = false;
    this.messageQueue = [];
    this.callbacks = new Map();

    // Listen for Unity ready signal
    window.addEventListener('message', this.handleUnityMessage.bind(this));
  }

  /**
   * Set Unity instance once loaded
   */
  setInstance(instance) {
    this.unityInstance = instance;
    this.ready = true;

    // Process queued messages
    this.messageQueue.forEach((msg) => this.sendToUnity(msg.method, msg.param));
    this.messageQueue = [];

    console.log('[UnityBridge] Connected to Unity');
  }

  /**
   * Send message to Unity
   */
  sendToUnity(methodName, parameter = '') {
    if (!this.ready) {
      this.messageQueue.push({ method: methodName, param: parameter });
      return;
    }

    try {
      this.unityInstance.SendMessage('KellyAvatar', methodName, parameter);
    } catch (e) {
      console.error('[UnityBridge] SendMessage failed:', e);
    }
  }

  /**
   * Start lip-sync with viseme data
   */
  startLipSync(visemeData) {
    // visemeData format: [{time: 0, viseme: 'aa', weight: 0.8}, ...]
    this.sendToUnity('StartLipSync', JSON.stringify(visemeData));
  }

  /**
   * Stop lip-sync
   */
  stopLipSync() {
    this.sendToUnity('StopLipSync');
  }

  /**
   * Set expression
   */
  setExpression(expression) {
    this.sendToUnity('SetExpression', expression);
  }

  /**
   * Play animation
   */
  playAnimation(animationName) {
    this.sendToUnity('PlayAnimation', animationName);
  }

  /**
   * Handle messages from Unity
   */
  handleUnityMessage(event) {
    if (event.data.type !== 'unity-message') return;

    const { action, payload } = event.data;

    switch (action) {
      case 'ready':
        this.dispatchEvent('unity-ready');
        break;
      case 'animation-complete':
        this.dispatchEvent('animation-complete', payload);
        break;
      case 'lipsync-complete':
        this.dispatchEvent('lipsync-complete');
        break;
    }
  }

  dispatchEvent(name, detail = {}) {
    document.dispatchEvent(new CustomEvent(`kelly-${name}`, { detail }));
  }
}
```

### 3.4 Unity Loader with Fallback

```javascript
/**
 * UnityKellyLoader
 * Loads Unity WebGL build with graceful fallback to 2D
 */
class UnityKellyLoader {
  constructor(options = {}) {
    this.options = {
      canvasId: options.canvasId || 'unity-canvas',
      buildPath: options.buildPath || '/unity/kelly/Build',
      fallbackTo2D: options.fallbackTo2D !== false,
      onProgress: options.onProgress || null,
      onLoad: options.onLoad || null,
      onError: options.onError || null,
      timeout: options.timeout || 30000 // 30 second timeout
    };

    this.unityInstance = null;
    this.bridge = new UnityKellyBridge();
    this.isLoaded = false;
    this.loadFailed = false;
  }

  /**
   * Check if device supports Unity WebGL
   */
  static isSupported() {
    // Check WebGL support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

    if (!gl) return false;

    // Check memory (Unity needs ~500MB)
    if (navigator.deviceMemory && navigator.deviceMemory < 2) {
      console.warn('[UnityLoader] Low memory device detected');
      return false;
    }

    // Check mobile (may have performance issues)
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (isMobile) {
      console.warn('[UnityLoader] Mobile device - 3D may be slow');
    }

    return true;
  }

  /**
   * Load Unity build
   */
  async load() {
    if (!UnityKellyLoader.isSupported()) {
      console.warn('[UnityLoader] WebGL not supported, falling back to 2D');
      if (this.options.onError) {
        this.options.onError('webgl_not_supported');
      }
      return false;
    }

    const canvas = document.getElementById(this.options.canvasId);
    if (!canvas) {
      console.error('[UnityLoader] Canvas not found');
      return false;
    }

    // Load Unity loader script
    const loaderUrl = `${this.options.buildPath}/Kelly_Web_Build.loader.js`;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Unity load timeout'));
      }, this.options.timeout);

      const script = document.createElement('script');
      script.src = loaderUrl;

      script.onload = () => {
        if (typeof createUnityInstance === 'undefined') {
          clearTimeout(timeout);
          reject(new Error('Unity loader failed'));
          return;
        }

        const config = {
          dataUrl: `${this.options.buildPath}/Kelly_Web_Build.data.unityweb`,
          frameworkUrl: `${this.options.buildPath}/Kelly_Web_Build.framework.js.unityweb`,
          codeUrl: `${this.options.buildPath}/Kelly_Web_Build.wasm.unityweb`,
          streamingAssetsUrl: 'StreamingAssets',
          companyName: 'CuriousKelly',
          productName: 'KellyAvatar',
          productVersion: '1.0'
        };

        createUnityInstance(canvas, config, (progress) => {
          if (this.options.onProgress) {
            this.options.onProgress(progress);
          }
        })
          .then((instance) => {
            clearTimeout(timeout);
            this.unityInstance = instance;
            this.bridge.setInstance(instance);
            this.isLoaded = true;

            if (this.options.onLoad) {
              this.options.onLoad(instance);
            }

            resolve(true);
          })
          .catch((error) => {
            clearTimeout(timeout);
            this.loadFailed = true;
            console.error('[UnityLoader] Load failed:', error);

            if (this.options.onError) {
              this.options.onError('load_failed', error);
            }

            reject(error);
          });
      };

      script.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Failed to load Unity loader script'));
      };

      document.body.appendChild(script);
    });
  }

  /**
   * Unload Unity to free memory
   */
  unload() {
    if (this.unityInstance) {
      this.unityInstance.Quit().then(() => {
        this.unityInstance = null;
        this.isLoaded = false;
        console.log('[UnityLoader] Unity unloaded');
      });
    }
  }
}
```

---

## 4. 2D/3D Switch System

### 4.1 Unified Avatar Controller

```javascript
/**
 * KellyAvatarController
 * Unified controller that manages both 2D and 3D modes
 */
class KellyAvatarController {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      defaultMode: options.defaultMode || '2D',
      allowModeSwitch: options.allowModeSwitch !== false,
      unityBuildPath: options.unityBuildPath || '/unity/kelly/Build',
      ...options
    };

    this.currentMode = null;
    this.avatar2D = null;
    this.avatar3D = null;
    this.unityLoader = null;

    this.state = {
      expression: 'curious',
      phase: 'welcome',
      isSpeaking: false
    };

    this.init();
  }

  async init() {
    // Create container structure
    this.container.innerHTML = `
            <div class="kelly-avatar-wrapper">
                <!-- 2D Layer -->
                <div class="kelly-2d-layer" id="kelly-2d"></div>
                
                <!-- 3D Layer (Unity canvas) -->
                <div class="kelly-3d-layer" id="kelly-3d" style="display: none;">
                    <canvas id="unity-canvas"></canvas>
                </div>
                
                <!-- Loading overlay -->
                <div class="kelly-loading" id="kelly-loading" style="display: none;">
                    <div class="kelly-loading-spinner"></div>
                    <div class="kelly-loading-text">Loading 3D Kelly...</div>
                    <div class="kelly-loading-progress" id="kelly-progress">0%</div>
                </div>
            </div>
        `;

    // Initialize 2D (always available)
    this.avatar2D = new Kelly2DAvatarPlayer(document.getElementById('kelly-2d'), { preload: true });

    // Set initial mode
    const savedMode = localStorage.getItem('kelly_mode') || this.options.defaultMode;
    await this.setMode(savedMode);
  }

  /**
   * Switch between 2D and 3D modes
   */
  async setMode(mode) {
    if (mode === this.currentMode) return;

    console.log(`[KellyAvatar] Switching mode: ${this.currentMode} → ${mode}`);

    if (mode === '3D') {
      // Check if 3D is supported
      if (!UnityKellyLoader.isSupported()) {
        console.warn('[KellyAvatar] 3D not supported, staying in 2D');
        this.dispatchEvent('mode-switch-failed', { reason: 'not_supported' });
        return;
      }

      // Show loading
      document.getElementById('kelly-loading').style.display = 'flex';

      // Load Unity if not already loaded
      if (!this.unityLoader?.isLoaded) {
        this.unityLoader = new UnityKellyLoader({
          canvasId: 'unity-canvas',
          buildPath: this.options.unityBuildPath,
          onProgress: (p) => {
            document.getElementById('kelly-progress').textContent = `${Math.round(p * 100)}%`;
          },
          onLoad: () => {
            this.finishModeSwitch('3D');
          },
          onError: (err) => {
            console.error('[KellyAvatar] 3D load failed:', err);
            document.getElementById('kelly-loading').style.display = 'none';
            this.dispatchEvent('mode-switch-failed', { reason: err });
          }
        });

        try {
          await this.unityLoader.load();
        } catch (e) {
          document.getElementById('kelly-loading').style.display = 'none';
          return;
        }
      } else {
        this.finishModeSwitch('3D');
      }
    } else {
      this.finishModeSwitch('2D');
    }
  }

  finishModeSwitch(mode) {
    const layer2D = document.getElementById('kelly-2d');
    const layer3D = document.getElementById('kelly-3d');
    const loading = document.getElementById('kelly-loading');

    loading.style.display = 'none';

    if (mode === '3D') {
      // Crossfade to 3D
      layer2D.style.opacity = '0';
      setTimeout(() => {
        layer2D.style.display = 'none';
        layer3D.style.display = 'block';
        layer3D.style.opacity = '1';
      }, 400);

      // Sync state to Unity
      this.unityLoader.bridge.setExpression(this.state.expression);
    } else {
      // Crossfade to 2D
      layer3D.style.opacity = '0';
      setTimeout(() => {
        layer3D.style.display = 'none';
        layer2D.style.display = 'block';
        layer2D.style.opacity = '1';
      }, 400);

      // Sync state to 2D
      this.avatar2D.setExpression(this.state.expression);
    }

    this.currentMode = mode;
    localStorage.setItem('kelly_mode', mode);
    this.dispatchEvent('mode-changed', { mode });
  }

  /**
   * Set expression (works in both modes)
   */
  setExpression(expression) {
    this.state.expression = expression;

    if (this.currentMode === '2D') {
      this.avatar2D.setExpression(expression);
    } else if (this.unityLoader?.isLoaded) {
      this.unityLoader.bridge.setExpression(expression);
    }
  }

  /**
   * Set phase (syncs expression based on phase)
   */
  setPhase(phase, choice = null) {
    this.state.phase = phase;

    if (this.currentMode === '2D') {
      this.avatar2D.setPhase(phase, choice);
    } else {
      // Map phase to expression and animation
      const expressionMap = {
        welcome: 'curious',
        q1: choice ? (choice === 'a' ? 'explaining' : 'celebrating') : 'curious',
        q2: choice ? (choice === 'a' ? 'explaining' : 'celebrating') : 'curious',
        q3: choice ? (choice === 'a' ? 'explaining' : 'celebrating') : 'listening',
        wisdom: 'wisdom',
        complete: 'celebrating'
      };

      this.unityLoader?.bridge.setExpression(expressionMap[phase] || 'curious');

      if (choice === 'b' || phase === 'complete') {
        this.unityLoader?.bridge.playAnimation('celebrate');
      }
    }
  }

  /**
   * Set speaking state (for audio sync)
   */
  setSpeaking(speaking, visemeData = null) {
    this.state.isSpeaking = speaking;

    if (this.currentMode === '2D') {
      this.avatar2D.setSpeaking(speaking);
    } else {
      if (speaking && visemeData) {
        this.unityLoader?.bridge.startLipSync(visemeData);
      } else {
        this.unityLoader?.bridge.stopLipSync();
      }
    }
  }

  /**
   * Toggle mode
   */
  toggleMode() {
    const newMode = this.currentMode === '2D' ? '3D' : '2D';
    return this.setMode(newMode);
  }

  /**
   * Get current mode
   */
  getMode() {
    return this.currentMode;
  }

  dispatchEvent(name, detail) {
    document.dispatchEvent(new CustomEvent(`kelly-${name}`, { detail }));
  }

  destroy() {
    this.avatar2D?.destroy();
    this.unityLoader?.unload();
    this.container.innerHTML = '';
  }
}
```

---

## 5. Integration with learn.html

### 5.1 Required Changes to learn.html

```html
<!-- ADD: Kelly Avatar Layers -->
<div class="kelly-frame" id="kelly-frame">
  <!-- 2D Layer (default) -->
  <div class="kelly-2d-layer" id="kelly-2d-layer">
    <img
      src="/images/kelly/kelly-directors-chair-curious.png"
      alt="Kelly"
      class="kelly-avatar"
      id="kelly-avatar-2d"
    />
  </div>

  <!-- 3D Layer (Unity) -->
  <div class="kelly-3d-layer" id="kelly-3d-layer" style="display: none;">
    <canvas id="unity-canvas" style="width: 100%; height: 100%;"></canvas>
  </div>

  <!-- Loading overlay for 3D -->
  <div class="kelly-loading-overlay" id="unity-loading" style="display: none;">
    <div class="loading-spinner"></div>
    <div class="loading-text">Loading 3D Kelly...</div>
    <div class="loading-progress" id="unity-progress">0%</div>
    <button class="loading-cancel" onclick="cancelUnityLoad()">Stay in 2D</button>
  </div>
</div>

<!-- ADD: Scripts -->
<script src="/js/kelly-2d-avatar.js"></script>
<script src="/js/unity-kelly-loader.js"></script>
<script src="/js/kelly-avatar-controller.js"></script>
```

### 5.2 JavaScript Integration

```javascript
// In learn.html <script>

// Initialize unified avatar controller
const kellyAvatar = new KellyAvatarController(document.getElementById('kelly-frame'), {
  defaultMode: localStorage.getItem('kelly_mode') || '2D',
  unityBuildPath: '/unity/kelly/Build'
});

// 2D/3D Toggle button handler
document.getElementById('btn-mode').onclick = async () => {
  const newMode = kellyAvatar.getMode() === '2D' ? '3D' : '2D';

  if (newMode === '3D') {
    // Warn user about load time
    const confirmed = await showConfirmDialog(
      'Load 3D Kelly?',
      'This will download ~40MB and may take 10-30 seconds. Your lesson will continue afterward.'
    );
    if (!confirmed) return;
  }

  await kellyAvatar.setMode(newMode);
  updateModeBadge(newMode);
};

// Update renderPhase to use avatar controller
function renderPhase(phase) {
  // ... existing code ...

  // Use unified avatar controller
  kellyAvatar.setPhase(phase.type, null);
  kellyAvatar.setSpeaking(true);

  // ... rest of existing code ...
}

// Update selectChoice to trigger reaction
function selectChoice(letter) {
  // ... existing code ...

  // Trigger avatar reaction
  kellyAvatar.setPhase(`q${state.currentPhase - 1}`, letter.toLowerCase());

  // ... rest of existing code ...
}

// Connect audio to avatar
const audioElement = document.getElementById('lesson-audio');
if (audioElement) {
  audioElement.addEventListener('play', () => kellyAvatar.setSpeaking(true));
  audioElement.addEventListener('pause', () => kellyAvatar.setSpeaking(false));
  audioElement.addEventListener('ended', () => kellyAvatar.setSpeaking(false));
}
```

---

## 6. Performance Requirements

### 6.1 2D Avatar Performance

| Metric            | Target     | Measurement              |
| ----------------- | ---------- | ------------------------ |
| Initial load      | <2 seconds | First image visible      |
| Expression switch | <400ms     | Crossfade complete       |
| Animation FPS     | 60 FPS     | CSS animation smoothness |
| Memory usage      | <20MB      | Total 2D avatar memory   |
| CPU usage         | <5%        | During idle animations   |

### 6.2 3D Avatar Performance

| Metric           | Target         | Measurement             |
| ---------------- | -------------- | ----------------------- |
| Initial load     | <30 seconds    | Unity fully loaded      |
| Progressive load | Show progress  | % indicator during load |
| Render FPS       | 30 FPS minimum | On mid-range devices    |
| Memory usage     | <500MB         | Unity WebGL memory      |
| GPU usage        | <50%           | On supported devices    |
| Fallback trigger | Auto after 45s | If load fails           |

### 6.3 Device Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVICE TIER MATRIX                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TIER 1: LOW-END (2D only)                                    │
│   ─────────────────────────                                     │
│   • Memory: <2GB RAM                                            │
│   • Examples: iPhone 6s, budget Android                        │
│   • Mode: 2D only, no 3D option                                │
│   • Animation: Reduced (prefers-reduced-motion)                │
│                                                                 │
│   TIER 2: MID-RANGE (2D default, 3D optional)                  │
│   ───────────────────────────────────────────                  │
│   • Memory: 2-4GB RAM                                          │
│   • Examples: iPhone 11, mid Android                           │
│   • Mode: 2D default, 3D available                             │
│   • Animation: Full                                            │
│                                                                 │
│   TIER 3: HIGH-END (2D default, 3D recommended)                │
│   ─────────────────────────────────────────────                │
│   • Memory: >4GB RAM                                           │
│   • Examples: iPhone 14+, flagship Android, desktops          │
│   • Mode: 2D default, 3D one-click enable                     │
│   • Animation: Full with extras                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Test Scenarios

### 7.1 2D Avatar Tests

| ID    | Scenario                             | Expected                   | Priority |
| ----- | ------------------------------------ | -------------------------- | -------- |
| 2D-01 | Page loads                           | Curious Kelly visible <2s  | P1       |
| 2D-02 | Expression: curious → explaining     | Smooth 400ms crossfade     | P1       |
| 2D-03 | Expression: explaining → celebrating | Smooth + bounce animation  | P1       |
| 2D-04 | All 5 expressions cycle              | Each image loads correctly | P1       |
| 2D-05 | Speaking state on                    | Speaking ring visible      | P1       |
| 2D-06 | Speaking state off                   | Speaking ring hidden       | P1       |
| 2D-07 | Breathing animation                  | Subtle 4s cycle visible    | P2       |
| 2D-08 | Image fails to load                  | Previous image stays       | P1       |
| 2D-09 | Rapid expression changes             | No flicker, queues changes | P2       |
| 2D-10 | Memory pressure                      | Images don't leak          | P2       |
| 2D-11 | Reduced motion preference            | Animations disabled        | P2       |
| 2D-12 | Mobile Safari                        | All animations work        | P1       |

### 7.2 3D Avatar Tests

| ID    | Scenario                 | Expected                 | Priority |
| ----- | ------------------------ | ------------------------ | -------- |
| 3D-01 | Unity load starts        | Progress indicator shows | P1       |
| 3D-02 | Unity load completes     | 3D Kelly visible         | P1       |
| 3D-03 | Unity load fails         | Fallback to 2D + message | P1       |
| 3D-04 | Unity load timeout (45s) | Fallback to 2D           | P1       |
| 3D-05 | WebGL not supported      | Stay in 2D + message     | P1       |
| 3D-06 | Low memory device        | Block 3D option          | P2       |
| 3D-07 | Lip-sync with audio      | Mouth moves correctly    | P1       |
| 3D-08 | Expression change        | Blendshapes animate      | P1       |
| 3D-09 | FPS stays above 30       | No major drops           | P2       |
| 3D-10 | Memory stays under 500MB | No OOM crashes           | P1       |
| 3D-11 | Unload Unity             | Memory released          | P2       |
| 3D-12 | Page refresh during load | Clean restart            | P2       |

### 7.3 Mode Switch Tests

| ID    | Scenario               | Expected                   | Priority |
| ----- | ---------------------- | -------------------------- | -------- |
| SW-01 | 2D → 3D switch         | Crossfade, no flicker      | P1       |
| SW-02 | 3D → 2D switch         | Instant, state preserved   | P1       |
| SW-03 | Switch during phase    | Expression syncs           | P1       |
| SW-04 | Switch during speaking | Speaking state syncs       | P1       |
| SW-05 | Preference persists    | Reload keeps mode          | P1       |
| SW-06 | Cancel 3D load         | Stay in 2D                 | P1       |
| SW-07 | Multiple rapid toggles | Debounced, no crash        | P2       |
| SW-08 | Switch on slow network | Progress shown, cancelable | P2       |

---

## 8. Content Production Requirements

### 8.1 2D Assets Required

| Asset Type              | Current | Needed | Gap   |
| ----------------------- | ------- | ------ | ----- |
| Adult expressions       | 10      | 10     | 0 ✅  |
| Age variant expressions | 0       | 50     | 50 🔴 |
| Total 2D images         | 10      | 60     | 50    |

**Timeline for full age support:**

- 50 images × 2 hours/image = 100 hours
- 2 artists = 50 hours = 2 weeks

### 8.2 3D Assets Required

| Asset Type          | Current | Needed | Status    |
| ------------------- | ------- | ------ | --------- |
| Kelly 3D model      | 1       | 1      | ✅ Ready  |
| Facial blendshapes  | 15      | 15     | ✅ Ready  |
| Viseme shapes       | 15      | 15     | ✅ Ready  |
| Idle animation      | 1       | 1      | ✅ Ready  |
| Celebrate animation | 0       | 1      | 🔴 Needed |
| Thinking animation  | 0       | 1      | 🔴 Needed |
| Age variant rigs    | 0       | 5      | 🔴 Future |

### 8.3 Audio Integration Requirements

| Component              | Status     | Notes              |
| ---------------------- | ---------- | ------------------ |
| Viseme timing data     | 🔴 Needed  | Generate from TTS  |
| Audio-to-viseme mapper | 🔴 Needed  | Build or integrate |
| Phoneme library        | ⚠️ Partial | CC4 has basics     |
| Lip-sync smoothing     | 🔴 Needed  | Interpolation code |

---

## 9. Scale Considerations

### 9.1 CDN Strategy for Avatar Assets

```
┌─────────────────────────────────────────────────────────────────┐
│                    AVATAR CDN ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   2D ASSETS (Always cached)                                    │
│   ─────────────────────────                                     │
│   • 10 images × ~500KB = 5MB total                             │
│   • Cache: 1 year (immutable)                                  │
│   • Preload: All 5 expressions on page load                   │
│   • CDN: Cloudflare with global edge                          │
│                                                                 │
│   3D ASSETS (Lazy loaded)                                      │
│   ─────────────────────────                                     │
│   • Unity build: ~40MB compressed                              │
│   • Cache: 1 week (versioned)                                  │
│   • Load: Only when user opts for 3D                          │
│   • CDN: Cloudflare R2 for large files                        │
│                                                                 │
│   ESTIMATED BANDWIDTH (1M DAU)                                 │
│   ────────────────────────────                                 │
│   • 2D assets: 5MB × 1M = 5TB/day (cached, actual ~500GB)    │
│   • 3D assets: 40MB × 100K (10%) = 4TB/day                   │
│   • Total: ~5TB/day peak, ~1.5TB/day average                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Memory Management at Scale

```javascript
// Memory-conscious avatar loading
const AvatarMemoryManager = {
  // Unload 3D when not in use
  autoUnload3D: true,
  unload3DAfterMs: 5 * 60 * 1000, // 5 minutes idle

  // Limit 2D image cache
  max2DImagesInMemory: 10,

  // Monitor memory pressure
  onMemoryPressure() {
    if (kellyAvatar.getMode() === '3D') {
      kellyAvatar.setMode('2D');
      kellyAvatar.unityLoader.unload();
      showToast('Switched to 2D to free memory', 'info');
    }
  }
};

// Listen for memory warnings (Chrome)
if ('memory' in performance) {
  setInterval(() => {
    const used = performance.memory.usedJSHeapSize;
    const total = performance.memory.jsHeapSizeLimit;
    if (used / total > 0.9) {
      AvatarMemoryManager.onMemoryPressure();
    }
  }, 10000);
}
```

---

## 10. Implementation Plan

### Phase 1: 2D Avatar Integration (This Week)

| Task                                     | Est. Hours | Owner | Status |
| ---------------------------------------- | ---------- | ----- | ------ |
| Create kelly-2d-avatar.js for learn.html | 4h         | Dev   | 🔴     |
| Add CSS animations to kelly-os.css       | 2h         | Dev   | 🔴     |
| Integrate with phase system              | 3h         | Dev   | 🔴     |
| Connect to audio for speaking state      | 2h         | Dev   | 🔴     |
| Test all 5 expressions                   | 2h         | QA    | 🔴     |
| Mobile Safari testing                    | 2h         | QA    | 🔴     |
| **Total**                                | **15h**    |       |        |

### Phase 2: 3D Avatar Integration (Next Week)

| Task                              | Est. Hours | Owner | Status |
| --------------------------------- | ---------- | ----- | ------ |
| Copy Unity build to public/unity/ | 1h         | Dev   | 🔴     |
| Create unity-kelly-loader.js      | 4h         | Dev   | 🔴     |
| Create unity-kelly-bridge.js      | 4h         | Dev   | 🔴     |
| Create kelly-avatar-controller.js | 6h         | Dev   | 🔴     |
| Add 2D/3D toggle UI               | 2h         | Dev   | 🔴     |
| Test Unity load/unload            | 3h         | QA    | 🔴     |
| Test mode switching               | 3h         | QA    | 🔴     |
| Performance testing               | 4h         | QA    | 🔴     |
| **Total**                         | **27h**    |       |        |

### Phase 3: Audio Lip-Sync (Week 3)

| Task                            | Est. Hours | Owner | Status |
| ------------------------------- | ---------- | ----- | ------ |
| Research viseme generation      | 4h         | Dev   | 🔴     |
| Build audio-to-viseme converter | 8h         | Dev   | 🔴     |
| Integrate with Unity controller | 4h         | Dev   | 🔴     |
| Test lip-sync accuracy          | 4h         | QA    | 🔴     |
| **Total**                       | **20h**    |       |        |

### Phase 4: Age Variants (Week 4+)

| Task                               | Est. Hours | Owner | Status |
| ---------------------------------- | ---------- | ----- | ------ |
| Commission 50 age variant images   | 100h       | Art   | 🔴     |
| Update kelly-2d-avatar.js for ages | 4h         | Dev   | 🔴     |
| Add age morphing transitions       | 4h         | Dev   | 🔴     |
| Test age → Kelly mapping           | 4h         | QA    | 🔴     |
| **Total**                          | **112h**   |       |        |

---

## Summary

### What We Have Now

| Component                 | Status    | Location                                               |
| ------------------------- | --------- | ------------------------------------------------------ |
| 2D Images (5 expressions) | ✅ Ready  | public/images/kelly/                                   |
| 2D Avatar JS (basic)      | ✅ Exists | daily-lesson-marketing/.../kelly-2d-avatar.js          |
| 2D CSS Animations         | ✅ Exists | daily-lesson-marketing/.../kelly-avatar-animations.css |
| Unity 3D Build            | ✅ Built  | digital-kelly/.../Kelly_Web_Build/                     |
| Unity Controller          | ✅ Exists | digital-kelly/.../KellyAvatarController.cs             |
| Unity Loader              | ✅ Exists | app/unity-loader.js                                    |
| Unity Bridge              | ✅ Exists | public/js/unity-bridge.js                              |

### What We Need to Build

| Component                    | Priority | Hours |
| ---------------------------- | -------- | ----- |
| Integrate 2D into learn.html | P1       | 15h   |
| Integrate 3D into learn.html | P2       | 27h   |
| 2D/3D toggle system          | P2       | 8h    |
| Audio lip-sync               | P3       | 20h   |
| Age variants (images)        | P4       | 100h  |
| Age variant code             | P4       | 12h   |

### Impact Assessment

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMPACT MATRIX                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   LEARNER EXPERIENCE                                           │
│   ──────────────────                                           │
│   • Kelly feels alive: Breathing, blinking, reacting           │
│   • Instant feedback: Expression changes on choice             │
│   • Immersion option: 3D for those who want it                │
│   • Universal access: 2D works everywhere                      │
│                                                                 │
│   SCALE (Millions Daily)                                       │
│   ──────────────────────                                       │
│   • 2D: Handles unlimited scale (CDN-served images)           │
│   • 3D: ~10% opt-in = manageable bandwidth                    │
│   • Memory: Auto-fallback prevents crashes                     │
│   • Global: Edge-cached assets everywhere                      │
│                                                                 │
│   PRODUCTION                                                    │
│   ──────────                                                   │
│   • 2D launch-ready: 10 images exist                          │
│   • 3D launch-ready: Unity build exists                       │
│   • Gap: Integration code (42 hours)                          │
│   • Future: Age variants (112 hours)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

_Document Version 2.0 — November 28, 2025_
_This is the authoritative specification for Kelly avatar integration_





