# Dual-Mode Kelly Architecture
## 2D Video (HeyGen) + 3D Live (Unity WebGL) with Learner Toggle

**Created:** December 16, 2025  
**Status:** Active Development  
**Goal:** Let learners choose their preferred Kelly experience

---

## 🎯 Core Concept

Kelly exists in TWO rendering modes that learners can toggle between (or combine):

| Mode | Technology | Best For | Characteristics |
|------|------------|----------|-----------------|
| **2D Video** | HeyGen + LoRA Images | Mobile, low-bandwidth, accessibility | Pre-rendered, consistent, smaller files |
| **3D Live** | Unity WebGL + CC5 Model | Desktop, immersive, interactive | Real-time, responsive, higher requirements |

**Key Insight:** Both modes use the SAME visual identity (trained LoRA + CC5 share the character design).

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KELLY ASSET PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  Kelly LoRA     │    │  Kelly CC5      │    │  Kelly Voice    │     │
│  │  (Flux Model)   │    │  (3D Model)     │    │  (ElevenLabs)   │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  Static Images  │    │  Unity WebGL    │    │  Audio Files    │     │
│  │  (PNG/JPEG)     │    │  Build          │    │  (MP3)          │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           ▼                      ▼                      │               │
│  ┌─────────────────┐    ┌─────────────────┐            │               │
│  │  HeyGen Upload  │    │  Kelly 3D       │            │               │
│  │  (Talking Photo)│    │  iframe         │◄───────────┘               │
│  └────────┬────────┘    └─────────────────┘                            │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  HeyGen Video   │                                                   │
│  │  Generation     │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LEARNER MODE SELECTOR                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ 2D Only      │  │ 3D Only      │  │ Hybrid (Both)        │   │   │
│  │  │ (HeyGen)     │  │ (Unity)      │  │ (2D + 3D optional)   │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Asset Storage Structure

```
public/
├── kelly/
│   ├── 2d/                          # 2D Video Mode Assets
│   │   ├── videos/                  # Pre-rendered HeyGen videos
│   │   │   ├── day-001/
│   │   │   │   ├── hook.mp4
│   │   │   │   ├── q1.mp4
│   │   │   │   ├── q2.mp4
│   │   │   │   └── wisdom.mp4
│   │   │   └── day-002/...
│   │   ├── poses/                   # Static images for fallback
│   │   │   ├── kelly_welcome.png
│   │   │   ├── kelly_explaining.png
│   │   │   └── ...
│   │   └── manifest.json            # 2D asset registry
│   │
│   ├── 3d/                          # 3D Unity Mode Assets
│   │   ├── build/                   # Unity WebGL build
│   │   │   ├── Kelly_Web_Build.wasm
│   │   │   ├── Kelly_Web_Build.data
│   │   │   └── Kelly_Web_Build.framework.js
│   │   ├── animations/              # Animation clips
│   │   └── manifest.json            # 3D asset registry
│   │
│   └── shared/                      # Shared assets (both modes)
│       ├── audio/                   # Voice audio (used by both)
│       └── expressions/             # Expression states
│
├── unity/kelly-live/                # Unity WebGL deployment
│   ├── index.html
│   └── Build/
│
└── js/
    └── kelly-mode-controller.js     # Mode switching logic
```

---

## 🎛️ Learner Mode Settings

### Settings UI Location
`Settings > Kelly Display > Rendering Mode`

### Mode Options

```javascript
const KELLY_MODES = {
  AUTO: 'auto',           // System chooses based on device/bandwidth
  VIDEO_2D: '2d',         // Force 2D HeyGen videos
  LIVE_3D: '3d',          // Force 3D Unity WebGL
  HYBRID: 'hybrid',       // 2D primary, 3D available on-demand
};
```

### User Preferences Storage

```javascript
// localStorage key
const KELLY_MODE_KEY = 'kelly_display_mode';

// Default based on device detection
function getDefaultMode() {
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  const hasGoodGPU = detectGPUCapability();
  const hasFastNetwork = navigator.connection?.effectiveType === '4g';
  
  if (isMobile || !hasGoodGPU) return 'video_2d';
  if (hasFastNetwork && hasGoodGPU) return 'hybrid';
  return 'auto';
}
```

---

## 🔄 Mode Controller Implementation

### `public/js/kelly-mode-controller.js`

```javascript
/**
 * Kelly Dual-Mode Controller
 * Manages switching between 2D video and 3D live modes
 */

class KellyModeController {
  constructor() {
    this.currentMode = this.loadPreference();
    this.videoPlayer = null;
    this.unityInstance = null;
    this.isUnityLoaded = false;
  }

  // ═══════════════════════════════════════════════════════════════════
  // MODE DETECTION & SWITCHING
  // ═══════════════════════════════════════════════════════════════════

  loadPreference() {
    return localStorage.getItem('kelly_display_mode') || this.detectOptimalMode();
  }

  savePreference(mode) {
    localStorage.setItem('kelly_display_mode', mode);
    this.currentMode = mode;
    this.applyMode();
  }

  detectOptimalMode() {
    // Mobile devices → 2D
    if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
      return '2d';
    }
    
    // Check for WebGL 2 support (required for Unity)
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return '2d';
    
    // Check network speed
    const connection = navigator.connection;
    if (connection && connection.effectiveType !== '4g') {
      return '2d';
    }
    
    // Good conditions → offer hybrid
    return 'hybrid';
  }

  // ═══════════════════════════════════════════════════════════════════
  // 2D VIDEO MODE (HeyGen)
  // ═══════════════════════════════════════════════════════════════════

  async play2DVideo(videoUrl, options = {}) {
    const container = document.getElementById('kelly-container');
    
    // Hide 3D if showing
    this.hide3D();
    
    // Create or reuse video element
    if (!this.videoPlayer) {
      this.videoPlayer = document.createElement('video');
      this.videoPlayer.id = 'kelly-2d-player';
      this.videoPlayer.className = 'kelly-video-player';
      this.videoPlayer.playsInline = true;
      container.appendChild(this.videoPlayer);
    }
    
    this.videoPlayer.style.display = 'block';
    this.videoPlayer.src = videoUrl;
    
    if (options.autoplay !== false) {
      await this.videoPlayer.play();
    }
    
    return this.videoPlayer;
  }

  show2DStatic(imageUrl) {
    const container = document.getElementById('kelly-container');
    this.hide3D();
    
    let img = container.querySelector('.kelly-static-image');
    if (!img) {
      img = document.createElement('img');
      img.className = 'kelly-static-image';
      container.appendChild(img);
    }
    
    img.src = imageUrl;
    img.style.display = 'block';
    
    if (this.videoPlayer) {
      this.videoPlayer.style.display = 'none';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // 3D LIVE MODE (Unity WebGL)
  // ═══════════════════════════════════════════════════════════════════

  async load3D() {
    if (this.isUnityLoaded) return;
    
    const container = document.getElementById('kelly-3d-container');
    if (!container) {
      console.error('Kelly 3D container not found');
      return;
    }
    
    // Create iframe for Unity WebGL
    const iframe = document.createElement('iframe');
    iframe.id = 'kelly-unity-iframe';
    iframe.src = '/unity/kelly-live/index.html';
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.border = 'none';
    iframe.allow = 'autoplay';
    
    container.appendChild(iframe);
    
    // Wait for Unity to load
    return new Promise((resolve) => {
      iframe.onload = () => {
        this.isUnityLoaded = true;
        this.unityIframe = iframe;
        resolve();
      };
    });
  }

  show3D() {
    const container = document.getElementById('kelly-3d-container');
    if (container) {
      container.style.display = 'block';
    }
    
    // Hide 2D
    if (this.videoPlayer) {
      this.videoPlayer.style.display = 'none';
    }
    const staticImg = document.querySelector('.kelly-static-image');
    if (staticImg) staticImg.style.display = 'none';
  }

  hide3D() {
    const container = document.getElementById('kelly-3d-container');
    if (container) {
      container.style.display = 'none';
    }
  }

  // Send command to Unity (via postMessage)
  sendToUnity(command, data) {
    if (this.unityIframe && this.unityIframe.contentWindow) {
      this.unityIframe.contentWindow.postMessage({
        type: 'kelly_command',
        command,
        data,
      }, '*');
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // HYBRID MODE
  // ═══════════════════════════════════════════════════════════════════

  async playHybrid(videoUrl, options = {}) {
    // In hybrid mode:
    // - Play 2D video for the main content
    // - Show 3D toggle button for learners who want to switch
    
    await this.play2DVideo(videoUrl, options);
    this.showModeToggle();
  }

  showModeToggle() {
    let toggle = document.getElementById('kelly-mode-toggle');
    if (!toggle) {
      toggle = document.createElement('button');
      toggle.id = 'kelly-mode-toggle';
      toggle.className = 'kelly-mode-toggle';
      toggle.innerHTML = '🎮 Switch to 3D';
      toggle.onclick = () => this.toggleMode();
      document.getElementById('kelly-container').appendChild(toggle);
    }
    toggle.style.display = 'block';
  }

  toggleMode() {
    if (this.currentMode === '2d' || !this.isUnityLoaded) {
      this.load3D().then(() => {
        this.show3D();
        this.currentMode = '3d';
        document.getElementById('kelly-mode-toggle').innerHTML = '📺 Switch to 2D';
      });
    } else {
      this.hide3D();
      if (this.videoPlayer) {
        this.videoPlayer.style.display = 'block';
      }
      this.currentMode = '2d';
      document.getElementById('kelly-mode-toggle').innerHTML = '🎮 Switch to 3D';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // LESSON INTEGRATION
  // ═══════════════════════════════════════════════════════════════════

  async playLessonPhase(phase, dayNumber) {
    const mode = this.currentMode;
    
    if (mode === '3d') {
      // Send phase data to Unity
      this.sendToUnity('play_phase', { phase, day: dayNumber });
    } else {
      // Play HeyGen video
      const videoUrl = `/kelly/2d/videos/day-${String(dayNumber).padStart(3, '0')}/${phase}.mp4`;
      await this.play2DVideo(videoUrl);
    }
  }

  setExpression(expression) {
    if (this.currentMode === '3d') {
      this.sendToUnity('set_expression', { expression });
    } else {
      // Show static expression image
      const imageUrl = `/kelly/poses/kelly_${expression}.png`;
      this.show2DStatic(imageUrl);
    }
  }
}

// Global instance
window.kellyMode = new KellyModeController();
```

---

## 🎬 HeyGen Video Generation Pipeline

### For 2D Mode Videos

```javascript
// Generate HeyGen video for a lesson phase
async function generateHeyGenVideo(day, phase, script, personaId) {
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': process.env.HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: personaId,  // From uploaded LoRA images
        },
        voice: {
          type: 'text',
          voice_id: 'KELLY_VOICE_ID',
          input_text: script,
        },
      }],
      dimension: { width: 1280, height: 720 },
      aspect_ratio: '16:9',
    }),
  });
  
  return response.json();
}
```

---

## 🎮 Unity 3D Integration

### Unity Bridge Commands

The Unity build accepts these postMessage commands:

| Command | Data | Description |
|---------|------|-------------|
| `play_phase` | `{ phase, day }` | Play lesson phase animation |
| `set_expression` | `{ expression }` | Set Kelly's facial expression |
| `speak` | `{ audioUrl, visemeData }` | Lip-sync to audio |
| `set_pose` | `{ pose }` | Set body pose |
| `idle` | - | Return to idle state |

### Unity → Web Communication

```javascript
// In Unity WebGL build, send events to parent
window.parent.postMessage({
  type: 'kelly_event',
  event: 'animation_complete',
  data: { phase: 'hook' }
}, '*');

// Parent page listens
window.addEventListener('message', (event) => {
  if (event.data.type === 'kelly_event') {
    handleKellyEvent(event.data);
  }
});
```

---

## 📱 Settings UI Component

### HTML Structure

```html
<div class="settings-section kelly-mode-settings">
  <h3>Kelly Display Mode</h3>
  
  <div class="mode-options">
    <label class="mode-option">
      <input type="radio" name="kelly_mode" value="auto" checked>
      <span class="mode-label">
        <strong>Auto</strong>
        <small>System chooses best option for your device</small>
      </span>
    </label>
    
    <label class="mode-option">
      <input type="radio" name="kelly_mode" value="2d">
      <span class="mode-label">
        <strong>2D Video</strong>
        <small>Pre-rendered videos (lower bandwidth, works everywhere)</small>
      </span>
    </label>
    
    <label class="mode-option">
      <input type="radio" name="kelly_mode" value="3d">
      <span class="mode-label">
        <strong>3D Live</strong>
        <small>Real-time 3D Kelly (requires modern browser + GPU)</small>
      </span>
    </label>
    
    <label class="mode-option">
      <input type="radio" name="kelly_mode" value="hybrid">
      <span class="mode-label">
        <strong>Hybrid</strong>
        <small>2D by default, with option to switch to 3D anytime</small>
      </span>
    </label>
  </div>
  
  <div class="mode-preview" id="mode-preview">
    <!-- Preview of selected mode -->
  </div>
</div>
```

---

## 📊 Mode Comparison

| Feature | 2D Video (HeyGen) | 3D Live (Unity) |
|---------|-------------------|-----------------|
| **File Size** | 2-10 MB per video | 50-80 MB initial load |
| **Bandwidth** | Stream on demand | One-time load |
| **Interactivity** | Limited | Full |
| **Expression Changes** | Pre-baked | Real-time |
| **Mobile Support** | Excellent | Limited |
| **Offline** | With caching | Harder |
| **Consistency** | Perfect (pre-rendered) | Real-time variance |
| **Updates** | Re-render required | Hot-swappable |

---

## 🚀 Implementation Priority

### Phase 1: LoRA Asset Generation (Now)
- [x] Create `kelly-lora-asset-factory.ts`
- [ ] Run generation for all 87 assets
- [ ] Upload to Supabase/R2

### Phase 2: HeyGen Integration (Week 1)
- [ ] Upload LoRA images as HeyGen talking photos
- [ ] Generate lesson videos for pilot days (1-10)
- [ ] Create video delivery pipeline

### Phase 3: Mode Controller (Week 1-2)
- [ ] Implement `kelly-mode-controller.js`
- [ ] Add Settings UI
- [ ] Test mode switching

### Phase 4: Unity Bridge (Week 2)
- [ ] Add postMessage handlers to Unity build
- [ ] Implement command protocol
- [ ] Test 3D mode

### Phase 5: Hybrid Mode (Week 3)
- [ ] Implement hybrid toggle
- [ ] A/B test user preferences
- [ ] Optimize performance

---

## 🔧 Environment Variables

```bash
# .env.local

# Replicate (LoRA image generation)
REPLICATE_API_TOKEN=r8_xxx

# HeyGen (2D video generation)
HEYGEN_API_KEY=xxx

# ElevenLabs (Voice)
ELEVENLABS_API_KEY=xxx
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0

# Supabase (Asset storage)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=xxx
```

---

**Document Owner:** Engineering  
**Last Updated:** December 16, 2025
