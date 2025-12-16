# 🔮 Kelly 3D System - Future-Proofing Strategy

**Created:** December 16, 2025  
**Purpose:** Document the WebGL → WebGPU migration path and optimization strategies

---

## 📊 Current Technology Stack (December 2025)

| Layer | Technology | Status |
|-------|------------|--------|
| **3D Engine** | Unity 6000.x | ✅ Production |
| **Web Rendering** | WebGL 2.0 | ✅ 97%+ browser support |
| **Build Target** | Unity WebGL | ✅ Stable |
| **Future Target** | Unity WebGPU | 🔜 Experimental (Unity 6+) |

---

## 🏛️ Architecture: Designed for Swappability

Kelly's 3D system uses a **decoupled architecture** that allows swapping rendering technologies without touching the web application:

```
┌─────────────────────────────────────────────────────┐
│          Web Application (Astro/HTML/JS)            │
│  ┌───────────────────────────────────────────────┐  │
│  │           Lesson Player / Kelly OS            │  │
│  └───────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                Bridge Layer (kbridge.js)            │
│  - postMessage communication                        │
│  - kelly-ready, kelly-loading, kelly-playing        │
│  - Engine-agnostic API                              │
├─────────────────────────────────────────────────────┤
│              3D Engine (iframe)                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Current: /unity/kelly-chair/ (WebGL 2.0)   │   │
│  │  Future:  /unity/kelly-webgpu/ (WebGPU)     │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Key Insight:** The web app only talks to `kbridge.js`. It doesn't know or care whether the iframe contains WebGL, WebGPU, or a future technology.

---

## 🚀 WebGPU: What It Means for Kelly

### Performance Benefits (Expected)

| Metric | WebGL 2.0 | WebGPU (Expected) | Improvement |
|--------|-----------|-------------------|-------------|
| Draw calls | CPU-bound | GPU-instanced | 30-50% faster |
| Shader compilation | Blocking | Async pipelines | Smoother load |
| Compute shaders | ❌ Not available | ✅ Available | New capabilities |
| Multi-threading | Limited | Full parallel | Less jank |

### New Capabilities for Kelly

1. **GPU Compute Shaders**
   - Real-time hair physics simulation
   - Cloth dynamics (sweater, jeans)
   - Particle effects (celebration confetti)

2. **Better Lip Sync**
   - ML-based viseme prediction on GPU
   - Lower latency audio-to-face

3. **Higher Fidelity**
   - More complex shaders without performance hit
   - Real-time subsurface scattering
   - Better skin/eye rendering

---

## 📋 Optimization Checklist (Benefits Both WebGL & WebGPU)

### Model Optimizations

- [ ] **LOD System** — Multiple detail levels for Kelly
  - LOD0: Full detail (close-up)
  - LOD1: Medium detail (normal view)
  - LOD2: Low detail (far/mobile)

- [ ] **Texture Compression**
  - BC7 for desktop (high quality)
  - ASTC for mobile (smaller)
  - Use Unity's texture compression pipeline

- [ ] **Mesh Optimization**
  - Draco compression for smaller downloads
  - Merge static meshes where possible
  - Remove invisible geometry

### Rendering Optimizations

- [ ] **Baked Lighting** — Pre-compute where possible
- [ ] **Single Pass Rendering** — Minimize render passes
- [ ] **Instanced Rendering** — For repeated elements
- [ ] **Occlusion Culling** — Don't render what's not visible

### Load Time Optimizations

- [ ] **Asset Bundles** — Load Kelly incrementally
- [ ] **Addressables** — On-demand asset loading
- [ ] **Streaming Assets** — Progressive loading

---

## 🔄 Progressive Enhancement Strategy

When WebGPU reaches production readiness, implement this loader:

```javascript
// kelly-loader.js — Progressive Enhancement

async function loadKellyEngine() {
  const container = document.getElementById('kelly-container');
  
  // Check for WebGPU support
  const hasWebGPU = 'gpu' in navigator;
  
  if (hasWebGPU) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        console.log('[Kelly] WebGPU available — loading high-fidelity version');
        return loadUnityBuild('/unity/kelly-webgpu/index.html');
      }
    } catch (e) {
      console.log('[Kelly] WebGPU failed, falling back to WebGL');
    }
  }
  
  // Fallback to WebGL (always works)
  console.log('[Kelly] Using WebGL 2.0');
  return loadUnityBuild('/unity/kelly-chair/index.html');
}

function loadUnityBuild(path) {
  const iframe = document.createElement('iframe');
  iframe.src = path;
  iframe.id = 'kelly-unity-frame';
  iframe.allow = 'autoplay; fullscreen';
  document.getElementById('kelly-container').appendChild(iframe);
  return iframe;
}

// Feature detection for capabilities
function getKellyCapabilities() {
  return {
    webgpu: 'gpu' in navigator,
    webgl2: !!document.createElement('canvas').getContext('webgl2'),
    webgl1: !!document.createElement('canvas').getContext('webgl'),
    compute: 'gpu' in navigator, // Only WebGPU has compute
    offscreen: 'OffscreenCanvas' in window,
  };
}
```

---

## 📅 Migration Timeline (Estimated)

| Phase | Timeframe | Action |
|-------|-----------|--------|
| **Now** | Dec 2025 | Ship with WebGL 2.0 ✅ |
| **Monitor** | Q1 2026 | Track Unity WebGPU progress |
| **Test** | Q2 2026 | Internal WebGPU builds |
| **Dual Build** | Q3 2026 | Ship both, detect & serve |
| **Default** | Q4 2026+ | WebGPU default, WebGL fallback |

---

## 🔒 What's Locked (Won't Change)

| Asset | Format | Reason |
|-------|--------|--------|
| **Kelly Model** | FBX | Engine-agnostic, works everywhere |
| **Textures** | PNG/JPG source | Can recompress for any format |
| **Blendshapes** | ARKit-compatible | Industry standard |
| **Skeleton** | Humanoid rig | Unity/Unreal/Godot compatible |
| **Bridge API** | postMessage | Web standard, engine-independent |

---

## 📁 Build Directory Structure (Future)

```
public/unity/
├── kelly-chair/           ← Current WebGL build
│   ├── Build/
│   ├── index.html
│   └── kbridge.js
├── kelly-webgpu/          ← Future WebGPU build (same Unity project)
│   ├── Build/
│   ├── index.html
│   └── kbridge.js         ← Same bridge API!
└── kelly-loader.js        ← Progressive enhancement detector
```

---

## ✅ Action Items for Launch (December 17)

1. **Ship with WebGL 2.0** — Stable, universal, proven
2. **Disable Unity compression** — Prevents double-zip bug
3. **Test on multiple browsers** — Chrome, Firefox, Safari, Edge
4. **Document current build settings** — For future reproduction

## 🔮 Future Considerations

1. **Unity 6 WebGPU** — Monitor Unity's WebGPU export maturity
2. **Three.js/Babylon.js** — Alternative engines if Unity licensing changes
3. **Native WebGPU** — Custom renderer for maximum control
4. **WASM optimization** — Smaller, faster Unity builds

---

**The Kelly model you're integrating today will work with whatever rendering technology comes next. The years of work are preserved.**

---

*Document lives at: `docs/KELLY_3D_FUTURE_PROOFING.md`*
