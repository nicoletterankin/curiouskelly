# 🎮 Unity Project Handoff Document
## Zero-Shot Prompt for Senior Engineer & Content Management Expert

**Project:** Curious Kelly - Unity Avatar Integration  
**Handoff Date:** [Current Date]  
**Unity Version:** 6000.2.10f1  
**Status:** Production-Ready with Known Limitations  
**Estimated Onboarding Time:** 2-4 hours

---

## 📋 EXECUTIVE SUMMARY

You are inheriting a Unity WebGL integration for an AI-powered educational platform called "Curious Kelly." The system renders a 3D avatar (Kelly) that teaches daily lessons. The architecture uses **dual integration patterns** (legacy canvas + modern iframe) with a sophisticated JavaScript bridge system for communication.

**Critical Context:** This project has **intentional fallbacks** - if Unity fails to load, the system gracefully degrades to static images + audio. This is by design, not a bug.

---

## 🏗️ ARCHITECTURE OVERVIEW

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    UNITY INTEGRATION LAYER                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  HTML Entry Points:                                          │
│  ├─ public/app.html          (Legacy - Direct Canvas)       │
│  ├─ app/index.html           (Modern - Iframe)              │
│  └─ public/player.html       (Fallback - No Unity)          │
│                                                               │
│  JavaScript Integration:                                      │
│  ├─ app/unity-loader.js      (Loads Unity builds)             │
│  ├─ app/unity-bridge.js      (JS ↔ Unity communication)      │
│  ├─ app/kelly-engine.js      (Master orchestrator)          │
│  └─ public/js/unity-bridge.js (Legacy bridge)                │
│                                                               │
│  Unity Builds:                                               │
│  ├─ unity/kelly-live/Build/  (Production build)             │
│  └─ unity/kelly-v1/Build/    (Legacy build)                 │
│                                                               │
│  Content Integration:                                        │
│  ├─ Supabase (lessons, user data)                            │
│  ├─ ElevenLabs (voice synthesis)                             │
│  └─ PhaseDNA system (lesson phases)                          │
└─────────────────────────────────────────────────────────────┘
```

### Integration Patterns

#### Pattern 1: Direct Canvas Embed (Legacy)
**File:** `public/app.html` (lines 704-740)

```javascript
// Direct Unity WebGL initialization
var script = document.createElement("script");
script.src = "unity/kelly-live/Build/Kelly_Web_Build.loader.js";
script.onload = () => {
  createUnityInstance(canvas, config, (progress) => {
    // Loading progress
  }).then((unityInstance) => {
    window.unityInstance = unityInstance; // Global reference
  });
};
```

**Characteristics:**
- ✅ Simple, direct initialization
- ❌ No error isolation
- ❌ Global namespace pollution
- ❌ Harder to debug

#### Pattern 2: Iframe Embed (Modern)
**File:** `app/index.html` + `app/unity-loader.js`

```javascript
// Iframe-based with error handling
this.unityLoader = new UnityLoader({
  buildUrl: '/unity/kelly-live/Build',
  useIframe: true,
  iframeId: 'unity-iframe',
  onLoad: (instance) => { /* success */ },
  onError: (type, error) => { /* graceful degradation */ }
});
```

**Characteristics:**
- ✅ Isolated execution context
- ✅ Better error handling
- ✅ Retry logic built-in
- ✅ Graceful fallback to static images

---

## 📁 FILE STRUCTURE & KEY FILES

### Unity Source Project
**Location:** `digital-kelly/engines/Kelly_Engine_V2/onlykelly/`

```
onlykelly/
├── Assets/                    # Unity assets (models, materials, scripts)
├── ProjectSettings/           # Unity project configuration
├── Packages/                  # Unity package dependencies
│   └── manifest.json         # Package manifest (URP, CC3 tools, etc.)
└── Kelly_Web_Build/          # WebGL build output (DO NOT EDIT)
```

**Unity Editor Path:** `C:\Program Files\Unity\Hub\Editor\6000.2.10f1\`  
**Project Opens In:** Unity Hub → Open Project → Select `onlykelly` folder

### WebGL Build Outputs

#### Production Build (Active)
**Path:** `unity/kelly-live/Build/`

```
Build/
├── Kelly_Web_Build.loader.js      # Unity loader script
├── Kelly_Web_Build.framework.js.br # Framework (Brotli compressed)
├── Kelly_Web_Build.wasm.br        # WebAssembly (Brotli compressed)
├── Kelly_Web_Build.data.br        # Asset data (Brotli compressed)
└── StreamingAssets/               # Streaming assets (if any)
```

**Build Configuration:**
- Platform: WebGL
- Compression: Brotli (.br)
- Template: Default WebGL template
- Resolution: 960x600 (configurable in Unity)

#### Legacy Build (Deprecated)
**Path:** `unity/kelly-v1/Build/`
- Older build, kept for reference
- Not actively used in production

### JavaScript Integration Files

#### Core Integration Modules

**`app/unity-loader.js`** (307 lines)
- **Purpose:** Unified Unity initialization
- **Methods:**
  - `load()` - Main entry point
  - `loadViaCanvas()` - Direct canvas embed
  - `loadViaIframe()` - Iframe wrapper
  - `retry()` - Retry failed loads
  - `disableUnity()` - Graceful fallback
- **Error Handling:** 3 retry attempts with exponential backoff
- **Used By:** `app/kelly-engine.js`

**`app/unity-bridge.js`** (200 lines)
- **Purpose:** Communication layer between JS and Unity
- **Transports:**
  - `postMessage` (iframe communication)
  - `WebSocket` (optional, for live streaming)
- **Events:** `bridge-handshake`, `audio-load`, `character-load`, etc.
- **Used By:** `app/kelly-engine.js`, `app/index.html`

**`app/kelly-engine.js`** (620 lines)
- **Purpose:** Master orchestrator
- **Responsibilities:**
  - Initialize Unity components
  - Load lessons from Supabase
  - Coordinate voice synthesis (ElevenLabs)
  - Manage lesson phases
  - Handle age/language/archetype changes
- **State Management:** Tracks current lesson, phase, user preferences
- **Used By:** `app/index.html`, `app/script.js`

### HTML Entry Points

**`public/app.html`** (747 lines)
- **Type:** Legacy direct canvas embed
- **Unity Integration:** Lines 704-740
- **Build Path:** `unity/kelly-live/Build`
- **Global Instance:** `window.unityInstance`
- **Auth Flow:** Redirects from `public/index.html` (lines 818, 839, 1187)
- **Dependencies:** Supabase, direct Unity loader

**`app/index.html`** (286 lines)
- **Type:** Modern iframe embed
- **Unity Integration:** Iframe with lazy loading (line 173)
- **Iframe ID:** `#unity-iframe`
- **Overlay:** `#unity-overlay` for status/errors
- **Dependencies:** `app/unity-loader.js`, `app/kelly-engine.js`

**`public/player.html`** (711 lines)
- **Type:** Fallback (NO Unity)
- **Purpose:** Static image + audio player
- **Fallback Images:** `/assets/kelly_canonical/core/chair/`
- **Audio:** ElevenLabs API + browser TTS
- **Use Case:** When Unity fails or unavailable

---

## 🔧 BUILD PROCESS

### Prerequisites

1. **Unity Hub** installed
2. **Unity Editor 6000.2.10f1** (exact version required)
3. **Project Dependencies:**
   - Universal Render Pipeline (URP) 17.2.0
   - Character Creator 3 (CC3) Unity Tools
   - WebGL build support module

### Building WebGL

#### Step 1: Open Project
```bash
# Via Unity Hub
1. Open Unity Hub
2. Click "Open" → Navigate to: digital-kelly/engines/Kelly_Engine_V2/onlykelly/
3. Wait for project to load (may take 2-3 minutes first time)
```

#### Step 2: Configure Build Settings
```
1. File → Build Settings
2. Platform: WebGL (select and click "Switch Platform" if needed)
3. Player Settings → WebGL:
   - Compression Format: Brotli
   - Template: Default
   - Resolution: 960x600 (or desired)
   - Memory Size: 512 MB (adjust if needed)
```

#### Step 3: Build
```
1. File → Build Settings → Build
2. Select output directory: unity/kelly-live/Build/
3. Wait for build (5-15 minutes depending on assets)
4. Verify output files exist:
   - Kelly_Web_Build.loader.js
   - Kelly_Web_Build.framework.js.br
   - Kelly_Web_Build.wasm.br
   - Kelly_Web_Build.data.br
```

#### Step 4: Deploy
```
# Build files are automatically served from:
unity/kelly-live/Build/

# Ensure web server can serve .br files with correct MIME type:
# Content-Type: application/brotli (or application/octet-stream)
```

### Build Troubleshooting

**Issue:** Build fails with "Out of memory"
- **Fix:** Increase Unity Editor memory allocation or reduce asset quality

**Issue:** Build succeeds but won't load in browser
- **Fix:** Check browser console for CORS errors, verify .br files are served correctly

**Issue:** "createUnityInstance is not defined"
- **Fix:** Loader script not loading - check path, verify loader.js exists

---

## 🔌 INTEGRATION POINTS

### JavaScript → Unity Communication

**Method 1: Direct Instance (Legacy)**
```javascript
// In public/app.html
window.unityInstance.SendMessage('GameObjectName', 'MethodName', 'parameter');
```

**Method 2: Bridge (Modern)**
```javascript
// In app/kelly-engine.js
this.unityBridge.emit('audio-load', {
  url: audioUrl,
  phase: 'welcome',
  autoplay: true
});
```

### Unity → JavaScript Communication

Unity sends messages via `postMessage` or WebSocket:
```javascript
// Unity C# code calls:
Application.ExternalCall("OnUnityMessage", jsonData);

// JavaScript receives:
window.addEventListener('message', (event) => {
  if (event.data.type === 'unity-bridge-handshake') {
    // Handle Unity message
  }
});
```

### Content Integration

#### Supabase (Lessons)
```javascript
// Load lesson data
const lesson = await supabaseService.getCoreLesson(dayNumber);

// Lesson structure:
{
  id: number,
  day_number: number,
  topic: string,
  topic_title: string,
  category: string,
  content: { welcome, q1, q2, q3, wisdom }
}
```

#### ElevenLabs (Voice)
```javascript
// Generate audio for phase
const audio = await voiceEngine.generatePhaseAudio(
  script,
  age,
  language,
  archetype,
  tone
);

// Send to Unity
unityBridge.emit('audio-load', { url: audio.audioUrl, phase });
```

---

## 🐛 KNOWN ISSUES & GOTCHAS

### Critical Issues

1. **Unity Build Size**
   - **Issue:** Build files are large (40+ MB uncompressed)
   - **Impact:** Slow initial load, especially on mobile
   - **Mitigation:** Brotli compression reduces to ~10-15 MB
   - **Status:** Acceptable for desktop, problematic for mobile

2. **Browser Compatibility**
   - **Issue:** WebGL not supported on all browsers/devices
   - **Impact:** Unity fails to load on older browsers, some mobile devices
   - **Mitigation:** Graceful fallback to `player.html` (static images)
   - **Status:** By design - fallback system works

3. **CORS & MIME Types**
   - **Issue:** `.br` files must be served with correct MIME type
   - **Impact:** Build won't load if server misconfigured
   - **Fix:** Configure server to serve `.br` as `application/brotli` or `application/octet-stream`

4. **Memory Limits**
   - **Issue:** Unity WebGL has memory constraints
   - **Impact:** May crash on low-memory devices
   - **Mitigation:** Reduce build memory size in Unity settings

### Architectural Decisions (Not Bugs)

1. **Dual Integration Patterns**
   - **Why:** Legacy code (`app.html`) uses direct canvas, new code (`index.html`) uses iframe
   - **Action:** Consider migrating `app.html` to iframe pattern for consistency

2. **Global `window.unityInstance`**
   - **Why:** Legacy pattern for easy access
   - **Action:** Consider refactoring to module-based access

3. **Fallback System**
   - **Why:** Unity may fail to load (network, browser, device)
   - **Action:** This is intentional - don't "fix" it

---

## 📊 CONTENT MANAGEMENT WORKFLOW

### Lesson Content Structure

Lessons are stored in Supabase with the following structure:

```sql
-- Table: core_lessons
{
  id: uuid,
  day_number: integer (1-365),
  topic: string,
  topic_title: string,
  category: string,
  description: text,
  content: jsonb {
    welcome: string,
    q1: string,
    q2: string,
    q3: string,
    wisdom: string
  }
}
```

### Adding/Editing Lessons

1. **Via Supabase Dashboard:**
   - Navigate to `core_lessons` table
   - Insert/update lesson records
   - Content must include all 5 phases (welcome, q1, q2, q3, wisdom)

2. **Via API:**
   ```javascript
   await supabase
     .from('core_lessons')
     .insert({
       day_number: 1,
       topic: 'Gravity',
       topic_title: 'Understanding Gravity',
       category: 'Physics',
       content: {
         welcome: 'Welcome text...',
         q1: 'Question 1...',
         q2: 'Question 2...',
         q3: 'Question 3...',
         wisdom: 'Wisdom text...'
       }
     });
   ```

### Unity Asset Updates

**Character Models:**
- Located in Unity project: `Assets/`
- Age variants managed via Unity Asset Manager
- To update: Edit in Unity, rebuild WebGL

**Animations:**
- Managed in Unity Animator Controller
- Expression system uses Unity's animation system
- To update: Edit animations in Unity, rebuild

**Audio Integration:**
- Audio generated dynamically via ElevenLabs
- Unity receives audio URLs via bridge
- No static audio files in Unity build

---

## 🧪 TESTING PROCEDURES

### Local Testing

1. **Start Local Server:**
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Or Node.js
   npx serve .
   ```

2. **Test Unity Load:**
   - Navigate to `http://localhost:8000/app.html`
   - Check browser console for errors
   - Verify Unity canvas appears
   - Test interaction (if Unity methods exposed)

3. **Test Fallback:**
   - Disable Unity build files (rename folder)
   - Navigate to `http://localhost:8000/app.html`
   - Verify fallback to static images works

### Production Testing

1. **Check Build Files:**
   - Verify all `.br` files exist in `unity/kelly-live/Build/`
   - Check file sizes (should be 5-15 MB total)

2. **Test on Multiple Browsers:**
   - Chrome/Edge (best support)
   - Firefox (good support)
   - Safari (may have issues)
   - Mobile browsers (likely fallback)

3. **Monitor Console:**
   - Check for CORS errors
   - Check for 404s on build files
   - Check for Unity initialization errors

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Deployment

- [ ] Unity build completed successfully
- [ ] All build files present in `unity/kelly-live/Build/`
- [ ] Build files are Brotli compressed (.br)
- [ ] Web server configured to serve `.br` files with correct MIME type
- [ ] CORS headers configured (if cross-origin)
- [ ] Fallback system tested (static images work)

### Deployment Steps

1. **Build Unity Project** (see Build Process above)
2. **Copy Build Files:**
   ```bash
   # Ensure build output is in correct location
   unity/kelly-live/Build/
   ```
3. **Configure Server:**
   - Ensure `.br` MIME type configured
   - Ensure CORS headers set (if needed)
   - Ensure gzip/brotli compression enabled
4. **Deploy HTML/JS Files:**
   - `public/app.html`
   - `app/index.html`
   - `app/unity-loader.js`
   - `app/unity-bridge.js`
   - `app/kelly-engine.js`
5. **Test:**
   - Load `app.html` in production
   - Verify Unity loads
   - Test fallback if Unity fails

---

## 🔍 TROUBLESHOOTING GUIDE

### Unity Won't Load

**Symptom:** Canvas stays blank, no Unity content

**Diagnosis:**
1. Open browser DevTools → Console
2. Check for errors:
   - `Failed to load resource` → Build files missing
   - `CORS error` → Server CORS misconfigured
   - `createUnityInstance is not defined` → Loader script not loading
   - `Out of memory` → Device/browser memory limit

**Solutions:**
- Verify build files exist: `unity/kelly-live/Build/Kelly_Web_Build.loader.js`
- Check server logs for 404s
- Verify MIME types for `.br` files
- Test in different browser
- Check device memory (mobile devices may fail)

### Unity Loads But No Communication

**Symptom:** Unity canvas appears but no interaction

**Diagnosis:**
1. Check `window.unityInstance` exists (legacy pattern)
2. Check Unity Bridge connection status
3. Verify Unity C# scripts are calling JavaScript methods

**Solutions:**
- Check browser console for bridge handshake messages
- Verify `unity-bridge.js` is loaded
- Check Unity build includes communication scripts
- Test with `unityInstance.SendMessage()` directly

### Build Files Too Large

**Symptom:** Slow load times, timeouts

**Solutions:**
1. Reduce Unity build memory size
2. Optimize textures/models in Unity
3. Enable better compression (Brotli)
4. Consider lazy loading non-critical assets
5. Use CDN for build files

### Fallback Not Working

**Symptom:** When Unity fails, nothing appears

**Solutions:**
- Verify `player.html` exists and is accessible
- Check fallback image paths: `/assets/kelly_canonical/`
- Ensure error handling in `unity-loader.js` calls `disableUnity()`

---

## 📚 ADDITIONAL RESOURCES

### Documentation Files

- `app/UNITY_INTEGRATION_GUIDE.md` - Integration examples
- `KELLY_AVATAR_DEPLOYMENT_GUIDE.md` - Deployment details
- `UNITY_INTEGRATION_PLAN.md` - Original implementation plan
- `UNITY_INTEGRATION_SUMMARY.md` - Implementation summary

### Code References

- Unity Source: `digital-kelly/engines/Kelly_Engine_V2/onlykelly/`
- WebGL Builds: `unity/kelly-live/Build/`
- Integration: `app/unity-loader.js`, `app/unity-bridge.js`, `app/kelly-engine.js`
- Entry Points: `public/app.html`, `app/index.html`

### External Dependencies

- **Supabase:** Database for lessons and user data
- **ElevenLabs:** Voice synthesis API
- **Unity WebGL:** Runtime for 3D rendering

---

## ⚠️ CRITICAL WARNINGS

1. **DO NOT** modify Unity build files directly - rebuild from source
2. **DO NOT** change Unity version without testing thoroughly
3. **DO NOT** remove fallback system - it's intentional
4. **DO** test on multiple browsers before deploying
5. **DO** verify build files after every Unity rebuild
6. **DO** keep backup of working build before changes

---

## 🎯 QUICK START FOR NEW ENGINEER

### First Day Tasks

1. **Understand Architecture** (1 hour)
   - Read this document
   - Review `app/unity-loader.js` code
   - Review `app/kelly-engine.js` code

2. **Set Up Environment** (30 minutes)
   - Install Unity Hub + Unity 6000.2.10f1
   - Open Unity project: `digital-kelly/engines/Kelly_Engine_V2/onlykelly/`
   - Verify project opens without errors

3. **Test Build Process** (1 hour)
   - Make a small change in Unity (e.g., change background color)
   - Build WebGL to `unity/kelly-live/Build/`
   - Test locally: `python -m http.server 8000` → `http://localhost:8000/app.html`
   - Verify change appears

4. **Understand Integration** (1 hour)
   - Trace code flow: `app.html` → `unity-loader.js` → Unity build
   - Test bridge communication (if Unity methods exposed)
   - Test fallback system (disable build files, verify fallback)

### First Week Tasks

1. **Code Audit**
   - Review all Unity integration files
   - Identify technical debt
   - Document any issues found

2. **Improvement Plan**
   - Migrate `app.html` to iframe pattern?
   - Optimize build size?
   - Improve error handling?

3. **Content Workflow**
   - Understand Supabase lesson structure
   - Test adding/editing lessons
   - Document content management process

---

## 📞 SUPPORT & ESCALATION

### When to Escalate

- Unity build consistently fails
- Critical production issue affecting users
- Need to change Unity version
- Major architectural changes needed

### Useful Commands

```bash
# Check Unity build files exist
ls -lh unity/kelly-live/Build/

# Test local server
python -m http.server 8000

# Check for Unity references in code
grep -r "unity" app/ public/ --include="*.js" --include="*.html"
```

---

## ✅ FINAL CHECKLIST

Before considering this handoff complete, ensure you:

- [ ] Can open Unity project in Unity Editor
- [ ] Can build WebGL successfully
- [ ] Understand the dual integration patterns
- [ ] Know where build files go
- [ ] Understand the fallback system
- [ ] Can troubleshoot common issues
- [ ] Know how to add/edit lessons
- [ ] Have tested locally
- [ ] Have read all referenced documentation

---

**End of Handoff Document**

*This document represents the complete state of the Unity integration as of handoff. All architectural decisions, known issues, and workflows are documented above. If something is unclear, it's likely intentional (fallback system) or needs investigation (new issue).*

*Good luck! 🚀*


