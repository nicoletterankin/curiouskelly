# 🔑 BYOK Lesson Experience Directive

**Mission:** Leverage user-provided API keys to enhance their lesson experience while building community flywheel.

**Generated:** December 21, 2025  
**Status:** READY TO IMPLEMENT

---

## 📊 Current State Analysis

### What Exists (COMPLETE)

| Component | Location | Status |
|-----------|----------|--------|
| **BYOKManager** | `public/js/byok-manager.js` | ✅ Full implementation |
| **Provider Support** | OpenAI, Anthropic, Google, HeyGen, ElevenLabs | ✅ All wired |
| **Key Storage** | `localStorage` with `kelly_byok_keys` | ✅ Working |
| **Settings UI** | `learn.html` Settings panel | ✅ Multi-provider cards |
| **Chat with BYOK** | `sendByokMessage()` in `learn.html` | ✅ OpenAI/Anthropic |
| **Generation Queue** | `kelly-generation-queue.js` | ✅ Schema ready |

### What's NOT Connected (GAPS)

| Feature | Current Behavior | Opportunity |
|---------|------------------|-------------|
| **TTS Audio** | Uses platform's ElevenLabs key only | User's ElevenLabs key could provide Kelly's voice |
| **HeyGen Video** | Not integrated into lesson player | User's HeyGen credits could generate response videos |
| **Image Generation** | Static infographics only | User's Google/OpenAI key could generate contextual visuals |
| **Community Pooling** | Queue exists but not active | User keys could generate content for everyone |

---

## 🏗️ Architecture

### Current TTS Flow (Platform-Only)
```
User clicks lesson → Kelly speaks
        ↓
KellyAudio.speak(text)
        ↓
fetch('/api/tts', { text, voiceId })
        ↓
Cloudflare Worker (tts.curiouskelly.com)
        ↓
Uses ELEVENLABS_API_KEY (platform secret)
        ↓
Returns audio/mpeg → plays
```

### Target TTS Flow (BYOK-Enhanced)
```
User clicks lesson → Kelly speaks
        ↓
KellyAudio.speak(text)
        ↓
Check: BYOKManager.hasProvider('elevenlabs') ?
        ↓
YES → BYOKManager.generateTTS({ text, voice })
      (Uses user's ElevenLabs key directly)
        ↓
NO → fetch('/api/tts') 
     (Platform fallback)
        ↓
Returns audio → plays
```

---

## 📁 Key Files

### Core BYOK System
```
public/js/byok-manager.js          # THE manager - handles all providers
  ├── providers{}                   # OpenAI, Anthropic, Google, HeyGen, ElevenLabs
  ├── saveKey(providerId, key)      # Validate + store
  ├── getKey(providerId)            # Retrieve
  ├── generate(capability, opts)    # Route to correct provider
  ├── generateChat()                # OpenAI/Anthropic/Google
  ├── generateTTS()                 # ElevenLabs/OpenAI
  ├── generateVideo()               # HeyGen
  └── generateImage()               # Google/OpenAI
```

### Lesson Player Integration Points
```
public/js/kelly-audio.js           # TTS playback
  └── _speakWithElevenLabs()        # ← INSERT BYOK CHECK HERE

public/learn.html                   # Main player
  ├── sendByokMessage()             # ← Already uses BYOK for chat
  ├── loadLessonRuntime()           # ← Could trigger BYOK generation
  └── playPhaseVideo()              # ← Could use BYOK HeyGen
```

### Settings UI
```
public/learn.html (lines 9227-9256)  # BYOK Settings Panel
  ├── Provider cards                  # OpenAI, Anthropic, Google, etc.
  ├── Key input + test button         # Validates key format + API
  └── Status indicators               # Shows connected providers
```

---

## 🎯 Implementation Tasks

### TASK 1: BYOK TTS in Lesson Player (HIGH IMPACT)

**File:** `public/js/kelly-audio.js`

**Current:** Lines 215-252 always call `/api/tts` (platform key)

**Change:** Check for user's ElevenLabs key first

```javascript
// In _speakWithElevenLabs() - around line 215
async _speakWithElevenLabs(text, options) {
  if (window.__KELLY_TTS_DISABLED || this.ttsAvailable === false) {
    throw new Error('TTS unavailable');
  }

  // Check cache first
  const cacheKey = `${text}-${options.language || 'en'}`;
  if (this.audioCache.has(cacheKey)) {
    return this._playAudioBuffer(this.audioCache.get(cacheKey));
  }

  // NEW: Try user's BYOK ElevenLabs key first
  if (window.BYOKManager?.hasProvider('elevenlabs')) {
    try {
      const result = await window.BYOKManager.generateTTS('elevenlabs', 
        window.BYOKManager.getKey('elevenlabs'),
        { 
          text, 
          voice: this.options.kellyVoiceId // wAdymQH5YucAkXwmrdL0
        }
      );
      if (result.success && result.audioUrl) {
        await this._playAudioUrl(result.audioUrl);
        return;
      }
    } catch (e) {
      console.warn('[KellyAudio] BYOK TTS failed, falling back to platform:', e);
    }
  }

  // Fallback: Platform API
  const response = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      voiceId: this.options.kellyVoiceId
    })
  });
  // ... rest of existing code
}
```

**Why This Matters:**
- User with ElevenLabs key gets instant, reliable TTS
- Doesn't depend on platform API quota
- User feels ownership ("my key powers my lessons")

---

### TASK 2: BYOK Status Badge in Player

**File:** `public/learn.html`

**Add visible indicator when user has BYOK configured:**

```javascript
// In initializeUI() or loadByokKey()
function updateBYOKStatusBadge() {
  const badge = document.getElementById('byok-status-badge');
  if (!badge || !window.BYOKManager) return;
  
  const summary = window.BYOKManager.getStatusSummary();
  
  if (summary.providersConfigured > 0) {
    badge.innerHTML = `🔑 ${summary.providersConfigured} key${summary.providersConfigured > 1 ? 's' : ''} active`;
    badge.classList.add('active');
    badge.title = summary.providers.map(p => `${p.emoji} ${p.name}`).join(', ');
  } else {
    badge.innerHTML = '🔑 Add API key';
    badge.classList.remove('active');
  }
}
```

**Add badge HTML near nav:**
```html
<button class="nav-action-btn byok-badge-btn" id="byok-status-badge" data-panel="settings" title="BYOK Status">
  🔑 Add API key
</button>
```

---

### TASK 3: "Powered by Your Key" Indicator

When BYOK is used for a specific action, show the user:

```javascript
// After successful BYOK TTS
function showByokIndicator(provider, action) {
  const indicator = document.createElement('div');
  indicator.className = 'byok-used-indicator';
  indicator.innerHTML = `
    <span class="byok-icon">${window.BYOKManager.providers[provider]?.emoji || '🔑'}</span>
    <span class="byok-text">Powered by your ${provider} key</span>
  `;
  indicator.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 12px;
    z-index: 1000;
    animation: fadeInUp 0.3s ease;
  `;
  document.body.appendChild(indicator);
  setTimeout(() => indicator.remove(), 3000);
}
```

---

### TASK 4: Community Flywheel Stats

Show users how their keys help everyone:

```html
<!-- In Settings panel BYOK section -->
<div class="byok-community-stats">
  <h5>🌀 Community Flywheel</h5>
  <div class="stat-row">
    <span>Videos generated with BYOK</span>
    <span id="byok-videos-count">--</span>
  </div>
  <div class="stat-row">
    <span>Credits saved for learners</span>
    <span id="byok-credits-saved">--</span>
  </div>
  <div class="stat-row">
    <span>Contributors this month</span>
    <span id="byok-contributors">--</span>
  </div>
</div>
```

```javascript
// Fetch from Supabase
async function loadCommunityStats() {
  const { data } = await supabase.rpc('get_byok_stats');
  if (data) {
    document.getElementById('byok-videos-count').textContent = data.videos_generated;
    document.getElementById('byok-credits-saved').textContent = `$${data.credits_saved}`;
    document.getElementById('byok-contributors').textContent = data.contributors;
  }
}
```

---

## 🔧 BYOKManager API Reference

### Check if provider is available
```javascript
window.BYOKManager.hasProvider('elevenlabs')  // true/false
window.BYOKManager.hasCapability('tts')       // true/false
```

### Get best provider for capability
```javascript
window.BYOKManager.getBestProviderForCapability('tts')  // 'elevenlabs' or 'openai'
window.BYOKManager.getBestProviderForCapability('chat') // 'anthropic', 'openai', or 'google'
```

### Generate with routing
```javascript
// Automatically routes to best available provider
const result = await window.BYOKManager.generate('tts', {
  text: "Hello, learner!",
  voice: 'wAdymQH5YucAkXwmrdL0'
});

if (result.success) {
  // result.audioUrl for TTS
  // result.content for chat
  // result.url for image
  // result.videoId for video
}
```

### Direct generation
```javascript
// TTS
await window.BYOKManager.generateTTS('elevenlabs', key, { text, voice });

// Chat
await window.BYOKManager.generateChat('openai', key, { messages, model });

// Image
await window.BYOKManager.generateImage('google', key, { prompt, size });

// Video (HeyGen)
await window.BYOKManager.generateVideo('heygen', key, { script, avatarId });
```

---

## 🚦 Implementation Order

```
1. TTS BYOK (Task 1) ─────────────────────────────────────
   Impact: HIGH - Every lesson uses TTS
   Effort: LOW - Just a conditional check
   Time: 30 minutes

2. Status Badge (Task 2) ─────────────────────────────────
   Impact: MEDIUM - User awareness
   Effort: LOW - UI addition
   Time: 20 minutes

3. "Powered by" Indicator (Task 3) ───────────────────────
   Impact: MEDIUM - User satisfaction
   Effort: LOW - Toast notification
   Time: 15 minutes

4. Community Stats (Task 4) ──────────────────────────────
   Impact: LOW - Flywheel motivation
   Effort: MEDIUM - Needs Supabase RPC
   Time: 1 hour
```

---

## 📈 Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Users with any BYOK key | Unknown | Track in Supabase |
| TTS calls using BYOK | 0% | 20%+ |
| Lesson completion with BYOK | Unknown | Higher than without |
| Community videos generated | 0 | 100/week |

---

## ⚠️ Constraints

### From CLAUDE.md
- **Never use browser TTS** - ElevenLabs only (BYOK or platform)
- **Kelly's voice ID:** `wAdymQH5YucAkXwmrdL0` (trained, locked)
- **Keys stored locally only** - Never sent to our servers

### Security
- BYOK keys never leave user's browser (except to provider APIs)
- No key logging, no key transmission to our backend
- Keys validated client-side before storage

### User Experience
- BYOK is **enhancement**, not requirement
- Lessons work without any keys (platform TTS)
- Clear messaging: "Add your own key for faster, unlimited Kelly"

---

## 🧪 Testing Checklist

```
□ No BYOK keys
  - Lessons play with platform TTS
  - Settings show "Add API key" prompt
  - No errors in console

□ ElevenLabs BYOK only
  - Kelly speaks using user's key
  - "Powered by your ElevenLabs" shows
  - Fallback works if key is invalid

□ OpenAI BYOK only
  - Chat with Kelly works
  - TTS falls back to platform (OpenAI TTS not as good)

□ Multiple BYOK keys
  - Best provider selected per capability
  - All status badges update
  - Switching works correctly

□ Invalid/expired key
  - Graceful fallback to platform
  - User notified of issue
  - Key can be removed and re-added
```

---

**This directive focuses on wiring BYOK into the actual lesson experience where users spend their time.**

*The foundation (BYOKManager) is complete. The integration points are clear. Execute Task 1 first for immediate impact.*

