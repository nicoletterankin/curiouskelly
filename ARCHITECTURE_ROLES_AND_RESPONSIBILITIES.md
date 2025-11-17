# Architecture: Roles, Tools, and Responsibilities

## The Three Platforms

### 1. **Claude (Content Agent)** - Lesson Author
**Location:** External (Claude.ai or content-agent-base)  
**Responsibility:** Write lesson content  
**Output:** Lesson JSON files (e.g., `balance-schema-compliant.json`)

**What Claude Does:**
- Writes lesson text content for all age variants (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- Writes content in all languages (EN, ES, FR)
- Defines voice profiles, expression cues, teaching moments
- Creates schema-compliant JSON files
- **NEVER generates audio** - Claude only writes text

**Output Files:**
- `lesson-player/balance-schema-compliant.json`
- `lesson-player/{lesson-id}-schema-compliant.json`

---

### 2. **Cursor (UI-TARS-desktop)** - Content Production Pipeline
**Location:** `C:\Users\user\UI-TARS-desktop`  
**Responsibility:** Generate all assets from lesson JSON  
**Output:** Audio files, viseme JSON, Unity builds

**What Cursor Does:**
- **Detects new/updated lesson JSON files**
- **Automatically generates audio** via ElevenLabs API
- **Automatically generates viseme JSON** via Audio2Face or viseme mapping
- **Builds Unity WebGL** and deploys to `public/unity/kelly-v1/`
- **Manages web server** and file serving
- **Never writes lesson content** - Cursor only processes what Claude creates

**Key Scripts:**
- `scripts/generate_all_lesson_audio.py` - Generates audio from lesson JSON
- `scripts/deploy_unity_webgl.ps1` - Builds and deploys Unity
- `scripts/dev-server.ps1` - Manages web server

**Output Files:**
- `lessons/audio/{lesson-id}/{age}-{lang}-{phase}.mp3` (54 files per lesson)
- `lessons/audio/{lesson-id}/{age}-{lang}-{phase}.a2f.json` (54 viseme files)
- `public/unity/kelly-v1/` (Unity WebGL build)

**Critical Rule:** Cursor runs **BUILD-TIME** processes. Audio generation happens ONCE when lesson is created/updated, NOT every time Unity runs.

---

### 3. **Unity (VS Code "My project")** - Runtime Player
**Location:** `digital-kelly/engines/kelly_unity_player`  
**Responsibility:** Play pre-generated assets  
**Output:** Visual rendering, animation, user interaction

**What Unity Does:**
- **Loads pre-generated audio files** via HTTP (never calls ElevenLabs API)
- **Loads pre-generated viseme JSON** via HTTP
- **Plays audio** synchronized with blendshape animation
- **Renders Kelly avatar** at 60 FPS
- **Handles user interactions** (play, pause, stop)
- **Never generates audio** - Unity only consumes what Cursor produced

**Key Scripts:**
- `KellyBridge.cs` - Receives lesson URLs, loads audio + viseme JSON
- `BlendshapeDriver.cs` - Animates face based on viseme data
- `LessonAudioPlayer.cs` - Plays audio synchronized with animation

**Input Files (via HTTP):**
- `http://localhost:4000/lessons/audio/balance/18-35-en-mainContent.mp3`
- `http://localhost:4000/lessons/audio/balance/18-35-en-mainContent.a2f.json`

**Critical Rule:** Unity is **RUNTIME ONLY**. It never calls external APIs. It only loads and plays pre-generated assets.

---

## The Flow: Who Does What, When, and Why

### Step 1: Claude Writes Lesson
```
Claude → Writes lesson JSON → Saves to lesson-player/balance-schema-compliant.json
```
**Why:** Claude is the content expert. Only Claude writes lesson text.

---

### Step 2: Cursor Detects & Generates Assets
```
Cursor detects new lesson JSON
  → Reads lesson JSON
  → Extracts all text content (54 segments: 6 ages × 3 languages × 3 phases)
  → Calls ElevenLabs API 54 times (BUILD-TIME)
  → Saves 54 MP3 files to lessons/audio/balance/
  → Generates 54 viseme JSON files (via Audio2Face or mapping)
  → Saves 54 JSON files to lessons/audio/balance/
  → Creates metadata.json listing all files
```
**Why:** Audio generation is expensive and slow. Do it ONCE at build-time, not every time Unity runs.

**When:** Automatically triggered when lesson JSON is created or updated.

---

### Step 3: Cursor Builds & Deploys Unity
```
Cursor → Builds Unity WebGL → Deploys to public/unity/kelly-v1/
```
**Why:** Unity build is a build-time process. Cursor manages deployment.

**When:** After audio generation completes, or manually via script.

---

### Step 4: Unity Loads & Plays (Runtime)
```
Web page → Sends PostMessage to Unity iframe:
  {
    destination: 'kelly-webgl',
    type: 'kelly-load',
    payload: {
      audioUrl: '/lessons/audio/balance/18-35-en-mainContent.mp3',
      jsonUrl: '/lessons/audio/balance/18-35-en-mainContent.a2f.json'
    }
  }

Unity → Loads audio via UnityWebRequest
Unity → Loads viseme JSON via UnityWebRequest
Unity → Plays audio + animates blendshapes
```
**Why:** Unity is the runtime player. It only consumes pre-generated assets.

**When:** When user requests to play a lesson.

---

## Critical Rules

### ✅ DO:
1. **Claude writes lesson JSON** → Cursor generates audio → Unity plays audio
2. **Audio generation happens in Cursor** (build-time, once per lesson)
3. **Unity only loads pre-generated files** (runtime, via HTTP)
4. **All assets pre-generated** before Unity needs them

### ❌ DON'T:
1. **Unity NEVER calls ElevenLabs API** - Too slow, too expensive, breaks runtime performance
2. **Claude NEVER generates audio** - Claude only writes text
3. **Cursor NEVER writes lesson content** - Cursor only processes what Claude creates
4. **No runtime audio generation** - Everything must be pre-generated

---

## Why This Architecture?

### Performance
- **Audio generation:** 2-5 seconds per file × 54 files = 2-5 minutes per lesson
- **Runtime loading:** <100ms per file
- **If Unity called API:** User waits 2-5 minutes every time they play a lesson ❌
- **Pre-generated:** User waits <100ms ✅

### Cost
- **Build-time generation:** $0.12 per lesson (one-time cost)
- **Runtime generation:** $0.12 per lesson × every play = expensive ❌
- **Pre-generated:** Free after initial generation ✅

### Reliability
- **Pre-generated:** Works offline, no API dependencies, consistent quality
- **Runtime generation:** Depends on API availability, rate limits, network issues ❌

### Scalability
- **Pre-generated:** Can serve thousands of users simultaneously
- **Runtime generation:** Limited by API rate limits ❌

---

## Automation: Who Triggers What?

### Automatic Triggers (Cursor):
1. **File watcher detects new lesson JSON** → Auto-generates audio + visemes
2. **Audio generation completes** → Auto-updates metadata.json
3. **Unity build completes** → Auto-deploys to public/unity/kelly-v1/

### Manual Triggers:
1. **Claude writes lesson** → Manual save to lesson-player/
2. **Unity build** → Manual: `scripts/deploy_unity_webgl.ps1`
3. **Test playback** → Manual: Open lesson-player/index.html

---

## Answer: Who Kicks Off ElevenLabs?

**ANSWER: Cursor (UI-TARS-desktop) kicks off ElevenLabs sync.**

**NOT Unity.** Unity never calls ElevenLabs.

**Flow:**
1. Claude writes lesson JSON
2. Cursor detects lesson JSON
3. Cursor calls ElevenLabs API (build-time)
4. Cursor saves audio files
5. Unity loads audio files (runtime)

**This is the correct architecture. Unity is a player, not a generator.**




