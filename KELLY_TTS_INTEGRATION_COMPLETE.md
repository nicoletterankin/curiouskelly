# Kelly TTS Integration - Setup Complete ✅

## What Was Done

### ✅ Backend Changes (COMPLETED)

1. **`curious-kellly/backend/src/services/voice.js`**
   - ✅ Updated `generateSpeech()` method to call ElevenLabs API
   - ✅ Returns audio binary (MPEG format)
   - ✅ Uses Kelly's voice ID: `wAdymQH5YucAkXwmrdL0`

2. **`curious-kellly/backend/src/api/voice.js`**
   - ✅ Added new route: `POST /api/voice/tts`
   - ✅ Accepts: `{ age: 35, text: "Hello" }`
   - ✅ Returns: Audio MPEG binary

### ✅ Unity Scripts (COMPLETED)

1. **`KellyTTSClient.cs`** (Created in both locations)
   - ✅ Requests TTS from backend
   - ✅ Receives audio and plays via KellyTalkTest
   - ✅ Context menu: "Speak (ElevenLabs)" for testing

---

## What You Need To Do Next

### Step 1: Add ElevenLabs API Key (REQUIRED)

1. Open: `curious-kellly/backend/.env`
2. Add this line (replace with your actual key):
   ```
   ELEVENLABS_API_KEY=your_actual_key_here
   ```
3. Optional: Add voice ID (defaults to Kelly25):
   ```
   ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0
   ```
4. Save the file

**Get your key from:** https://elevenlabs.io/app/settings/api-keys

### Step 2: Start Backend Server

Run this command:
```powershell
.\start_backend.ps1
```

Or manually:
```powershell
cd curious-kellly\backend
npm run dev
```

You should see: `Server listening on http://localhost:3000`

### Step 3: Unity Setup

1. **Add Component:**
   - Select `kelly_character` in Hierarchy
   - Inspector → Add Component → Search "Kelly TTS Client" → Add

2. **Configure:**
   - `Text To Speak`: "Hello! I'm Kelly and I'm here to help you learn."
   - `Learner Age`: 35

3. **Disable Gaze** (to prevent flipping):
   - In `Blendshape Driver 60fps` → Uncheck "Enable Gaze"

### Step 4: Test

1. **Press Play** ▶️ in Unity
2. **In Inspector**, find `Kelly TTS Client` component
3. **Right-click** the component name → Select **"Speak (ElevenLabs)"**
4. **Listen** - Kelly should speak via ElevenLabs!

---

## How It Works

```
Unity → POST /api/voice/tts { age: 35, text: "Hello" }
   ↓
Backend → ElevenLabs API (with Kelly's voice)
   ↓
Backend → Returns audio MPEG
   ↓
Unity → Plays audio → Kelly's mouth moves
```

---

## Troubleshooting

### Backend won't start
- Check `.env` has `ELEVENLABS_API_KEY` set
- Check `OPENAI_API_KEY` is also set (required for other features)

### Unity can't connect
- Make sure backend is running (`npm run dev`)
- Check backend shows: `Server listening on http://localhost:3000`
- Check Unity Console for connection errors

### No audio plays
- Check Unity Console for error messages
- Verify `KellyTalkTest` component is assigned to `KellyTTSClient`
- Check backend terminal for API errors

### Lips don't move
- Volume-based lip-sync is basic (mouth opens/closes)
- For better lip-sync: Use Audio2Face JSON data or viseme updates
- Ensure `Enable Gaze` is unchecked (prevents flipping)

---

## Files Changed

- ✅ `curious-kellly/backend/src/services/voice.js` - ElevenLabs integration
- ✅ `curious-kellly/backend/src/api/voice.js` - TTS endpoint
- ✅ `digital-kelly/engines/kelly_unity_player/Assets/Kelly/Scripts/KellyTTSClient.cs`
- ✅ `digital-kelly/engines/kelly_unity_player/My project/Assets/Kelly/Scripts/KellyTTSClient.cs`

---

## Next Steps After Testing

1. **Better Lip-Sync:** Generate Audio2Face JSON for accurate mouth movements
2. **Lesson Integration:** Connect to lesson system to auto-generate audio
3. **UI Button:** Add Unity UI button to trigger TTS
4. **Error Handling:** Add retry logic and better error messages

---

**Status:** ✅ Code Complete - Ready for API Key & Testing!










