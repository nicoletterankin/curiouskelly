# ElevenLabs Conversational AI Setup - Complete Guide

**Created:** December 3, 2025  
**Agent ID:** `agent_3501kbg14w37er08w0mq13bvhy64`  
**Goal:** Voice chat with Kelly on curiouskelly.com/learn + avatar expression driving

---

## 📋 Diagnostic Summary

### What Was Found

| Component | Status | Issue |
|-----------|--------|-------|
| Agent ID | ✅ Configured | `agent_3501kbg14w37er08w0mq13bvhy64` in `config.js` |
| Voice ID | ✅ Configured | `wAdymQH5YucAkXwmrdL0` (Kelly's voice) |
| SDK | ❌ Not installed | Was using custom WebSocket |
| Audio Format | ❌ Wrong format | Was sending webm/opus, needed PCM 16-bit |
| Signed URLs | ❌ Not implemented | No backend endpoint existed |
| Expression Bridge | ⚠️ Partial | Only worked for some events |

### Root Causes

1. **Audio format mismatch**: The original code sent audio as webm/opus, but ElevenLabs expects PCM 16-bit audio
2. **No authentication flow**: Private agents require signed URLs from the backend
3. **Incomplete message handling**: Not all ElevenLabs message types were handled

---

## ✅ What Was Fixed

### 1. New API Endpoints Created

#### `/api/elevenlabs-signed-url` (POST)
- Generates secure signed URLs for private agents
- Falls back gracefully if agent is public
- Location: `api/elevenlabs-signed-url.ts`

#### `/api/elevenlabs-webhook` (POST)
- Receives real-time events from ElevenLabs
- Handles: conversation.started, conversation.ended, agent.response, user.transcript
- Location: `api/elevenlabs-webhook.ts`

### 2. Rewritten Conversation Handler

`public/js/kelly-conversation.js` v2.0 now includes:

- ✅ **Proper PCM 16-bit audio encoding** for microphone input
- ✅ **Signed URL support** for private agents
- ✅ **Expression bridge** connecting voice to Kelly avatar (2D, Unity, and production assets)
- ✅ **Complete message type handling** for all ElevenLabs events
- ✅ **Audio playback queue** for smooth Kelly voice output
- ✅ **Better error handling** with user-friendly messages
- ✅ **UI state management** for the Talk to Kelly button

### 3. Expression Bridge

Voice events now trigger Kelly avatar states:

| Voice Event | Kelly Expression | Avatar Action |
|-------------|------------------|---------------|
| Listening | `listening` | Shows attentive pose |
| Thinking | `thinking` | Shows curious/processing pose |
| Speaking | `explaining` | Triggers lip-sync animation |
| Idle | `hello` | Returns to neutral |

---

## 🔧 Required: ElevenLabs Dashboard Configuration

### 1. Check Agent Visibility

Go to [ElevenLabs Conversational AI](https://elevenlabs.io/app/conversational-ai) and verify:

1. Click on your agent (Curious Kelly v2)
2. Go to **Settings** or **Configuration**
3. Check **Agent Visibility**:
   - If **Public**: Direct WebSocket connection works ✅
   - If **Private**: Signed URLs are required (now implemented ✅)

### 2. Add Allowed Origins (IMPORTANT!)

In agent settings → **Allowed Origins**, add:

```
https://curiouskelly.com
https://www.curiouskelly.com
https://*.vercel.app
http://localhost:*
```

⚠️ **If this is not configured, WebSocket connections will be rejected!**

### 3. Configure Webhook (Optional but Recommended)

In agent settings → **Webhooks**, add:

```
URL: https://curiouskelly.com/api/elevenlabs-webhook
Events: conversation.started, conversation.ended, agent.response, user.transcript
```

### 4. Add Tools (Currently Empty)

In agent settings → **Tools**, add these for enhanced functionality:

#### Tool 1: Get Current Lesson
```json
{
  "name": "get_current_lesson",
  "description": "Retrieves the current lesson content for the user",
  "parameters": {
    "type": "object",
    "properties": {
      "lesson_id": {
        "type": "string",
        "description": "The lesson identifier"
      }
    }
  }
}
```

#### Tool 2: Navigate to Lesson
```json
{
  "name": "navigate_to_lesson",
  "description": "Navigates the user to a specific lesson",
  "parameters": {
    "type": "object",
    "properties": {
      "lesson_number": {
        "type": "integer",
        "description": "Day number 1-365"
      }
    }
  }
}
```

### 5. Recommended Advanced Settings

| Setting | Current | Recommended |
|---------|---------|-------------|
| Eagerness | Normal | **High** (Kelly should be responsive) |
| Take turn after silence | 7s | **3s** (faster for kids) |
| End conversation after silence | -1 | **120s** (auto-end after 2 min) |
| Max conversation duration | 600s | **300s** (5 min for focused learning) |

### 6. Knowledge Base (Recommended)

Upload to the agent's Knowledge Base:
- Lesson summaries/content
- Kelly's personality guide
- Age-appropriate response guidelines

---

## 🔑 Environment Variables Required

### Vercel Dashboard

Add these environment variables in your Vercel project:

| Variable | Value | Required |
|----------|-------|----------|
| `ELEVENLABS_API_KEY` | `sk_...` | ✅ Yes |
| `ELEVENLABS_AGENT_ID` | `agent_3501kbg14w37er08w0mq13bvhy64` | Optional (hardcoded) |
| `ELEVENLABS_VOICE_ID` | `wAdymQH5YucAkXwmrdL0` | Optional (hardcoded) |

### Local Development

Create `.env.local` in project root:

```bash
ELEVENLABS_API_KEY=sk_your_api_key_here
ELEVENLABS_AGENT_ID=agent_3501kbg14w37er08w0mq13bvhy64
ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0
```

---

## 🧪 Testing Guide

### 1. Test Signed URL Endpoint

```bash
curl -X POST https://curiouskelly.com/api/elevenlabs-signed-url \
  -H "Content-Type: application/json"
```

Expected response (public agent):
```json
{
  "signedUrl": null,
  "agentId": "agent_3501kbg14w37er08w0mq13bvhy64",
  "isPublic": true,
  "message": "Agent is public, use direct connection with agent ID"
}
```

### 2. Test Voice Chat

1. Go to https://curiouskelly.com/learn.html
2. Click "Talk to Kelly" button
3. Allow microphone access when prompted
4. Speak to Kelly
5. Check browser console for debug messages

### 3. Debug Console Messages

Look for these in browser console:
```
[KellyConversation v2] Initialized
[KellyConversation v2] Starting conversation...
[KellyConversation v2] Using public agent connection
[KellyConversation v2] WebSocket connected
[KellyConversation v2] Listening started
```

### 4. Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Connection failed" | Origin not whitelisted | Add domain to Allowed Origins |
| "Microphone access denied" | Browser permission | Allow microphone in browser |
| "Agent not available" | Agent ID wrong | Verify agent ID in config.js |
| WebSocket closes immediately | Auth failed | Check API key / signed URL |

---

## 📁 Files Changed/Created

### New Files
- `api/elevenlabs-signed-url.ts` - Signed URL generator
- `api/elevenlabs-webhook.ts` - Webhook handler

### Modified Files
- `public/js/kelly-conversation.js` - Complete rewrite (v2.0)

### Existing Files (Referenced)
- `public/config.js` - Agent ID configuration
- `public/learn.html` - Talk to Kelly button
- `api/tts.ts` - TTS proxy (separate from ConvAI)

---

## 🔄 Audio Flow (Fixed)

```
User speaks → getUserMedia (16kHz mono)
            → Float32 to Int16 PCM conversion
            → Base64 encoding
            → WebSocket → ElevenLabs ASR
            → LLM processing
            → TTS generation
            → MP3 audio chunks
            → WebSocket → Browser
            → AudioContext decoding
            → Kelly speaks + Expression updates
```

---

## 📱 Mobile Considerations

- Touch events properly handle mic button
- Transcript positioned above keyboard area
- AudioContext requires user interaction to start (handled)
- Safari requires `webkitAudioContext` (supported)

---

## 💰 Cost Considerations

Conversational AI is charged per minute:
- ~$0.08/minute on Pro plan
- Set reasonable timeouts
- Auto-end after 2 min silence

Monitor usage at: https://elevenlabs.io/app/usage

---

## 🚀 Next Steps

1. **Configure ElevenLabs Dashboard** - Add allowed origins
2. **Set Environment Variables** - Add API key to Vercel
3. **Test on Production** - Verify voice chat works
4. **Monitor Usage** - Track conversation minutes
5. **Add Knowledge Base** - Upload lesson content

---

## Support

- ElevenLabs Docs: https://docs.elevenlabs.io/conversational-ai
- Kelly Support: hello@curiouskelly.com



