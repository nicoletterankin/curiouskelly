# ElevenLabs Optimal Setup Guide for Curious Kelly

**Last Updated:** December 2, 2025  
**Purpose:** Complete guide for setting up ElevenLabs TTS and Conversational AI for Kelly

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Account Setup](#account-setup)
3. [Voice Cloning (Kelly's Voice)](#voice-cloning-kellys-voice)
4. [TTS API Setup](#tts-api-setup)
5. [Conversational AI Setup](#conversational-ai-setup)
6. [Environment Configuration](#environment-configuration)
7. [Cost Optimization](#cost-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Curious Kelly uses ElevenLabs for two distinct features:

| Feature | Purpose | API Used |
|---------|---------|----------|
| **Text-to-Speech (TTS)** | Kelly narrates lessons | `/v1/text-to-speech` |
| **Conversational AI** | Real-time voice chat with Kelly | Conversational AI WebSocket |

---

## Account Setup

### 1. Create ElevenLabs Account

1. Go to [elevenlabs.io](https://elevenlabs.io)
2. Sign up with email or Google
3. Verify your email

### 2. Choose a Plan

**Recommended Plans for Curious Kelly:**

| Plan | Characters/Month | Voices | Price | Best For |
|------|-----------------|--------|-------|----------|
| **Starter** | 30,000 | 10 | $5/mo | Development/Testing |
| **Creator** | 100,000 | 30 | $22/mo | Small user base |
| **Pro** | 500,000 | 160 | $99/mo | Production (recommended) |
| **Scale** | 2,000,000 | 660 | $330/mo | High traffic |

**For Curious Kelly Production:**
- Start with **Creator** ($22/mo) for MVP
- Upgrade to **Pro** ($99/mo) when you have 500+ daily users
- Average lesson = ~2,000 characters
- 100,000 chars = ~50 full lessons/day

### 3. Get Your API Key

1. Go to [Profile Settings](https://elevenlabs.io/app/settings/api-keys)
2. Click "Create API Key"
3. Copy the key (starts with `sk_`)
4. **Never expose this in client-side code!**

---

## Voice Cloning (Kelly's Voice)

### Option A: Use Existing Voice ID

Kelly already has a trained voice: `wAdymQH5YucAkXwmrdL0`

This is configured in `public/config.js`:
```javascript
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
```

### Option B: Create a New Voice Clone

If you want to create a new Kelly voice:

#### Requirements (per CLAUDE.md):
- **Minimum 60 minutes** of training audio
- High-quality recordings (no background noise)
- Consistent microphone and environment
- Sample rate: 44.1kHz or 48kHz
- Format: WAV or MP3

#### Steps:

1. **Go to Voice Lab:**
   - Navigate to [Voice Lab](https://elevenlabs.io/app/voice-lab)
   - Click "Add Voice" → "Instant Voice Cloning" or "Professional Voice Cloning"

2. **For Instant Clone (Quick):**
   - Upload 1-5 minutes of audio
   - Name: "Kelly - Curious Educator"
   - Description: "Warm, curious, encouraging teacher voice"
   - Click "Add Voice"

3. **For Professional Clone (Best Quality):**
   - Apply for Professional Voice Cloning
   - Upload 60+ minutes of audio
   - Wait for ElevenLabs to process (24-48 hours)

4. **Voice Settings (Optimal for Kelly):**
   ```json
   {
     "stability": 0.50,
     "similarity_boost": 0.80,
     "style": 0.35,
     "use_speaker_boost": true
   }
   ```

5. **Copy the Voice ID:**
   - Click on your voice in Voice Lab
   - Find "Voice ID" in the settings
   - Copy and update `public/config.js`

---

## TTS API Setup

### Server-Side Proxy (Already Implemented)

Kelly uses a secure server-side proxy at `api/tts.ts` to protect the API key.

**Environment Variables Required:**
```bash
# In Vercel Dashboard or .env.local
ELEVENLABS_API_KEY=sk_your_api_key_here
ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0  # Optional, has default
```

### Set Up in Vercel:

1. Go to your Vercel project
2. Settings → Environment Variables
3. Add:
   - Name: `ELEVENLABS_API_KEY`
   - Value: Your API key (sk_...)
   - Environment: Production, Preview, Development

### Test the TTS API:

```bash
curl -X POST https://curiouskelly.com/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am Kelly!", "voiceId": "wAdymQH5YucAkXwmrdL0"}'
```

---

## Conversational AI Setup

### 1. Create a Conversational AI Agent

1. Go to [Conversational AI](https://elevenlabs.io/app/conversational-ai)
2. Click "Create Agent"
3. Configure:

**Basic Settings:**
```
Name: Kelly - Daily Lesson Teacher
Voice: [Select your Kelly voice]
Language: English
```

**System Prompt (paste this):**
```
You are Kelly, a warm and curious educator who helps people learn something new every day.

YOUR PERSONALITY:
- Warm, encouraging, genuinely curious
- You celebrate small wins enthusiastically
- You make complex topics feel approachable
- You use analogies and real-world examples
- You're playful but never condescending

YOUR GOAL:
- Help learners engage with their daily lesson
- Keep responses concise (2-3 sentences)
- Always be supportive, never judgmental
- Relate everything back to the current lesson topic

IMPORTANT:
- Sound natural, like a friendly teacher
- Express genuine excitement about learning
- Use "we" to create togetherness
```

**First Message:**
```
I'm here! What's on your mind about today's lesson?
```

**Advanced Settings:**
- Response Length: Short to Medium
- Temperature: 0.7 (balanced creativity)
- Interruption Sensitivity: Medium
- End Call On Silence: 30 seconds

### 2. Get Your Agent ID

1. After creating the agent, click on it
2. Go to "Settings" or "Configuration"
3. Find "Agent ID" (format: `agent_xxxxxxxx`)
4. Copy it

### 3. Configure in Curious Kelly

**Option A: In config.js (for testing):**
```javascript
window.ELEVENLABS_AGENT_ID = 'agent_your_agent_id_here';
```

**Option B: Via Environment Variable (recommended):**
```javascript
// In config.js
window.ELEVENLABS_AGENT_ID = '%ELEVENLABS_AGENT_ID%'; // Replaced at build time
```

Then in Vercel:
- Name: `ELEVENLABS_AGENT_ID`
- Value: `agent_your_agent_id_here`

### 4. Enable WebSocket Access (Important!)

By default, Conversational AI only works from localhost. To enable production:

1. In Agent Settings → "Allowed Origins"
2. Add:
   - `https://curiouskelly.com`
   - `https://*.vercel.app` (for previews)
   - `http://localhost:*` (for development)

---

## Environment Configuration

### Complete .env.local file:

```bash
# Supabase
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...

# ElevenLabs TTS (server-side only)
ELEVENLABS_API_KEY=sk_your_api_key_here
ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0

# ElevenLabs Conversational AI
ELEVENLABS_AGENT_ID=agent_your_agent_id_here

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Anthropic (for comment generation)
ANTHROPIC_API_KEY=sk-ant-...
```

### Vercel Environment Variables:

| Variable | Value | Environments |
|----------|-------|--------------|
| `ELEVENLABS_API_KEY` | sk_... | All |
| `ELEVENLABS_VOICE_ID` | wAdymQH5YucAkXwmrdL0 | All |
| `ELEVENLABS_AGENT_ID` | agent_... | All |

---

## Cost Optimization

### TTS Cost Reduction:

1. **Cache Audio:**
   - Kelly audio already caches in browser
   - Consider R2/S3 caching for common phrases

2. **Pre-generate Lesson Audio:**
   ```bash
   # Generate audio for all 365 lessons offline
   python scripts/generate_lesson_audio.py --all
   ```

3. **Use Turbo Model:**
   ```javascript
   // In api/tts.ts
   model_id: 'eleven_turbo_v2'  // Faster, cheaper
   ```

4. **Optimize Text:**
   - Remove filler words
   - Use contractions
   - Average lesson: aim for <2000 characters

### Conversational AI Cost:

- Charged per minute of conversation
- ~$0.08/minute on Pro plan
- Set reasonable timeouts (30s silence = end)

### Monthly Budget Estimate:

| Usage | TTS Cost | ConvAI Cost | Total |
|-------|----------|-------------|-------|
| 100 users/day | ~$20 | ~$15 | ~$35 |
| 500 users/day | ~$80 | ~$60 | ~$140 |
| 1000 users/day | ~$150 | ~$100 | ~$250 |

---

## Troubleshooting

### TTS Not Working

**Symptom:** Kelly is silent, no audio plays

**Solutions:**
1. Check Vercel logs for API errors
2. Verify `ELEVENLABS_API_KEY` is set
3. Test the API directly:
   ```bash
   curl https://curiouskelly.com/api/tts \
     -X POST -H "Content-Type: application/json" \
     -d '{"text":"Test"}'
   ```

### Conversational AI Not Connecting

**Symptom:** Mic button does nothing or shows error

**Solutions:**
1. Check `ELEVENLABS_AGENT_ID` is set correctly
2. Verify agent's "Allowed Origins" includes your domain
3. Check browser console for WebSocket errors
4. Ensure HTTPS (WebSocket requires secure context)

### Voice Sounds Wrong

**Symptom:** Kelly's voice sounds robotic or different

**Solutions:**
1. Check voice settings in ElevenLabs dashboard
2. Adjust stability (lower = more expressive)
3. Ensure using correct Voice ID
4. Try regenerating with different settings

### Rate Limiting

**Symptom:** 429 errors, "Too Many Requests"

**Solutions:**
1. Implement request queuing
2. Add delays between requests
3. Upgrade to higher tier
4. Use caching aggressively

---

## Quick Start Checklist

- [ ] Create ElevenLabs account
- [ ] Choose appropriate plan (Creator for MVP)
- [ ] Get API key from settings
- [ ] Set `ELEVENLABS_API_KEY` in Vercel
- [ ] Create Conversational AI agent
- [ ] Copy Agent ID
- [ ] Set `ELEVENLABS_AGENT_ID` in Vercel/config.js
- [ ] Add your domain to agent's Allowed Origins
- [ ] Test TTS: `/api/tts` endpoint
- [ ] Test ConvAI: Click mic button on lesson page
- [ ] Monitor usage in ElevenLabs dashboard

---

## Support Resources

- [ElevenLabs Documentation](https://docs.elevenlabs.io)
- [API Reference](https://docs.elevenlabs.io/api-reference)
- [Conversational AI Docs](https://docs.elevenlabs.io/conversational-ai)
- [Discord Community](https://discord.gg/elevenlabs)
- [Status Page](https://status.elevenlabs.io)

---

## Contact

For Curious Kelly-specific issues:
- Email: hello@curiouskelly.com
- Check `/docs/` for other implementation guides



