# Curious Kelly TTS Worker (ElevenLabs + Cache Fallback)

Cloudflare Worker that provides **live ElevenLabs TTS** for the browser player with **cache-first resilience**.

Why this exists:
- `curiouskelly.com` is deployed as a **static site** (see `vercel.json`), so `/api/tts` is not available in production.
- For the Matthew Prince demo (Cloudflare CEO), routing Kelly’s voice through a Worker is both **architecturally elegant** and **reliable**.

## Endpoints

### `POST /tts`
Returns `audio/mpeg` (MP3).

Request JSON:
```json
{
  "text": "Hello!",
  "voiceId": "wAdymQH5YucAkXwmrdL0",
  "language": "en",
  "speed": 1.0,
  "day": 1,
  "phase": "hook",
  "cacheKeyHint": "day1_hook_en"
}
```

Notes:
- The worker uses a deterministic cache key derived from `voiceId + language + speed + phase + textHash`.
- If ElevenLabs fails, it will attempt **fallback audio** if `day` and `phase` are provided (see below).

Fallback behavior:
- If ElevenLabs fails (timeout/non-200), the worker tries to serve a pre-generated MP3 from:
  - `AUDIO_PREGEN` (R2) at key: `pregen/{day}/{phase}/{language}.mp3`
  - If not found, returns `503` with JSON payload describing the failure.

## Bindings & Secrets

R2 bindings (recommended):
- `AUDIO_CACHE`: generated MP3 cache
- `AUDIO_PREGEN`: optional pre-generated MP3 library (fallback)

Secrets:
- `ELEVENLABS_API_KEY`: set with `wrangler secret put ELEVENLABS_API_KEY`

## Deploy
```bash
cd infrastructure/cloudflare/tts-worker
npm install
npx wrangler deploy
```

## CORS
This worker locks CORS to:
- `https://www.curiouskelly.com`
- `https://curiouskelly.com`
- localhost dev origins



