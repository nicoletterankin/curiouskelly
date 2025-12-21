/**
 * Curious Kelly TTS Worker
 * - POST /tts -> ElevenLabs TTS -> returns audio/mpeg
 * - Cache generated MP3 in R2 (AUDIO_CACHE)
 * - On ElevenLabs failure, fallback to pregen MP3 in R2 (AUDIO_PREGEN)
 *
 * Secrets:
 * - ELEVENLABS_API_KEY
 *
 * Optional bindings:
 * - AUDIO_CACHE (R2Bucket)
 * - AUDIO_PREGEN (R2Bucket)
 */

const ALLOWED_ORIGINS = new Set([
  'https://www.curiouskelly.com',
  'https://curiouskelly.com',
  'http://localhost:3000',
  'http://localhost:5173',
  'http://localhost:8787',
]);

function corsHeaders(request) {
  const origin = request.headers.get('Origin') || '';
  const allowOrigin = ALLOWED_ORIGINS.has(origin) ? origin : '';
  return {
    ...(allowOrigin ? { 'Access-Control-Allow-Origin': allowOrigin } : {}),
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

function json(status, data, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...extraHeaders,
    },
  });
}

async function sha256Hex(input) {
  const data = new TextEncoder().encode(input);
  const hash = await crypto.subtle.digest('SHA-256', data);
  const bytes = new Uint8Array(hash);
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function safePhase(phase) {
  return String(phase || '').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '');
}

function safeLang(lang) {
  const l = String(lang || 'en').trim().toLowerCase();
  return /^[a-z]{2}(-[a-z]{2})?$/.test(l) ? l : 'en';
}

function clampSpeed(speed) {
  const n = Number(speed);
  if (!Number.isFinite(n)) return 1.0;
  return Math.max(0.7, Math.min(1.3, n));
}

async function getR2Object(bucket, key) {
  if (!bucket) return null;
  try {
    return await bucket.get(key);
  } catch (_) {
    return null;
  }
}

async function putR2Object(bucket, key, body, httpMetadata = {}) {
  if (!bucket) return false;
  try {
    await bucket.put(key, body, {
      httpMetadata,
    });
    return true;
  } catch (_) {
    return false;
  }
}

function audioResponse(arrayBuffer, request, extra = {}) {
  return new Response(arrayBuffer, {
    status: 200,
    headers: {
      'Content-Type': 'audio/mpeg',
      'Cache-Control': 'no-store',
      ...corsHeaders(request),
      ...extra,
    },
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(request) });
    }

    if (url.pathname !== '/tts' || request.method !== 'POST') {
      return json(
        404,
        { error: 'Not found' },
        { ...corsHeaders(request) }
      );
    }

    // NOTE: We intentionally parse JSON via request.text() rather than request.json().
    // If request.json() throws, it may have already consumed the request body stream,
    // which prevents fallback parsing and results in persistent "Invalid JSON" errors.
    // Parsing text also lets us strip an optional UTF-8 BOM.
    let payload = null;
    try {
      const raw = await request.text();
      const cleaned = raw.startsWith('\uFEFF') ? raw.slice(1) : raw;
      payload = JSON.parse(cleaned);
    } catch (_) {
      return json(400, { error: 'Invalid JSON' }, { ...corsHeaders(request) });
    }

    const text = String(payload?.text || '').trim();
    // Accept both camelCase and snake_case for compatibility with existing clients/scripts.
    // - `voiceId` is used by the Kelly web app
    // - `voice_id` is used by some scripts and older payloads
    const voiceId = String(payload?.voiceId || payload?.voice_id || '').trim();
    const language = safeLang(payload?.language || 'en');
    const phase = safePhase(payload?.phase || '');
    const day = Number.isFinite(Number(payload?.day)) ? Number(payload.day) : null;
    const speed = clampSpeed(payload?.speed ?? 1.0);

    if (!text || text.length < 1) {
      return json(400, { error: 'Missing text' }, { ...corsHeaders(request) });
    }
    if (!voiceId) {
      return json(400, { error: 'Missing voiceId' }, { ...corsHeaders(request) });
    }

    // Deterministic cache key
    const textHash = await sha256Hex(text);
    const cacheKey = `tts/v1/${voiceId}/${language}/s${speed.toFixed(2)}/${phase || 'phase'}/${textHash}.mp3`;

    // 1) Cache hit -> return
    const cached = await getR2Object(env.AUDIO_CACHE, cacheKey);
    if (cached) {
      const buf = await cached.arrayBuffer();
      return audioResponse(buf, request, {
        'X-Kelly-TTS': 'cache_hit',
        'X-Kelly-Cache-Key': cacheKey,
      });
    }

    // 2) Cache miss -> call ElevenLabs
    const elevenKey = env.ELEVENLABS_API_KEY;
    if (!elevenKey) {
      // If no key, try pregen fallback if possible
      const fallback = await tryPregenFallback({ env, request, day, phase, language });
      if (fallback) return fallback;
      return json(
        503,
        { error: 'TTS unavailable', code: 'MISSING_API_KEY', ttsAvailable: false },
        { ...corsHeaders(request) }
      );
    }

    try {
      const ttsUrl = `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}`;

      // Keep it minimal; advanced voice settings can be added later.
      const body = {
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.45,
          similarity_boost: 0.8,
          style: 0.4,
          use_speaker_boost: true,
        },
      };

      const res = await fetch(ttsUrl, {
        method: 'POST',
        headers: {
          'xi-api-key': elevenKey,
          'Content-Type': 'application/json',
          'Accept': 'audio/mpeg',
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        // Pull upstream error body for debugging (no secrets). Keep it bounded.
        let upstream = '';
        try {
          upstream = (await res.text()) || '';
        } catch (_) {
          upstream = '';
        }
        const fallback = await tryPregenFallback({ env, request, day, phase, language });
        if (fallback) return fallback;
        return json(
          503,
          {
            error: 'ElevenLabs TTS failed',
            code: 'ELEVENLABS_ERROR',
            status: res.status,
            upstream: upstream.slice(0, 2000),
          },
          { ...corsHeaders(request) }
        );
      }

      const audio = await res.arrayBuffer();

      // Cache in background
      ctx.waitUntil(
        putR2Object(env.AUDIO_CACHE, cacheKey, audio, {
          contentType: 'audio/mpeg',
          cacheControl: 'public, max-age=31536000, immutable',
        })
      );

      return audioResponse(audio, request, {
        'X-Kelly-TTS': 'generated',
        'X-Kelly-Cache-Key': cacheKey,
      });
    } catch (err) {
      const fallback = await tryPregenFallback({ env, request, day, phase, language });
      if (fallback) return fallback;
      return json(
        503,
        {
          error: 'TTS fetch failed',
          code: 'TTS_FETCH_FAILED',
          message: String(err?.message || err || ''),
        },
        { ...corsHeaders(request) }
      );
    }
  },
};

async function tryPregenFallback({ env, request, day, phase, language }) {
  if (!env.AUDIO_PREGEN) return null;
  if (!day || !phase) return null;

  const dayNum = Math.max(1, Math.min(365, Math.floor(day)));
  const key = `pregen/${String(dayNum).padStart(3, '0')}/${safePhase(phase)}/${safeLang(language)}.mp3`;
  const obj = await getR2Object(env.AUDIO_PREGEN, key);
  if (!obj) return null;
  const buf = await obj.arrayBuffer();

  return audioResponse(buf, request, {
    'X-Kelly-TTS': 'pregen_fallback',
    'X-Kelly-Pregen-Key': key,
  });
}



