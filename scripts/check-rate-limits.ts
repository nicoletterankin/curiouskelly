/**
 * Gemini API sanity + rate-limit header probe
 *
 * Safe by default (no image generation).
 *
 * Usage:
 *   npx tsx scripts/check-rate-limits.ts
 *   npx tsx scripts/check-rate-limits.ts --imagen-probe   # will generate ONE image (costs)
 */
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });
dotenv.config();

const API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY;
const IMAGEN_MODEL = process.env.IMAGEN_MODEL || 'imagen-3.0-generate-002';

function pickHeaders(headers: Headers, keys: string[]) {
  const out: Record<string, string> = {};
  for (const k of keys) {
    const v = headers.get(k);
    if (v) out[k] = v;
  }
  return out;
}

async function probeModels() {
  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${API_KEY}`;
  const res = await fetch(url);
  const text = await res.text();
  let data: any = null;
  try {
    data = JSON.parse(text);
  } catch {
    // ignore
  }

  console.log(`Status: ${res.status}`);
  console.log(
    'Rate-limit headers (if present):',
    pickHeaders(res.headers, [
      'x-ratelimit-limit-requests',
      'x-ratelimit-remaining-requests',
      'x-ratelimit-reset-requests',
      'x-ratelimit-limit-tokens',
      'x-ratelimit-remaining-tokens',
      'x-ratelimit-reset-tokens',
      'retry-after',
    ])
  );

  const names: string[] =
    data?.models?.map((m: any) => m?.name).filter(Boolean).slice(0, 10) || [];
  if (names.length) {
    console.log('First models:', names.join(', '));
  } else {
    console.log('Response (truncated):', text.slice(0, 400));
  }
}

async function probeImagenOnce() {
  console.log('\n⚠️  Imagen probe will generate 1 image (costs + counts against quotas).');

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${IMAGEN_MODEL}:predict?key=${API_KEY}`;
  const body = {
    instances: [{ prompt: 'A simple abstract gradient background, 16:9, no text.' }],
    parameters: { sampleCount: 1, aspectRatio: '16:9' },
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  console.log(`Imagen status: ${res.status}`);
  console.log(
    'Rate-limit headers (if present):',
    pickHeaders(res.headers, [
      'x-ratelimit-limit-requests',
      'x-ratelimit-remaining-requests',
      'x-ratelimit-reset-requests',
      'retry-after',
    ])
  );

  const text = await res.text();
  console.log('Response (truncated):', text.slice(0, 400));
}

async function main() {
  if (!API_KEY) {
    console.error('❌ Missing GEMINI_API_KEY (or GOOGLE_AI_API_KEY / GOOGLE_API_KEY)');
    process.exit(1);
  }

  const args = process.argv.slice(2);
  const imagenProbe = args.includes('--imagen-probe');

  console.log('🔎 Gemini API probe');
  await probeModels();

  if (imagenProbe) {
    await probeImagenOnce();
  }
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});















