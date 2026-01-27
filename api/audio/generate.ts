import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

/**
 * POST /api/audio/generate
 * 
 * Generates audio for a specific lesson phase using ElevenLabs TTS
 * and stores it in Supabase Storage.
 * 
 * Request body:
 *   { day: number, phase: string, text: string, voiceId?: string }
 * 
 * Response:
 *   { audioUrl: string, duration?: number, cached: boolean }
 */

const ELEVENLABS_API_URL = 'https://api.elevenlabs.io/v1/text-to-speech';
const DEFAULT_VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'; // Rachel voice
const STORAGE_BUCKET = 'lesson-audio';

interface GenerateRequest {
  day: number;
  phase: string;
  text: string;
  voiceId?: string;
  language?: string;
  archetype?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get API key
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    console.error('[Audio Generate] ELEVENLABS_API_KEY not set');
    return res.status(503).json({
      error: 'Audio generation service not configured',
      code: 'MISSING_API_KEY',
    });
  }

  // Parse request body
  const { day, phase, text, voiceId, language = 'en', archetype = 'default' }: GenerateRequest = req.body;

  // Validate required fields
  if (!day || !phase || !text) {
    return res.status(400).json({
      error: 'Missing required fields',
      required: ['day', 'phase', 'text'],
      received: { day: !!day, phase: !!phase, text: !!text },
    });
  }

  if (typeof day !== 'number' || day < 1 || day > 365) {
    return res.status(400).json({ error: 'Invalid day number (1-365)' });
  }

  const validPhases = ['hook', 'q1', 'q2', 'q3', 'wisdom', 'welcome', 'socratic', 'reveal', 'explore', 'wonder', 'reflect'];
  if (!validPhases.includes(phase.toLowerCase())) {
    return res.status(400).json({
      error: `Invalid phase. Use: ${validPhases.join(', ')}`,
    });
  }

  const voice = voiceId || process.env.KELLY_VOICE_ID || DEFAULT_VOICE_ID;
  const normalizedPhase = phase.toLowerCase();

  try {
    const supabase = getSupabaseAdmin();

    // Generate storage path
    const storagePath = `day-${String(day).padStart(3, '0')}/${language}/${archetype}/${normalizedPhase}.mp3`;

    // Check if audio already exists
    const { data: existingFile } = await supabase.storage
      .from(STORAGE_BUCKET)
      .createSignedUrl(storagePath, 3600);

    if (existingFile?.signedUrl) {
      console.log(`[Audio Generate] Using cached audio: ${storagePath}`);
      return res.status(200).json({
        audioUrl: existingFile.signedUrl,
        cached: true,
        path: storagePath,
      });
    }

    console.log(`[Audio Generate] Generating audio for day ${day}, phase ${normalizedPhase}`);

    // Call ElevenLabs API
    const elevenLabsResponse = await fetch(`${ELEVENLABS_API_URL}/${voice}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': apiKey,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    });

    if (!elevenLabsResponse.ok) {
      const errorText = await elevenLabsResponse.text();
      console.error('[Audio Generate] ElevenLabs API error:', elevenLabsResponse.status, errorText);
      
      let errorDetails: unknown = { raw: errorText };
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        // Not JSON, use raw text
      }

      return res.status(elevenLabsResponse.status).json({
        error: 'Audio generation failed',
        status: elevenLabsResponse.status,
        elevenlabsError: errorDetails,
        hint: elevenLabsResponse.status === 401 ? 'API key may be invalid' :
              elevenLabsResponse.status === 429 ? 'Rate limit exceeded' :
              'Check ElevenLabs dashboard for details',
      });
    }

    // Get audio buffer
    const audioBuffer = await elevenLabsResponse.arrayBuffer();
    const audioBytes = new Uint8Array(audioBuffer);

    console.log(`[Audio Generate] Generated ${audioBytes.length} bytes of audio`);

    // Upload to Supabase Storage
    const { error: uploadError } = await supabase.storage
      .from(STORAGE_BUCKET)
      .upload(storagePath, audioBytes, {
        contentType: 'audio/mpeg',
        upsert: true,
      });

    if (uploadError) {
      console.error('[Audio Generate] Storage upload error:', uploadError);
      // Still return the audio even if storage fails
      res.setHeader('Content-Type', 'audio/mpeg');
      res.setHeader('Content-Length', audioBytes.length.toString());
      return res.send(Buffer.from(audioBuffer));
    }

    // Get signed URL for the uploaded file
    const { data: signedUrlData, error: signedUrlError } = await supabase.storage
      .from(STORAGE_BUCKET)
      .createSignedUrl(storagePath, 86400); // 24 hour expiry

    if (signedUrlError || !signedUrlData?.signedUrl) {
      console.error('[Audio Generate] Failed to create signed URL:', signedUrlError);
      // Fallback: return public URL pattern
      const publicUrl = `${process.env.PUBLIC_SUPABASE_URL}/storage/v1/object/public/${STORAGE_BUCKET}/${storagePath}`;
      return res.status(200).json({
        audioUrl: publicUrl,
        cached: false,
        path: storagePath,
        byteLength: audioBytes.length,
      });
    }

    // Log success to database for tracking
    try {
      await supabase.from('audio_generation_log').insert({
        day_number: day,
        phase: normalizedPhase,
        language,
        archetype,
        storage_path: storagePath,
        voice_id: voice,
        text_length: text.length,
        audio_bytes: audioBytes.length,
        generated_at: new Date().toISOString(),
      });
    } catch (logError) {
      // Non-fatal - logging failure shouldn't break the response
      console.warn('[Audio Generate] Failed to log generation:', logError);
    }

    console.log(`[Audio Generate] Success: ${storagePath}`);

    return res.status(200).json({
      audioUrl: signedUrlData.signedUrl,
      cached: false,
      path: storagePath,
      byteLength: audioBytes.length,
    });

  } catch (error) {
    console.error('[Audio Generate] Unexpected error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
