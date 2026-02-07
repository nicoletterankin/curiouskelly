import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

/**
 * POST /api/audio/batch
 * 
 * Batch generates audio for multiple lesson days.
 * Rate limited to 3 requests/second to respect ElevenLabs limits.
 * 
 * Request body:
 *   { startDay: number, endDay: number, phases?: string[], voiceId?: string }
 * 
 * Response:
 *   { 
 *     success: boolean,
 *     generated: number,
 *     failed: number,
 *     skipped: number,
 *     results: Array<{ day, phase, status, audioUrl?, error? }>
 *   }
 */

const ELEVENLABS_API_URL = 'https://api.elevenlabs.io/v1/text-to-speech';
const DEFAULT_VOICE_ID = 'wAdymQH5YucAkXwmrdL0'; // Kelly voice
const STORAGE_BUCKET = 'lesson-audio';
const RATE_LIMIT_MS = 334; // ~3 requests per second
const MAX_BATCH_SIZE = 50; // Safety limit

// Phase mapping from short names to DB names
const PHASE_MAP: Record<string, string> = {
  'hook': 'Hook',
  'q1': 'Fact1',
  'q2': 'Fact2',
  'q3': 'Fact3',
  'wisdom': 'Wisdom',
  'welcome': 'Welcome',
  'socratic': 'Socratic',
  'reveal': 'Reveal',
  'explore': 'Explore',
  'wonder': 'Wonder',
  'reflect': 'Reflect',
};

const DEFAULT_PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom'];

interface BatchRequest {
  startDay: number;
  endDay: number;
  phases?: string[];
  voiceId?: string;
  language?: string;
  archetype?: string;
  skipExisting?: boolean;
}

interface BatchResult {
  day: number;
  phase: string;
  status: 'generated' | 'cached' | 'failed' | 'skipped';
  audioUrl?: string;
  error?: string;
}

// Rate limiter helper
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
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
    console.error('[Audio Batch] ELEVENLABS_API_KEY not set');
    return res.status(503).json({
      error: 'Audio generation service not configured',
      code: 'MISSING_API_KEY',
    });
  }

  // Parse request
  const {
    startDay,
    endDay,
    phases = DEFAULT_PHASES,
    voiceId,
    language = 'en',
    archetype = 'The Scientist',
    skipExisting = true,
  }: BatchRequest = req.body;

  // Validate
  if (!startDay || !endDay) {
    return res.status(400).json({
      error: 'Missing required fields',
      required: ['startDay', 'endDay'],
    });
  }

  if (startDay < 1 || startDay > 365 || endDay < 1 || endDay > 365) {
    return res.status(400).json({ error: 'Day numbers must be between 1-365' });
  }

  if (startDay > endDay) {
    return res.status(400).json({ error: 'startDay must be <= endDay' });
  }

  const totalRequests = (endDay - startDay + 1) * phases.length;
  if (totalRequests > MAX_BATCH_SIZE) {
    return res.status(400).json({
      error: `Batch too large. Max ${MAX_BATCH_SIZE} audio files per request.`,
      requested: totalRequests,
      hint: `Try smaller range or fewer phases. Current: ${endDay - startDay + 1} days × ${phases.length} phases = ${totalRequests}`,
    });
  }

  const voice = voiceId || process.env.KELLY_VOICE_ID || DEFAULT_VOICE_ID;

  console.log(`[Audio Batch] Starting batch: days ${startDay}-${endDay}, phases: ${phases.join(', ')}`);

  const supabase = getSupabaseAdmin();
  const results: BatchResult[] = [];
  let generated = 0;
  let failed = 0;
  let skipped = 0;
  let cached = 0;

  try {
    // Process each day and phase
    for (let day = startDay; day <= endDay; day++) {
      // Fetch lesson data for this day
      const { data: lesson, error: lessonError } = await supabase
        .from('core_lessons')
        .select('id, topic')
        .eq('day_number', day)
        .single();

      if (lessonError || !lesson) {
        console.warn(`[Audio Batch] No lesson found for day ${day}`);
        for (const phase of phases) {
          results.push({
            day,
            phase,
            status: 'skipped',
            error: 'Lesson not found',
          });
          skipped++;
        }
        continue;
      }

      for (const phase of phases) {
        const normalizedPhase = phase.toLowerCase();
        const dbPhase = PHASE_MAP[normalizedPhase] || normalizedPhase;

        // Check if audio already exists
        const storagePath = `day-${String(day).padStart(3, '0')}/${language}/${archetype}/${normalizedPhase}.mp3`;

        if (skipExisting) {
          const { data: existingFile } = await supabase.storage
            .from(STORAGE_BUCKET)
            .createSignedUrl(storagePath, 60);

          if (existingFile?.signedUrl) {
            console.log(`[Audio Batch] Cached: day ${day}, phase ${normalizedPhase}`);
            results.push({
              day,
              phase: normalizedPhase,
              status: 'cached',
              audioUrl: existingFile.signedUrl,
            });
            cached++;
            continue;
          }
        }

        // Get script text from lesson_atoms
        const { data: atom } = await supabase
          .from('lesson_atoms')
          .select('content')
          .eq('core_lesson_id', lesson.id)
          .eq('archetype', archetype)
          .eq('phase', dbPhase)
          .single();

        const scriptText = atom?.content?.script || atom?.content?.text;

        if (!scriptText) {
          console.warn(`[Audio Batch] No script for day ${day}, phase ${normalizedPhase}`);
          results.push({
            day,
            phase: normalizedPhase,
            status: 'skipped',
            error: 'No script content',
          });
          skipped++;
          continue;
        }

        // Rate limit before API call
        await sleep(RATE_LIMIT_MS);

        try {
          // Generate audio via ElevenLabs
          const elevenLabsResponse = await fetch(`${ELEVENLABS_API_URL}/${voice}`, {
            method: 'POST',
            headers: {
              'Accept': 'audio/mpeg',
              'Content-Type': 'application/json',
              'xi-api-key': apiKey,
            },
            body: JSON.stringify({
              text: scriptText,
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
            console.error(`[Audio Batch] ElevenLabs error for day ${day}: ${errorText}`);
            results.push({
              day,
              phase: normalizedPhase,
              status: 'failed',
              error: `ElevenLabs API error: ${elevenLabsResponse.status}`,
            });
            failed++;
            continue;
          }

          // Upload to storage
          const audioBuffer = await elevenLabsResponse.arrayBuffer();
          const audioBytes = new Uint8Array(audioBuffer);

          const { error: uploadError } = await supabase.storage
            .from(STORAGE_BUCKET)
            .upload(storagePath, audioBytes, {
              contentType: 'audio/mpeg',
              upsert: true,
            });

          if (uploadError) {
            console.error(`[Audio Batch] Upload error for day ${day}:`, uploadError);
            results.push({
              day,
              phase: normalizedPhase,
              status: 'failed',
              error: 'Storage upload failed',
            });
            failed++;
            continue;
          }

          // Get signed URL
          const { data: signedUrlData } = await supabase.storage
            .from(STORAGE_BUCKET)
            .createSignedUrl(storagePath, 86400);

          console.log(`[Audio Batch] Generated: day ${day}, phase ${normalizedPhase}`);
          results.push({
            day,
            phase: normalizedPhase,
            status: 'generated',
            audioUrl: signedUrlData?.signedUrl,
          });
          generated++;

        } catch (phaseError) {
          console.error(`[Audio Batch] Error for day ${day}, phase ${normalizedPhase}:`, phaseError);
          results.push({
            day,
            phase: normalizedPhase,
            status: 'failed',
            error: phaseError instanceof Error ? phaseError.message : 'Unknown error',
          });
          failed++;
        }
      }
    }

    // Log batch completion
    try {
      await supabase.from('audio_batch_log').insert({
        start_day: startDay,
        end_day: endDay,
        phases: phases,
        language,
        archetype,
        generated_count: generated,
        cached_count: cached,
        failed_count: failed,
        skipped_count: skipped,
        completed_at: new Date().toISOString(),
      });
    } catch (logError) {
      console.warn('[Audio Batch] Failed to log batch:', logError);
    }

    console.log(`[Audio Batch] Complete: ${generated} generated, ${cached} cached, ${failed} failed, ${skipped} skipped`);

    return res.status(200).json({
      success: failed === 0,
      generated,
      cached,
      failed,
      skipped,
      total: results.length,
      results,
    });

  } catch (error) {
    console.error('[Audio Batch] Unexpected error:', error);
    return res.status(500).json({
      error: 'Batch processing failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      partialResults: results,
    });
  }
}
