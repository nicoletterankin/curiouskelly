import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

/**
 * ElevenLabs Omnihuman 1.5 Video Generation API
 * 
 * Generates lip-synced Kelly videos from static images + TTS audio
 * 
 * POST /api/elevenlabs-video
 * Body: { lessonDay, phase, ageBucket, language, text, archetype? }
 * Returns: { success, videoUrl, generationId, durationMs }
 */

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_BASE_URL = 'https://api.elevenlabs.io/v1';
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

// Kelly pose images mapped to lesson phases
const PHASE_TO_POSE: Record<string, string> = {
  'welcome': '/kelly/poses/kelly_welcome.png',
  'q1': '/kelly/poses/kelly_hint.png',
  'q2': '/kelly/poses/kelly_listening.png',
  'q3': '/kelly/poses/kelly_hint_flip.png',
  'wisdom': '/kelly/poses/kelly_clasp.png',
};

// Fallback poses if primary not available
const FALLBACK_POSES = [
  '/kelly/poses/kelly_idle.png',
  '/kelly/poses/kelly_clasp.png',
];

interface VideoGenerationRequest {
  lessonDay: number;
  phase: 'welcome' | 'q1' | 'q2' | 'q3' | 'wisdom';
  ageBucket: string;
  language: string;
  text: string;
  archetype?: string;
  forceRegenerate?: boolean;
}

interface TTSResponse {
  success: boolean;
  audioBuffer?: ArrayBuffer;
  error?: string;
}

interface VideoResponse {
  success: boolean;
  videoBuffer?: ArrayBuffer;
  generationId?: string;
  durationMs?: number;
  error?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method not allowed' });
  }

  // Validate API key
  if (!ELEVENLABS_API_KEY) {
    console.error('[ElevenLabs Video] ELEVENLABS_API_KEY not configured');
    return res.status(500).json({ 
      success: false, 
      error: 'ElevenLabs API not configured' 
    });
  }

  try {
    const body: VideoGenerationRequest = req.body;
    const { lessonDay, phase, ageBucket, language = 'en', text, archetype, forceRegenerate } = body;

    // Validate required fields
    if (!lessonDay || !phase || !ageBucket || !text) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: lessonDay, phase, ageBucket, text'
      });
    }

    // Validate phase
    const validPhases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
    if (!validPhases.includes(phase)) {
      return res.status(400).json({
        success: false,
        error: `Invalid phase. Must be one of: ${validPhases.join(', ')}`
      });
    }

    console.log(`[ElevenLabs Video] Generating video for Day ${lessonDay}, Phase: ${phase}`);

    // Initialize Supabase
    const supabase = createClient(
      process.env.PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Check if video already exists (unless force regenerate)
    if (!forceRegenerate) {
      const { data: existing } = await supabase
        .from('kelly_video_assets')
        .select('video_public_url, video_duration_ms, status')
        .eq('lesson_day', lessonDay)
        .eq('phase', phase)
        .eq('age_bucket', ageBucket)
        .eq('language', language)
        .eq('status', 'completed')
        .single();

      if (existing?.video_public_url) {
        console.log(`[ElevenLabs Video] Using cached video: ${existing.video_public_url}`);
        return res.status(200).json({
          success: true,
          videoUrl: existing.video_public_url,
          durationMs: existing.video_duration_ms,
          cached: true
        });
      }
    }

    // Create/update pending record
    const { data: pendingRecord, error: insertError } = await supabase
      .from('kelly_video_assets')
      .upsert({
        lesson_day: lessonDay,
        phase,
        age_bucket: ageBucket,
        language,
        archetype,
        source_image_path: PHASE_TO_POSE[phase] || FALLBACK_POSES[0],
        script_text: text.substring(0, 5000), // Limit stored text
        status: 'generating',
        generation_started_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }, {
        onConflict: 'lesson_day,phase,age_bucket,language'
      })
      .select()
      .single();

    if (insertError) {
      console.error('[ElevenLabs Video] Failed to create pending record:', insertError);
    }

    // Step 1: Generate TTS audio
    console.log('[ElevenLabs Video] Step 1: Generating TTS audio...');
    const audioResponse = await generateTTSAudio(text);
    
    if (!audioResponse.success || !audioResponse.audioBuffer) {
      await updateVideoStatus(supabase, lessonDay, phase, ageBucket, language, 'failed', audioResponse.error);
      return res.status(500).json({
        success: false,
        error: `TTS generation failed: ${audioResponse.error}`
      });
    }
    console.log(`[ElevenLabs Video] TTS audio generated: ${audioResponse.audioBuffer.byteLength} bytes`);

    // Step 2: Get the appropriate Kelly pose image
    const sourceImagePath = PHASE_TO_POSE[phase] || FALLBACK_POSES[0];
    console.log(`[ElevenLabs Video] Step 2: Using image: ${sourceImagePath}`);

    // Step 3: Generate lip-sync video via Omnihuman 1.5
    console.log('[ElevenLabs Video] Step 3: Calling Omnihuman 1.5 API...');
    const videoResponse = await generateLipSyncVideo(sourceImagePath, audioResponse.audioBuffer);

    if (!videoResponse.success || !videoResponse.videoBuffer) {
      await updateVideoStatus(supabase, lessonDay, phase, ageBucket, language, 'failed', videoResponse.error);
      return res.status(500).json({
        success: false,
        error: `Video generation failed: ${videoResponse.error}`
      });
    }
    console.log(`[ElevenLabs Video] Video generated: ${videoResponse.videoBuffer.byteLength} bytes`);

    // Step 4: Upload video to Supabase Storage
    console.log('[ElevenLabs Video] Step 4: Uploading to Supabase Storage...');
    const storagePath = `kelly-videos/${lessonDay}/${phase}/${ageBucket}-${language}.mp4`;

    const { error: uploadError } = await supabase.storage
      .from('lesson-assets')
      .upload(storagePath, videoResponse.videoBuffer, {
        contentType: 'video/mp4',
        upsert: true,
        cacheControl: '31536000' // 1 year cache
      });

    if (uploadError) {
      console.error('[ElevenLabs Video] Storage upload failed:', uploadError);
      await updateVideoStatus(supabase, lessonDay, phase, ageBucket, language, 'failed', uploadError.message);
      return res.status(500).json({
        success: false,
        error: `Storage upload failed: ${uploadError.message}`
      });
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('lesson-assets')
      .getPublicUrl(storagePath);

    // Step 5: Update database record with success
    const { error: dbError } = await supabase
      .from('kelly_video_assets')
      .update({
        video_storage_path: storagePath,
        video_public_url: urlData.publicUrl,
        video_duration_ms: videoResponse.durationMs,
        video_file_size_bytes: videoResponse.videoBuffer.byteLength,
        elevenlabs_generation_id: videoResponse.generationId,
        model_used: 'omnihuman-1.5',
        status: 'completed',
        generation_completed_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('lesson_day', lessonDay)
      .eq('phase', phase)
      .eq('age_bucket', ageBucket)
      .eq('language', language);

    if (dbError) {
      console.error('[ElevenLabs Video] Database update failed:', dbError);
      // Don't fail the request - video was generated successfully
    }

    console.log(`[ElevenLabs Video] ✅ Success! Video URL: ${urlData.publicUrl}`);

    return res.status(200).json({
      success: true,
      videoUrl: urlData.publicUrl,
      generationId: videoResponse.generationId,
      durationMs: videoResponse.durationMs,
      storagePath,
      cached: false
    });

  } catch (error) {
    console.error('[ElevenLabs Video] Unexpected error:', error);
    return res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

/**
 * Generate TTS audio using ElevenLabs
 */
async function generateTTSAudio(text: string): Promise<TTSResponse> {
  try {
    const response = await fetch(
      `${ELEVENLABS_BASE_URL}/text-to-speech/${KELLY_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': ELEVENLABS_API_KEY!,
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
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `TTS API error ${response.status}: ${errorText}` };
    }

    const audioBuffer = await response.arrayBuffer();
    return { success: true, audioBuffer };

  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'TTS request failed'
    };
  }
}

/**
 * Generate lip-sync video using ElevenLabs Omnihuman 1.5
 * 
 * Note: This uses the expected API structure for ElevenLabs Image-to-Video.
 * Verify endpoint and parameters against actual ElevenLabs documentation.
 */
async function generateLipSyncVideo(
  imagePath: string,
  audioBuffer: ArrayBuffer
): Promise<VideoResponse> {
  try {
    // Fetch the source image
    // In production, this should use the full URL or read from local filesystem
    const baseUrl = process.env.PUBLIC_SITE_URL || process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}` 
      : 'http://localhost:3000';
    
    const imageUrl = `${baseUrl}${imagePath}`;
    console.log(`[Omnihuman] Fetching image from: ${imageUrl}`);
    
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      return { success: false, error: `Failed to fetch image: ${imageUrl} (${imageResponse.status})` };
    }
    const imageBuffer = await imageResponse.arrayBuffer();
    console.log(`[Omnihuman] Image fetched: ${imageBuffer.byteLength} bytes`);

    // Create form data for multipart upload
    const formData = new FormData();
    formData.append('source_image', new Blob([imageBuffer], { type: 'image/png' }), 'kelly.png');
    formData.append('audio', new Blob([audioBuffer], { type: 'audio/mpeg' }), 'speech.mp3');
    
    // Optional parameters for Omnihuman
    formData.append('crop_to_face', 'false');  // Keep full image
    formData.append('output_format', 'mp4');

    // Call ElevenLabs Image-to-Video / Omnihuman endpoint
    // Note: Endpoint may be /v1/image-to-video or /v1/text-to-video/image
    // Verify against actual ElevenLabs API documentation
    const endpoint = `${ELEVENLABS_BASE_URL}/image-to-video`;
    console.log(`[Omnihuman] Calling: ${endpoint}`);

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY!,
        // Don't set Content-Type for FormData - let fetch set it with boundary
      },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[Omnihuman] API error: ${response.status}`, errorText);
      
      // Try to parse error for more details
      try {
        const errorJson = JSON.parse(errorText);
        return { 
          success: false, 
          error: `Omnihuman API error: ${errorJson.detail || errorJson.message || errorText}` 
        };
      } catch {
        return { success: false, error: `Omnihuman API error ${response.status}: ${errorText}` };
      }
    }

    const result = await response.json();
    console.log('[Omnihuman] API response:', JSON.stringify(result, null, 2));

    // Handle async generation (polling for completion)
    if (result.status === 'processing' || result.status === 'pending') {
      if (result.generation_id || result.id) {
        console.log(`[Omnihuman] Async generation started: ${result.generation_id || result.id}`);
        return await pollForVideoCompletion(result.generation_id || result.id);
      }
    }

    // Handle synchronous response with video URL
    if (result.video_url || result.output_url || result.url) {
      const videoUrl = result.video_url || result.output_url || result.url;
      console.log(`[Omnihuman] Downloading video from: ${videoUrl}`);
      
      const videoResponse = await fetch(videoUrl);
      if (!videoResponse.ok) {
        return { success: false, error: `Failed to download video: ${videoResponse.status}` };
      }
      
      const videoBuffer = await videoResponse.arrayBuffer();
      return {
        success: true,
        videoBuffer,
        generationId: result.generation_id || result.id,
        durationMs: result.duration_ms || result.duration * 1000
      };
    }

    // Handle response with video data directly
    if (result.video || result.data) {
      const videoData = result.video || result.data;
      // If base64 encoded
      if (typeof videoData === 'string') {
        const videoBuffer = Buffer.from(videoData, 'base64').buffer;
        return {
          success: true,
          videoBuffer,
          generationId: result.generation_id,
          durationMs: result.duration_ms
        };
      }
    }

    return { success: false, error: 'Unexpected API response format' };

  } catch (error) {
    console.error('[Omnihuman] Error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Video generation failed'
    };
  }
}

/**
 * Poll for async video generation completion
 */
async function pollForVideoCompletion(
  generationId: string,
  maxAttempts: number = 60,
  intervalMs: number = 5000
): Promise<VideoResponse> {
  console.log(`[Omnihuman] Polling for completion: ${generationId}`);

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, intervalMs));

    try {
      // Try different possible status endpoints
      const statusEndpoints = [
        `${ELEVENLABS_BASE_URL}/image-to-video/${generationId}`,
        `${ELEVENLABS_BASE_URL}/image-to-video/status/${generationId}`,
        `${ELEVENLABS_BASE_URL}/generations/${generationId}`,
      ];

      for (const statusUrl of statusEndpoints) {
        const response = await fetch(statusUrl, {
          headers: { 'xi-api-key': ELEVENLABS_API_KEY! }
        });

        if (!response.ok) continue;

        const result = await response.json();
        console.log(`[Omnihuman] Poll attempt ${attempt + 1}: status = ${result.status}`);

        // Check for completion
        if (result.status === 'completed' || result.status === 'done' || result.status === 'succeeded') {
          const videoUrl = result.video_url || result.output_url || result.url;
          
          if (videoUrl) {
            const videoResponse = await fetch(videoUrl);
            if (videoResponse.ok) {
              const videoBuffer = await videoResponse.arrayBuffer();
              return {
                success: true,
                videoBuffer,
                generationId,
                durationMs: result.duration_ms || result.duration * 1000
              };
            }
          }
        }

        // Check for failure
        if (result.status === 'failed' || result.status === 'error') {
          return { 
            success: false, 
            error: result.error || result.message || 'Generation failed',
            generationId 
          };
        }

        // Found a valid endpoint, no need to try others
        break;
      }
    } catch (error) {
      console.warn(`[Omnihuman] Poll attempt ${attempt + 1} error:`, error);
      // Continue polling on error
    }
  }

  return { 
    success: false, 
    error: 'Polling timeout - video generation took too long',
    generationId 
  };
}

/**
 * Helper to update video status in database
 */
async function updateVideoStatus(
  supabase: ReturnType<typeof createClient>,
  lessonDay: number,
  phase: string,
  ageBucket: string,
  language: string,
  status: string,
  errorMessage?: string
): Promise<void> {
  try {
    await supabase
      .from('kelly_video_assets')
      .update({
        status,
        error_message: errorMessage,
        retry_count: status === 'failed' ? 1 : 0,
        updated_at: new Date().toISOString()
      })
      .eq('lesson_day', lessonDay)
      .eq('phase', phase)
      .eq('age_bucket', ageBucket)
      .eq('language', language);
  } catch (error) {
    console.error('[ElevenLabs Video] Failed to update status:', error);
  }
}

