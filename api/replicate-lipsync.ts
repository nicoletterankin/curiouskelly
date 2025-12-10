/**
 * 🎬 Kelly 4K Lip-Sync Video Generation API
 * 
 * Production endpoint for generating lip-synced videos of Kelly
 * Uses: ElevenLabs TTS + Replicate SadTalker + Real-ESRGAN upscaling
 * 
 * POST /api/replicate-lipsync
 * {
 *   "lessonDay": 1,           // 1-365
 *   "phase": "welcome",       // welcome, question, wisdom, etc.
 *   "text": "Hello!",         // Text for Kelly to speak
 *   "imageType": "hero",      // hero, guide-point, reaction, prop, bg
 *   "upscale": true,          // Enable 4K upscaling
 *   "model": "sadtalker"      // sadtalker, liveportrait, hallo2
 * }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import Replicate from 'replicate';

// Configuration
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
const KELLY_LORA_URL = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';

// Lip-sync models available on Replicate
const LIPSYNC_MODELS = {
  sadtalker: {
    id: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
    name: 'SadTalker',
    quality: 'good',
    speed: 'fast',
  },
  sadtalker_alt: {
    id: 'lucataco/sadtalker:85f79f4a1d369fc190998c3dbbf6e67a8b6bee9fcbae33ff6be3261aaaefd85e',
    name: 'SadTalker (Lucataco)',
    quality: 'good',
    speed: 'fast',
  },
  liveportrait: {
    id: 'fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146',
    name: 'LivePortrait',
    quality: 'excellent',
    speed: 'medium',
  },
  wav2lip: {
    id: 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    name: 'Wav2Lip',
    quality: 'good',
    speed: 'fast',
  },
} as const;

// Video upscaler
const UPSCALER_MODEL = 'lucataco/real-esrgan-video:c23768236472c41b7a121ee735c8073e29080c02d343419c4b7f0e56e045cb4d';

// CORS headers
function setCorsHeaders(res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

// Generate Kelly image URL from lesson/phase
function getKellyImageUrl(lessonDay: number, imageType: string): string {
  const baseUrl = 'https://curiouskelly.com';
  const paddedDay = String(lessonDay).padStart(3, '0');
  
  // Map image types to file names
  const imageMap: Record<string, string> = {
    hero: `lesson-${lessonDay}-hero.png`,
    'guide-point': `lesson-${lessonDay}-guide-point.png`,
    reaction: `lesson-${lessonDay}-reaction.png`,
    prop: `lesson-${lessonDay}-prop.png`,
    bg: `lesson-${lessonDay}-bg.png`,
  };
  
  const filename = imageMap[imageType] || imageMap.hero;
  return `${baseUrl}/kelly/lessons/${paddedDay}/${filename}`;
}

// Get core pose URL
function getKellyPoseUrl(pose: string): string {
  const baseUrl = 'https://curiouskelly.com';
  return `${baseUrl}/kelly/poses/kelly_${pose}.png`;
}

// Generate TTS audio with ElevenLabs
async function generateTTS(text: string, apiKey: string): Promise<ArrayBuffer> {
  console.log('[TTS] Generating audio for:', text.substring(0, 50) + '...');
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
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
          similarity_boost: 0.85,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`TTS failed: ${response.status} - ${error}`);
  }

  return response.arrayBuffer();
}

// Upload to temporary storage (Replicate needs URLs)
async function uploadToReplicate(
  replicate: Replicate, 
  buffer: ArrayBuffer, 
  filename: string
): Promise<string> {
  // Convert to base64 data URL for Replicate
  const base64 = Buffer.from(buffer).toString('base64');
  const mimeType = filename.endsWith('.mp3') ? 'audio/mpeg' : 'image/png';
  return `data:${mimeType};base64,${base64}`;
}

// Generate lip-sync video
async function generateLipSync(
  replicate: Replicate,
  imageUrl: string,
  audioDataUrl: string,
  modelKey: keyof typeof LIPSYNC_MODELS = 'sadtalker'
): Promise<string> {
  const model = LIPSYNC_MODELS[modelKey] || LIPSYNC_MODELS.sadtalker;
  console.log(`[LipSync] Using ${model.name}...`);

  // Fetch image and convert to data URL
  const imageResponse = await fetch(imageUrl);
  if (!imageResponse.ok) {
    throw new Error(`Failed to fetch image: ${imageUrl} (${imageResponse.status})`);
  }
  const imageBuffer = await imageResponse.arrayBuffer();
  const imageDataUrl = `data:image/png;base64,${Buffer.from(imageBuffer).toString('base64')}`;
  
  console.log(`[LipSync] Image size: ${(imageBuffer.byteLength / 1024).toFixed(1)} KB`);

  // Different input formats for different models
  let input: Record<string, any>;
  
  if (modelKey === 'liveportrait') {
    input = {
      image: imageDataUrl,
      video: audioDataUrl, // LivePortrait uses video for driving
    };
  } else if (modelKey === 'wav2lip') {
    input = {
      face: imageDataUrl,
      audio: audioDataUrl,
    };
  } else {
    // SadTalker variants
    input = {
      source_image: imageDataUrl,
      driven_audio: audioDataUrl,
      enhancer: 'gfpgan',
      preprocess: 'crop',
      still_mode: false,
      expression_scale: 1.0,
    };
  }

  const output = await replicate.run(model.id as `${string}/${string}:${string}`, { input });
  
  // Handle different output formats
  let videoUrl: string | null = null;
  
  if (typeof output === 'string') {
    videoUrl = output;
  } else if (Array.isArray(output)) {
    videoUrl = output[0];
  } else if (output && typeof output === 'object') {
    videoUrl = (output as any).output || (output as any).video || (output as any).url;
  }

  if (!videoUrl || typeof videoUrl !== 'string') {
    throw new Error(`Invalid output from ${model.name}: ${JSON.stringify(output)}`);
  }

  console.log(`[LipSync] Video generated successfully`);
  return videoUrl;
}

// Upscale video to 4K
async function upscaleVideo(
  replicate: Replicate,
  videoUrl: string
): Promise<string> {
  console.log('[Upscale] Upscaling to 4K...');
  
  try {
    const output = await replicate.run(
      UPSCALER_MODEL as `${string}/${string}:${string}`,
      {
        input: {
          video: videoUrl,
          scale: 4,
          face_enhance: true,
        },
      }
    );

    const upscaledUrl = typeof output === 'string' ? output : (output as any)?.output;
    
    if (upscaledUrl && typeof upscaledUrl === 'string') {
      console.log('[Upscale] 4K video ready');
      return upscaledUrl;
    }
  } catch (error: any) {
    console.warn('[Upscale] Failed:', error.message);
  }

  // Return original if upscaling fails
  console.log('[Upscale] Returning original video');
  return videoUrl;
}

// Main handler
export default async function handler(req: VercelRequest, res: VercelResponse) {
  setCorsHeaders(res);

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Check required env vars
  const replicateToken = process.env.REPLICATE_API_TOKEN;
  const elevenLabsKey = process.env.ELEVENLABS_API_KEY;

  if (!replicateToken) {
    return res.status(500).json({ 
      success: false, 
      error: 'REPLICATE_API_TOKEN not configured',
      setup: 'Add REPLICATE_API_TOKEN to Vercel environment variables'
    });
  }

  if (!elevenLabsKey) {
    return res.status(500).json({ 
      success: false, 
      error: 'ELEVENLABS_API_KEY not configured' 
    });
  }

  try {
    const {
      lessonDay,
      phase = 'welcome',
      text,
      imageType = 'hero',
      pose,
      upscale = false,
      model = 'sadtalker',
      imageUrl: customImageUrl,
    } = req.body;

    // Validate input
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return res.status(400).json({ error: 'Text is required' });
    }

    if (text.length > 5000) {
      return res.status(400).json({ error: 'Text too long (max 5000 chars)' });
    }

    // Determine image URL
    let imageUrl: string;
    if (customImageUrl) {
      imageUrl = customImageUrl;
    } else if (pose) {
      imageUrl = getKellyPoseUrl(pose);
    } else if (lessonDay && lessonDay >= 1 && lessonDay <= 365) {
      imageUrl = getKellyImageUrl(lessonDay, imageType);
    } else {
      // Default to welcome pose
      imageUrl = getKellyPoseUrl('welcome');
    }

    console.log('[Pipeline] Starting Kelly lip-sync generation');
    console.log('[Pipeline] Image:', imageUrl);
    console.log('[Pipeline] Text:', text.substring(0, 100) + (text.length > 100 ? '...' : ''));
    console.log('[Pipeline] Model:', model);
    console.log('[Pipeline] Upscale:', upscale);

    const startTime = Date.now();
    const replicate = new Replicate({ auth: replicateToken });

    // Step 1: Generate TTS
    console.log('[Pipeline] Step 1: Generating TTS...');
    const audioBuffer = await generateTTS(text, elevenLabsKey);
    const audioDataUrl = `data:audio/mpeg;base64,${Buffer.from(audioBuffer).toString('base64')}`;
    console.log(`[Pipeline] Audio: ${(audioBuffer.byteLength / 1024).toFixed(1)} KB`);

    // Step 2: Generate lip-sync video
    console.log('[Pipeline] Step 2: Generating lip-sync...');
    let videoUrl = await generateLipSync(
      replicate, 
      imageUrl, 
      audioDataUrl, 
      model as keyof typeof LIPSYNC_MODELS
    );

    // Step 3: Upscale to 4K (optional)
    if (upscale) {
      console.log('[Pipeline] Step 3: Upscaling to 4K...');
      videoUrl = await upscaleVideo(replicate, videoUrl);
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Pipeline] Complete in ${totalTime}s`);

    return res.status(200).json({
      success: true,
      videoUrl,
      metadata: {
        lessonDay,
        phase,
        imageUrl,
        model,
        upscaled: upscale,
        duration: `${totalTime}s`,
        textLength: text.length,
      },
    });

  } catch (error: any) {
    console.error('[Pipeline] Error:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Video generation failed',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined,
    });
  }
}



