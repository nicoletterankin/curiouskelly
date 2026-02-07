/**
 * Day Generation Pipeline API
 * 
 * End-to-end generation for a single day:
 * 1. Get lesson content
 * 2. Generate audio (ElevenLabs)
 * 3. Submit to video queue (multi-provider fallback)
 * 4. Run eval gates
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import { evaluateContent, evaluateAudio } from '../../lib/eval-gates';
import { submitWithFallback, getAvailableProviders } from '../../lib/fallback-queue';
import { notifyPipelineError } from '../../lib/email-alerts';
import type { VideoJob, Phase, AgeCategory, EngineInputPayload } from '../../lib/engines/types';

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

// Kelly base images for video generation
const KELLY_BASE_IMAGES = {
  default: 'https://storage.googleapis.com/curious-kelly-assets/kelly/kelly-presenter-01.png',
  talking: 'https://storage.googleapis.com/curious-kelly-assets/kelly/kelly-talking-01.mp4',
};

interface GenerateDayRequest {
  day_of_year: number;
  phases?: Phase[];
  age_categories?: AgeCategory[];
  dry_run?: boolean;
}

interface GenerateDayResponse {
  success: boolean;
  day_of_year: number;
  jobs_created: number;
  jobs: Array<{
    id: string;
    phase: Phase;
    age_category: AgeCategory;
    status: string;
  }>;
  errors: string[];
  provider_status: Record<string, boolean>;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  
  try {
    const body = req.body as GenerateDayRequest;
    const { day_of_year, dry_run = false } = body;
    const phases: Phase[] = body.phases || ['hook', 'story', 'wonder', 'action', 'wisdom'];
    const age_categories: AgeCategory[] = body.age_categories || ['adult'];
    
    if (!day_of_year || day_of_year < 1 || day_of_year > 365) {
      res.status(400).json({ error: 'day_of_year must be between 1 and 365' });
      return;
    }
    
    const result = await generateDay({
      day_of_year,
      phases,
      age_categories,
      dry_run,
    });
    
    res.status(200).json(result);
    
  } catch (error: any) {
    console.error('Pipeline error:', error);
    await notifyPipelineError(error.message, { endpoint: 'generate-day' });
    res.status(500).json({ error: error.message });
  }
}

/**
 * Generate all videos for a day
 */
export async function generateDay(config: {
  day_of_year: number;
  phases: Phase[];
  age_categories: AgeCategory[];
  dry_run?: boolean;
}): Promise<GenerateDayResponse> {
  const { day_of_year, phases, age_categories, dry_run = false } = config;
  const supabase = getSupabaseAdmin();
  
  const response: GenerateDayResponse = {
    success: true,
    day_of_year,
    jobs_created: 0,
    jobs: [],
    errors: [],
    provider_status: {},
  };
  
  // Check provider availability
  const providers = await getAvailableProviders();
  for (const p of ['heygen', 'sync_so', 'fal_latentsync', 'replicate']) {
    response.provider_status[p] = providers.includes(p as any);
  }
  
  if (providers.length === 0) {
    response.success = false;
    response.errors.push('No video providers available');
    return response;
  }
  
  // Get lesson content for this day
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_of_year', day_of_year)
    .single();
  
  if (lessonError || !lesson) {
    response.errors.push(`No lesson found for day ${day_of_year}`);
    response.success = false;
    return response;
  }
  
  console.log(`\n📚 Generating Day ${day_of_year}: ${lesson.topic}`);
  console.log(`   Phases: ${phases.join(', ')}`);
  console.log(`   Age categories: ${age_categories.join(', ')}`);
  
  // Generate for each phase and age category
  for (const phase of phases) {
    for (const age_category of age_categories) {
      try {
        // Get phase content
        const { data: atom, error: atomError } = await supabase
          .from('lesson_atoms')
          .select('*')
          .eq('core_lesson_id', lesson.id)
          .eq('phase', phase)
          .single();
        
        if (atomError || !atom) {
          response.errors.push(`No content for phase ${phase}`);
          continue;
        }
        
        const text = atom.content || atom.text || '';
        
        // 1. CONTENT EVAL GATE
        const contentEval = evaluateContent({ text, phase, day: day_of_year });
        if (!contentEval.passed) {
          response.errors.push(`Content eval failed for ${phase}: ${contentEval.issues.join('; ')}`);
          continue;
        }
        
        // 2. GENERATE AUDIO
        let audioUrl: string | undefined;
        if (!dry_run && ELEVENLABS_API_KEY) {
          audioUrl = await generateAudio(text, ELEVENLABS_VOICE_ID);
          
          // Audio eval gate
          const audioEval = evaluateAudio({ url: audioUrl });
          if (!audioEval.passed) {
            response.errors.push(`Audio eval failed for ${phase}: ${audioEval.issues.join('; ')}`);
            continue;
          }
        }
        
        // 3. CREATE VIDEO JOB
        const jobId = `day${day_of_year}-${phase}-${age_category}-${Date.now()}`;
        
        const input_payload: EngineInputPayload = {
          text,
          audio_url: audioUrl,
          source_image_url: KELLY_BASE_IMAGES.default,
          video_url: KELLY_BASE_IMAGES.talking,
        };
        
        const job: VideoJob = {
          id: jobId,
          day_of_year,
          phase,
          age_category,
          language: 'en',
          engine: providers[0], // Will be set by fallback queue
          status: 'queued',
          input_payload,
          priority: phase === 'hook' ? 10 : 5, // Hook phase gets priority
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        
        if (dry_run) {
          console.log(`   [DRY RUN] Would create job: ${jobId}`);
          response.jobs.push({
            id: jobId,
            phase,
            age_category,
            status: 'dry_run',
          });
          response.jobs_created++;
          continue;
        }
        
        // Insert job
        const { error: insertError } = await supabase
          .from('video_jobs')
          .insert(job);
        
        if (insertError) {
          response.errors.push(`Failed to insert job: ${insertError.message}`);
          continue;
        }
        
        // 4. SUBMIT TO QUEUE WITH FALLBACK
        const submitResult = await submitWithFallback(job, {
          enableEvalGates: true,
          enableAlerts: true,
        });
        
        response.jobs.push({
          id: jobId,
          phase,
          age_category,
          status: submitResult.success ? 'submitted' : 'failed',
        });
        
        if (submitResult.success) {
          response.jobs_created++;
          console.log(`   ✅ ${phase}/${age_category} → ${submitResult.engine_used}`);
        } else {
          response.errors.push(`Job ${jobId} failed: ${submitResult.error}`);
        }
        
      } catch (error: any) {
        response.errors.push(`Error processing ${phase}/${age_category}: ${error.message}`);
      }
    }
  }
  
  response.success = response.errors.length === 0;
  
  console.log(`\n📊 Day ${day_of_year} summary:`);
  console.log(`   Jobs created: ${response.jobs_created}`);
  console.log(`   Errors: ${response.errors.length}`);
  
  return response;
}

/**
 * Generate audio with ElevenLabs and upload to Supabase Storage
 */
async function generateAudio(text: string, voiceId: string): Promise<string> {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY not configured');
  }
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.3,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ElevenLabs error: ${response.status} - ${errorText}`);
  }
  
  const audioBuffer = await response.arrayBuffer();
  const audioBytes = new Uint8Array(audioBuffer);
  
  // Upload to Supabase Storage
  const supabase = getSupabaseAdmin();
  const timestamp = Date.now();
  const storagePath = `pipeline-audio/${timestamp}-${Math.random().toString(36).substring(7)}.mp3`;
  
  const { error: uploadError } = await supabase.storage
    .from('lesson-audio')
    .upload(storagePath, audioBytes, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (uploadError) {
    console.error('Storage upload error:', uploadError);
    throw new Error(`Storage upload failed: ${uploadError.message}`);
  }
  
  // Get public URL
  const { data: urlData } = supabase.storage
    .from('lesson-audio')
    .getPublicUrl(storagePath);
  
  if (!urlData?.publicUrl) {
    throw new Error('Failed to get public URL for uploaded audio');
  }
  
  console.log(`   🎙️ Audio uploaded: ${storagePath} (${audioBytes.length} bytes)`);
  
  return urlData.publicUrl;
}
