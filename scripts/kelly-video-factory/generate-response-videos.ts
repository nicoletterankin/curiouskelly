#!/usr/bin/env npx tsx
/**
 * 🎬 GENERATE RESPONSE VIDEOS FOR DAY 1
 * 
 * Creates HD lip-sync videos for Kelly's responses to user choices.
 * 
 * For Day 1, we have:
 * - 3 archetypes (Explorer, Rebel, Scientist)
 * - 3 question phases (Fact1, Fact2, Fact3)
 * - 3 choices per phase (A, B, C)
 * - Each choice has a response script
 * 
 * Total: 3 × 3 × 3 = 27 response videos per day
 * 
 * Output naming: day-001/{archetype}/{phase}_response_{letter}.mp4
 * Example: day-001/explorer/fact1_response_a.mp4
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/generate-response-videos.ts
 *   npx tsx scripts/kelly-video-factory/generate-response-videos.ts --archetype "The Explorer"
 *   npx tsx scripts/kelly-video-factory/generate-response-videos.ts --dry-run
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // API Keys
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Kelly Voice
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'responses'),
  BUCKET_NAME: 'kelly-videos',
  
  // Day number
  DAY_NUMBER: 1,
};

// Replicate model versions
const MODELS = {
  FLUX_LORA: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  MINIMAX: 'minimax/video-01:5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912',
};

// Kelly visual identity
const KELLY = {
  identity: 'Kelly, a warm and engaging AI teacher with auburn hair in loose waves, olive skin, bright curious eyes, wearing her signature deep teal sweater',
  responseExpression: 'warm supportive smile, engaging eye contact, gentle nod',
  responseGesture: 'hands gesturing supportively at chest level, slightly leaning forward with encouragement',
  background: 'soft-focused cozy learning environment with warm lighting',
};

// Voice settings for responses (warmer, more encouraging)
const VOICE_SETTINGS = {
  stability: 0.55,
  similarity_boost: 0.85,
  style: 0.35,
};

// =============================================================================
// TYPES
// =============================================================================

interface ResponseContent {
  archetype: string;
  phase: string;
  choiceLetter: string;
  responseScript: string;
}

interface GenerationResult {
  success: boolean;
  archetype: string;
  phase: string;
  choiceLetter: string;
  outputPath?: string;
  publicUrl?: string;
  error?: string;
}

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

let supabase: SupabaseClient;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  }
  return supabase;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function log(emoji: string, message: string, indent = 0): void {
  const prefix = '   '.repeat(indent);
  console.log(`${prefix}${emoji} ${message}`);
}

// =============================================================================
// FETCH RESPONSE SCRIPTS FROM DATABASE
// =============================================================================

async function fetchResponseScripts(dayNumber: number): Promise<ResponseContent[]> {
  const sb = getSupabase();
  const responses: ResponseContent[] = [];
  
  // Get lesson ID
  const { data: lesson, error: lessonError } = await sb
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    throw new Error(`Lesson not found for day ${dayNumber}`);
  }
  
  // Get all atoms for question phases
  const { data: atoms, error: atomsError } = await sb
    .from('lesson_atoms')
    .select('archetype, phase, content')
    .eq('core_lesson_id', lesson.id)
    .in('phase', ['Fact1', 'Fact2', 'Fact3']);
  
  if (atomsError || !atoms) {
    throw new Error(`Failed to fetch atoms: ${atomsError?.message}`);
  }
  
  for (const atom of atoms) {
    const content = atom.content as any;
    const options = content?.options || [];
    const responseMap = content?.responses || {};
    
    // Process each option
    options.forEach((option: any, index: number) => {
      const letter = String.fromCharCode(65 + index); // A, B, C
      
      // Response can be in option.response or in responseMap
      let responseScript = option.response || responseMap[letter] || '';
      
      if (responseScript) {
        responses.push({
          archetype: atom.archetype,
          phase: atom.phase,
          choiceLetter: letter,
          responseScript,
        });
      }
    });
  }
  
  return responses;
}

// =============================================================================
// AUDIO GENERATION
// =============================================================================

async function generateAudio(script: string, outputPath: string): Promise<string> {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          ...VOICE_SETTINGS,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
  
  return outputPath;
}

// =============================================================================
// IMAGE GENERATION
// =============================================================================

async function generateSourceImage(): Promise<string> {
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const prompt = `${KELLY.identity}, ${KELLY.responseExpression}, ${KELLY.responseGesture}, ${KELLY.background}, professional studio lighting, cinematic quality, 8k`;
  
  const output = await replicate.run(MODELS.FLUX_LORA as `${string}/${string}:${string}`, {
    input: {
      prompt,
      hf_lora: CONFIG.KELLY_LORA_URL,
      lora_scale: 0.85,
      num_outputs: 1,
      aspect_ratio: '16:9',
      output_format: 'png',
      guidance_scale: 3.5,
      output_quality: 100,
      num_inference_steps: 35,
      disable_safety_checker: true,
    },
  });
  
  return Array.isArray(output) ? String(output[0]) : String(output);
}

// =============================================================================
// MOTION VIDEO GENERATION
// =============================================================================

async function generateMotionVideo(imageUrl: string): Promise<string> {
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const motionPrompt = `${KELLY.identity}. ${KELLY.responseExpression}. Motion: ${KELLY.responseGesture}. Natural subtle head movements, blinking, breathing. Professional video quality.`;
  
  const prediction = await replicate.predictions.create({
    version: MODELS.MINIMAX.split(':')[1],
    input: {
      prompt: motionPrompt,
      first_frame_image: imageUrl,
      prompt_optimizer: true,
    },
  });
  
  // Poll for completion
  for (let i = 0; i < 120; i++) {
    const status = await replicate.predictions.get(prediction.id);
    
    if (status.status === 'succeeded') {
      if (typeof status.output === 'string') {
        return status.output;
      } else if (Array.isArray(status.output)) {
        return status.output[0];
      }
      throw new Error('Unexpected output format');
    }
    
    if (status.status === 'failed' || status.status === 'canceled') {
      throw new Error(`MiniMax failed: ${status.error || 'Unknown error'}`);
    }
    
    await sleep(5000);
  }
  
  throw new Error('MiniMax timed out');
}

// =============================================================================
// LIP-SYNC APPLICATION
// =============================================================================

async function applyLipSync(videoUrl: string, audioPath: string): Promise<string> {
  // Upload audio to Supabase for public URL
  const audioBuffer = fs.readFileSync(audioPath);
  const audioFileName = `sync_audio_${Date.now()}.mp3`;
  
  const sb = getSupabase();
  const { error: uploadError } = await sb.storage
    .from('kelly-templates')
    .upload(`sync-audio/${audioFileName}`, audioBuffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (uploadError) {
    throw new Error(`Audio upload failed: ${uploadError.message}`);
  }
  
  const { data: urlData } = sb.storage
    .from('kelly-templates')
    .getPublicUrl(`sync-audio/${audioFileName}`);
  
  const audioUrl = urlData.publicUrl;
  
  // Submit to Sync Labs
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl },
      ],
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Sync Labs error: ${response.status}`);
  }
  
  const job = await response.json();
  
  // Poll for completion
  for (let i = 0; i < 180; i++) {
    const pollResponse = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    const status = await pollResponse.json();
    
    if (status.status === 'COMPLETED') {
      return status.output?.[0]?.url || status.outputUrl;
    }
    
    if (status.status === 'FAILED' || status.status === 'REJECTED') {
      throw new Error(`Sync Labs failed: ${status.error || status.message}`);
    }
    
    await sleep(5000);
  }
  
  throw new Error('Sync Labs timed out');
}

// =============================================================================
// DOWNLOAD AND UPLOAD
// =============================================================================

async function downloadAndUpload(
  videoUrl: string,
  response: ResponseContent,
  dayNumber: number
): Promise<string> {
  // Download video
  const fetchResponse = await fetch(videoUrl);
  if (!fetchResponse.ok) {
    throw new Error(`Download failed: ${fetchResponse.status}`);
  }
  
  const buffer = Buffer.from(await fetchResponse.arrayBuffer());
  
  // Normalize names for storage
  const archetype = response.archetype.replace(/^The\s+/, '').toLowerCase();
  const phase = response.phase.toLowerCase();
  const letter = response.choiceLetter.toLowerCase();
  const dayStr = String(dayNumber).padStart(3, '0');
  
  // Storage path: day-001/explorer/fact1_response_a.mp4
  const storagePath = `day-${dayStr}/${archetype}/${phase}_response_${letter}.mp4`;
  
  // Upload to Supabase
  const sb = getSupabase();
  const { error: uploadError } = await sb.storage
    .from(CONFIG.BUCKET_NAME)
    .upload(storagePath, buffer, {
      contentType: 'video/mp4',
      upsert: true,
    });
  
  if (uploadError) {
    throw new Error(`Upload failed: ${uploadError.message}`);
  }
  
  // Get public URL
  const { data: urlData } = sb.storage
    .from(CONFIG.BUCKET_NAME)
    .getPublicUrl(storagePath);
  
  return urlData.publicUrl;
}

// =============================================================================
// GENERATE SINGLE RESPONSE VIDEO
// =============================================================================

async function generateResponseVideo(
  response: ResponseContent,
  dayNumber: number,
  dryRun: boolean
): Promise<GenerationResult> {
  const result: GenerationResult = {
    success: false,
    archetype: response.archetype,
    phase: response.phase,
    choiceLetter: response.choiceLetter,
  };
  
  console.log(`\n${'─'.repeat(60)}`);
  log('🎬', `${response.archetype} - ${response.phase} - Choice ${response.choiceLetter}`);
  log('📝', `"${response.responseScript.substring(0, 60)}..."`, 1);
  
  if (dryRun) {
    log('✅', '[DRY RUN] Would generate video', 1);
    result.success = true;
    return result;
  }
  
  try {
    // Create output directory
    const archetype = response.archetype.replace(/^The\s+/, '').toLowerCase();
    const baseDir = path.join(
      CONFIG.OUTPUT_DIR,
      `day_${String(dayNumber).padStart(3, '0')}`,
      archetype,
      `${response.phase.toLowerCase()}_response_${response.choiceLetter.toLowerCase()}`
    );
    fs.mkdirSync(baseDir, { recursive: true });
    
    // Step 1: Generate audio
    log('🎤', 'Generating audio...', 1);
    const audioPath = path.join(baseDir, 'audio.mp3');
    await generateAudio(response.responseScript, audioPath);
    log('✅', 'Audio generated', 2);
    
    // Step 2: Generate source image
    log('🎨', 'Generating source image...', 1);
    const imageUrl = await generateSourceImage();
    log('✅', 'Image generated', 2);
    
    // Step 3: Generate motion video
    log('🎬', 'Generating motion video (2-4 min)...', 1);
    const motionVideoUrl = await generateMotionVideo(imageUrl);
    log('✅', 'Motion video generated', 2);
    
    // Step 4: Apply lip-sync
    log('👄', 'Applying lip-sync...', 1);
    const lipsyncVideoUrl = await applyLipSync(motionVideoUrl, audioPath);
    log('✅', 'Lip-sync applied', 2);
    
    // Step 5: Download and upload to Supabase
    log('📤', 'Uploading to Supabase...', 1);
    const publicUrl = await downloadAndUpload(lipsyncVideoUrl, response, dayNumber);
    log('✅', `Uploaded: ${publicUrl}`, 2);
    
    result.success = true;
    result.publicUrl = publicUrl;
    
  } catch (error: any) {
    log('❌', `Error: ${error.message}`, 1);
    result.error = error.message;
  }
  
  return result;
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  
  let filterArchetype: string | undefined;
  const archetypeIdx = args.indexOf('--archetype');
  if (archetypeIdx >= 0 && args[archetypeIdx + 1]) {
    filterArchetype = args[archetypeIdx + 1];
  }
  
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  🎬 GENERATE RESPONSE VIDEOS - DAY 1                                 ║');
  console.log('╚' + '═'.repeat(70) + '╝');
  
  if (dryRun) {
    console.log('\n⚠️  DRY RUN MODE - No videos will be generated\n');
  }
  
  // Validate API keys
  const missing: string[] = [];
  if (!CONFIG.REPLICATE_API_TOKEN) missing.push('REPLICATE_API_TOKEN');
  if (!CONFIG.ELEVENLABS_API_KEY) missing.push('ELEVENLABS_API_KEY');
  if (!CONFIG.SYNC_LABS_API_KEY) missing.push('SYNC_LABS_API_KEY');
  if (!CONFIG.SUPABASE_URL) missing.push('SUPABASE_URL');
  if (!CONFIG.SUPABASE_KEY) missing.push('SUPABASE_SERVICE_ROLE_KEY');
  
  if (missing.length > 0) {
    console.error(`❌ Missing API keys: ${missing.join(', ')}`);
    process.exit(1);
  }
  
  // Fetch response scripts
  log('📚', 'Fetching response scripts from database...');
  let responses = await fetchResponseScripts(CONFIG.DAY_NUMBER);
  
  if (filterArchetype) {
    responses = responses.filter(r => r.archetype === filterArchetype);
    log('🔍', `Filtered to ${filterArchetype}: ${responses.length} responses`);
  }
  
  log('✅', `Found ${responses.length} response scripts`);
  
  if (responses.length === 0) {
    console.log('\n⚠️ No response scripts found. Make sure lesson_atoms have options with responses.');
    process.exit(0);
  }
  
  // Estimate time and cost
  const estimatedMinutes = responses.length * 5;
  const estimatedCost = responses.length * 0.50; // ~$0.50 per video
  
  console.log(`\n📊 Estimated time: ${estimatedMinutes} minutes (${(estimatedMinutes / 60).toFixed(1)} hours)`);
  console.log(`💰 Estimated cost: ~$${estimatedCost.toFixed(2)}`);
  
  if (!dryRun) {
    console.log('\n⏳ Starting in 5 seconds... (Ctrl+C to cancel)');
    await sleep(5000);
  }
  
  // Generate videos
  const results: GenerationResult[] = [];
  
  for (let i = 0; i < responses.length; i++) {
    console.log(`\n[${i + 1}/${responses.length}]`);
    const result = await generateResponseVideo(responses[i], CONFIG.DAY_NUMBER, dryRun);
    results.push(result);
    
    // Brief pause between generations
    if (!dryRun && i < responses.length - 1) {
      await sleep(2000);
    }
  }
  
  // Summary
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  
  console.log('\n');
  console.log('═'.repeat(72));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(72));
  console.log(`   ✅ Successful: ${successful}/${responses.length}`);
  console.log(`   ❌ Failed: ${failed}/${responses.length}`);
  
  if (failed > 0) {
    console.log('\n   Failed videos:');
    results.filter(r => !r.success).forEach(r => {
      console.log(`   - ${r.archetype} / ${r.phase} / ${r.choiceLetter}: ${r.error}`);
    });
  }
  
  // Save results
  if (!dryRun) {
    const resultsPath = path.join(CONFIG.OUTPUT_DIR, `results_${Date.now()}.json`);
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`\n📁 Results saved: ${resultsPath}`);
  }
  
  console.log('\n🎉 Done!\n');
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});


