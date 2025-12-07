/**
 * 🎬 PRODUCTION TEMPLATE PIPELINE
 * 
 * Creates base video templates with ACTUAL MOTION (not just talking heads).
 * 
 * Architecture:
 * ┌──────────────────────────────────────────────────────────────────┐
 * │  PHASE 1: MOTION GENERATION                                      │
 * │  Tool: MiniMax Video-01 / Runway / Luma                         │
 * │  Input: Production-grade prompt with specific gestures           │
 * │  Output: 6s video with Kelly performing template motion          │
 * └──────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌──────────────────────────────────────────────────────────────────┐
 * │  PHASE 2: LIP-SYNC OVERLAY                                       │
 * │  Tool: Wav2Lip (V2V) / Sync Labs                                │
 * │  Input: Motion video + ElevenLabs audio                         │
 * │  Output: Same motion with accurate lip movements                 │
 * └──────────────────────────────────────────────────────────────────┘
 * 
 * This is the CORRECT pipeline - motion FIRST, lip-sync SECOND.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'template-forge', 'production-templates'),
};

// Create output directory
fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });

// =============================================================================
// KELLY CHARACTER PROMPTS (Film-Quality Motion Descriptions)
// =============================================================================

// Load production templates from JSON
const TEMPLATES_FILE = path.join(process.cwd(), 'template-forge', 'BASE_VIDEO_TEMPLATES.json');

interface TemplateSpec {
  id: string;
  name: string;
  internal_name: string;
  category: string;
  purpose: string;
  duration?: { total_sec: number };
  camera: any;
  motion_breakdown: any;
  emotional_arc: any;
  prompt_guidance: {
    minimax: string;
    runway: string;
    luma: string;
  };
  lip_sync_integration: any;
}

function loadTemplates(): { templates: TemplateSpec[] } {
  const data = JSON.parse(fs.readFileSync(TEMPLATES_FILE, 'utf-8'));
  return data;
}

// =============================================================================
// VIDEO GENERATION MODELS
// =============================================================================

const MINIMAX_MODEL = 'minimax/video-01:5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912';

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// PHASE 1: MOTION VIDEO GENERATION (MiniMax)
// =============================================================================

async function generateMotionVideo(
  prompt: string,
  templateId: string
): Promise<string> {
  console.log('\n🎬 PHASE 1: Generating MOTION Video');
  console.log('━'.repeat(60));
  console.log(`   Template: ${templateId}`);
  console.log(`   Prompt: "${prompt.substring(0, 100)}..."`);
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  console.log('   🚀 Submitting to MiniMax Video-01...');
  
  const prediction = await replicate.predictions.create({
    version: MINIMAX_MODEL.split(':')[1],
    input: {
      prompt,
      prompt_optimizer: true,
    },
  });
  
  console.log(`   Prediction ID: ${prediction.id}`);
  
  // Poll for completion (MiniMax takes 2-5 minutes)
  let attempts = 0;
  const maxAttempts = 120; // 10 minutes max
  
  while (attempts < maxAttempts) {
    const status = await replicate.predictions.get(prediction.id);
    const elapsed = Math.round((attempts * 5) / 60);
    
    if (status.status === 'succeeded') {
      console.log(`\n   ✅ Motion video generated! (${elapsed}m)`);
      
      // Extract video URL
      let videoUrl: string;
      if (typeof status.output === 'string') {
        videoUrl = status.output;
      } else if (Array.isArray(status.output)) {
        videoUrl = status.output[0];
      } else {
        throw new Error(`Unexpected output format: ${JSON.stringify(status.output)}`);
      }
      
      console.log(`   URL: ${videoUrl.substring(0, 80)}...`);
      return videoUrl;
    }
    
    if (status.status === 'failed') {
      throw new Error(`MiniMax failed: ${status.error}`);
    }
    
    if (status.status === 'canceled') {
      throw new Error('MiniMax job was canceled');
    }
    
    process.stdout.write(`\r   ⏳ Status: ${status.status} (${elapsed}m)...`);
    await sleep(5000);
    attempts++;
  }
  
  throw new Error('MiniMax timed out after 10 minutes');
}

// =============================================================================
// AUDIO GENERATION (ElevenLabs)
// =============================================================================

async function generateAudio(text: string): Promise<{ buffer: Buffer; url: string }> {
  console.log('\n🎤 Generating Kelly Voice (ElevenLabs)');
  console.log('━'.repeat(60));
  console.log(`   Text: "${text.substring(0, 60)}..."`);
  
  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`, {
    method: 'POST',
    headers: {
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json',
      'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
    },
    body: JSON.stringify({
      text,
      model_id: 'eleven_turbo_v2_5',
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.85,
        use_speaker_boost: true,
      },
    }),
  });
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  console.log(`   ✅ Audio: ${(buffer.length / 1024).toFixed(1)} KB`);
  
  // Upload to Supabase for URL access
  const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  const filename = `kelly_audio_${Date.now()}.mp3`;
  
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(`production-pipeline/${filename}`, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (error) {
    console.log(`   ⚠️ Supabase upload failed: ${error.message}`);
    // Return data URI as fallback
    return { 
      buffer, 
      url: `data:audio/mpeg;base64,${buffer.toString('base64')}` 
    };
  }
  
  const { data: urlData } = supabase.storage
    .from('kelly-templates')
    .getPublicUrl(`production-pipeline/${filename}`);
  
  console.log(`   📤 Uploaded: ${urlData.publicUrl.substring(0, 60)}...`);
  
  return { buffer, url: urlData.publicUrl };
}

// =============================================================================
// PHASE 2: V2V LIP-SYNC (Wav2Lip on Motion Video)
// =============================================================================

async function applyV2VLipSync(
  motionVideoUrl: string,
  audioUrl: string,
  templateId: string
): Promise<string> {
  console.log('\n👄 PHASE 2: Applying V2V Lip-Sync');
  console.log('━'.repeat(60));
  console.log(`   Motion Video: ${motionVideoUrl.substring(0, 60)}...`);
  console.log(`   Audio: ${audioUrl.substring(0, 60)}...`);
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  console.log('   🚀 Running Wav2Lip V2V...');
  
  const prediction = await replicate.predictions.create({
    version: '8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    input: {
      face: motionVideoUrl,  // VIDEO input (not image!)
      audio: audioUrl,
      fps: 25,
      smooth: true,
      resize_factor: 1,
      pads: '0 10 0 0',
    },
  });
  
  console.log(`   Prediction ID: ${prediction.id}`);
  
  // Poll for completion
  let attempts = 0;
  const maxAttempts = 120;
  
  while (attempts < maxAttempts) {
    const status = await replicate.predictions.get(prediction.id);
    const elapsed = Math.round((attempts * 5) / 60);
    
    if (status.status === 'succeeded') {
      console.log(`\n   ✅ Lip-sync applied! (${elapsed}m)`);
      
      const videoUrl = typeof status.output === 'string' 
        ? status.output 
        : (status.output as string[])?.[0] || status.output;
      
      console.log(`   URL: ${String(videoUrl).substring(0, 80)}...`);
      return String(videoUrl);
    }
    
    if (status.status === 'failed') {
      throw new Error(`Wav2Lip failed: ${status.error}`);
    }
    
    process.stdout.write(`\r   ⏳ Status: ${status.status} (${elapsed}m)...`);
    await sleep(5000);
    attempts++;
  }
  
  throw new Error('Wav2Lip timed out');
}

// =============================================================================
// DOWNLOAD & SAVE VIDEO
// =============================================================================

async function downloadVideo(url: string, outputPath: string): Promise<string> {
  console.log(`\n💾 Downloading video...`);
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(outputPath, buffer);
  
  const sizeMB = (buffer.length / (1024 * 1024)).toFixed(2);
  console.log(`   ✅ Saved: ${outputPath} (${sizeMB} MB)`);
  
  return outputPath;
}

// =============================================================================
// FULL PIPELINE: Motion + Lip-Sync
// =============================================================================

interface PipelineResult {
  templateId: string;
  templateName: string;
  success: boolean;
  motionVideoUrl?: string;
  finalVideoUrl?: string;
  localPath?: string;
  error?: string;
  duration?: number;
}

async function runProductionPipeline(
  template: TemplateSpec,
  scriptText: string
): Promise<PipelineResult> {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log(`║  🎬 PRODUCTION TEMPLATE: ${template.id} - ${template.name.padEnd(30)}   ║`);
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  
  const startTime = Date.now();
  
  try {
    // PHASE 1: Generate motion video with MiniMax
    const motionVideoUrl = await generateMotionVideo(
      template.prompt_guidance.minimax,
      template.id
    );
    
    // Save motion-only video
    const motionOnlyPath = path.join(
      CONFIG.OUTPUT_DIR,
      `${template.id}_motion_only_${Date.now()}.mp4`
    );
    await downloadVideo(motionVideoUrl, motionOnlyPath);
    
    // Generate audio
    const audio = await generateAudio(scriptText);
    
    // PHASE 2: Apply V2V lip-sync to motion video
    const finalVideoUrl = await applyV2VLipSync(
      motionVideoUrl,
      audio.url,
      template.id
    );
    
    // Save final video
    const finalPath = path.join(
      CONFIG.OUTPUT_DIR,
      `${template.id}_${template.internal_name}_${Date.now()}.mp4`
    );
    await downloadVideo(finalVideoUrl, finalPath);
    
    const duration = (Date.now() - startTime) / 1000;
    
    console.log('\n');
    console.log('═'.repeat(70));
    console.log(`✅ ${template.id} COMPLETE`);
    console.log('═'.repeat(70));
    console.log(`   Duration: ${duration.toFixed(1)}s`);
    console.log(`   Motion Video: ${motionOnlyPath}`);
    console.log(`   Final Video: ${finalPath}`);
    
    return {
      templateId: template.id,
      templateName: template.name,
      success: true,
      motionVideoUrl,
      finalVideoUrl,
      localPath: finalPath,
      duration,
    };
    
  } catch (error: any) {
    console.log(`\n❌ FAILED: ${error.message}`);
    return {
      templateId: template.id,
      templateName: template.name,
      success: false,
      error: error.message,
    };
  }
}

// =============================================================================
// SAMPLE SCRIPTS FOR EACH TEMPLATE
// =============================================================================

const TEMPLATE_SCRIPTS: Record<string, string> = {
  T01: "Hello, curious learner! I'm so excited to see you today. Welcome to a new adventure in learning!",
  T02: "Let me explain how this works. When we break down a big idea into smaller parts, it becomes so much easier to understand.",
  T03: "Hmm, that's really interesting. Let me take a closer look at this. What do you notice about it?",
  T04: "I want to share something meaningful with you. This lesson comes from my heart, because I truly believe it will help you.",
  T05: "Oh wow! Did you see that? This is amazing! We just discovered something incredible together!",
  T06: "Let me think about that for a moment. Sometimes the best insights come when we pause and reflect.",
  T07: "Yes! You did it! I'm so proud of you! That was fantastic work!",
  T08: "I hear you. Tell me more about that. I'm really listening.",
  T09: "It's okay if this feels challenging. Learning something new is always a little uncomfortable at first.",
  T10: "Thank you so much for learning with me today. Remember, every small step forward counts. See you next time!",
};

// =============================================================================
// MAIN EXECUTION
// =============================================================================

async function main() {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 PRODUCTION TEMPLATE PIPELINE                                             ║');
  console.log('║     Motion Generation + V2V Lip-Sync                                        ║');
  console.log('╠══════════════════════════════════════════════════════════════════════════════╣');
  console.log('║  CORRECT ARCHITECTURE:                                                       ║');
  console.log('║  1. MiniMax Video-01 → Generates MOTION (walk, gesture, wave)               ║');
  console.log('║  2. Wav2Lip V2V → Applies lip-sync ON TOP of motion                         ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝');
  
  const args = process.argv.slice(2);
  const templateArg = args.find(a => a.startsWith('--template'))?.split('=')[1]?.toUpperCase();
  const allTemplates = args.includes('--all');
  const motionOnly = args.includes('--motion-only');
  
  // Load templates
  const { templates } = loadTemplates();
  console.log(`\n📋 Loaded ${templates.length} production templates`);
  
  // Select templates to process
  let selectedTemplates: TemplateSpec[];
  
  if (allTemplates) {
    selectedTemplates = templates;
  } else if (templateArg) {
    const template = templates.find(t => t.id === templateArg);
    if (!template) {
      console.log(`❌ Unknown template: ${templateArg}`);
      console.log(`   Available: ${templates.map(t => t.id).join(', ')}`);
      process.exit(1);
    }
    selectedTemplates = [template];
  } else {
    // Default: generate T02 (Present & Explain) as test
    selectedTemplates = [templates.find(t => t.id === 'T02')!];
    console.log('   (Using T02 as default test template)');
  }
  
  console.log(`\n🎯 Processing ${selectedTemplates.length} template(s):\n`);
  for (const t of selectedTemplates) {
    console.log(`   ${t.id}: ${t.name} (${t.category})`);
  }
  
  // Process templates
  const results: PipelineResult[] = [];
  
  for (const template of selectedTemplates) {
    const script = TEMPLATE_SCRIPTS[template.id] || TEMPLATE_SCRIPTS.T02;
    
    if (motionOnly) {
      // Only generate motion video, skip lip-sync
      console.log('\n⚠️ Motion-only mode: Skipping lip-sync phase');
      
      try {
        const motionVideoUrl = await generateMotionVideo(
          template.prompt_guidance.minimax,
          template.id
        );
        
        const motionPath = path.join(
          CONFIG.OUTPUT_DIR,
          `${template.id}_motion_only_${Date.now()}.mp4`
        );
        await downloadVideo(motionVideoUrl, motionPath);
        
        results.push({
          templateId: template.id,
          templateName: template.name,
          success: true,
          motionVideoUrl,
          localPath: motionPath,
        });
      } catch (error: any) {
        results.push({
          templateId: template.id,
          templateName: template.name,
          success: false,
          error: error.message,
        });
      }
    } else {
      // Full pipeline: motion + lip-sync
      const result = await runProductionPipeline(template, script);
      results.push(result);
    }
  }
  
  // Summary
  console.log('\n\n');
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║  📊 PIPELINE SUMMARY                                                         ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝');
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n   Total: ${results.length}`);
  console.log(`   ✅ Success: ${successful.length}`);
  console.log(`   ❌ Failed: ${failed.length}`);
  
  if (successful.length > 0) {
    console.log('\n   Successful:');
    for (const r of successful) {
      console.log(`      ${r.templateId}: ${r.templateName}`);
      if (r.localPath) console.log(`         → ${r.localPath}`);
    }
  }
  
  if (failed.length > 0) {
    console.log('\n   Failed:');
    for (const r of failed) {
      console.log(`      ${r.templateId}: ${r.error}`);
    }
  }
  
  // Save report
  const reportPath = path.join(CONFIG.OUTPUT_DIR, `pipeline_report_${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    architecture: 'motion-first-then-lipsync',
    results,
  }, null, 2));
  
  console.log(`\n   📄 Report: ${reportPath}`);
  console.log('\n' + '═'.repeat(78));
}

main().catch(console.error);

