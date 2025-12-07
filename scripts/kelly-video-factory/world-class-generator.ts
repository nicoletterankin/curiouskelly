#!/usr/bin/env npx tsx
/**
 * WORLD-CLASS KELLY TEMPLATE GENERATOR
 * 
 * Creates 10 visually distinct templates with TRUE animation layers
 * Each template has a SIGNATURE visual element that makes it unique
 * 
 * Usage:
 *   npx tsx world-class-generator.ts --all
 *   npx tsx world-class-generator.ts --template T05
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ES Module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize clients
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Constants
const OUTPUT_DIR = path.join(__dirname, '../../template-forge/world-class-templates');
const TEMPLATES_FILE = path.join(__dirname, '../../template-forge/WORLD_CLASS_TEMPLATES.json');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Load template specs
const templateSpecs = JSON.parse(fs.readFileSync(TEMPLATES_FILE, 'utf-8'));

interface TemplateSpec {
  id: string;
  name: string;
  visual_signature: string;
  what_makes_it_unique: string;
  generation_prompt: {
    video_model: string;
    prompt: string;
    negative: string;
  };
}

// Kelly's identity for consistent character
const KELLY_IDENTITY = {
  face: "young woman, late 20s, warm friendly face, brown eyes",
  hair: "long wavy chestnut brown hair with caramel highlights, past shoulders",
  clothing: "powder blue crew neck sweater",
  style: "professional, approachable, teacher energy",
  background: "soft focus classroom or studio, warm lighting"
};

// Enhanced prompts with Kelly identity baked in
function buildEnhancedPrompt(template: TemplateSpec): string {
  const identity = `${KELLY_IDENTITY.face}, ${KELLY_IDENTITY.hair}, wearing ${KELLY_IDENTITY.clothing}`;
  
  // Template-specific motion emphasis
  const motionEmphasis: Record<string, string> = {
    'T01': 'walking movement, hair swinging with steps, natural gait',
    'T02': 'animated hand gestures throughout, counting on fingers, presenting with palms',
    'T03': 'head tilted to the right, one eyebrow raised, hand touching chin',
    'T04': 'hand placed on heart, soft emotional expression, sincere gesture',
    'T05': 'hands raised in celebration near face, excited bounce, wide eyes',
    'T06': 'chin resting on hand, gazing into distance, very still and contemplative',
    'T07': 'clapping hands with joy, then pointing at camera, celebration',
    'T08': 'leaning forward, nodding continuously, attentive listening pose',
    'T09': 'slow calming hand movements pressing downward, gentle pace',
    'T10': 'hands clasped together in gratitude, gentle goodbye wave'
  };

  const motion = motionEmphasis[template.id] || '';
  
  return `Professional cinematic video of ${identity}. ${template.generation_prompt.prompt} Key motion: ${motion}. ${KELLY_IDENTITY.background}. Smooth natural movement, high quality, 8 seconds.`;
}

// Generate image with Flux (standard, no LoRA)
async function generateKellyImage(template: TemplateSpec): Promise<string> {
  console.log(`   📸 Generating source image for ${template.id}...`);
  
  // Kelly's consistent identity description - MUST specify exact color
  const KELLY_DESC = `a young woman in her late 20s with long wavy chestnut brown hair with caramel highlights flowing past her shoulders, warm brown eyes, light olive skin, wearing a LIGHT STEEL BLUE crewneck sweater (hex color #B0C4DE, similar to light denim blue or dusty blue)`;
  
  // Pose-specific image prompts - HIGHLY SPECIFIC to ensure differentiation
  const posePrompts: Record<string, string> = {
    'T01': `${KELLY_DESC}. Standing pose with right hand raised to shoulder height in a friendly wave, fingers together, palm facing forward. She has a warm welcoming smile showing teeth. Full body shot, facing camera directly. Professional studio with soft warm lighting, shallow depth of field.`,
    'T02': `${KELLY_DESC}. Teaching pose, medium shot, chest up. Right hand extended with open palm facing up at chest height presenting an idea. Left hand supporting below. Engaged enthusiastic expression, eyebrows slightly raised, confident teacher energy. Studio background.`,
    'T03': `${KELLY_DESC}. Close portrait, head tilted 15 degrees to the RIGHT, right eyebrow raised higher than left showing curiosity. Right hand touching chin in thinking pose. Intrigued inquisitive expression, slight narrowing of eyes examining something. Studio lighting.`,
    'T04': `${KELLY_DESC}. Close portrait, right hand placed flat on heart/center of chest with fingers together in sincere gesture. Soft emotional expression, eyes warm and slightly glistening, gentle closed-lip smile showing vulnerability. Intimate studio lighting.`,
    'T05': `${KELLY_DESC}. CELEBRATION POSE - both hands raised up near face level with fingers spread wide like jazz hands, eyes wide open with excitement and wonder, HUGE genuine smile showing teeth with eye crinkles, expressing pure joy and excitement. Bright energetic studio lighting.`,
    'T06': `${KELLY_DESC}. Classic thinker pose - chin resting on right hand with elbow supported, gazing slightly to the left into middle distance NOT at camera. Contemplative thoughtful expression, slight furrow between brows, very still composed posture. Soft studio lighting.`,
    'T07': `${KELLY_DESC}. CLAPPING pose - hands together at chest level in applause position, capturing the moment of clap. Joyful proud expression, big smile with eye crinkles, head slightly tilted with pride. Celebratory energy, studio background.`,
    'T08': `${KELLY_DESC}. Leaning slightly forward in attentive listening posture. Soft encouraging smile, eyes focused and interested, head tilted slightly. Hands relaxed and open. Active listening expression showing engagement. Studio setting.`,
    'T09': `${KELLY_DESC}. Calming gesture - both hands in front at waist level, palms facing downward in gentle pressing motion. Soft empathetic expression, eyes warm and caring, gentle reassuring smile. Slow calming energy. Warm studio lighting.`,
    'T10': `${KELLY_DESC}. Hands clasped together at chest in grateful prayer-like gesture. Soft thankful expression with gentle smile, eyes warm possibly slightly glistening with gratitude. Peaceful thankful demeanor. Warm soft studio lighting.`
  };

  const prompt = posePrompts[template.id] || `${KELLY_DESC}. Professional portrait, friendly expression, studio lighting.`;

  try {
    // Use standard Flux Dev for high quality
    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt,
          aspect_ratio: "16:9",
          output_format: "png",
          output_quality: 100,
          num_inference_steps: 28,
          guidance: 3.5
        }
      }
    ) as string[];

    console.log(`   ✅ Image generated`);
    return Array.isArray(output) ? output[0] : output as string;
  } catch (error) {
    console.error(`   ❌ Image generation failed:`, error);
    throw error;
  }
}

// Generate base video with motion
async function generateBaseVideo(imageUrl: string, template: TemplateSpec): Promise<string> {
  console.log(`   🎬 Generating base video...`);
  
  const motionPrompts: Record<string, string> = {
    'T01': 'Woman walks into frame from left side, natural gait with hair swinging, turns to face camera, raises hand to wave at shoulder height, settles into welcoming pose',
    'T02': 'Woman gestures actively while explaining, open palm presentation, counts on fingers one-two-three, brings hands together with emphatic nod',
    'T03': 'Woman tilts head to right with curious expression, raises one eyebrow, touches chin in thinking pose, then eyes brighten with realization',
    'T04': 'Woman takes deep breath, places hand gently on heart, soft sincere expression, then extends hand outward in sharing gesture',
    'T05': 'Woman\'s eyes go wide, raises both hands near face with fingers spread in celebration, bounces with excitement, full joyful smile',
    'T06': 'Woman rests chin on hand in thinking pose, gazes to middle distance contemplatively, very still, then refocuses with knowing smile',
    'T07': 'Woman claps hands joyfully 3-4 times at chest level, bouncing slightly, then points encouragingly at camera with proud expression',
    'T08': 'Woman leans forward attentively, nods continuously in small encouraging movements, eyes tracking, then larger affirming nod',
    'T09': 'Woman makes slow calming gesture pressing hands downward gently, soft empathetic expression, slow supportive nods',
    'T10': 'Woman clasps hands together at chest in grateful gesture, small thankful bow, then gentle goodbye wave at face height'
  };

  const motion = motionPrompts[template.id];

  // Try Minimax Video-01 first (best quality)
  try {
    console.log(`   Trying Minimax Video-01...`);
    const output = await replicate.run(
      "minimax/video-01",
      {
        input: {
          prompt: `${motion}. Natural fluid movement, professional quality, warm lighting.`,
          first_frame_image: imageUrl,
          prompt_optimizer: true
        }
      }
    );

    const videoUrl = typeof output === 'string' ? output : (output as any)?.output || (output as string[])?.[0];
    if (videoUrl) {
      console.log(`   ✅ Minimax video generated`);
      return videoUrl;
    }
    throw new Error('No video URL returned');
  } catch (error: any) {
    console.error(`   ⚠️ Minimax failed: ${error.message?.substring(0, 50)}...`);
  }

  // Fallback to LumaLabs Dream Machine
  try {
    console.log(`   Trying Luma Dream Machine...`);
    const output = await replicate.run(
      "luma/dream-machine",
      {
        input: {
          prompt: motion,
          start_image_url: imageUrl,
          aspect_ratio: "16:9"
        }
      }
    );
    
    const videoUrl = typeof output === 'string' ? output : (output as any)?.video || (output as string[])?.[0];
    if (videoUrl) {
      console.log(`   ✅ Luma video generated`);
      return videoUrl;
    }
    throw new Error('No video URL returned');
  } catch (error: any) {
    console.error(`   ⚠️ Luma failed: ${error.message?.substring(0, 50)}...`);
  }

  // Fallback to SVD (Stable Video Diffusion)
  try {
    console.log(`   Trying Stable Video Diffusion...`);
    const output = await replicate.run(
      "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
      {
        input: {
          input_image: imageUrl,
          video_length: "25_frames_with_svd_xt",
          sizing_strategy: "maintain_aspect_ratio",
          motion_bucket_id: 127,
          fps: 7
        }
      }
    );

    const videoUrl = typeof output === 'string' ? output : (output as string[])?.[0];
    if (videoUrl) {
      console.log(`   ✅ SVD video generated`);
      return videoUrl;
    }
    throw new Error('No video URL returned');
  } catch (error: any) {
    console.error(`   ⚠️ SVD failed: ${error.message?.substring(0, 50)}...`);
  }

  // Final fallback - return image as a video would need post-processing
  console.log(`   ⚠️ All video models failed, using image directly with Sync Labs for animation`);
  return imageUrl;
}

// Use Hedra to animate image with audio (can create video from still image)
async function animateWithHedra(imageUrl: string, audioUrl: string): Promise<string> {
  console.log(`   🎭 Animating image with Hedra...`);
  
  const HEDRA_API_KEY = process.env.HEDRA_API_KEY;
  
  if (!HEDRA_API_KEY) {
    console.log(`   ⚠️ No Hedra key, trying Sync Labs directly`);
    return imageUrl;
  }

  try {
    // Initialize character
    const initRes = await fetch('https://api.hedra.com/v1/characters', {
      method: 'POST',
      headers: {
        'X-API-Key': HEDRA_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        avatarImageInput: { type: 'url', url: imageUrl }
      })
    });

    if (!initRes.ok) throw new Error(`Init failed: ${await initRes.text()}`);
    const character = await initRes.json();

    // Generate video
    const genRes = await fetch('https://api.hedra.com/v1/characters/generate', {
      method: 'POST',
      headers: {
        'X-API-Key': HEDRA_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        characterId: character.id || character.characterId,
        audioUrl: audioUrl,
        aspectRatio: '16:9'
      })
    });

    if (!genRes.ok) throw new Error(`Gen failed: ${await genRes.text()}`);
    const job = await genRes.json();

    // Poll for completion
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 5000));
      
      const statusRes = await fetch(`https://api.hedra.com/v1/characters/${job.jobId || job.id}/status`, {
        headers: { 'X-API-Key': HEDRA_API_KEY }
      });
      const status = await statusRes.json();

      if (status.status === 'complete' || status.videoUrl) {
        console.log(`\n   ✅ Hedra animation complete`);
        return status.videoUrl || status.url;
      }
      if (status.status === 'failed') throw new Error(status.error || 'Failed');
      
      process.stdout.write('.');
    }
    throw new Error('Timeout');
  } catch (error: any) {
    console.error(`   ❌ Hedra failed: ${error.message}`);
    return imageUrl;
  }
}

// Apply Sync Labs lip-sync enhancement (needs video input for v2)
async function applyLipSync(mediaUrl: string, audioUrl: string): Promise<string> {
  console.log(`   👄 Applying Sync Labs lip-sync...`);
  
  const SYNC_LABS_API_KEY = process.env.SYNC_LABS_API_KEY;
  
  if (!SYNC_LABS_API_KEY) {
    console.log(`   ⚠️ No Sync Labs key, skipping lip-sync`);
    return typeof mediaUrl === 'string' ? mediaUrl : String(mediaUrl);
  }

  // Ensure mediaUrl is a string
  const urlStr = typeof mediaUrl === 'string' ? mediaUrl : String(mediaUrl);
  
  // Determine if input is image or video
  const isImage = urlStr.includes('.png') || urlStr.includes('.jpg') || urlStr.includes('.jpeg') || urlStr.includes('.webp');
  
  // If image, need to convert to video first using Hedra
  if (isImage) {
    console.log(`   Input is image, using Hedra to animate first...`);
    const hedraResult = await animateWithHedra(urlStr, audioUrl);
    if (hedraResult !== urlStr) {
      // Hedra succeeded, now apply Sync Labs for better quality
      return applyLipSync(hedraResult, audioUrl);
    }
    // Hedra failed, return image
    return urlStr;
  }
  
  const mediaType = 'video';
  console.log(`   Input type: ${mediaType}`);

  try {
    // Submit job
    const submitRes = await fetch('https://api.sync.so/v2/generate', {
      method: 'POST',
      headers: {
        'x-api-key': SYNC_LABS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'lipsync-2',
        input: [
          { type: mediaType, url: urlStr },
          { type: 'audio', url: audioUrl }
        ]
      })
    });

    if (!submitRes.ok) {
      const errorText = await submitRes.text();
      throw new Error(`Submit failed: ${submitRes.status} - ${errorText}`);
    }

    const job = await submitRes.json();
    console.log(`   Job ID: ${job.id}`);

    // Poll for completion
    for (let i = 0; i < 120; i++) {
      await new Promise(r => setTimeout(r, 5000));
      
      const statusRes = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
        headers: { 'x-api-key': SYNC_LABS_API_KEY }
      });
      const status = await statusRes.json();

      if (status.status === 'COMPLETED') {
        const outputUrl = status.output?.[0]?.url || status.outputUrl || status.output;
        console.log(`\n   ✅ Lip-sync complete`);
        return outputUrl;
      }
      if (status.status === 'FAILED' || status.status === 'REJECTED') {
        throw new Error(status.error || status.message || 'Generation failed');
      }
      
      if (i % 12 === 0) {
        console.log(`\n      Status: ${status.status} (${Math.round(i * 5 / 60)}m)`);
      }
      process.stdout.write('.');
    }

    throw new Error('Lip-sync timed out after 10 minutes');
  } catch (error: any) {
    console.error(`   ❌ Lip-sync failed:`, error.message);
    return urlStr;
  }
}

// Generate Kelly's voice audio
async function generateAudio(text: string): Promise<string> {
  console.log(`   🎤 Generating audio...`);
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0'}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': process.env.ELEVENLABS_API_KEY!
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
          use_speaker_boost: true
        }
      })
    }
  );

  if (!response.ok) throw new Error(`ElevenLabs error: ${response.status}`);

  const buffer = await response.arrayBuffer();
  const audioPath = `sota-pipeline/world_class_audio_${Date.now()}.mp3`;
  
  await supabase.storage
    .from('kelly-templates')
    .upload(audioPath, Buffer.from(buffer), { contentType: 'audio/mpeg', upsert: true });

  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(audioPath);
  console.log(`   ✅ Audio ready`);
  return data.publicUrl;
}

// Sample dialogue for each template
const TEMPLATE_DIALOGUES: Record<string, string> = {
  'T01': "Hello there! I'm so glad you're here. I'm Kelly, and I'm really excited to explore something amazing with you today!",
  'T02': "Let me show you something fascinating. First, notice this. Second, see how it connects here. And third - this is the really cool part!",
  'T03': "Hmm, that's really interesting when you think about it. What if we looked at it from this angle? Oh! I think I see it now!",
  'T04': "You know, I want to share something that really means a lot to me. This is something I carry with me, and I'm sharing it with you.",
  'T05': "Oh wow! This is SO exciting! Can you believe it? This is absolutely amazing - I just have to celebrate this with you!",
  'T06': "Let me think about this for a moment... Yes, I see it now. The answer was there all along, waiting for us to notice.",
  'T07': "Yes! You did it! That's wonderful! I knew you could do it. I'm so proud of what you accomplished!",
  'T08': "Mmhmm, I hear you. Yes, keep going. That's a really good point. I understand what you're saying.",
  'T09': "It's okay. Take your time. You've got this, I believe in you. There's no rush at all.",
  'T10': "Thank you so much for spending this time with me today. I really appreciate you. See you next time!"
};

// Main generation function for a single template
async function generateTemplate(templateId: string): Promise<void> {
  const template = templateSpecs.templates.find((t: TemplateSpec) => t.id === templateId);
  
  if (!template) {
    console.error(`Template ${templateId} not found`);
    return;
  }

  console.log(`\n${'═'.repeat(70)}`);
  console.log(`🎬 GENERATING: ${template.id} - ${template.name}`);
  console.log(`   Signature: ${template.visual_signature}`);
  console.log(`${'═'.repeat(70)}`);

  const outputDir = path.join(OUTPUT_DIR, template.id);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  try {
    // Step 1: Generate audio
    const dialogue = TEMPLATE_DIALOGUES[template.id];
    const audioUrl = await generateAudio(dialogue);

    // Step 2: Generate source image with signature pose
    const imageUrl = await generateKellyImage(template);
    
    // Save image locally
    const imageRes = await fetch(imageUrl);
    const imageBuffer = await imageRes.arrayBuffer();
    const imagePath = path.join(outputDir, `${template.id.toLowerCase()}_source.png`);
    fs.writeFileSync(imagePath, Buffer.from(imageBuffer));

    // Step 3: Generate base video with motion
    const baseVideoUrl = await generateBaseVideo(imageUrl, template);
    
    // Save base video
    const videoRes = await fetch(baseVideoUrl);
    const videoBuffer = await videoRes.arrayBuffer();
    const baseVideoPath = path.join(outputDir, `${template.id.toLowerCase()}_base.mp4`);
    fs.writeFileSync(baseVideoPath, Buffer.from(videoBuffer));

    // Step 4: Apply lip-sync
    const finalVideoUrl = await applyLipSync(baseVideoUrl, audioUrl);
    
    // Save final video
    const finalRes = await fetch(finalVideoUrl);
    const finalBuffer = await finalRes.arrayBuffer();
    const finalVideoPath = path.join(outputDir, `${template.id.toLowerCase()}_final.mp4`);
    fs.writeFileSync(finalVideoPath, Buffer.from(finalBuffer));

    // Step 5: Upload to Supabase
    const storagePath = `world-class-templates/${template.id}/${template.id.toLowerCase()}_${Date.now()}.mp4`;
    await supabase.storage
      .from('kelly-templates')
      .upload(storagePath, Buffer.from(finalBuffer), { contentType: 'video/mp4', upsert: true });

    const { data: publicUrl } = supabase.storage.from('kelly-templates').getPublicUrl(storagePath);

    // Save metadata
    const metadata = {
      templateId: template.id,
      templateName: template.name,
      visual_signature: template.visual_signature,
      what_makes_it_unique: template.what_makes_it_unique,
      generated: new Date().toISOString(),
      files: {
        source_image: imagePath,
        base_video: baseVideoPath,
        final_video: finalVideoPath,
        supabase_url: publicUrl.publicUrl
      },
      urls: {
        image: imageUrl,
        base_video: baseVideoUrl,
        final_video: finalVideoUrl,
        audio: audioUrl
      }
    };
    
    fs.writeFileSync(
      path.join(outputDir, `${template.id.toLowerCase()}_metadata.json`),
      JSON.stringify(metadata, null, 2)
    );

    console.log(`\n   ✅ ${template.id} COMPLETE`);
    console.log(`   📁 Output: ${outputDir}`);
    console.log(`   🔗 URL: ${publicUrl.publicUrl}`);

  } catch (error) {
    console.error(`   ❌ Failed to generate ${template.id}:`, error);
    throw error;
  }
}

// Generate all templates
async function generateAllTemplates(): Promise<void> {
  console.log('═'.repeat(70));
  console.log('🌟 WORLD-CLASS KELLY TEMPLATE GENERATOR');
  console.log('   Creating 10 visually distinct templates with true animation layers');
  console.log('═'.repeat(70));

  const results: Array<{ id: string; success: boolean; error?: string }> = [];

  for (const template of templateSpecs.templates) {
    try {
      await generateTemplate(template.id);
      results.push({ id: template.id, success: true });
    } catch (error: any) {
      results.push({ id: template.id, success: false, error: error.message });
    }

    // Rate limiting between templates
    await new Promise(r => setTimeout(r, 3000));
  }

  // Summary
  console.log('\n');
  console.log('═'.repeat(70));
  console.log('📊 GENERATION SUMMARY');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`   ✅ Successful: ${successful.length}`);
  console.log(`   ❌ Failed: ${failed.length}`);
  
  if (failed.length > 0) {
    console.log('\n   Failed templates:');
    failed.forEach(f => console.log(`      ${f.id}: ${f.error}`));
  }

  // Save batch report
  const report = {
    timestamp: new Date().toISOString(),
    total: results.length,
    successful: successful.length,
    failed: failed.length,
    results
  };

  fs.writeFileSync(
    path.join(OUTPUT_DIR, `batch_report_${Date.now()}.json`),
    JSON.stringify(report, null, 2)
  );

  console.log('═'.repeat(70));
}

// CLI handler
async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--all')) {
    await generateAllTemplates();
  } else if (args.includes('--template')) {
    const idx = args.indexOf('--template');
    const templateId = args[idx + 1];
    await generateTemplate(templateId);
  } else {
    console.log(`
Usage:
  npx tsx world-class-generator.ts --all              # Generate all 10 templates
  npx tsx world-class-generator.ts --template T05    # Generate specific template
    `);
  }
}

main().catch(console.error);

