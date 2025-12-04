/**
 * Kelly Video Lip-Sync Production Pipeline
 * 
 * Complete pipeline for generating Kelly talking videos:
 * 1. Generate Kelly image with trained LoRA
 * 2. Generate audio with ElevenLabs (Kelly's voice)
 * 3. Create lip-synced video with SadTalker/Hedra/Omnihuman
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-lipsync-pipeline.ts
 *   npx tsx scripts/kelly-video-lipsync-pipeline.ts --text "Hello learners!"
 *   npx tsx scripts/kelly-video-lipsync-pipeline.ts --pose thinking --text "Let me think about that..."
 * 
 * Prerequisites:
 *   - REPLICATE_API_TOKEN in .env.local
 *   - ELEVENLABS_API_KEY in .env.local
 *   - FAL_KEY in .env.local (optional, for fal.ai models)
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // API Keys
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  FAL_KEY: process.env.FAL_KEY,
  
  // Kelly's Voice
  KELLY_VOICE_ID: 'wAdymQH5YucAkXwmrdL0',
  
  // Kelly's LoRA (Civitai)
  CIVITAI_LORA_VERSION: '2455956',
  get CIVITAI_LORA_URL() {
    return `https://civitai.com/api/download/models/${this.CIVITAI_LORA_VERSION}`;
  },
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos'),
  
  // Lip-sync model preference order
  LIPSYNC_MODELS: ['sadtalker', 'hedra', 'wav2lip', 'omnihuman'] as const,
};

// =============================================================================
// KELLY VISUAL IDENTITY (Locked)
// =============================================================================

const KELLY = {
  character: `kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, white leather sneakers`,
  
  scene: `pure white cyclorama photography studio, director's chair with black canvas fabric seat and natural warm wood frame with round finials, professional studio lighting with soft natural window light from upper right casting gentle diagonal shadows on light gray seamless floor, clean minimal background, shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, professional fashion photography, 8K UHD`,
  
  poses: {
    idle: `seated in director's chair, relaxed natural posture, genuine warm smile, looking directly at camera, hands resting on armrests`,
    thinking: `seated in director's chair, chin resting thoughtfully on right hand, elbow on armrest, looking upward and slightly to side, contemplative curious expression`,
    explaining: `seated in director's chair, leaning slightly forward, one hand gesturing outward, engaged teaching expression`,
    excited: `seated in director's chair, energetic forward-leaning posture, bright enthusiastic eyes, excited anticipating smile, hands clasped together eagerly`,
    celebrating: `seated in director's chair, both arms raised joyfully in celebration, big genuine smile, bright excited eyes, triumphant victorious pose`,
    supportive: `seated in director's chair, warm empathetic caring expression, gentle head tilt showing understanding, reassuring smile, one hand on heart`,
    curious: `seated in director's chair, head tilted slightly, eyebrows raised with interest, engaged curious expression`,
    welcome: `seated in director's chair, warm open welcoming expression, slight lean forward, friendly inviting gesture`,
  } as Record<string, string>,
};

// =============================================================================
// PIPELINE FUNCTIONS
// =============================================================================

interface PipelineResult {
  success: boolean;
  imagePath?: string;
  audioPath?: string;
  videoPath?: string;
  error?: string;
  model?: string;
}

/**
 * Step 1: Generate Kelly image with LoRA
 */
async function generateKellyImage(
  replicate: Replicate, 
  pose: string = 'idle'
): Promise<string | null> {
  console.log('\n🎨 Step 1: Generating Kelly image with LoRA...');
  console.log(`   Pose: ${pose}`);
  
  const posePrompt = KELLY.poses[pose] || KELLY.poses.idle;
  const fullPrompt = `${KELLY.character}, ${posePrompt}, ${KELLY.scene}`;
  
  try {
    // Try FLUX Dev with LoRA
    console.log('   Using FLUX Dev + Civitai LoRA...');
    
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: fullPrompt,
          hf_lora: CONFIG.CIVITAI_LORA_URL,
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          prompt_strength: 0.8,
          num_inference_steps: 28,
          disable_safety_checker: true,
        }
      }
    ) as string[];

    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    // Download image
    console.log('   📥 Downloading image...');
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    const imagePath = path.join(CONFIG.OUTPUT_DIR, `kelly_${pose}_${Date.now()}.png`);
    fs.writeFileSync(imagePath, buffer);
    
    console.log(`   ✅ Image saved: ${imagePath}`);
    return imagePath;
    
  } catch (error: any) {
    console.error(`   ❌ Image generation failed: ${error.message}`);
    
    // Try fallback with FLUX Pro (no LoRA)
    console.log('   🔄 Trying fallback with FLUX 1.1 Pro...');
    
    try {
      const output = await replicate.run(
        "black-forest-labs/flux-1.1-pro",
        {
          input: {
            prompt: fullPrompt,
            aspect_ratio: "16:9",
            output_format: "png",
            output_quality: 100,
            safety_tolerance: 2,
            prompt_upsampling: true,
          }
        }
      ) as string;
      
      const imageUrl = typeof output === 'string' ? output : (Array.isArray(output) ? output[0] : String(output));
      
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error(`Download failed: ${response.status}`);
      
      const buffer = Buffer.from(await response.arrayBuffer());
      const imagePath = path.join(CONFIG.OUTPUT_DIR, `kelly_${pose}_fallback_${Date.now()}.png`);
      fs.writeFileSync(imagePath, buffer);
      
      console.log(`   ✅ Fallback image saved: ${imagePath}`);
      return imagePath;
      
    } catch (fallbackError: any) {
      console.error(`   ❌ Fallback also failed: ${fallbackError.message}`);
      return null;
    }
  }
}

/**
 * Step 2: Generate audio with ElevenLabs
 */
async function generateAudio(text: string): Promise<string | null> {
  console.log('\n🔊 Step 2: Generating audio with ElevenLabs...');
  console.log(`   Text: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
  
  try {
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
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.4,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`TTS API error: ${response.status} - ${errorText}`);
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    const audioPath = path.join(CONFIG.OUTPUT_DIR, `kelly_audio_${Date.now()}.mp3`);
    fs.writeFileSync(audioPath, audioBuffer);
    
    console.log(`   ✅ Audio saved: ${audioPath} (${audioBuffer.byteLength} bytes)`);
    return audioPath;
    
  } catch (error: any) {
    console.error(`   ❌ Audio generation failed: ${error.message}`);
    return null;
  }
}

/**
 * Step 3: Generate lip-synced video
 */
async function generateLipSyncVideo(
  replicate: Replicate,
  imagePath: string,
  audioPath: string
): Promise<{ videoPath: string | null; model: string }> {
  console.log('\n🎬 Step 3: Generating lip-sync video...');
  
  // Convert to data URLs for Replicate
  const imageBuffer = fs.readFileSync(imagePath);
  const audioBuffer = fs.readFileSync(audioPath);
  const imageDataUrl = `data:image/png;base64,${imageBuffer.toString('base64')}`;
  const audioDataUrl = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  // Try each model in preference order
  for (const modelName of CONFIG.LIPSYNC_MODELS) {
    console.log(`   Trying: ${modelName}...`);
    
    try {
      let output: any;
      
      switch (modelName) {
        case 'sadtalker':
          output = await replicate.run(
            "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
            {
              input: {
                source_image: imageDataUrl,
                driven_audio: audioDataUrl,
                enhancer: "gfpgan",
                preprocess: "crop",
                still_mode: false,
                use_ref_video: false,
              }
            }
          );
          break;
          
        case 'wav2lip':
          output = await replicate.run(
            "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
            {
              input: {
                face: imageDataUrl,
                audio: audioDataUrl,
                pads: "0 10 0 0",
                smooth: true,
                fps: 25,
                resize_factor: 1,
              }
            }
          );
          break;
          
        case 'hedra':
          // Hedra requires public URLs, so we need to upload first
          // For now, skip if no public URL available
          console.log(`   ⏭️ Skipping Hedra (requires public URLs)`);
          continue;
          
        case 'omnihuman':
          // Omnihuman is on fal.ai, skip if no FAL_KEY
          if (!CONFIG.FAL_KEY) {
            console.log(`   ⏭️ Skipping Omnihuman (no FAL_KEY)`);
            continue;
          }
          // Would need to use fal.ai client here
          continue;
          
        default:
          continue;
      }
      
      // Extract video URL from output
      const videoUrl = typeof output === 'string' 
        ? output 
        : (Array.isArray(output) ? output[0] : output?.url || output?.video);
      
      if (!videoUrl) {
        console.log(`   ⚠️ ${modelName} returned no video URL`);
        continue;
      }
      
      // Download video
      console.log(`   📥 Downloading video from ${modelName}...`);
      const response = await fetch(videoUrl);
      if (!response.ok) {
        console.log(`   ⚠️ Download failed: ${response.status}`);
        continue;
      }
      
      const videoBuffer = Buffer.from(await response.arrayBuffer());
      const videoPath = path.join(CONFIG.OUTPUT_DIR, `kelly_lipsync_${modelName}_${Date.now()}.mp4`);
      fs.writeFileSync(videoPath, videoBuffer);
      
      console.log(`   ✅ Video saved: ${videoPath} (${videoBuffer.byteLength} bytes)`);
      return { videoPath, model: modelName };
      
    } catch (error: any) {
      console.log(`   ❌ ${modelName} failed: ${error.message?.substring(0, 80)}`);
    }
  }
  
  return { videoPath: null, model: 'none' };
}

/**
 * Main pipeline function
 */
async function runPipeline(options: {
  text: string;
  pose?: string;
  useExistingImage?: string;
}): Promise<PipelineResult> {
  const { text, pose = 'idle', useExistingImage } = options;
  
  // Validate API keys
  if (!CONFIG.REPLICATE_API_TOKEN) {
    return { success: false, error: 'REPLICATE_API_TOKEN not set' };
  }
  if (!CONFIG.ELEVENLABS_API_KEY) {
    return { success: false, error: 'ELEVENLABS_API_KEY not set' };
  }
  
  // Create output directory
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  // Step 1: Get or generate Kelly image
  let imagePath: string | null = null;
  
  if (useExistingImage && fs.existsSync(useExistingImage)) {
    console.log(`\n🖼️ Using existing image: ${useExistingImage}`);
    imagePath = useExistingImage;
  } else {
    imagePath = await generateKellyImage(replicate, pose);
    if (!imagePath) {
      return { success: false, error: 'Image generation failed' };
    }
  }
  
  // Step 2: Generate audio
  const audioPath = await generateAudio(text);
  if (!audioPath) {
    return { success: false, imagePath, error: 'Audio generation failed' };
  }
  
  // Step 3: Generate lip-sync video
  const { videoPath, model } = await generateLipSyncVideo(replicate, imagePath, audioPath);
  if (!videoPath) {
    return { success: false, imagePath, audioPath, error: 'Video generation failed' };
  }
  
  return {
    success: true,
    imagePath,
    audioPath,
    videoPath,
    model,
  };
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 KELLY VIDEO LIP-SYNC PRODUCTION PIPELINE');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
  
  // Parse CLI arguments
  const args = process.argv.slice(2);
  let text = "Hello! I'm Kelly, and I'm so excited to learn with you today. Let's explore something amazing together!";
  let pose = 'idle';
  let useExistingImage: string | undefined;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--text' && args[i + 1]) {
      text = args[i + 1];
      i++;
    } else if (args[i] === '--pose' && args[i + 1]) {
      pose = args[i + 1];
      i++;
    } else if (args[i] === '--image' && args[i + 1]) {
      useExistingImage = args[i + 1];
      i++;
    }
  }
  
  console.log('📋 Configuration:');
  console.log(`   Text: "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"`);
  console.log(`   Pose: ${pose}`);
  console.log(`   Image: ${useExistingImage || 'Will generate new'}`);
  console.log('');
  
  // Check API keys
  console.log('🔑 API Keys:');
  console.log(`   REPLICATE: ${CONFIG.REPLICATE_API_TOKEN ? '✅' : '❌'}`);
  console.log(`   ELEVENLABS: ${CONFIG.ELEVENLABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   FAL: ${CONFIG.FAL_KEY ? '✅' : '⏭️ (optional)'}`);
  
  if (!CONFIG.REPLICATE_API_TOKEN || !CONFIG.ELEVENLABS_API_KEY) {
    console.error('\n❌ Missing required API keys. Add them to .env.local');
    process.exit(1);
  }
  
  // Run pipeline
  const startTime = Date.now();
  const result = await runPipeline({ text, pose, useExistingImage });
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  
  if (result.success) {
    console.log('   ✅ PIPELINE COMPLETE!');
    console.log(`   ⏱️ Total time: ${duration}s`);
    console.log('');
    console.log('   📁 Output files:');
    console.log(`      Image: ${result.imagePath}`);
    console.log(`      Audio: ${result.audioPath}`);
    console.log(`      Video: ${result.videoPath}`);
    console.log(`      Model: ${result.model}`);
  } else {
    console.log('   ❌ PIPELINE FAILED');
    console.log(`   Error: ${result.error}`);
    if (result.imagePath) console.log(`   Image was generated: ${result.imagePath}`);
    if (result.audioPath) console.log(`   Audio was generated: ${result.audioPath}`);
  }
  
  console.log('═══════════════════════════════════════════════════════════════');
}

// Export for use as module
export { runPipeline, generateKellyImage, generateAudio, generateLipSyncVideo, CONFIG, KELLY };

// Run if called directly
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

