/**
 * Generate Lip-Sync Videos for a Lesson
 * 
 * Generates all phase videos for a given lesson day using the Kelly quick lip-sync pipeline.
 * 
 * Usage:
 *   npx tsx scripts/generate-lesson-videos.ts --day 1
 *   npx tsx scripts/generate-lesson-videos.ts --day 1 --phases "welcome,q1,q2,q3,wisdom"
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIG
// =============================================================================

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN!;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

// =============================================================================
// LESSON CONTENT (Example - in production, load from lesson DNA files)
// =============================================================================

function getPhaseContent(dayNumber: number, phase: string): { text: string; image: string } {
  const dayStr = dayNumber.toString().padStart(3, '0');
  
  // Phase-specific content templates
  const templates: Record<string, { text: string; imageType: string }> = {
    welcome: {
      text: `Welcome to day ${dayNumber}! I'm so excited to explore today's lesson with you. Let's discover something amazing together!`,
      imageType: 'hero',
    },
    q1: {
      text: `Here's our first question to get us thinking. Take your time and consider what you already know about this topic.`,
      imageType: 'guide-point',
    },
    q2: {
      text: `Great thinking! Now let's go a bit deeper with our second question. What connections can you make?`,
      imageType: 'guide-point',
    },
    q3: {
      text: `You're doing wonderfully! This final question will really make us think. Trust your instincts!`,
      imageType: 'guide-point',
    },
    wisdom: {
      text: `What a journey we've had today! Remember, every day we learn something new is a day well spent. See you tomorrow!`,
      imageType: 'reaction',
    },
  };
  
  const template = templates[phase] || templates.welcome;
  
  return {
    text: template.text,
    image: `public/kelly/lessons/${dayStr}/lesson-${dayNumber}-${template.imageType}.png`,
  };
}

// =============================================================================
// VIDEO GENERATION
// =============================================================================

async function generateAudio(text: string): Promise<Buffer> {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
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
    throw new Error(`TTS error: ${response.status}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

async function generateVideo(
  replicate: Replicate,
  imageBuffer: Buffer,
  audioBuffer: Buffer
): Promise<Buffer> {
  // Upload files
  const imageFile = await replicate.files.create(imageBuffer, {
    filename: 'kelly.png',
    content_type: 'image/png',
  });
  
  const audioFile = await replicate.files.create(audioBuffer, {
    filename: 'audio.mp3',
    content_type: 'audio/mpeg',
  });
  
  // Run SadTalker
  const prediction = await replicate.predictions.create({
    version: "3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    input: {
      source_image: imageFile.urls.get,
      driven_audio: audioFile.urls.get,
      enhancer: "gfpgan",
      preprocess: "crop",
      still_mode: false,
    },
  });
  
  // Poll for completion
  let result = prediction;
  while (result.status !== 'succeeded' && result.status !== 'failed') {
    await new Promise(r => setTimeout(r, 2000));
    result = await replicate.predictions.get(prediction.id);
  }
  
  if (result.status === 'failed') {
    throw new Error(`SadTalker failed: ${result.error}`);
  }
  
  // Download video
  const videoResponse = await fetch(result.output as string);
  return Buffer.from(await videoResponse.arrayBuffer());
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 KELLY LESSON VIDEO GENERATOR');
  console.log('═══════════════════════════════════════════════════════════════');
  
  // Parse args
  const args = process.argv.slice(2);
  let dayNumber = 1;
  let phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      dayNumber = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--phases' && args[i + 1]) {
      phases = args[i + 1].split(',').map(p => p.trim());
      i++;
    }
  }
  
  const dayStr = dayNumber.toString().padStart(3, '0');
  const outputDir = path.join(process.cwd(), 'public', 'kelly', 'videos', dayStr);
  
  console.log(`\n📅 Day: ${dayNumber}`);
  console.log(`📁 Output: ${outputDir}`);
  console.log(`🎬 Phases: ${phases.join(', ')}`);
  
  // Check prerequisites
  if (!REPLICATE_API_TOKEN || !ELEVENLABS_API_KEY) {
    console.error('\n❌ Missing API keys!');
    process.exit(1);
  }
  
  // Create output directory
  fs.mkdirSync(outputDir, { recursive: true });
  
  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });
  const results: Array<{ phase: string; success: boolean; path?: string; error?: string }> = [];
  
  // Generate each phase
  for (const phase of phases) {
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`📹 Generating: ${phase}`);
    
    const content = getPhaseContent(dayNumber, phase);
    const imagePath = path.resolve(process.cwd(), content.image);
    
    // Check if image exists
    if (!fs.existsSync(imagePath)) {
      console.log(`   ⚠️ Image not found: ${content.image}`);
      results.push({ phase, success: false, error: 'Image not found' });
      continue;
    }
    
    try {
      const startTime = Date.now();
      
      // Load image
      const imageBuffer = fs.readFileSync(imagePath);
      console.log(`   ✅ Image: ${(imageBuffer.byteLength / 1024).toFixed(1)} KB`);
      
      // Generate audio
      console.log(`   🔊 Generating audio...`);
      const audioBuffer = await generateAudio(content.text);
      console.log(`   ✅ Audio: ${(audioBuffer.byteLength / 1024).toFixed(1)} KB`);
      
      // Generate video
      console.log(`   🎬 Generating video...`);
      const videoBuffer = await generateVideo(replicate, imageBuffer, audioBuffer);
      
      // Save video
      const videoPath = path.join(outputDir, `${phase}.mp4`);
      fs.writeFileSync(videoPath, videoBuffer);
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`   ✅ Saved: ${videoPath}`);
      console.log(`   ⏱️ Time: ${duration}s`);
      
      results.push({ phase, success: true, path: videoPath });
      
    } catch (error: any) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.push({ phase, success: false, error: error.message });
    }
    
    // Rate limiting
    console.log(`   ⏳ Waiting 5s before next...`);
    await new Promise(r => setTimeout(r, 5000));
  }
  
  // Summary
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('   📊 GENERATION SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n✅ Successful: ${successful.length}/${results.length}`);
  successful.forEach(r => console.log(`   • ${r.phase}: ${r.path}`));
  
  if (failed.length > 0) {
    console.log(`\n❌ Failed: ${failed.length}/${results.length}`);
    failed.forEach(r => console.log(`   • ${r.phase}: ${r.error}`));
  }
  
  console.log(`\n📁 Output directory: ${outputDir}`);
}

main().catch(err => {
  console.error('\n❌ Fatal error:', err);
  process.exit(1);
});


