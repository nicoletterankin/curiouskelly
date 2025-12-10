import 'dotenv/config';
import { fal } from '@fal-ai/client';
import * as fs from 'fs';
import * as path from 'path';

fal.config({ credentials: process.env.FAL_KEY! });

const imageUrl = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png';
const audioUrl = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/audio_1765361262640.mp3';

console.log('🎬 Testing SadTalker with REAL Kelly photo...');
console.log('   Image:', imageUrl);
console.log('   Audio:', audioUrl);

async function main() {
  try {
    console.log('\n⏳ Processing...');
    const result = await fal.subscribe('fal-ai/sadtalker', {
      input: {
        source_image_url: imageUrl,
        driven_audio_url: audioUrl,
        still: true,
        enhancer: 'gfpgan',
        preprocess: 'full',
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') process.stdout.write('.');
      }
    });
    
    console.log('\n\n✅ SUCCESS!');
    console.log('Full result:', JSON.stringify(result, null, 2));
    const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
    console.log('🎬 Video URL:', videoUrl);
    
    // Download and save
    if (videoUrl) {
      const response = await fetch(videoUrl);
      const buffer = Buffer.from(await response.arrayBuffer());
      const outPath = 'generated-videos/photorealistic-test/kelly_sadtalker.mp4';
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, buffer);
      console.log('💾 Saved to:', outPath);
    }
  } catch (e: any) {
    console.error('\n❌ Error:', e.message);
  }
}

main();

