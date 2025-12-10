import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });

const imageUrl = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png';
const audioUrl = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/audio_1765361262640.mp3';

console.log('🎬 Testing Replicate Wav2Lip with REAL Kelly photo...');

async function main() {
  try {
    console.log('⏳ Processing with Wav2Lip (high quality)...');
    
    const output = await replicate.run(
      "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
      {
        input: {
          face: imageUrl,
          audio: audioUrl,
          pads: "0 10 0 0",
          smooth: true,
          fps: 25,
          face_det_batch_size: 16,
          wav2lip_batch_size: 128,
        }
      }
    );
    
    console.log('✅ Wav2Lip result:', output);
    
    // Download if URL
    if (typeof output === 'string' && output.startsWith('http')) {
      const response = await fetch(output);
      const buffer = Buffer.from(await response.arrayBuffer());
      const outPath = 'generated-videos/photorealistic-test/kelly_wav2lip.mp4';
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, buffer);
      console.log('💾 Saved to:', outPath);
    }
  } catch (e: any) {
    console.error('❌ Wav2Lip error:', e.message);
    
    // Try alternative model
    console.log('\n⏳ Trying alternative: SadTalker on Replicate...');
    try {
      const output = await replicate.run(
        "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
        {
          input: {
            source_image: imageUrl,
            driven_audio: audioUrl,
            enhancer: "gfpgan",
            preprocess: "full",
          }
        }
      );
      
      console.log('✅ SadTalker result:', output);
      
      if (typeof output === 'string' && output.startsWith('http')) {
        const response = await fetch(output);
        const buffer = Buffer.from(await response.arrayBuffer());
        const outPath = 'generated-videos/photorealistic-test/kelly_sadtalker_replicate.mp4';
        fs.writeFileSync(outPath, buffer);
        console.log('💾 Saved to:', outPath);
      }
    } catch (e2: any) {
      console.error('❌ SadTalker error:', e2.message);
    }
  }
}

main();

