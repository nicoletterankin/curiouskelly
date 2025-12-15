#!/usr/bin/env npx tsx
/**
 * Generate Day 17 Thumbnail
 * Topic: Why Bodies Need to Move
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const LORA_URL = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';

// Style constants from the thumbnail system
const KELLY_ANCHOR = `kelly, young woman walking in profile view, long brown wavy hair, light blue crewneck sweater, blue jeans cuffed at ankles, white sneakers, mid-stride walking pose, natural arm movement, looking ahead, soft natural lighting on figure, subtle ground shadow`;

const STYLE_LOCKS = `full body shot including feet, wide shot, photorealistic editorial photography, clean composition, cinematic color grading, soft shadows, 8k, professional photography, 16:9 aspect ratio`;

const NEGATIVE_PROMPT = `cropped feet, cut off feet, close up, cartoon, illustration, anime, painting, drawing, sketch, blurry, low quality, watermark, text, logo, extra limbs, extra fingers, deformed, distorted face, wrong outfit, different clothes, holding objects, sitting, standing still, bad anatomy, distorted body`;

// Day 17: Why Bodies Need to Move
const DAY17_PROMPT = `walking with energetic stride through a vibrant outdoor park, warm morning sunlight, sense of movement and vitality, athletic dynamic environment, soft green grass and blue sky, healthy and active atmosphere, fresh air and freedom of motion`;

async function generateThumbnail(): Promise<void> {
  console.log('═'.repeat(60));
  console.log('🎨 GENERATING DAY 17 THUMBNAIL');
  console.log('   Topic: Why Bodies Need to Move');
  console.log('═'.repeat(60));
  
  const fullPrompt = `${KELLY_ANCHOR}, ${DAY17_PROMPT}, ${STYLE_LOCKS}`;
  
  console.log('\n📝 Prompt:');
  console.log(fullPrompt.substring(0, 200) + '...');
  console.log('');
  
  try {
    console.log('⏳ Generating with Flux LoRA...');
    
    const output = await replicate.run(
      'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d' as `${string}/${string}:${string}`,
      {
        input: {
          prompt: fullPrompt,
          hf_lora: LORA_URL,
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: '16:9',
          output_format: 'png',
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28,
          negative_prompt: NEGATIVE_PROMPT,
        },
      }
    );
    
    let imageUrl: string;
    if (Array.isArray(output)) {
      imageUrl = String(output[0]);
    } else {
      imageUrl = String(output);
    }
    
    console.log('✅ Image generated:', imageUrl.substring(0, 80) + '...');
    
    // Download image
    const response = await fetch(imageUrl);
    const buffer = Buffer.from(await response.arrayBuffer());
    
    // Save locally
    const outputDir = path.join(process.cwd(), 'generated-images', 'thumbnails');
    fs.mkdirSync(outputDir, { recursive: true });
    const localPath = path.join(outputDir, '017-why-bodies-need-to-move.png');
    fs.writeFileSync(localPath, buffer);
    console.log('💾 Saved locally:', localPath);
    
    // Upload to Supabase
    const storagePath = 'thumbnails/017-why-bodies-need-to-move.png';
    const { error: uploadError } = await supabase.storage
      .from('lesson-visuals')
      .upload(storagePath, buffer, {
        contentType: 'image/png',
        upsert: true,
      });
    
    if (uploadError) {
      console.error('❌ Upload error:', uploadError.message);
    } else {
      const { data: urlData } = supabase.storage
        .from('lesson-visuals')
        .getPublicUrl(storagePath);
      
      const thumbnailUrl = urlData.publicUrl;
      console.log('☁️ Uploaded to Supabase:', thumbnailUrl);
      
      // Update core_lessons
      const { error: updateError } = await supabase
        .from('core_lessons')
        .update({ 
          thumbnail_url: thumbnailUrl,
          hero_image_url: thumbnailUrl, // Use same for hero
        })
        .eq('day_number', 17);
      
      if (updateError) {
        console.error('❌ DB update error:', updateError.message);
      } else {
        console.log('📝 Database updated!');
      }
    }
    
    console.log('');
    console.log('═'.repeat(60));
    console.log('✅ DAY 17 THUMBNAIL COMPLETE');
    console.log('═'.repeat(60));
    
  } catch (error: any) {
    console.error('❌ Generation failed:', error.message);
    process.exit(1);
  }
}

generateThumbnail();
