#!/usr/bin/env npx tsx
/**
 * 🔧 KELLY LOGO POLISH - Generate multiple variations to find flawless version
 * 
 * Generates the curious Kelly pose with multiple seeds to find one with
 * perfectly clean eyes and no artifacts.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import https from 'https';
import http from 'http';

const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.92,
  FLUX_LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-logo-polish'),
};

// The winning prompt - curious Kelly looking up and right
const CURIOUS_PROMPT = `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, beautiful clear brown eyes with perfect reflections, wearing soft light blue ribbed sweater with crew neck, 

looking UP and to the RIGHT with genuine curiosity, slight soft smile with lips gently closed, eyebrows slightly raised in wonder, eyes sparkling with interest and perfectly clear, head tilted slightly, hand gently resting under chin in classic thinking pose,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting with soft catchlights in eyes, pure white seamless background, photorealistic quality, sharp focus on eyes, flawless skin, corporate headshot style, 8K detail`;

async function downloadImage(url: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outputPath);
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadImage(redirectUrl, outputPath).then(resolve).catch(reject);
          return;
        }
      }
      
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(outputPath, () => {});
      reject(err);
    });
  });
}

async function generateVariation(replicate: Replicate, seed: number, index: number): Promise<string> {
  console.log(`\n🎬 Generating variation ${index} with seed ${seed}...`);
  
  const input = {
    prompt: CURIOUS_PROMPT,
    hf_lora: CONFIG.KELLY_LORA_URL,
    lora_scale: CONFIG.LORA_SCALE,
    num_outputs: 1,
    aspect_ratio: '1:1',
    output_format: 'png',
    guidance_scale: 4.0, // Slightly higher for more precision
    output_quality: 100,
    prompt_strength: 0.8,
    num_inference_steps: 50,
    seed: seed,
    disable_safety_checker: true,
  };
  
  const output = await replicate.run(CONFIG.FLUX_LORA_MODEL as `${string}/${string}:${string}`, { input });
  
  let imageUrl: string;
  if (Array.isArray(output)) {
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  const outputPath = path.join(CONFIG.OUTPUT_DIR, `kelly-curious-v${index}-seed${seed}.png`);
  console.log(`   📥 Downloading...`);
  await downloadImage(imageUrl, outputPath);
  console.log(`   ✅ Saved: ${path.basename(outputPath)}`);
  
  return outputPath;
}

async function main() {
  console.log('═'.repeat(60));
  console.log('🔧 KELLY LOGO POLISH - Finding Flawless Version');
  console.log('═'.repeat(60));
  
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }
  
  if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  }
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  // Generate 4 variations with different seeds
  const seeds = [
    11111111,  // Fresh seed 1
    22222222,  // Fresh seed 2
    33333333,  // Fresh seed 3
    55555555,  // Fresh seed 4
  ];
  
  const generatedPaths: string[] = [];
  
  for (let i = 0; i < seeds.length; i++) {
    try {
      const outputPath = await generateVariation(replicate, seeds[i], i + 1);
      generatedPaths.push(outputPath);
    } catch (error) {
      console.error(`❌ Error with seed ${seeds[i]}:`, error);
    }
  }
  
  // Generate press kit HTML for comparison
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>✨ Kelly Logo Polish - Pick the Best</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f0f11;
      color: white;
      padding: 2rem;
    }
    h1 { text-align: center; margin-bottom: 2rem; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 2rem;
      max-width: 1400px;
      margin: 0 auto;
    }
    .card {
      background: #1a1a1f;
      border-radius: 16px;
      overflow: hidden;
    }
    .card img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      cursor: zoom-in;
    }
    .card img:hover {
      transform: scale(1.02);
    }
    .card-info {
      padding: 1rem;
      text-align: center;
    }
    .card-info h3 { margin-bottom: 0.5rem; }
    .card-info a {
      display: inline-block;
      margin-top: 0.5rem;
      padding: 0.5rem 1rem;
      background: #d97757;
      color: white;
      text-decoration: none;
      border-radius: 8px;
    }
    .instructions {
      text-align: center;
      margin: 2rem 0;
      color: #888;
    }
    .zoom-overlay {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.95);
      z-index: 1000;
      cursor: zoom-out;
    }
    .zoom-overlay img {
      max-width: 95vw;
      max-height: 95vh;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
    .zoom-overlay.active { display: block; }
  </style>
</head>
<body>
  <h1>🔍 Kelly Logo Polish - Pick the Flawless One</h1>
  <p class="instructions">Click any image to zoom and inspect eyes for artifacts. Download the best one!</p>
  
  <div class="grid">
    ${generatedPaths.map((p, i) => `
    <div class="card">
      <img src="${path.basename(p)}" alt="Variation ${i + 1}" onclick="zoom(this.src)">
      <div class="card-info">
        <h3>Variation ${i + 1}</h3>
        <p>Seed: ${seeds[i]}</p>
        <a href="${path.basename(p)}" download>Download PNG</a>
      </div>
    </div>`).join('')}
  </div>
  
  <div class="zoom-overlay" id="zoom" onclick="this.classList.remove('active')">
    <img id="zoom-img" src="">
  </div>
  
  <script>
    function zoom(src) {
      document.getElementById('zoom-img').src = src;
      document.getElementById('zoom').classList.add('active');
    }
  </script>
</body>
</html>`;

  const htmlPath = path.join(CONFIG.OUTPUT_DIR, 'pick-best.html');
  fs.writeFileSync(htmlPath, html);
  
  console.log('\n' + '═'.repeat(60));
  console.log('✅ Generated', generatedPaths.length, 'variations');
  console.log('═'.repeat(60));
  console.log('\n📁 Location:', CONFIG.OUTPUT_DIR);
  console.log('🔍 Open pick-best.html to compare and choose the flawless one!');
}

main().catch(console.error);


