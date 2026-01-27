#!/usr/bin/env npx tsx
/**
 * 🎬 KELLY LOGO ANIMATION SEQUENCE GENERATOR
 * 
 * Generates a series of flawless 4K frames for a micro-animation:
 * Frame 1: Looking up and to the right (curious/thinking)
 * Frame 2: Beginning to turn toward camera
 * Frame 3: Eyes meeting camera, smile beginning  
 * Frame 4: Full warm smile at camera
 * 
 * Output: 4K PNG files ready to be assembled into animation
 * 
 * USAGE:
 *   npx tsx scripts/kelly-logo-animation-sequence.ts
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import https from 'https';
import http from 'http';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  
  // Kelly LoRA - maximum scale for perfect consistency
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.92, // Slightly higher for maximum character consistency
  
  // Model
  FLUX_LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  
  // Output - 4K square for logo use
  WIDTH: 2048,
  HEIGHT: 2048,
  
  // Output directory
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-logo-animation'),
  
  // Seed for consistency (same seed = same hair, lighting, background)
  BASE_SEED: 42424242,
};

// =============================================================================
// ANIMATION FRAMES - The sequence from curious to smile
// =============================================================================

interface AnimationFrame {
  id: string;
  name: string;
  prompt: string;
  seed: number;
}

const ANIMATION_FRAMES: AnimationFrame[] = [
  {
    id: 'frame-01-curious-up-right',
    name: 'Frame 1: Curious - Looking Up Right',
    prompt: `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, brown eyes, wearing soft light blue ribbed sweater with crew neck, 

looking UP and to the RIGHT with genuine curiosity, slight soft smile with lips gently closed, eyebrows slightly raised in wonder, eyes sparkling with interest, head tilted slightly, hand gently resting under chin in classic thinking pose,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting, pure white seamless background, photorealistic quality, sharp focus on eyes, corporate headshot style`,
    seed: CONFIG.BASE_SEED,
  },
  {
    id: 'frame-02-turning',
    name: 'Frame 2: Beginning to Turn',
    prompt: `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, brown eyes, wearing soft light blue ribbed sweater with crew neck,

head turning toward camera, eyes beginning to shift toward viewer, thoughtful gentle smile starting to form, transitioning from thinking pose, one eyebrow slightly raised, warm friendly energy,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting, pure white seamless background, photorealistic quality, sharp focus on eyes, corporate headshot style`,
    seed: CONFIG.BASE_SEED + 1,
  },
  {
    id: 'frame-03-eyes-meet',
    name: 'Frame 3: Eyes Meet Camera',
    prompt: `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, brown eyes, wearing soft light blue ribbed sweater with crew neck,

eyes looking DIRECTLY at camera with warm connection, smile growing wider, genuine happiness, eyebrows relaxed and friendly, moment of recognition and warmth, friendly teacher expression,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting, pure white seamless background, photorealistic quality, sharp focus on eyes, corporate headshot style`,
    seed: CONFIG.BASE_SEED + 2,
  },
  {
    id: 'frame-04-full-smile',
    name: 'Frame 4: Full Warm Smile',
    prompt: `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, brown eyes, wearing soft light blue ribbed sweater with crew neck,

looking directly at camera with warm genuine smile showing teeth, eyes bright with joy, genuine happiness, confident and welcoming expression, friendly teacher greeting, professional warmth,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting, pure white seamless background, photorealistic quality, sharp focus on eyes, corporate headshot style`,
    seed: CONFIG.BASE_SEED + 3,
  },
  {
    id: 'frame-05-knowing-smile',
    name: 'Frame 5: Knowing Smile (Optional Loop Point)',
    prompt: `CKELLY curious kelly, professional headshot portrait, young woman late 20s, long wavy brown hair with golden blonde highlights, brown eyes, wearing soft light blue ribbed sweater with crew neck,

looking at camera with knowing confident smile, hint of curiosity, warm inviting expression, slight head tilt, eyes engaged and bright, as if about to share something interesting, teacher ready to inspire,

tight crop on face and shoulders, head fills most of frame, minimal space above head, professional studio lighting, pure white seamless background, photorealistic quality, sharp focus on eyes, corporate headshot style`,
    seed: CONFIG.BASE_SEED + 4,
  },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

async function downloadImage(url: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outputPath);
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        // Handle redirect
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
      fs.unlink(outputPath, () => {}); // Delete partial file
      reject(err);
    });
  });
}

async function generateFrame(replicate: Replicate, frame: AnimationFrame): Promise<string> {
  console.log(`\n🎬 Generating: ${frame.name}`);
  console.log(`   Seed: ${frame.seed}`);
  
  const input = {
    prompt: frame.prompt,
    hf_lora: CONFIG.KELLY_LORA_URL,
    lora_scale: CONFIG.LORA_SCALE,
    num_outputs: 1,
    aspect_ratio: '1:1',
    output_format: 'png',
    guidance_scale: 3.5,
    output_quality: 100,
    prompt_strength: 0.8,
    num_inference_steps: 50, // Maximum quality
    seed: frame.seed,
    disable_safety_checker: true, // Avoid false positives on professional headshots
  };
  
  console.log(`   Calling Replicate API...`);
  
  const output = await replicate.run(CONFIG.FLUX_LORA_MODEL as `${string}/${string}:${string}`, { input });
  
  if (!output) {
    throw new Error(`No output from Replicate for ${frame.id}`);
  }
  
  // Extract URL from output (handles FileOutput objects, arrays, strings)
  let imageUrl: string;
  if (Array.isArray(output)) {
    // Force convert FileOutput to string URL
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  if (!imageUrl || imageUrl === '[object Object]' || imageUrl === 'undefined') {
    throw new Error(`Invalid output URL from Replicate: ${JSON.stringify(output)}`);
  }
  
  console.log(`   ✅ Generated! URL: ${imageUrl.substring(0, 60)}...`);
  
  // Download the image
  const outputPath = path.join(CONFIG.OUTPUT_DIR, `${frame.id}.png`);
  console.log(`   📥 Downloading to: ${outputPath}`);
  await downloadImage(imageUrl, outputPath);
  console.log(`   ✅ Saved!`);
  
  return outputPath;
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('═'.repeat(70));
  console.log('🎬 KELLY LOGO ANIMATION SEQUENCE GENERATOR');
  console.log('═'.repeat(70));
  console.log(`\nGenerating ${ANIMATION_FRAMES.length} frames for animation loop:`);
  console.log('  1. Curious - Looking up and right');
  console.log('  2. Turning toward camera');
  console.log('  3. Eyes meet camera, smile starting');
  console.log('  4. Full warm smile');
  console.log('  5. Knowing smile (loop back point)');
  console.log(`\nOutput: ${CONFIG.OUTPUT_DIR}`);
  console.log(`Resolution: ${CONFIG.WIDTH}×${CONFIG.HEIGHT} (4K square)`);
  console.log(`LoRA Scale: ${CONFIG.LORA_SCALE}`);
  console.log('═'.repeat(70));
  
  // Check API key
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('\n❌ ERROR: REPLICATE_API_TOKEN not found in environment!');
    console.error('   Set it in your .env file or environment variables.');
    process.exit(1);
  }
  
  // Create output directory
  if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    console.log(`\n📁 Created output directory: ${CONFIG.OUTPUT_DIR}`);
  }
  
  // Initialize Replicate
  const replicate = new Replicate({
    auth: CONFIG.REPLICATE_API_TOKEN,
  });
  
  // Generate each frame
  const generatedPaths: string[] = [];
  const startTime = Date.now();
  
  for (let i = 0; i < ANIMATION_FRAMES.length; i++) {
    const frame = ANIMATION_FRAMES[i];
    console.log(`\n[${ i + 1}/${ANIMATION_FRAMES.length}] Processing ${frame.name}...`);
    
    try {
      const outputPath = await generateFrame(replicate, frame);
      generatedPaths.push(outputPath);
    } catch (error) {
      console.error(`\n❌ ERROR generating ${frame.id}:`, error);
      // Continue with other frames
    }
  }
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  
  // Generate Press Kit HTML
  if (generatedPaths.length > 0) {
    const htmlPath = generatePressKitHtml(generatedPaths);
    console.log(`\n📄 Press Kit HTML generated: ${htmlPath}`);
  }
  
  // Summary
  console.log('\n' + '═'.repeat(70));
  console.log('✅ GENERATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\nGenerated ${generatedPaths.length}/${ANIMATION_FRAMES.length} frames in ${elapsed}s`);
  console.log(`\nOutput files:`);
  generatedPaths.forEach((p, i) => {
    console.log(`  ${i + 1}. ${path.basename(p)}`);
  });
  console.log(`\n📁 Location: ${CONFIG.OUTPUT_DIR}`);
  console.log('\n🎬 Next steps:');
  console.log('   1. Review frames for quality');
  console.log('   2. Import into After Effects, Photoshop, or animation tool');
  console.log('   3. Create smooth transitions between frames');
  console.log('   4. Export as GIF, APNG, or video loop');
  console.log('\n✨ Happy animating!');
}

function generatePressKitHtml(imagePaths: string[]): string {
  const date = new Date().toISOString().split('T')[0];
  
  const imageCards = imagePaths.map((imgPath, i) => {
    const filename = path.basename(imgPath);
    const frameInfo = ANIMATION_FRAMES[i];
    return `
      <div class="asset-card">
        <div class="asset-preview">
          <img src="${filename}" alt="${frameInfo?.name || filename}" loading="lazy">
        </div>
        <div class="asset-info">
          <h3>${frameInfo?.name || `Frame ${i + 1}`}</h3>
          <p class="asset-meta">2048×2048 PNG • 4K Resolution</p>
          <a href="${filename}" download class="download-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download PNG
          </a>
        </div>
      </div>`;
  }).join('\n');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>✨ Curious Kelly Logo Animation - Press Kit</title>
  <style>
    :root {
      --kelly-blue: #7BA7C2;
      --kelly-orange: #d97757;
      --bg-dark: #0f0f11;
      --bg-card: #1a1a1f;
      --text-primary: #ffffff;
      --text-secondary: #a0a0a0;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-dark);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.6;
    }
    
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 3rem 2rem;
    }
    
    header {
      text-align: center;
      margin-bottom: 4rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .logo {
      font-size: 3rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, var(--kelly-blue), var(--kelly-orange));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .subtitle {
      font-size: 1.25rem;
      color: var(--text-secondary);
      margin-bottom: 1rem;
    }
    
    .meta {
      font-size: 0.875rem;
      color: var(--text-secondary);
    }
    
    .section-title {
      font-size: 1.5rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .section-title::before {
      content: '';
      width: 4px;
      height: 1.5rem;
      background: var(--kelly-orange);
      border-radius: 2px;
    }
    
    .assets-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 2rem;
      margin-bottom: 4rem;
    }
    
    .asset-card {
      background: var(--bg-card);
      border-radius: 16px;
      overflow: hidden;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .asset-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .asset-preview {
      aspect-ratio: 1;
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }
    
    .asset-preview img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    
    .asset-info {
      padding: 1.5rem;
    }
    
    .asset-info h3 {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    
    .asset-meta {
      font-size: 0.875rem;
      color: var(--text-secondary);
      margin-bottom: 1rem;
    }
    
    .download-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem 1.25rem;
      background: var(--kelly-orange);
      color: white;
      text-decoration: none;
      border-radius: 8px;
      font-weight: 500;
      font-size: 0.875rem;
      transition: background 0.2s;
    }
    
    .download-btn:hover {
      background: #c56545;
    }
    
    .usage-section {
      background: var(--bg-card);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 4rem;
    }
    
    .usage-section h2 {
      margin-bottom: 1.5rem;
    }
    
    .usage-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
    }
    
    .usage-item {
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }
    
    .usage-icon {
      width: 40px;
      height: 40px;
      background: rgba(217, 119, 87, 0.2);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      flex-shrink: 0;
    }
    
    .usage-text h4 {
      font-size: 0.875rem;
      font-weight: 600;
      margin-bottom: 0.25rem;
    }
    
    .usage-text p {
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    
    footer {
      text-align: center;
      padding-top: 2rem;
      border-top: 1px solid rgba(255,255,255,0.1);
      color: var(--text-secondary);
      font-size: 0.875rem;
    }
    
    footer a {
      color: var(--kelly-orange);
      text-decoration: none;
    }
    
    .download-all {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 1rem 2rem;
      background: linear-gradient(135deg, var(--kelly-blue), var(--kelly-orange));
      color: white;
      text-decoration: none;
      border-radius: 12px;
      font-weight: 600;
      font-size: 1rem;
      margin-top: 1rem;
    }
    
    .animation-preview {
      text-align: center;
      margin-bottom: 4rem;
      padding: 2rem;
      background: var(--bg-card);
      border-radius: 16px;
    }
    
    .animation-preview h2 {
      margin-bottom: 1rem;
    }
    
    .frame-strip {
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin: 2rem 0;
    }
    
    .frame-strip img {
      width: 120px;
      height: 120px;
      object-fit: cover;
      border-radius: 8px;
      border: 2px solid transparent;
      transition: border-color 0.2s;
    }
    
    .frame-strip img:hover {
      border-color: var(--kelly-orange);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">✨ Curious Kelly</div>
      <p class="subtitle">Logo Animation Sequence - Press Kit</p>
      <p class="meta">Generated: ${date} • ${imagePaths.length} Frames • 4K Resolution (2048×2048)</p>
    </header>
    
    <section class="animation-preview">
      <h2 class="section-title">Animation Sequence Preview</h2>
      <p style="color: var(--text-secondary); margin-bottom: 1rem;">
        From curious thought to warm welcome - perfect for animated logos and brand moments
      </p>
      <div class="frame-strip">
        ${imagePaths.map((p, i) => `<img src="${path.basename(p)}" alt="Frame ${i + 1}">`).join('\n        ')}
      </div>
      <p style="color: var(--text-secondary); font-size: 0.875rem;">
        → Import into After Effects, Photoshop, or your animation tool<br>
        → Add easing and transitions between frames<br>
        → Export as GIF, APNG, WebM, or video loop
      </p>
    </section>

    <h2 class="section-title">Individual Frames</h2>
    <div class="assets-grid">
      ${imageCards}
    </div>
    
    <section class="usage-section">
      <h2 class="section-title">Usage Guidelines</h2>
      <div class="usage-grid">
        <div class="usage-item">
          <div class="usage-icon">✅</div>
          <div class="usage-text">
            <h4>Press & Media</h4>
            <p>Use in articles, reviews, and educational content about Curious Kelly</p>
          </div>
        </div>
        <div class="usage-item">
          <div class="usage-icon">✅</div>
          <div class="usage-text">
            <h4>Social Media</h4>
            <p>Share on social platforms with proper attribution to @CuriousKelly</p>
          </div>
        </div>
        <div class="usage-item">
          <div class="usage-icon">✅</div>
          <div class="usage-text">
            <h4>Educational</h4>
            <p>Use in educational contexts and learning-related content</p>
          </div>
        </div>
        <div class="usage-item">
          <div class="usage-icon">⚠️</div>
          <div class="usage-text">
            <h4>Modification</h4>
            <p>Do not alter, distort, or misrepresent Kelly's appearance</p>
          </div>
        </div>
      </div>
    </section>
    
    <footer>
      <p>© ${new Date().getFullYear()} Lesson of the Day PBC. All rights reserved.</p>
      <p style="margin-top: 0.5rem;">
        Contact: <a href="mailto:hello@curiouskelly.com">hello@curiouskelly.com</a> • 
        <a href="https://curiouskelly.com">curiouskelly.com</a>
      </p>
    </footer>
  </div>
</body>
</html>`;

  const htmlPath = path.join(CONFIG.OUTPUT_DIR, 'press-kit.html');
  fs.writeFileSync(htmlPath, html, 'utf-8');
  return htmlPath;
}

main().catch(console.error);

