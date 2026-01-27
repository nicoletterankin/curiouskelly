#!/usr/bin/env npx tsx
/**
 * 🌟 KELLY LOGO - GENUINE WONDER
 * 
 * Regenerating Kelly with TRUE curiosity and wonder - not posed, but that
 * genuine spark of discovery. The "aha!" moment. Eyes alive with interest.
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
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-logo-wonder'),
};

// Different approaches to genuine wonder
const WONDER_PROMPTS = [
  {
    id: 'wonder-discovery',
    name: 'Discovery Moment',
    prompt: `CKELLY curious kelly, candid portrait photo, young woman late 20s, long wavy brown hair with golden highlights, sparkling brown eyes, wearing soft light blue sweater,

GENUINE WONDER: eyes bright and WIDE with sudden interest, eyebrows raised high in delighted surprise, mouth slightly open in an excited "oh!", natural spontaneous expression like she just discovered something amazing, head tilted with eager anticipation, NO posed hand gestures, authentic candid moment of discovery,

professional headshot, tight crop on face filling frame, minimal headroom, pure white background, soft natural lighting, photorealistic, sharp focus on eyes, 8K quality`,
    seed: 77777777,
  },
  {
    id: 'wonder-fascinated',
    name: 'Fascinated Gaze',
    prompt: `CKELLY curious kelly, natural portrait, young woman late 20s, long wavy brown hair with blonde highlights, bright curious brown eyes, light blue knit sweater,

TRUE FASCINATION: looking slightly upward with wide-eyed wonder, eyebrows lifted in genuine amazement, soft smile of delighted discovery, eyes sparkling with the joy of learning something new, expression of a child seeing magic for the first time but on an adult, completely natural and unposed, caught in a moment of pure curiosity,

close-up headshot, face fills 80% of frame, white seamless background, studio lighting with catchlights, photorealistic quality, crystal clear eyes, no hand in frame`,
    seed: 88888888,
  },
  {
    id: 'wonder-lightbulb',
    name: 'Lightbulb Moment',
    prompt: `CKELLY curious kelly, authentic portrait, young woman late 20s, wavy brown hair with golden streaks, expressive brown eyes full of wonder, cozy light blue sweater,

AHA MOMENT: the exact instant of understanding, eyes wide and bright with excitement, eyebrows arched in surprise, genuine smile breaking through, radiating the joy of sudden insight, looking upward and slightly to the side as if an idea just struck, completely candid and real expression, not posing for camera,

tight headshot crop, head large in frame, minimal white space above, clean white background, soft professional lighting, photorealistic, pin-sharp focus on eyes`,
    seed: 99999999,
  },
  {
    id: 'wonder-enchanted',
    name: 'Enchanted Wonder',
    prompt: `CKELLY curious kelly, spontaneous portrait, young woman late 20s, flowing wavy brown hair with honey highlights, big wondering brown eyes, soft blue sweater,

ENCHANTED: gazing with childlike wonder, eyes wide and glistening with amazement, subtle open-mouth smile of awe, eyebrows raised in gentle surprise, expression of someone seeing something beautiful for the first time, natural and unguarded, pure innocent curiosity without any posed elements,

extreme close portrait, face dominant in frame, almost no headroom, pristine white background, beautiful soft lighting, photorealistic detail, sparkling clear eyes`,
    seed: 12121212,
  },
  {
    id: 'wonder-intrigued',
    name: 'Intrigued Interest', 
    prompt: `CKELLY curious kelly, real candid portrait, young woman late 20s, natural wavy brown hair with sun-kissed highlights, warm brown eyes alive with curiosity, light blue casual sweater,

DEEPLY INTRIGUED: leaning forward slightly with genuine interest, eyes wide and focused with fascination, one eyebrow slightly higher showing intrigue, lips parted in engaged concentration, the look of someone who just heard something fascinating and wants to know more, absolutely authentic not staged,

intimate headshot, large face filling frame, tight crop above head, pure white backdrop, flattering natural light, ultra photorealistic, crystal clear eye detail`,
    seed: 34343434,
  },
  {
    id: 'wonder-delighted',
    name: 'Delighted Curiosity',
    prompt: `CKELLY curious kelly, genuine moment portrait, young woman late 20s, beautiful wavy brown hair, bright expressive eyes, cozy blue sweater,

DELIGHTED CURIOSITY: eyes sparkling wide with joyful wonder, beaming smile of excited discovery, eyebrows lifted in happy surprise, the expression of "oh that's amazing!", radiating warmth and genuine enthusiasm, completely natural candid capture, no artificial posing,

close crop headshot, minimal space above head, face fills frame, clean white background, soft studio lighting, hyper realistic, perfectly sharp eyes full of life`,
    seed: 56565656,
  },
];

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
      file.on('finish', () => { file.close(); resolve(); });
    }).on('error', (err) => { fs.unlink(outputPath, () => {}); reject(err); });
  });
}

async function generateVariation(replicate: Replicate, prompt: typeof WONDER_PROMPTS[0]): Promise<string> {
  console.log(`\n🌟 Generating: ${prompt.name}`);
  console.log(`   Seed: ${prompt.seed}`);
  
  const input = {
    prompt: prompt.prompt,
    hf_lora: CONFIG.KELLY_LORA_URL,
    lora_scale: CONFIG.LORA_SCALE,
    num_outputs: 1,
    aspect_ratio: '1:1',
    output_format: 'png',
    guidance_scale: 4.5, // Higher for more prompt adherence
    output_quality: 100,
    prompt_strength: 0.85,
    num_inference_steps: 50,
    seed: prompt.seed,
    disable_safety_checker: true,
  };
  
  console.log(`   Calling Replicate...`);
  const output = await replicate.run(CONFIG.FLUX_LORA_MODEL as `${string}/${string}:${string}`, { input });
  
  let imageUrl: string;
  if (Array.isArray(output)) {
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  const outputPath = path.join(CONFIG.OUTPUT_DIR, `${prompt.id}.png`);
  console.log(`   📥 Downloading...`);
  await downloadImage(imageUrl, outputPath);
  console.log(`   ✅ Saved!`);
  
  return outputPath;
}

function generateHtml(generatedPaths: string[]): void {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🌟 Kelly Logo - Genuine Wonder</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root { --orange: #d97757; --blue: #7BA7C2; --bg: #0a0a0c; --card: #141418; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: white; padding: 2rem; }
    
    h1 { 
      text-align: center; 
      font-size: 2.5rem; 
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, var(--blue), var(--orange));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 2rem;
      max-width: 1600px;
      margin: 0 auto;
    }
    
    .card {
      background: var(--card);
      border-radius: 20px;
      overflow: hidden;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .preview {
      aspect-ratio: 1;
      background: white;
      cursor: pointer;
      position: relative;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .preview:hover::after {
      content: 'Click to zoom';
      position: absolute;
      bottom: 1rem;
      right: 1rem;
      background: rgba(0,0,0,0.8);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
    }
    
    .info { padding: 1.5rem; }
    .info h3 { font-size: 1.1rem; margin-bottom: 0.25rem; }
    .info p { color: #888; font-size: 0.85rem; margin-bottom: 1rem; }
    
    .download-btn {
      display: block;
      width: 100%;
      padding: 1rem;
      background: var(--orange);
      color: white;
      text-decoration: none;
      text-align: center;
      border-radius: 12px;
      font-weight: 600;
      transition: background 0.2s;
    }
    .download-btn:hover { background: #c56545; }
    
    .zoom-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.95);
      z-index: 1000;
      cursor: zoom-out;
    }
    .zoom-modal.active { display: flex; align-items: center; justify-content: center; }
    .zoom-modal img { max-width: 95vw; max-height: 95vh; border-radius: 12px; }
    
    .instructions {
      text-align: center;
      max-width: 600px;
      margin: 0 auto 3rem;
      padding: 1.5rem;
      background: var(--card);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .instructions h2 { font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--orange); }
    .instructions p { color: #aaa; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>🌟 Genuine Wonder</h1>
  <p class="subtitle">Kelly with authentic curiosity - not posed, truly wondering</p>
  
  <div class="instructions">
    <h2>Pick the One with Real Wonder</h2>
    <p>Click each image to zoom. Look for: wide eyes, raised eyebrows, genuine surprise, that "aha!" spark. No staged poses!</p>
  </div>
  
  <div class="grid">
    ${WONDER_PROMPTS.map((p, i) => `
    <div class="card">
      <div class="preview" onclick="zoom('${p.id}.png')">
        <img src="${p.id}.png" alt="${p.name}">
      </div>
      <div class="info">
        <h3>${p.name}</h3>
        <p>Seed: ${p.seed}</p>
        <a href="${p.id}.png" download="curious-kelly-${p.id}.png" class="download-btn">Download PNG</a>
      </div>
    </div>`).join('')}
  </div>
  
  <div class="zoom-modal" id="modal" onclick="this.classList.remove('active')">
    <img id="zoomImg" src="">
  </div>
  
  <script>
    function zoom(src) {
      document.getElementById('zoomImg').src = src;
      document.getElementById('modal').classList.add('active');
    }
    document.addEventListener('keydown', e => { if (e.key === 'Escape') document.getElementById('modal').classList.remove('active'); });
  </script>
</body>
</html>`;

  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, 'pick-wonder.html'), html);
}

async function main() {
  console.log('═'.repeat(60));
  console.log('🌟 KELLY LOGO - GENUINE WONDER GENERATOR');
  console.log('═'.repeat(60));
  console.log('\nNo more posed "thinking" shots. This is REAL curiosity.\n');
  
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }
  
  if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  }
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const generatedPaths: string[] = [];
  
  for (let i = 0; i < WONDER_PROMPTS.length; i++) {
    console.log(`\n[${i + 1}/${WONDER_PROMPTS.length}]`);
    try {
      const outputPath = await generateVariation(replicate, WONDER_PROMPTS[i]);
      generatedPaths.push(outputPath);
    } catch (error) {
      console.error(`❌ Error:`, error);
    }
  }
  
  console.log('\n📄 Generating picker HTML...');
  generateHtml(generatedPaths);
  
  console.log('\n' + '═'.repeat(60));
  console.log('✅ WONDER GENERATION COMPLETE');
  console.log('═'.repeat(60));
  console.log(`\n🌟 Generated ${generatedPaths.length} wonder variations`);
  console.log(`📁 Location: ${CONFIG.OUTPUT_DIR}`);
  console.log(`🔍 Open pick-wonder.html to find the one with REAL curiosity!`);
}

main().catch(console.error);


