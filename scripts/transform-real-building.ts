#!/usr/bin/env npx tsx
/**
 * 🏛️ TRANSFORM THE REAL BUILDING
 * 
 * Takes ACTUAL historical photos of the Chet Holifield Federal Building
 * and transforms them to show the LED concept overlays.
 * 
 * This is what we SHOULD have done from the start.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'ziggurat-transforms');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// REAL source images from the historical gallery
const REAL_SOURCES = {
  'wikimedia-2020': {
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Chet_Holifield_Federal_Building_2020.jpg/1920px-Chet_Holifield_Federal_Building_2020.jpg',
    description: 'Contemporary view (2020) - clear day, full building visible',
    transformPrompt: `The same stepped pyramid brutalist building but now with bright blue LED strip lighting (#3B82F6) along each of the 13 horizontal terrace edges, glowing electric blue accent lines highlighting the ziggurat form, futuristic educational beacon, the LED strips are integrated into the concrete terrace edges creating horizontal bands of blue light, evening twilight sky, architectural photography`,
    strength: 0.45,
  },
  
  'loc-exterior-01700': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01700v.jpg',
    description: 'Library of Congress - Carol M. Highsmith exterior shot',
    transformPrompt: `The same brutalist stepped pyramid building with glowing blue LED strips (#3B82F6) integrated along each terrace edge, 13 levels of horizontal blue light bands creating a futuristic ziggurat beacon, the concrete structure remains but now illuminated with electric blue accent lighting on every terrace, twilight atmosphere, architectural photography`,
    strength: 0.45,
  },
  
  'loc-exterior-01701': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01701v.jpg',
    description: 'Library of Congress - another angle',
    transformPrompt: `The same massive stepped pyramid federal building now transformed with blue LED edge lighting (#3B82F6) on each of its 13 terraced levels, horizontal bands of electric blue glow along every concrete terrace edge, futuristic educational landmark, architectural beacon, twilight sky`,
    strength: 0.45,
  },
  
  'flickr-aerial-1975': {
    url: 'https://live.staticflickr.com/3233/2882120279_b01a4ff374_h.jpg',
    description: 'OC Archives aerial view circa 1975',
    transformPrompt: `The same aerial view of the stepped pyramid building but now with glowing blue LED strips (#3B82F6) visible along each terrace edge from above, 13 concentric rings of blue light highlighting the ziggurat form, futuristic educational campus beacon, architectural transformation`,
    strength: 0.45,
  },
  
  'flickr-construction': {
    url: 'https://live.staticflickr.com/65535/53233116910_585eeeee43_h.jpg',
    description: 'Construction/early period photo',
    transformPrompt: `The same stepped pyramid brutalist building with blue LED lighting (#3B82F6) integrated along each horizontal terrace edge, glowing blue accent bands on every level, futuristic transformation of the ziggurat form, architectural photography`,
    strength: 0.45,
  },
};

async function downloadImage(url: string, name: string): Promise<string | null> {
  console.log(`   📥 Downloading: ${name}`);
  
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (educational project)',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const buffer = Buffer.from(await response.arrayBuffer());
    const ext = url.includes('.png') ? 'png' : 'jpg';
    const outputPath = path.join(OUTPUT_DIR, `source-${name}.${ext}`);
    fs.writeFileSync(outputPath, buffer);
    console.log(`   ✅ Saved source: ${outputPath}`);
    
    return outputPath;
  } catch (error: any) {
    console.error(`   ❌ Download failed: ${error.message}`);
    return null;
  }
}

async function transformImage(
  sourcePath: string,
  name: string,
  prompt: string,
  strength: number
): Promise<string | null> {
  console.log(`   🎨 Transforming: ${name}`);
  
  try {
    // Read source and convert to data URL
    const buffer = fs.readFileSync(sourcePath);
    const base64 = buffer.toString('base64');
    const ext = sourcePath.endsWith('.png') ? 'png' : 'jpeg';
    const dataUrl = `data:image/${ext};base64,${base64}`;
    
    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt,
          image: dataUrl,
          prompt_strength: strength,
          num_outputs: 1,
          output_format: "png",
          output_quality: 100,
          num_inference_steps: 50,
          guidance: 3.5,
        }
      }
    ) as any;

    let imageUrl: string | null = null;
    if (Array.isArray(output) && output.length > 0) {
      imageUrl = typeof output[0] === 'string' ? output[0] : String(output[0]);
    } else if (typeof output === 'string') {
      imageUrl = output;
    }
    
    if (imageUrl) {
      console.log(`   ✅ Generated: ${imageUrl.substring(0, 60)}...`);
      
      const response = await fetch(imageUrl);
      const resultBuffer = Buffer.from(await response.arrayBuffer());
      const outputPath = path.join(OUTPUT_DIR, `transformed-${name}.png`);
      fs.writeFileSync(outputPath, resultBuffer);
      console.log(`   💾 Saved: ${outputPath}`);
      
      return outputPath;
    }
    
    return null;
  } catch (error: any) {
    console.error(`   ❌ Transform error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🏛️  TRANSFORM THE REAL BUILDING');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('\nUsing ACTUAL historical photos as source images.\n');

  const results: Record<string, { source: string | null; transformed: string | null }> = {};

  for (const [name, config] of Object.entries(REAL_SOURCES)) {
    console.log(`\n━━━ ${name} ━━━`);
    console.log(`Description: ${config.description}`);
    
    // Download source
    const sourcePath = await downloadImage(config.url, name);
    if (!sourcePath) {
      results[name] = { source: null, transformed: null };
      continue;
    }
    
    // Transform
    const transformedPath = await transformImage(
      sourcePath,
      name,
      config.transformPrompt,
      config.strength
    );
    
    results[name] = { source: sourcePath, transformed: transformedPath };
    
    await new Promise(r => setTimeout(r, 1000));
  }

  // Generate comparison HTML
  await generateComparisonHTML(results);

  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('📊 TRANSFORMATION SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  
  const successful = Object.entries(results).filter(([_, r]) => r.transformed !== null);
  console.log(`\n✅ Successful: ${successful.length}/${Object.keys(REAL_SOURCES).length}`);
  successful.forEach(([name, _]) => console.log(`   - ${name}`));
  
  console.log('\n🎉 Done! Open ZIGGURAT-BEFORE-AFTER.html to see the comparisons.');
}

async function generateComparisonHTML(results: Record<string, { source: string | null; transformed: string | null }>) {
  const pairs = Object.entries(results)
    .filter(([_, r]) => r.source && r.transformed)
    .map(([name, r]) => ({
      name,
      source: `ziggurat-transforms/source-${name}.${r.source!.endsWith('.png') ? 'png' : 'jpg'}`,
      transformed: `ziggurat-transforms/transformed-${name}.png`,
      description: REAL_SOURCES[name as keyof typeof REAL_SOURCES].description,
    }));

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Ziggurat — Before & After Transformation</title>
    <style>
        :root {
            --blue: #3B82F6;
            --amber: #F59E0B;
            --dark: #050508;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--dark);
            color: #fff;
            min-height: 100vh;
        }
        
        .nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 1rem 2rem;
            background: rgba(5, 5, 8, 0.95);
            backdrop-filter: blur(10px);
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .nav-brand {
            font-weight: 700;
            color: var(--blue);
        }
        
        .nav-links a {
            color: rgba(255,255,255,0.6);
            text-decoration: none;
            margin-left: 2rem;
            font-size: 0.9rem;
        }
        
        .nav-links a:hover { color: var(--blue); }
        
        header {
            padding: 8rem 2rem 4rem;
            text-align: center;
            background: linear-gradient(180deg, rgba(59,130,246,0.1) 0%, transparent 100%);
        }
        
        header h1 {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 800;
            margin-bottom: 1rem;
        }
        
        header p {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.7);
            max-width: 700px;
            margin: 0 auto;
        }
        
        .instruction {
            display: inline-block;
            margin-top: 1.5rem;
            padding: 0.75rem 1.5rem;
            background: rgba(59,130,246,0.2);
            border: 1px solid rgba(59,130,246,0.3);
            border-radius: 2rem;
            font-size: 0.9rem;
            color: var(--blue);
        }
        
        .comparisons {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .comparison-card {
            margin-bottom: 4rem;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1.5rem;
            overflow: hidden;
        }
        
        .comparison-header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .comparison-header h2 {
            font-size: 1.3rem;
        }
        
        .comparison-header .source-badge {
            font-size: 0.8rem;
            padding: 0.3rem 0.8rem;
            background: rgba(255,255,255,0.1);
            border-radius: 1rem;
            color: rgba(255,255,255,0.6);
        }
        
        /* THE SLIDER */
        .comparison-container {
            position: relative;
            width: 100%;
            overflow: hidden;
            cursor: ew-resize;
        }
        
        .comparison-container img {
            display: block;
            width: 100%;
            height: auto;
        }
        
        .img-before {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        
        .img-before img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .slider-handle {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--blue);
            cursor: ew-resize;
            z-index: 10;
            box-shadow: 0 0 20px rgba(59,130,246,0.5);
        }
        
        .slider-handle::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 48px;
            height: 48px;
            background: var(--blue);
            border-radius: 50%;
            box-shadow: 0 0 30px rgba(59,130,246,0.5);
        }
        
        .slider-handle::after {
            content: '◀ ▶';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 4px;
            white-space: nowrap;
        }
        
        .labels {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            right: 1rem;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        
        .label {
            padding: 0.5rem 1rem;
            background: rgba(0,0,0,0.8);
            border-radius: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .label.before { color: rgba(255,255,255,0.8); }
        .label.after { color: var(--blue); }
        
        footer {
            text-align: center;
            padding: 4rem 2rem;
            color: rgba(255,255,255,0.4);
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        footer a { color: var(--blue); text-decoration: none; }
        
        .footer-quote {
            font-size: 1.3rem;
            font-style: italic;
            color: var(--blue);
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-brand">THE ZIGGURAT</div>
        <div class="nav-links">
            <a href="PACKAGE-INDEX.html">Package</a>
            <a href="ZIGGURAT-PREMIUM-GALLERY.html">Gallery</a>
            <a href="chet-holifield-federal-building-gallery.html">Historical Photos</a>
        </div>
    </nav>
    
    <header>
        <h1>Before & After</h1>
        <p>The actual Chet Holifield Federal Building transformed with LED edge lighting. Drag the slider to reveal the vision.</p>
        <div class="instruction">◀ Drag the slider ▶</div>
    </header>
    
    <div class="comparisons">
        ${pairs.map(pair => `
        <div class="comparison-card">
            <div class="comparison-header">
                <h2>${pair.name.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</h2>
                <span class="source-badge">${pair.description}</span>
            </div>
            <div class="comparison-container" data-before="${pair.source}" data-after="${pair.transformed}">
                <img src="${pair.transformed}" alt="After - with LED" class="img-after">
                <div class="img-before" style="width: 50%;">
                    <img src="${pair.source}" alt="Before - original">
                </div>
                <div class="slider-handle" style="left: 50%;"></div>
                <div class="labels">
                    <span class="label before">ORIGINAL</span>
                    <span class="label after">TRANSFORMED</span>
                </div>
            </div>
        </div>
        `).join('')}
    </div>
    
    <footer>
        <p class="footer-quote">"A building that teaches by being seen."</p>
        <p>Real photos transformed • January 2026</p>
        <p style="margin-top: 0.5rem;"><a href="PACKAGE-INDEX.html">Laguna Ridge Acquisition</a> confidential package</p>
    </footer>
    
    <script>
        document.querySelectorAll('.comparison-container').forEach(container => {
            const handle = container.querySelector('.slider-handle');
            const beforeDiv = container.querySelector('.img-before');
            let isDragging = false;
            
            function updateSlider(x) {
                const rect = container.getBoundingClientRect();
                let percent = ((x - rect.left) / rect.width) * 100;
                percent = Math.max(0, Math.min(100, percent));
                
                handle.style.left = percent + '%';
                beforeDiv.style.width = percent + '%';
            }
            
            container.addEventListener('mousedown', (e) => {
                isDragging = true;
                updateSlider(e.clientX);
            });
            
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                updateSlider(e.clientX);
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
            // Touch support
            container.addEventListener('touchstart', (e) => {
                isDragging = true;
                updateSlider(e.touches[0].clientX);
            });
            
            container.addEventListener('touchmove', (e) => {
                if (!isDragging) return;
                e.preventDefault();
                updateSlider(e.touches[0].clientX);
            });
            
            container.addEventListener('touchend', () => {
                isDragging = false;
            });
        });
    </script>
</body>
</html>`;

  const outputPath = path.join(process.cwd(), 'public', 'ZIGGURAT-BEFORE-AFTER.html');
  fs.writeFileSync(outputPath, html);
  console.log(`\n📄 Comparison gallery saved: ${outputPath}`);
}

main().catch(console.error);
