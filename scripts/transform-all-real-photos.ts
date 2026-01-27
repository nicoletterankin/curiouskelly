#!/usr/bin/env npx tsx
/**
 * 🏛️ TRANSFORM ALL REAL BUILDING PHOTOS
 * 
 * Transforms EVERY historical photo of the Chet Holifield Federal Building.
 * - 9 Library of Congress photos (Carol M. Highsmith, 2006)
 * - 10 Orange County Archives photos (1975-1989)
 * - 2 Modern photos (2020)
 * 
 * Total: 21 source images → 21 transformed images
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

// ═══════════════════════════════════════════════════════════════════════════
// ALL REAL SOURCE IMAGES
// ═══════════════════════════════════════════════════════════════════════════

interface SourceImage {
  url: string;
  title: string;
  date: string;
  source: string;
  type: 'exterior' | 'interior' | 'aerial';
}

const ALL_SOURCES: Record<string, SourceImage> = {
  // ─────────────────────────────────────────────────────────────────────────
  // LIBRARY OF CONGRESS - Carol M. Highsmith (July 3, 2006)
  // Using 'v' suffix for larger versions
  // ─────────────────────────────────────────────────────────────────────────
  'loc-01700': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01700v.jpg',
    title: 'Full View of Front Facade #1',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  'loc-01701': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01701v.jpg',
    title: 'Detail View of Front Facade #1',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  'loc-01702': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01702v.jpg',
    title: 'Detail View of Front Facade #2',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  'loc-01703': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01703v.jpg',
    title: 'Full View of Front Facade #2',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  'loc-01704': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01704v.jpg',
    title: 'Detail View of Front Facade #3',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  'loc-01705': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01705v.jpg',
    title: 'Interior Lobby View #1',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'interior',
  },
  'loc-01706': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01706v.jpg',
    title: 'Interior Office Area',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'interior',
  },
  'loc-01707': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01707v.jpg',
    title: 'Interior Lobby View #2',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'interior',
  },
  'loc-01708': {
    url: 'https://tile.loc.gov/storage-services/service/pnp/pplot/13800/13819/01708v.jpg',
    title: 'Detail View of Rooftop',
    date: 'July 3, 2006',
    source: 'Library of Congress',
    type: 'exterior',
  },
  
  // ─────────────────────────────────────────────────────────────────────────
  // ORANGE COUNTY ARCHIVES (1975-1989)
  // Using _h suffix for high resolution where available
  // ─────────────────────────────────────────────────────────────────────────
  'oc-1975-aerial': {
    url: 'https://live.staticflickr.com/3233/2882120279_b01a4ff374_h.jpg',
    title: 'Building with Regional Park',
    date: 'circa 1975',
    source: 'OC Archives',
    type: 'aerial',
  },
  'oc-1980s-reservoir': {
    url: 'https://live.staticflickr.com/65535/53233116910_585eeeee43_h.jpg',
    title: 'With Sulphur Creek Reservoir',
    date: 'circa 1980s',
    source: 'OC Archives',
    type: 'aerial',
  },
  'oc-1985-exterior': {
    url: 'https://live.staticflickr.com/65535/53232976214_f4f589821e_b.jpg',
    title: 'Exterior View',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1985-facade': {
    url: 'https://live.staticflickr.com/65535/53232976364_f151a818ef_b.jpg',
    title: 'Facade Detail',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1985-wide': {
    url: 'https://live.staticflickr.com/65535/53232976499_63f44a2cec_b.jpg',
    title: 'Wide Angle View',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1985-approach': {
    url: 'https://live.staticflickr.com/65535/53232897243_2ff5d823bf_b.jpg',
    title: 'Approach View',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1985-side': {
    url: 'https://live.staticflickr.com/65535/53231730252_59c3492c29_b.jpg',
    title: 'Side Elevation',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1985-panorama': {
    url: 'https://live.staticflickr.com/65535/53233099255_178e00fa10_b.jpg',
    title: 'Panoramic View',
    date: 'May 31, 1985',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1989-view1': {
    url: 'https://live.staticflickr.com/65535/53232976169_aa2904a408_b.jpg',
    title: 'Late 1980s View #1',
    date: 'November 17, 1989',
    source: 'OC Archives',
    type: 'exterior',
  },
  'oc-1989-view2': {
    url: 'https://live.staticflickr.com/65535/53232598311_92dfd94421_b.jpg',
    title: 'Late 1980s View #2',
    date: 'November 17, 1989',
    source: 'OC Archives',
    type: 'exterior',
  },
  
  // ─────────────────────────────────────────────────────────────────────────
  // MODERN PHOTOS (2020)
  // ─────────────────────────────────────────────────────────────────────────
  'modern-wikimedia-2020': {
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Chet_Holifield_Federal_Building_2020.jpg/1920px-Chet_Holifield_Federal_Building_2020.jpg',
    title: 'Contemporary View',
    date: 'October 22, 2020',
    source: 'Wikimedia Commons',
    type: 'exterior',
  },
  'modern-gsa-official': {
    url: 'https://origin-www.gsa.gov/system/files/holifield_new.jpg',
    title: 'GSA Official Documentation',
    date: '2020s',
    source: 'GSA.gov',
    type: 'exterior',
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFORMATION PROMPTS BY TYPE
// ═══════════════════════════════════════════════════════════════════════════

const TRANSFORM_PROMPTS = {
  exterior: `The same stepped pyramid brutalist building but now transformed with bright blue LED strip lighting (#3B82F6) integrated along each horizontal terrace edge, glowing electric blue accent bands highlighting every level of the ziggurat form, the concrete structure remains identical but now illuminated with futuristic blue edge lighting creating horizontal bands of light on each terrace, educational beacon, twilight atmosphere, architectural photography`,
  
  interior: `The same interior space but now with subtle blue LED accent lighting (#3B82F6) integrated along ceiling edges and architectural details, maintaining the original wood and concrete materials but adding a futuristic blue glow to highlight the space, modern educational facility aesthetic, architectural interior photography`,
  
  aerial: `The same aerial view of the stepped pyramid building but now with glowing blue LED strips (#3B82F6) visible along each terrace edge from above, 13 concentric rings of blue light highlighting the ziggurat form, the concrete structure unchanged but illuminated with futuristic edge lighting, educational campus beacon, architectural aerial photography`,
};

// ═══════════════════════════════════════════════════════════════════════════
// PROCESSING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function downloadImage(url: string, name: string): Promise<string | null> {
  const outputPath = path.join(OUTPUT_DIR, `source-${name}.jpg`);
  
  // Skip if already downloaded
  if (fs.existsSync(outputPath)) {
    console.log(`   📁 Already exists: ${name}`);
    return outputPath;
  }
  
  console.log(`   📥 Downloading: ${name}`);
  
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(outputPath, buffer);
    console.log(`   ✅ Saved: ${outputPath}`);
    
    return outputPath;
  } catch (error: any) {
    console.error(`   ❌ Download failed: ${error.message}`);
    return null;
  }
}

async function transformImage(
  sourcePath: string,
  name: string,
  type: 'exterior' | 'interior' | 'aerial'
): Promise<string | null> {
  const outputPath = path.join(OUTPUT_DIR, `transformed-${name}.png`);
  
  // Skip if already transformed
  if (fs.existsSync(outputPath)) {
    console.log(`   📁 Transform exists: ${name}`);
    return outputPath;
  }
  
  console.log(`   🎨 Transforming: ${name} (${type})`);
  
  try {
    const buffer = fs.readFileSync(sourcePath);
    const base64 = buffer.toString('base64');
    const dataUrl = `data:image/jpeg;base64,${base64}`;
    
    const prompt = TRANSFORM_PROMPTS[type];
    
    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt,
          image: dataUrl,
          prompt_strength: 0.42,
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
      const response = await fetch(imageUrl);
      const resultBuffer = Buffer.from(await response.arrayBuffer());
      fs.writeFileSync(outputPath, resultBuffer);
      console.log(`   ✅ Transformed: ${outputPath}`);
      return outputPath;
    }
    
    return null;
  } catch (error: any) {
    console.error(`   ❌ Transform error: ${error.message}`);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🏛️  TRANSFORM ALL HISTORICAL PHOTOS');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`\nTotal sources: ${Object.keys(ALL_SOURCES).length}`);
  console.log(`Output: ${OUTPUT_DIR}\n`);

  const results: Record<string, { source: string | null; transformed: string | null; meta: SourceImage }> = {};
  
  let completed = 0;
  const total = Object.keys(ALL_SOURCES).length;

  for (const [name, meta] of Object.entries(ALL_SOURCES)) {
    completed++;
    console.log(`\n━━━ [${completed}/${total}] ${name} ━━━`);
    console.log(`   Title: ${meta.title}`);
    console.log(`   Date: ${meta.date}`);
    console.log(`   Source: ${meta.source}`);
    
    // Download
    const sourcePath = await downloadImage(meta.url, name);
    if (!sourcePath) {
      results[name] = { source: null, transformed: null, meta };
      continue;
    }
    
    // Transform
    const transformedPath = await transformImage(sourcePath, name, meta.type);
    results[name] = { source: sourcePath, transformed: transformedPath, meta };
    
    // Rate limiting
    await new Promise(r => setTimeout(r, 500));
  }

  // Generate HTML comparison gallery
  await generateFullGallery(results);

  // Summary
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('📊 FINAL SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  
  const downloaded = Object.values(results).filter(r => r.source !== null).length;
  const transformed = Object.values(results).filter(r => r.transformed !== null).length;
  
  console.log(`\n📥 Downloaded: ${downloaded}/${total}`);
  console.log(`🎨 Transformed: ${transformed}/${total}`);
  
  console.log('\n🎉 Done! Open ZIGGURAT-BEFORE-AFTER.html to see all comparisons.');
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATE FULL COMPARISON GALLERY
// ═══════════════════════════════════════════════════════════════════════════

async function generateFullGallery(results: Record<string, { source: string | null; transformed: string | null; meta: SourceImage }>) {
  const pairs = Object.entries(results)
    .filter(([_, r]) => r.source && r.transformed)
    .map(([name, r]) => ({
      name,
      source: `ziggurat-transforms/source-${name}.jpg`,
      transformed: `ziggurat-transforms/transformed-${name}.png`,
      ...r.meta,
    }));

  // Group by source
  const locPairs = pairs.filter(p => p.source.includes('Library of Congress'));
  const ocPairs = pairs.filter(p => p.source.includes('OC Archives'));
  const modernPairs = pairs.filter(p => p.source.includes('Wikimedia') || p.source.includes('GSA'));

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Ziggurat — Complete Before & After Gallery</title>
    <style>
        :root {
            --blue: #3B82F6;
            --amber: #F59E0B;
            --dark: #050508;
            --darker: #020203;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--darker);
            color: #fff;
            min-height: 100vh;
        }
        
        .nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 1rem 2rem;
            background: rgba(2, 2, 3, 0.95);
            backdrop-filter: blur(10px);
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .nav-brand { font-weight: 700; color: var(--blue); }
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
            background: linear-gradient(180deg, rgba(59,130,246,0.15) 0%, transparent 100%);
        }
        
        header h1 {
            font-size: clamp(2.5rem, 6vw, 4.5rem);
            font-weight: 800;
            margin-bottom: 1rem;
        }
        
        header .count {
            font-size: 3rem;
            font-weight: 800;
            color: var(--blue);
            margin-bottom: 0.5rem;
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
        
        .section {
            max-width: 1400px;
            margin: 0 auto;
            padding: 3rem 2rem;
        }
        
        .section-header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .section-header h2 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        
        .section-header p {
            color: rgba(255,255,255,0.6);
        }
        
        .comparison-grid {
            display: grid;
            gap: 2rem;
        }
        
        .comparison-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1rem;
            overflow: hidden;
            transition: border-color 0.3s;
        }
        
        .comparison-card:hover {
            border-color: rgba(59, 130, 246, 0.3);
        }
        
        .comparison-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .comparison-header h3 { font-size: 1.1rem; }
        
        .meta-badges {
            display: flex;
            gap: 0.5rem;
        }
        
        .badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.6rem;
            border-radius: 1rem;
            background: rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.7);
        }
        
        .badge.date { background: rgba(245, 158, 11, 0.2); color: var(--amber); }
        .badge.type { background: rgba(59, 130, 246, 0.2); color: var(--blue); }
        
        /* SLIDER */
        .comparison-container {
            position: relative;
            width: 100%;
            overflow: hidden;
            cursor: ew-resize;
            touch-action: none;
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
            width: 3px;
            background: var(--blue);
            cursor: ew-resize;
            z-index: 10;
            box-shadow: 0 0 15px rgba(59,130,246,0.6);
        }
        
        .slider-handle::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: var(--blue);
            border-radius: 50%;
            box-shadow: 0 0 20px rgba(59,130,246,0.5);
        }
        
        .slider-handle::after {
            content: '◀ ▶';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 2px;
            white-space: nowrap;
        }
        
        .labels {
            position: absolute;
            bottom: 0.75rem;
            left: 0.75rem;
            right: 0.75rem;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        
        .label {
            padding: 0.4rem 0.8rem;
            background: rgba(0,0,0,0.85);
            border-radius: 0.4rem;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .label.before { color: rgba(255,255,255,0.9); }
        .label.after { color: var(--blue); }
        
        footer {
            text-align: center;
            padding: 4rem 2rem;
            color: rgba(255,255,255,0.4);
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        footer a { color: var(--blue); text-decoration: none; }
        
        .footer-quote {
            font-size: 1.4rem;
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
            <a href="chet-holifield-federal-building-gallery.html">Historical</a>
        </div>
    </nav>
    
    <header>
        <div class="count">${pairs.length}</div>
        <h1>Before & After</h1>
        <p>Every historical photo of the Chet Holifield Federal Building, transformed to show the LED edge lighting vision.</p>
        <div class="instruction">◀ Drag the slider to compare ▶</div>
    </header>
    
    ${locPairs.length > 0 ? `
    <section class="section">
        <div class="section-header">
            <h2>Library of Congress Collection</h2>
            <p>Carol M. Highsmith Archive — ${locPairs.length} photographs from July 3, 2006</p>
        </div>
        <div class="comparison-grid">
            ${locPairs.map(p => renderCard(p)).join('')}
        </div>
    </section>
    ` : ''}
    
    ${ocPairs.length > 0 ? `
    <section class="section">
        <div class="section-header">
            <h2>Orange County Archives Collection</h2>
            <p>Historical photographs from 1975-1989 — ${ocPairs.length} images</p>
        </div>
        <div class="comparison-grid">
            ${ocPairs.map(p => renderCard(p)).join('')}
        </div>
    </section>
    ` : ''}
    
    ${modernPairs.length > 0 ? `
    <section class="section">
        <div class="section-header">
            <h2>Modern Photography</h2>
            <p>Recent documentation — ${modernPairs.length} images</p>
        </div>
        <div class="comparison-grid">
            ${modernPairs.map(p => renderCard(p)).join('')}
        </div>
    </section>
    ` : ''}
    
    <footer>
        <p class="footer-quote">"A building that teaches by being seen."</p>
        <p>${pairs.length} real photos transformed • January 2026</p>
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
            
            container.addEventListener('mousedown', e => { isDragging = true; updateSlider(e.clientX); });
            document.addEventListener('mousemove', e => { if (isDragging) updateSlider(e.clientX); });
            document.addEventListener('mouseup', () => { isDragging = false; });
            
            container.addEventListener('touchstart', e => { isDragging = true; updateSlider(e.touches[0].clientX); });
            container.addEventListener('touchmove', e => { if (isDragging) { e.preventDefault(); updateSlider(e.touches[0].clientX); } });
            container.addEventListener('touchend', () => { isDragging = false; });
        });
    </script>
</body>
</html>`;

  function renderCard(p: any) {
    return `
        <div class="comparison-card">
            <div class="comparison-header">
                <h3>${p.title}</h3>
                <div class="meta-badges">
                    <span class="badge date">${p.date}</span>
                    <span class="badge type">${p.type}</span>
                </div>
            </div>
            <div class="comparison-container">
                <img src="${p.transformed}" alt="After - with LED" class="img-after">
                <div class="img-before" style="width: 50%;">
                    <img src="${p.source}" alt="Before - original">
                </div>
                <div class="slider-handle" style="left: 50%;"></div>
                <div class="labels">
                    <span class="label before">ORIGINAL</span>
                    <span class="label after">TRANSFORMED</span>
                </div>
            </div>
        </div>
    `;
  }

  const outputPath = path.join(process.cwd(), 'public', 'ZIGGURAT-BEFORE-AFTER.html');
  fs.writeFileSync(outputPath, html);
  console.log(`\n📄 Full gallery saved: ${outputPath}`);
}

main().catch(console.error);
