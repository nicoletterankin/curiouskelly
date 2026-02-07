#!/usr/bin/env npx tsx
/**
 * 🏛️ ZIGGURAT CONCEPT GENERATOR - REPLICATE VERSION
 * 
 * Generates high-quality concept visualizations for The Ziggurat transformation.
 * Uses Replicate's Flux models for photorealistic architectural renders.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'ziggurat-concepts');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ═══════════════════════════════════════════════════════════════════════════
// ZIGGURAT PROMPTS - HIGH QUALITY ARCHITECTURAL VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════

const ZIGGURAT_PROMPTS = {
  'dawn-awakening': {
    prompt: `Architectural photography of a massive stepped pyramid building at dawn, 13 terraced levels in inverted ziggurat form where each level is wider than the one above, brutalist concrete architecture from the 1970s, each horizontal terrace edge illuminated with thin blue LED strips glowing blue #3B82F6, lighting sequence visible from bottom terraces to top, Pacific coast sunrise with pink orange and purple sky in background, Southern California coastal hills landscape, dramatic silhouette against colorful sky, professional architectural photography, ultra high resolution, cinematic composition, photorealistic`,
    aspectRatio: '16:9',
  },
  
  'night-silhouette': {
    prompt: `Architectural night photography of a stepped pyramid building against starry sky, ziggurat form with 13 horizontal terrace levels, each terrace level outlined with subtle thin blue LED strip lighting at low brightness, dark navy sky with visible stars, building silhouette with minimal light pollution, respectful ambient lighting, Southern California coastal hills in background, architectural night photography, moody contemplative atmosphere, professional real estate photography, photorealistic`,
    aspectRatio: '16:9',
  },
  
  'morning-broadcast': {
    prompt: `Futuristic educational building with LED mesh facade displaying soft blue glow, massive stepped pyramid structure in ziggurat form, 13 horizontal terraces, modern brutalist concrete architecture, morning golden hour lighting with warm sun rays, visitors and families walking on plaza below looking up at building, inspirational educational atmosphere, building as a beacon of learning, clean minimalist aesthetic, professional architectural visualization render, cinematic composition, photorealistic`,
    aspectRatio: '16:9',
  },
  
  'aerial-overview': {
    prompt: `Aerial drone photography of massive stepped pyramid building, inverted ziggurat form where each of 13 levels is wider than the one above creating distinctive terraced silhouette, brutalist concrete architecture with blue LED edge lighting along each terrace, 92-acre campus with radiating parking lots and landscaping, Southern California coastal landscape with ocean visible in distance, evening golden hour light, professional drone photography perspective, architectural masterpiece educational institution, photorealistic`,
    aspectRatio: '16:9',
  },
  
  'observatory-interior': {
    prompt: `Interior architectural photography of modern broadcast studio on 13th floor penthouse level of brutalist building, floor-to-ceiling windows with panoramic Pacific Ocean view at sunrise, minimalist production setup with professional cameras and lighting, warm natural wood accent wall contrasting with exposed concrete ceiling, blue accent LED lighting, educational broadcast origin point aesthetic, architectural interior photography, natural light streaming in, professional real estate photography, photorealistic`,
    aspectRatio: '16:9',
  },
  
  'twilight-transition': {
    prompt: `Stepped pyramid building during blue hour twilight, ziggurat form with 13 terraces clearly visible, LED strips along each horizontal terrace edge transitioning colors, gradient sky from warm orange on horizon to deep blue overhead, building silhouette with blue accent lighting creating beacon effect, coastal California setting with hills, dramatic atmospheric perspective, architectural photography during magic hour, photorealistic, professional`,
    aspectRatio: '16:9',
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

async function generateImage(
  name: string, 
  prompt: string, 
  aspectRatio: string
): Promise<string | null> {
  console.log(`\n🎨 Generating: ${name}`);
  console.log(`   Aspect ratio: ${aspectRatio}`);
  
  try {
    // Use black-forest-labs/flux-schnell for fast generation
    const output = await replicate.run(
      "black-forest-labs/flux-schnell",
      {
        input: {
          prompt,
          aspect_ratio: aspectRatio,
          num_outputs: 1,
          output_format: "png",
          output_quality: 90,
        }
      }
    ) as any;

    // Handle output - could be array of URLs or FileOutput objects
    let imageUrl: string | null = null;
    
    if (Array.isArray(output) && output.length > 0) {
      // Could be URL string or FileOutput object
      const firstOutput = output[0];
      if (typeof firstOutput === 'string') {
        imageUrl = firstOutput;
      } else if (firstOutput && typeof firstOutput === 'object') {
        imageUrl = String(firstOutput);
      }
    } else if (typeof output === 'string') {
      imageUrl = output;
    }
    
    if (imageUrl) {
      console.log(`   ✅ Generated: ${imageUrl.substring(0, 80)}...`);
      
      // Download and save locally
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to download: ${response.status}`);
      }
      const buffer = Buffer.from(await response.arrayBuffer());
      const outputPath = path.join(OUTPUT_DIR, `${name}.png`);
      fs.writeFileSync(outputPath, buffer);
      console.log(`   💾 Saved: ${outputPath}`);
      
      return outputPath;
    }
    
    console.log(`   ⚠️ No image URL in response`);
    return null;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🏛️  THE ZIGGURAT - Concept Image Generator (Replicate)');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`\nOutput directory: ${OUTPUT_DIR}`);
  console.log(`Generating ${Object.keys(ZIGGURAT_PROMPTS).length} concept images...\n`);

  const results: Record<string, string | null> = {};
  
  // Generate all images
  for (const [name, config] of Object.entries(ZIGGURAT_PROMPTS)) {
    results[name] = await generateImage(name, config.prompt, config.aspectRatio);
    
    // Small delay between generations to be nice to the API
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // Summary
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('📊 GENERATION SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  
  const successful = Object.entries(results).filter(([_, path]) => path !== null);
  const failed = Object.entries(results).filter(([_, path]) => path === null);
  
  console.log(`\n✅ Successful: ${successful.length}`);
  successful.forEach(([name, path]) => console.log(`   - ${name}: ${path}`));
  
  if (failed.length > 0) {
    console.log(`\n❌ Failed: ${failed.length}`);
    failed.forEach(([name]) => console.log(`   - ${name}`));
  }

  // Generate HTML gallery if we have images
  if (successful.length > 0) {
    await generateGalleryHTML(results);
  }
  
  console.log('\n🎉 Done!');
}

async function generateGalleryHTML(results: Record<string, string | null>) {
  const images = Object.entries(results)
    .filter(([_, path]) => path !== null)
    .map(([name, _]) => ({
      name,
      title: name.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
      path: `ziggurat-concepts/${name}.png`
    }));

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Ziggurat - Generated Concepts</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #fff;
            padding: 2rem;
        }
        h1 {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #fff, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.6);
            margin-bottom: 3rem;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            max-width: 1600px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1rem;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2);
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .card-info {
            padding: 1.5rem;
        }
        .card-info h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }
        .card-info a {
            color: #3B82F6;
            text-decoration: none;
            font-size: 0.9rem;
        }
        .meta {
            text-align: center;
            margin-top: 3rem;
            color: rgba(255,255,255,0.4);
            font-size: 0.9rem;
        }
        .meta a { color: #3B82F6; }
        .nav {
            text-align: center;
            margin-bottom: 2rem;
        }
        .nav a {
            color: #3B82F6;
            text-decoration: none;
            margin: 0 1rem;
        }
    </style>
</head>
<body>
    <div class="nav">
        <a href="PACKAGE-INDEX.html">← Package Index</a>
        <a href="ZIGGURAT-VISUAL-CONCEPTS.html">Visual Concepts</a>
        <a href="chet-holifield-federal-building-gallery.html">Historical Photos</a>
    </div>
    
    <h1>THE ZIGGURAT</h1>
    <p class="subtitle">AI-Generated Transformation Concepts</p>
    
    <div class="gallery">
        ${images.map(img => `
        <div class="card">
            <img src="${img.path}" alt="${img.title}" loading="lazy">
            <div class="card-info">
                <h3>${img.title}</h3>
                <a href="${img.path}" target="_blank">View Full Size →</a>
            </div>
        </div>
        `).join('')}
    </div>
    
    <p class="meta">
        Generated ${new Date().toLocaleDateString()} using Flux Schnell via Replicate<br>
        Part of the <a href="PACKAGE-INDEX.html">Laguna Ridge Acquisition</a> package
    </p>
</body>
</html>`;

  const galleryPath = path.join(process.cwd(), 'public', 'ZIGGURAT-GENERATED-GALLERY.html');
  fs.writeFileSync(galleryPath, html);
  console.log(`\n📄 Gallery HTML saved: ${galleryPath}`);
}

main().catch(console.error);
