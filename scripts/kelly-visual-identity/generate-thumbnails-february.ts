/**
 * Kelly Thumbnail Generator - February (Days 32-59)
 * 
 * Implements the "Curious Kelly Thumbnail Generation System"
 * Lessons 032-059
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

// === STYLE BIBLE CONSTANTS ===

const KELLY_ANCHOR = `kelly, young woman walking in profile view, long brown wavy hair, light blue crewneck sweater, blue jeans cuffed at ankles, white sneakers, mid-stride walking pose, natural arm movement, looking ahead, soft natural lighting on figure, subtle ground shadow`;

const STYLE_LOCKS = `full body shot including feet, wide shot, photorealistic editorial photography, clean composition, cinematic color grading, soft shadows, 8k, professional photography, 16:9 aspect ratio`;

const NEGATIVE_PROMPT = `cropped feet, cut off feet, close up, cartoon, illustration, anime, painting, drawing, sketch, blurry, low quality, watermark, text, logo, extra limbs, extra fingers, deformed, distorted face, wrong outfit, different clothes, holding objects, sitting, standing still, bad anatomy, distorted body`;

// === FEBRUARY LESSONS (032-059) ===

const LESSONS = [
  {
    id: "032",
    slug: "the-moon-and-the-tides",
    prompt_middle: `walking along moonlit beach at night, gentle ocean waves in background, large luminous moon reflecting on water, cool blue and silver color palette, tidal patterns visible on sand, serene nocturnal coastal mood`
  },
  {
    id: "033",
    slug: "what-gravity-actually-does",
    prompt_middle: `floating objects subtly suspended around her, abstract visualization of gravitational pull, deep blue space-like environment, gentle curved lines suggesting orbital paths, weightless ethereal atmosphere`
  },
  {
    id: "034",
    slug: "how-magnets-work",
    prompt_middle: `abstract magnetic field lines flowing around her, metallic particles attracted in graceful arcs, deep blue and silver color palette, invisible forces made visible, scientific wonder atmosphere`
  },
  {
    id: "035",
    slug: "how-electricity-flows",
    prompt_middle: `subtle electric blue energy patterns flowing in background, glowing circuit-like pathways, dynamic energy visualization, cool blue and electric cyan palette, powerful yet controlled energy mood`
  },
  {
    id: "036",
    slug: "what-fire-really-is",
    prompt_middle: `warm campfire glow illuminating the scene, dancing flame reflections, warm orange and amber color palette, cozy firelight atmosphere, sense of primal discovery and warmth`
  },
  {
    id: "037",
    slug: "why-ice-floats",
    prompt_middle: `crystalline ice formations and floating icebergs, cool arctic blue environment, water surface with ice chunks, crisp cold atmosphere, beautiful frozen landscape, scientific wonder`
  },
  {
    id: "038",
    slug: "what-makes-wind-blow",
    prompt_middle: `dynamic wind patterns visible as flowing lines, leaves and particles carried by breeze, open landscape with moving air currents, fresh blue and green palette, sense of invisible force in motion`
  },
  {
    id: "039",
    slug: "where-rain-comes-from",
    prompt_middle: `gentle rain falling around her, dramatic clouds in sky, water droplets suspended in air, grey and silver atmosphere with subtle blue undertones, cleansing rain mood, fresh and alive`
  },
  {
    id: "040",
    slug: "what-causes-thunder",
    prompt_middle: `dramatic stormy sky with lightning in distance, dark dramatic clouds, electric energy in atmosphere, deep purple and electric blue palette, powerful natural forces, awe-inspiring storm`
  },
  {
    id: "041",
    slug: "how-rainbows-form",
    prompt_middle: `brilliant rainbow arcing across sky after rain, soft sunlight breaking through clouds, prismatic light effects, full spectrum of colors, magical atmospheric moment, wonder and joy`
  },
  {
    id: "042",
    slug: "why-seasons-change",
    prompt_middle: `environment transitioning between seasons, autumn leaves and spring blossoms together, warm and cool colors blending, sense of cyclical transformation, natural rhythm of change`
  },
  {
    id: "043",
    slug: "why-we-have-day-and-night",
    prompt_middle: `split environment showing day and night, half sun half moon sky, dramatic transition between light and dark, cosmic perspective, earth rotation visualization, beautiful duality`
  },
  {
    id: "044",
    slug: "how-shadows-work",
    prompt_middle: `dramatic long shadows cast by afternoon sun, interplay of light and dark, geometric shadow patterns, warm golden hour lighting, playful exploration of shadow and light`
  },
  {
    id: "045",
    slug: "why-mirrors-reflect",
    prompt_middle: `reflective surfaces creating mirror images, subtle light bouncing effects, silver and crystal elements, sense of reflection and symmetry, clean and bright atmosphere`
  },
  {
    id: "046",
    slug: "how-sound-bounces-back",
    prompt_middle: `canyon or cave environment with echo visualization, sound waves bouncing off walls, concentric circles representing echoes, earth tones with blue sound wave effects, acoustic wonder`
  },
  {
    id: "047",
    slug: "how-waves-carry-energy",
    prompt_middle: `ocean waves in dynamic motion, energy patterns visible in water movement, deep blue and teal ocean colors, powerful yet rhythmic wave patterns, sense of energy transfer`
  },
  {
    id: "048",
    slug: "the-science-of-bubbles",
    prompt_middle: `iridescent soap bubbles floating all around, rainbow reflections on bubble surfaces, soft dreamy atmosphere, delicate and magical, playful scientific wonder, light and airy mood`
  },
  {
    id: "049",
    slug: "how-crystals-form",
    prompt_middle: `crystalline formations growing in geometric patterns, purple amethyst and clear quartz elements, deep cave or geode environment, sparkle and refraction, natural geometric wonder`
  },
  {
    id: "050",
    slug: "stories-trapped-in-stone",
    prompt_middle: `ancient fossils visible in rock formations, layered geological strata, amber and earth tones, sense of deep time and preserved history, paleontology wonder`
  },
  {
    id: "051",
    slug: "when-dinosaurs-ruled",
    prompt_middle: `prehistoric landscape with lush ancient vegetation, volcanic mountains in distance, warm amber and green prehistoric atmosphere, sense of ancient earth, dinosaur era wonder`
  },
  {
    id: "052",
    slug: "whats-inside-a-volcano",
    prompt_middle: `dramatic volcanic landscape with glowing lava, orange and red molten rock, smoke and steam rising, powerful geological forces, intense heat visualization, primal earth energy`
  },
  {
    id: "053",
    slug: "why-the-ground-shakes",
    prompt_middle: `cracked earth surface showing tectonic stress, seismic wave patterns visualized, dramatic geological environment, earth tones with energy lines, powerful underground forces`
  },
  {
    id: "054",
    slug: "how-mountains-are-made",
    prompt_middle: `majestic mountain range rising dramatically, layers of rock showing formation, snow-capped peaks, grand scale landscape, sense of geological time and upward force, awe-inspiring heights`
  },
  {
    id: "055",
    slug: "the-deep-ocean-mystery",
    prompt_middle: `deep ocean environment with bioluminescent creatures, dark blue depths with glowing life forms, mysterious underwater atmosphere, sense of unknown depths, ocean exploration wonder`
  },
  {
    id: "056",
    slug: "how-rivers-shape-the-land",
    prompt_middle: `winding river cutting through landscape, canyon formation visible, flowing water carving rock, aerial perspective of river patterns, blue water against earth tones, erosion beauty`
  },
  {
    id: "057",
    slug: "where-lakes-come-from",
    prompt_middle: `pristine mountain lake with crystal clear water, reflective surface showing sky and mountains, serene blue and green palette, peaceful natural sanctuary, geological formation beauty`
  },
  {
    id: "058",
    slug: "life-in-the-desert",
    prompt_middle: `dramatic desert landscape with sand dunes, warm orange and gold sunset colors, resilient desert plants, vast open space, sense of adaptation and survival, beautiful harsh environment`
  },
  {
    id: "059",
    slug: "the-secret-life-of-forests",
    prompt_middle: `lush forest interior with dappled sunlight, rich green foliage layers, magical forest atmosphere, life everywhere visible, interconnected ecosystem, wonder of biodiversity`
  }
];

// === GENERATION LOGIC ===

const OUTPUT_DIR = path.join(process.cwd(), "public", "assets", "kelly", "production", "thumbnails", "february");

async function generateThumbnail(lesson: { id: string; slug: string; prompt_middle: string }, retries = 3): Promise<boolean> {
  const filename = `lesson-${parseInt(lesson.id)}.webp`;
  const filepath = path.join(OUTPUT_DIR, filename);
  
  // Skip if already exists
  if (fs.existsSync(filepath)) {
    console.log(`\n⏭️  Skipping ${filename} (already exists)`);
    return true;
  }
  
  // Prompt construction: Anchor + Scene + Style Locks
  const fullPrompt = `${KELLY_ANCHOR}, ${lesson.prompt_middle}, ${STYLE_LOCKS}`;
  
  console.log(`\n🎨 Generating: ${filename} (${lesson.slug})`);
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const output = await replicate.run(
        "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
        {
          input: {
            prompt: fullPrompt,
            hf_lora: LORA_URL,
            lora_scale: 0.95, 
            num_outputs: 1,
            aspect_ratio: "16:9",
            output_format: "webp",
            guidance_scale: 3.5,
            output_quality: 90,
            num_inference_steps: 30,
            extra_lora_scale: 0.8 
          }
        }
      ) as any;
      
      const imageUrl = Array.isArray(output) ? output[0] : output;
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error(`Download failed`);
      
      const buffer = Buffer.from(await response.arrayBuffer());
      
      // Ensure directory exists
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
      fs.writeFileSync(filepath, buffer);
      
      console.log(`   ✅ Saved (attempt ${attempt})`);
      return true;
      
    } catch (error: any) {
      const msg = error.message || '';
      
      // Handle rate limiting
      if (msg.includes('429') || msg.includes('rate limit') || msg.includes('throttled')) {
        const waitTime = 15;
        console.log(`   ⏳ Rate limited, waiting ${waitTime}s... (attempt ${attempt}/${retries})`);
        await new Promise(r => setTimeout(r, waitTime * 1000));
        continue;
      }
      
      console.log(`   ❌ Error (attempt ${attempt}/${retries}): ${msg.substring(0, 100)}`);
      
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 5000));
      }
    }
  }
  
  return false;
}

async function main() {
  const args = process.argv.slice(2);
  const isDryRun = args.includes('--dry-run');
  
  console.log("🖼️ KELLY THUMBNAIL BATCH - FEBRUARY (032-059)");
  console.log("=".repeat(60));
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log(`Lessons: ${LESSONS.length}`);
  console.log(`Estimated cost: $${(LESSONS.length * 0.04).toFixed(2)}`);
  console.log(`Mode: ${isDryRun ? '🔍 DRY RUN' : '🚀 LIVE'}`);
  console.log("");
  
  if (isDryRun) {
    console.log("📋 Would generate these files:");
    console.log("─".repeat(50));
    for (const lesson of LESSONS) {
      const filename = `lesson-${parseInt(lesson.id)}.webp`;
      const manifestKey = parseInt(lesson.id);
      console.log(`  Day ${manifestKey}: ${filename}`);
      console.log(`    → february/${filename}`);
    }
    console.log("─".repeat(50));
    console.log(`\n✅ Dry run complete. ${LESSONS.length} files would be generated.`);
    console.log(`📝 Manifest entries needed: Days 32-59`);
    return;
  }
  
  // Ensure output directory exists
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  let success = 0;
  const DELAY_MS = 12000; // 12 seconds between requests
  
  for (let i = 0; i < LESSONS.length; i++) {
    const lesson = LESSONS[i];
    console.log(`\n[${i + 1}/${LESSONS.length}]`);
    if (await generateThumbnail(lesson)) success++;
    
    if (i < LESSONS.length - 1) {
      console.log(`   ⏱️  Waiting ${DELAY_MS/1000}s before next...`);
      await new Promise(r => setTimeout(r, DELAY_MS));
    }
  }
  
  console.log("\n" + "=".repeat(60));
  console.log(`✅ Batch Complete: ${success}/${LESSONS.length}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
}

main().catch(console.error);










