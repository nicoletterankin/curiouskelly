/**
 * Kelly Thumbnail Generator - January Pilot (Full Batch)
 * 
 * Implements the "Curious Kelly Thumbnail Generation System"
 * Lessons 001-031
 * 
 * UPDATE: Added "full body shot including feet" to Style Locks to ensure grounding.
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

// ADDED: "full body shot including feet, wide shot" to prevent cropping
const STYLE_LOCKS = `full body shot including feet, wide shot, photorealistic editorial photography, clean composition, cinematic color grading, soft shadows, 8k, professional photography, 16:9 aspect ratio`;

const NEGATIVE_PROMPT = `cropped feet, cut off feet, close up, cartoon, illustration, anime, painting, drawing, sketch, blurry, low quality, watermark, text, logo, extra limbs, extra fingers, deformed, distorted face, wrong outfit, different clothes, holding objects, sitting, standing still, bad anatomy, distorted body`;

// === JANUARY LESSONS (001-031) ===

const LESSONS = [
  // ALREADY GENERATED: 001 (Skipping to save time/cost if needed, but re-running for consistent feet framing is safer)
  {
    id: "001",
    slug: "starting-fresh",
    prompt_middle: `walking toward golden sunrise horizon, vast open landscape, morning mist in distance, warm golden hour lighting, sense of new beginning and possibility, soft orange and gold gradient sky`
  },
  {
    id: "002",
    slug: "the-three-lives-of-water",
    prompt_middle: `ethereal environment with water elements, subtle mist and water droplets floating in air, ice crystals and steam wisps in background, blue and teal color palette, dreamy aquatic atmosphere, soft diffused lighting`
  },
  {
    id: "003",
    slug: "where-clouds-come-from",
    prompt_middle: `standing among soft fluffy clouds, vast open sky environment, white and soft blue cloud formations surrounding her, ethereal atmospheric setting, gentle sunlight filtering through clouds, airy and light mood`
  },
  {
    id: "004",
    slug: "how-light-travels",
    prompt_middle: `dramatic rays of light streaming through vast dark space, subtle prismatic rainbow light effects in background, beams of light cutting through atmospheric haze, deep blue and silver color palette with light accents, ethereal and scientific mood`
  },
  {
    id: "005",
    slug: "how-sound-moves",
    prompt_middle: `abstract environment with visible sound wave patterns, concentric circular ripples in the air, deep blue and teal atmosphere, subtle particle effects suggesting vibration, rhythmic visual patterns in background, scientific and ethereal mood`
  },
  {
    id: "006",
    slug: "whats-inside-a-seed",
    prompt_middle: `lush botanical environment, soft green and earth tone palette, subtle floating seeds and seedlings in atmosphere, warm natural daylight, sense of growth and potential, organic textures in background, fresh and alive mood`
  },
  {
    id: "007",
    slug: "what-stars-are-made-of",
    prompt_middle: `vast cosmic environment with glowing stars and nebula, deep purple and blue space backdrop, subtle stardust and glowing particles, luminous celestial bodies in distance, sense of infinite space, scientific wonder mood`
  },
  {
    id: "008",
    slug: "what-makes-a-real-friend",
    prompt_middle: `warm abstract gradient background, soft magenta to purple tones, gentle glowing orbs of light suggesting connection, warm and embracing atmosphere, minimal and emotionally resonant, soft diffused lighting`
  },
  {
    id: "009",
    slug: "how-kindness-spreads",
    prompt_middle: `soft rippling circular patterns emanating outward in background, warm magenta and pink gradient atmosphere, gentle concentric rings of light, sense of expanding warmth and connection, abstract and emotionally warm`
  },
  {
    id: "010",
    slug: "the-art-of-really-listening",
    prompt_middle: `looking ahead with attentive expression, soft abstract magenta and purple gradient background, subtle sound wave patterns fading gently, quiet contemplative atmosphere, sense of stillness and attention, warm and focused mood`
  },
  {
    id: "011",
    slug: "why-patience-pays-off",
    prompt_middle: `soft golden hour horizon environment, long path stretching into warm distance, sense of journey and gradual progress, orange and gold color palette, peaceful anticipation mood, open landscape with warm light`
  },
  {
    id: "012",
    slug: "how-gratitude-changes-you",
    prompt_middle: `looking ahead with gentle smile, warm glowing atmosphere, soft golden and amber light surrounding her, sense of inner radiance, abstract warmth emanating outward, peaceful and grateful mood`
  },
  {
    id: "013",
    slug: "why-we-dream",
    prompt_middle: `surreal dreamlike environment, soft purple and midnight blue atmosphere, floating abstract shapes and gentle clouds, ethereal and mysterious mood, sense of subconscious exploration, soft diffused lighting`
  },
  {
    id: "014",
    slug: "the-power-of-questions",
    prompt_middle: `looking ahead with curious expression, abstract environment with subtle light particles rising upward, sense of discovery and wonder, soft blue and silver tones, open and inquisitive atmosphere`
  },
  {
    id: "015",
    slug: "how-your-brain-learns",
    prompt_middle: `abstract environment with subtle glowing network patterns, soft blue and purple tones suggesting neural pathways, gentle points of light connected by soft lines, scientific and wonder-filled mood`
  },
  {
    id: "016",
    slug: "what-makes-music-feel",
    prompt_middle: `abstract environment with flowing wave patterns suggesting music, warm purple and magenta tones, sense of rhythm and movement in background, emotional and artistic atmosphere`
  },
  {
    id: "017",
    slug: "why-mistakes-matter",
    prompt_middle: `looking ahead with determined expression, environment transitioning from darker tones to warm golden light, sense of overcoming and growth, dawn breaking through clouds, hopeful and resilient mood`
  },
  {
    id: "018",
    slug: "how-plants-eat-sunlight",
    prompt_middle: `lush green botanical environment, warm sunlight streaming through leaves, soft golden and green color palette, sense of energy and life, dappled natural light, fresh and vibrant mood`
  },
  {
    id: "019",
    slug: "the-story-in-your-blood",
    prompt_middle: `abstract environment with flowing red and warm tones, subtle organic patterns suggesting flow and life, deep crimson and warm amber colors, sense of life force and connection, scientific yet warm mood`
  },
  {
    id: "020",
    slug: "why-we-need-sleep",
    prompt_middle: `peaceful twilight environment, soft deep blue and purple tones, gentle stars beginning to appear, calm and restful atmosphere, sense of peaceful transition to night, serene mood`
  },
  {
    id: "021",
    slug: "how-to-disagree-well",
    prompt_middle: `looking ahead with thoughtful expression, abstract environment with two complementary color gradients meeting harmoniously, soft purple and teal tones balancing, sense of respectful tension and resolution, calm and mature mood`
  },
  {
    id: "022",
    slug: "what-gravity-really-does",
    prompt_middle: `vast cosmic environment with curved space-like distortions, deep blue and purple space backdrop, subtle orbital patterns and celestial bodies, sense of invisible forces at work, scientific wonder mood`
  },
  {
    id: "023",
    slug: "the-gift-of-boredom",
    prompt_middle: `looking ahead with contemplative expression, minimal open environment with vast empty space, soft neutral tones with subtle warmth emerging, sense of possibility in emptiness, quiet contemplative mood, space for imagination`
  },
  {
    id: "024",
    slug: "how-memory-works",
    prompt_middle: `abstract environment with soft glowing fragments and light trails, sense of pieces connecting and floating, soft blue and lavender tones, dreamy yet structured atmosphere`
  },
  {
    id: "025",
    slug: "why-stories-change-us",
    prompt_middle: `warm atmospheric environment with soft pages or abstract text-like patterns floating gently, amber and warm brown tones, sense of narrative and journey, storytelling mood`
  },
  {
    id: "026",
    slug: "the-science-of-smiling",
    prompt_middle: `looking ahead with natural smile, warm bright environment with soft golden light, cheerful and uplifting atmosphere, soft yellow and warm tones, sense of genuine happiness, bright positive mood`
  },
  {
    id: "027",
    slug: "how-ice-shapes-land",
    prompt_middle: `vast icy landscape environment, cool blue and white tones, subtle glacial formations in background, sense of slow powerful transformation, crisp cold atmosphere, majestic natural mood`
  },
  {
    id: "028",
    slug: "what-courage-really-means",
    prompt_middle: `looking ahead with determined confident expression, dramatic environment with light breaking through darkness, warm golden light emerging from deeper shadows, sense of moving forward despite uncertainty, brave and inspiring mood`
  },
  {
    id: "029",
    slug: "the-hidden-life-of-soil",
    prompt_middle: `rich earthy environment with warm brown and ochre tones, subtle root patterns and organic textures in background, sense of hidden complexity beneath surface, grounded and organic mood`
  },
  {
    id: "030",
    slug: "why-we-help-strangers",
    prompt_middle: `looking ahead with warm open expression, warm abstract environment with soft interconnected light points, sense of invisible bonds between people, soft magenta and warm amber tones, compassionate and hopeful mood`
  },
  {
    id: "031",
    slug: "how-change-happens",
    prompt_middle: `environment showing subtle transformation from one state to another, gradient shifting from cool to warm tones, sense of gradual but inevitable change, abstract metamorphosis mood, hopeful and dynamic`
  }
];

// === GENERATION LOGIC ===

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "thumbnails", "raw");

async function generateThumbnail(lesson: { id: string; slug: string; prompt_middle: string }, retries = 3): Promise<boolean> {
  const filename = `lesson-${lesson.id}-${lesson.slug}.png`;
  const filepath = path.join(OUTPUT_DIR, filename);
  
  // Skip if already exists
  if (fs.existsSync(filepath)) {
    console.log(`\n⏭️  Skipping ${filename} (already exists)`);
    return true;
  }
  
  // Prompt construction: Anchor + Scene + Style Locks + Negative
  const fullPrompt = `${KELLY_ANCHOR}, ${lesson.prompt_middle}, ${STYLE_LOCKS}`;
  
  console.log(`\n🎨 Generating: ${filename}`);
  
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
            output_format: "png",
            guidance_scale: 3.5,
            output_quality: 100,
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
        const waitTime = 15; // Wait 15 seconds on rate limit
        console.log(`   ⏳ Rate limited, waiting ${waitTime}s... (attempt ${attempt}/${retries})`);
        await new Promise(r => setTimeout(r, waitTime * 1000));
        continue;
      }
      
      console.log(`   ❌ Error (attempt ${attempt}/${retries}): ${msg.substring(0, 100)}`);
      
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 5000)); // Wait 5s before retry
      }
    }
  }
  
  return false;
}

async function main() {
  console.log("🖼️ KELLY THUMBNAIL BATCH - JANUARY (001-031)");
  console.log("=".repeat(60));
  console.log("Pose: Walking (Full Body including Feet)");
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log("");
  
  let success = 0;
  
  // Process sequentially with rate limit awareness
  // With low credit: 6 requests/min = 10s delay minimum
  const DELAY_MS = 12000; // 12 seconds between requests (safe for low credit accounts)
  
  for (let i = 0; i < LESSONS.length; i++) {
    const lesson = LESSONS[i];
    console.log(`\n[${i + 1}/${LESSONS.length}]`);
    if (await generateThumbnail(lesson)) success++;
    
    // Don't delay after last item
    if (i < LESSONS.length - 1) {
      console.log(`   ⏱️  Waiting ${DELAY_MS/1000}s before next...`);
      await new Promise(r => setTimeout(r, DELAY_MS));
    }
  }
  
  console.log("\n" + "=".repeat(60));
  console.log(`✅ Batch Complete: ${success}/${LESSONS.length}`);
  
  try {
    const { execSync } = require("child_process");
    execSync(`explorer "${OUTPUT_DIR}"`);
  } catch (e) {}
}

main().catch(console.error);



