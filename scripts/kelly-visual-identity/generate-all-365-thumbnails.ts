/**
 * KELLY THUMBNAIL GENERATOR - FULL YEAR (365 Days)
 * 
 * Generates all lesson thumbnails using FLUX Dev + Kelly LoRA
 * Runs continuously with rate limiting until complete
 * 
 * Usage: npx tsx scripts/kelly-visual-identity/generate-all-365-thumbnails.ts
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

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "thumbnails", "raw");

// === ENVIRONMENT TEMPLATES BY TOPIC KEYWORDS ===
const ENVIRONMENTS: Record<string, string> = {
  // NATURE / SCIENCE
  'water': 'ethereal environment with water elements, subtle mist and water droplets floating in air, blue and teal color palette, dreamy aquatic atmosphere',
  'cloud': 'standing among soft fluffy clouds, vast open sky environment, white and soft blue cloud formations, ethereal atmospheric setting',
  'rain': 'gentle rain environment with soft droplets, misty atmosphere, cool blue and silver tones, peaceful rain shower mood',
  'ocean': 'vast ocean horizon, deep blue waters meeting sky, gentle waves, maritime atmosphere, sense of depth and wonder',
  'ice': 'ethereal glacial landscape with ice formations and soft blue light, crisp cold atmosphere, majestic natural mood',
  'snow': 'peaceful winter wonderland, soft white snow covering landscape, gentle snowfall, serene and quiet atmosphere',
  'light': 'dramatic rays of light streaming through vast space, prismatic rainbow effects, deep blue and silver color palette',
  'star': 'vast cosmic environment with glowing stars and nebula, deep purple and blue space backdrop, luminous celestial bodies',
  'space': 'cosmic environment with galaxies and nebulae, deep purple and blue space, sense of infinite wonder',
  'moon': 'moonlit environment, soft silver light, night sky with visible moon, peaceful nocturnal atmosphere',
  'sun': 'golden sunlit environment, warm rays streaming through, golden hour atmosphere, radiant and hopeful',
  'planet': 'cosmic environment with planets visible, orbital rings and celestial bodies, sense of scale and wonder',
  'gravity': 'vast cosmic environment with curved space-like distortions, deep blue space, subtle orbital patterns',
  'sound': 'abstract environment with visible sound wave patterns, concentric ripples in air, deep blue and teal atmosphere',
  'music': 'abstract environment with flowing musical wave patterns, warm purple and magenta tones, sense of rhythm',
  'seed': 'lush botanical environment, soft green and earth tones, floating seeds in atmosphere, sense of growth',
  'plant': 'lush green botanical environment, warm sunlight streaming through leaves, dappled natural light',
  'tree': 'majestic forest environment with tall trees, dappled sunlight filtering through canopy, green and gold tones',
  'forest': 'enchanted forest setting, tall trees with sunlight filtering through, magical green atmosphere',
  'flower': 'beautiful flower garden environment, colorful blooms, soft natural lighting, fresh and vibrant',
  'soil': 'rich earthy environment with warm brown and ochre tones, subtle root patterns, organic textures',
  'earth': 'vast natural landscape, earth tones and greens, sense of groundedness and connection to nature',
  'mountain': 'majestic mountain landscape, dramatic peaks, golden hour lighting, sense of achievement and scale',
  'volcano': 'dramatic volcanic landscape, warm orange and red tones, powerful natural forces, dynamic environment',
  'rock': 'geological environment with interesting rock formations, earth tones, sense of permanence and time',
  'wind': 'dynamic environment with visible wind patterns, flowing elements, sense of movement and change',
  'fire': 'warm environment with soft fire glow, orange and amber tones, sense of warmth and transformation',
  'electricity': 'dynamic environment with subtle electrical patterns, blue and white energy, scientific wonder',
  'magnet': 'abstract environment with magnetic field visualizations, blues and silvers, scientific atmosphere',
  'atom': 'abstract microscopic environment, atomic structures and particles, scientific and wonder-filled',
  
  // BIOLOGY / BODY
  'brain': 'abstract environment with soft glowing network patterns, soft blue and purple neural pathways',
  'memory': 'abstract environment with soft glowing fragments and light trails, dreamy and structured',
  'dream': 'surreal dreamlike environment, soft purple and midnight blue, floating abstract shapes',
  'sleep': 'peaceful twilight environment, soft deep blue and purple tones, gentle stars appearing',
  'blood': 'abstract environment with flowing red and warm tones, organic patterns, sense of life force',
  'heart': 'warm environment with soft red and pink tones, sense of love and connection, heartbeat patterns',
  'bone': 'abstract anatomical environment, clean white and soft tones, scientific and educational',
  'cell': 'abstract microscopic environment, soft biological visualization, cells and organic structures',
  'body': 'abstract anatomical environment, soft educational tones, sense of wonder about human form',
  'breath': 'soft ethereal environment, flowing air patterns, light and airy atmosphere',
  'sense': 'abstract sensory environment, multiple subtle elements representing different senses',
  'eye': 'abstract environment with light and vision themes, soft rays and reflections',
  'ear': 'abstract environment with sound wave patterns, soft acoustic atmosphere',
  
  // EMOTIONS / SOCIAL
  'friend': 'warm abstract gradient background, soft magenta to purple tones, sense of connection',
  'kindness': 'soft rippling patterns emanating outward, warm magenta and pink gradient, sense of warmth',
  'listen': 'quiet contemplative atmosphere, soft magenta and purple gradient, sense of stillness',
  'love': 'warm glowing environment, soft pink and magenta tones, sense of warmth and connection',
  'happy': 'bright cheerful environment, warm golden light, uplifting and joyful atmosphere',
  'smile': 'warm bright environment with soft golden light, cheerful atmosphere, positive mood',
  'laugh': 'joyful bright environment, warm colors, sense of happiness and levity',
  'cry': 'soft blue and gentle environment, sense of release and emotion, peaceful sadness',
  'anger': 'dramatic environment transitioning from dark to light, sense of energy and resolution',
  'fear': 'environment transitioning from shadows to light, sense of courage emerging',
  'courage': 'dramatic environment with light breaking through darkness, warm golden light emerging',
  'brave': 'inspiring horizon with light breaking through clouds, sense of determination',
  'help': 'warm inviting community space with soft golden light, sense of connection',
  'share': 'warm collaborative environment, soft golden tones, sense of generosity',
  'trust': 'stable warm environment, soft earth tones, sense of reliability and safety',
  'family': 'warm home-like environment, soft golden lighting, sense of belonging',
  'home': 'cozy warm interior environment, soft amber lighting, sense of comfort and safety',
  
  // GROWTH / MINDSET
  'change': 'environment showing transformation, gradient shifting from cool to warm, metamorphosis mood',
  'grow': 'environment transitioning upward, sense of progress and development, warm hopeful tones',
  'learn': 'inspiring educational environment, soft light of discovery, sense of wonder',
  'mistake': 'environment transitioning from darker tones to warm golden light, hopeful and resilient',
  'patience': 'soft golden hour horizon, long path stretching into distance, peaceful anticipation',
  'gratitude': 'warm glowing atmosphere, soft golden and amber light, sense of inner radiance',
  'question': 'abstract environment with light particles rising upward, sense of discovery',
  'curious': 'bright inquisitive environment, soft light of wonder, open and exploratory',
  'think': 'contemplative environment, soft abstract patterns, sense of deep thought',
  'idea': 'bright inspiring environment, lightbulb-like glow, sense of innovation',
  'create': 'vibrant creative studio environment, colorful and energetic, artistic atmosphere',
  'art': 'artistic studio environment, colorful swatches and creative elements',
  'story': 'warm atmospheric environment with floating pages, amber and warm brown tones',
  'book': 'magical library environment with floating books, warm golden light',
  'write': 'contemplative writing environment, soft warm lighting, sense of expression',
  
  // TIME / PHILOSOPHY
  'time': 'abstract temporal environment, flowing patterns suggesting passage of time',
  'past': 'nostalgic environment with sepia tones, sense of memory and history',
  'future': 'forward-looking environment, bright hopeful horizon, sense of possibility',
  'begin': 'sunrise environment, golden hour lighting, sense of new beginning',
  'fresh': 'walking toward golden sunrise horizon, vast open landscape, morning mist',
  'end': 'sunset environment, warm golden tones, sense of completion and peace',
  'cycle': 'environment showing cyclical patterns, seasons blending, sense of continuity',
  'season': 'environment with seasonal elements, nature in transition, sense of change',
  
  // DEFAULT
  'default': 'clean minimal environment with soft abstract shapes, soft daylight, approachable atmosphere'
};

// === LOAD ALL 365 LESSONS ===
function loadLessons(): Array<{day: number; title: string; lesson_id: string}> {
  const calendarPath = path.join(process.cwd(), "lessons", "365_day_calendar.json");
  const data = JSON.parse(fs.readFileSync(calendarPath, 'utf-8'));
  return data.lessons.map((l: any) => ({
    day: l.day,
    title: l.title,
    lesson_id: l.lesson_id
  }));
}

// === GET ENVIRONMENT FOR LESSON ===
function getEnvironment(title: string): string {
  const lowerTitle = title.toLowerCase();
  
  // Check each keyword
  for (const [keyword, env] of Object.entries(ENVIRONMENTS)) {
    if (keyword !== 'default' && lowerTitle.includes(keyword)) {
      return env;
    }
  }
  
  // Additional compound checks
  if (lowerTitle.includes('grav')) return ENVIRONMENTS.gravity;
  if (lowerTitle.includes('electr')) return ENVIRONMENTS.electricity;
  if (lowerTitle.includes('magnet')) return ENVIRONMENTS.magnet;
  if (lowerTitle.includes('atom')) return ENVIRONMENTS.atom;
  if (lowerTitle.includes('energy')) return 'dynamic environment with energy patterns, flowing light, sense of transformation';
  if (lowerTitle.includes('move') || lowerTitle.includes('motion')) return 'dynamic environment with motion blur effects, sense of movement';
  if (lowerTitle.includes('color')) return 'vibrant environment with color spectrum, rainbow effects, artistic';
  if (lowerTitle.includes('pattern')) return 'abstract environment with geometric patterns, mathematical beauty';
  if (lowerTitle.includes('number') || lowerTitle.includes('math')) return 'abstract environment with mathematical patterns, clean and logical';
  if (lowerTitle.includes('word') || lowerTitle.includes('language')) return 'environment with floating text elements, literary atmosphere';
  if (lowerTitle.includes('animal')) return 'natural habitat environment, wildlife setting, connection to nature';
  if (lowerTitle.includes('bird')) return 'open sky environment, birds in flight, sense of freedom';
  if (lowerTitle.includes('fish')) return 'underwater environment, soft blue tones, aquatic atmosphere';
  if (lowerTitle.includes('insect') || lowerTitle.includes('bug')) return 'macro nature environment, detailed natural world';
  if (lowerTitle.includes('food') || lowerTitle.includes('eat')) return 'warm kitchen or dining environment, sense of nourishment';
  if (lowerTitle.includes('cook')) return 'cozy kitchen environment, warm and inviting';
  if (lowerTitle.includes('game') || lowerTitle.includes('play')) return 'playful environment, colorful and fun atmosphere';
  if (lowerTitle.includes('sport')) return 'athletic environment, dynamic and energetic';
  if (lowerTitle.includes('work')) return 'productive environment, sense of accomplishment';
  if (lowerTitle.includes('rest')) return 'peaceful restful environment, soft calming tones';
  if (lowerTitle.includes('quiet') || lowerTitle.includes('silence')) return 'serene quiet environment, minimal and peaceful';
  if (lowerTitle.includes('loud') || lowerTitle.includes('noise')) return 'dynamic environment with sound waves, energetic';
  if (lowerTitle.includes('dark')) return 'environment with dramatic lighting, shadows and light contrast';
  if (lowerTitle.includes('bright')) return 'brightly lit environment, radiant and hopeful';
  if (lowerTitle.includes('cold')) return 'cool blue environment, crisp and refreshing';
  if (lowerTitle.includes('warm') || lowerTitle.includes('hot')) return 'warm environment, orange and amber tones';
  if (lowerTitle.includes('wet')) return 'environment with water elements, soft blue tones';
  if (lowerTitle.includes('dry')) return 'arid environment, warm earth tones, desert-like';
  
  return ENVIRONMENTS.default;
}

// === GENERATE SINGLE THUMBNAIL ===
async function generateThumbnail(lesson: {day: number; title: string; lesson_id: string}, retries = 5): Promise<boolean> {
  const dayStr = String(lesson.day).padStart(3, '0');
  const filename = `lesson-${dayStr}-${lesson.lesson_id}.png`;
  const filepath = path.join(OUTPUT_DIR, filename);
  
  // Skip if already exists
  if (fs.existsSync(filepath)) {
    console.log(`⏭️  [Day ${lesson.day}] Skipping "${lesson.title}" (exists)`);
    return true;
  }
  
  const environment = getEnvironment(lesson.title);
  const fullPrompt = `${KELLY_ANCHOR}, ${environment}, ${STYLE_LOCKS}`;
  
  console.log(`\n🎨 [Day ${lesson.day}] Generating: "${lesson.title}"`);
  console.log(`   Environment: ${environment.substring(0, 60)}...`);
  
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
      if (!response.ok) throw new Error(`Download failed: ${response.status}`);
      
      const buffer = Buffer.from(await response.arrayBuffer());
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
      fs.writeFileSync(filepath, buffer);
      
      console.log(`   ✅ Saved (attempt ${attempt})`);
      return true;
      
    } catch (error: any) {
      const msg = error.message || '';
      
      // Handle rate limiting
      if (msg.includes('429') || msg.includes('rate limit') || msg.includes('throttled')) {
        // Extract wait time from error message
        const match = msg.match(/resets in ~(\d+)s/);
        const waitTime = match ? parseInt(match[1]) + 2 : 15;
        console.log(`   ⏳ Rate limited, waiting ${waitTime}s... (attempt ${attempt}/${retries})`);
        await new Promise(r => setTimeout(r, waitTime * 1000));
        continue;
      }
      
      // Handle insufficient credits
      if (msg.includes('credit') || msg.includes('payment') || msg.includes('billing')) {
        console.log(`   💳 Insufficient credits! Please add funds to Replicate.`);
        console.log(`   Visit: https://replicate.com/account/billing`);
        // Wait 5 minutes and retry (user might add credits)
        console.log(`   Waiting 5 minutes before retry...`);
        await new Promise(r => setTimeout(r, 5 * 60 * 1000));
        continue;
      }
      
      console.log(`   ❌ Error (attempt ${attempt}/${retries}): ${msg.substring(0, 100)}`);
      
      if (attempt < retries) {
        const backoff = Math.min(attempt * 5, 30); // Exponential backoff, max 30s
        await new Promise(r => setTimeout(r, backoff * 1000));
      }
    }
  }
  
  return false;
}

// === MAIN EXECUTION ===
async function main() {
  console.log("═".repeat(70));
  console.log("🖼️  KELLY THUMBNAIL GENERATOR - FULL 365 DAYS");
  console.log("═".repeat(70));
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log(`LoRA: ${LORA_URL}`);
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    console.error("Set it in .env.local or .env");
    process.exit(1);
  }
  
  // Load all lessons
  const lessons = loadLessons();
  console.log(`📚 Loaded ${lessons.length} lessons`);
  
  // Check existing
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const existing = fs.readdirSync(OUTPUT_DIR).filter(f => f.endsWith('.png'));
  console.log(`📁 Found ${existing.length} existing thumbnails`);
  console.log("");
  
  // Rate limiting: 6 requests/min with low credit
  // To be safe, wait 12 seconds between requests
  const DELAY_MS = 12000;
  
  let generated = 0;
  let skipped = 0;
  let failed = 0;
  const startTime = Date.now();
  
  for (let i = 0; i < lessons.length; i++) {
    const lesson = lessons[i];
    const progress = `[${i + 1}/${lessons.length}]`;
    
    const elapsed = (Date.now() - startTime) / 1000 / 60;
    const rate = generated / Math.max(elapsed, 1);
    const remaining = lessons.length - i;
    const eta = remaining / Math.max(rate, 0.1);
    
    console.log(`\n${progress} ETA: ${eta.toFixed(0)}min | Generated: ${generated} | Skipped: ${skipped} | Failed: ${failed}`);
    
    const dayStr = String(lesson.day).padStart(3, '0');
    const filename = `lesson-${dayStr}-${lesson.lesson_id}.png`;
    const filepath = path.join(OUTPUT_DIR, filename);
    
    if (fs.existsSync(filepath)) {
      skipped++;
      console.log(`⏭️  [Day ${lesson.day}] Already exists: "${lesson.title}"`);
      continue;
    }
    
    const success = await generateThumbnail(lesson);
    
    if (success) {
      generated++;
    } else {
      failed++;
      // Log failed lesson for retry later
      fs.appendFileSync(
        path.join(OUTPUT_DIR, 'failed.txt'),
        `${lesson.day}|${lesson.lesson_id}|${lesson.title}\n`
      );
    }
    
    // Wait between requests (unless this is the last one)
    if (i < lessons.length - 1) {
      console.log(`   ⏱️  Waiting ${DELAY_MS/1000}s...`);
      await new Promise(r => setTimeout(r, DELAY_MS));
    }
  }
  
  // Final summary
  const totalTime = (Date.now() - startTime) / 1000 / 60;
  console.log("\n" + "═".repeat(70));
  console.log("✨ GENERATION COMPLETE");
  console.log("═".repeat(70));
  console.log(`Total time: ${totalTime.toFixed(1)} minutes`);
  console.log(`Generated: ${generated}`);
  console.log(`Skipped (existing): ${skipped}`);
  console.log(`Failed: ${failed}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  
  if (failed > 0) {
    console.log(`\n⚠️  ${failed} images failed. Check ${OUTPUT_DIR}/failed.txt`);
    console.log("Re-run this script to retry failed images.");
  }
}

// Run
main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});



