/**
 * GENERATE MISSING THUMBNAILS
 * 
 * Wrapper script that focuses on generating only missing thumbnails.
 * Uses the FLUX Dev + Kelly LoRA pipeline with smart rate limiting.
 * 
 * Usage: 
 *   npx tsx scripts/generate-missing-thumbnails.ts              # Generate all missing
 *   npx tsx scripts/generate-missing-thumbnails.ts --month 2    # Generate February only
 *   npx tsx scripts/generate-missing-thumbnails.ts --day 45     # Generate single day
 * 
 * Requirements:
 *   - REPLICATE_API_TOKEN in .env
 *   - Sufficient Replicate credits (~$0.10 per image)
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";
import { createClient } from "@supabase/supabase-js";

// === CONFIG ===
const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

const OUTPUT_DIRS = {
  raw: path.join(process.cwd(), "public", "kelly", "thumbnails", "raw"),
  production: path.join(process.cwd(), "public", "assets", "kelly", "production", "thumbnails")
};

const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 
                'july', 'august', 'september', 'october', 'november', 'december'];
const DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

// === KELLY STYLE BIBLE ===
const KELLY_ANCHOR = `kelly, young woman walking in profile view, long brown wavy hair, light blue crewneck sweater, blue jeans cuffed at ankles, white sneakers, mid-stride walking pose, natural arm movement, looking ahead, soft natural lighting on figure, subtle ground shadow`;

const STYLE_LOCKS = `full body shot including feet, wide shot, photorealistic editorial photography, clean composition, cinematic color grading, soft shadows, 8k, professional photography, 16:9 aspect ratio`;

// Environment mapping (simplified - see full version in generate-all-365-thumbnails.ts)
const ENVIRONMENTS: Record<string, string> = {
  'water': 'ethereal environment with water elements, subtle mist and water droplets, blue and teal palette',
  'cloud': 'standing among soft fluffy clouds, vast open sky, white and soft blue',
  'light': 'dramatic rays of light streaming through space, prismatic rainbow effects',
  'star': 'vast cosmic environment with glowing stars and nebula, deep purple and blue',
  'brain': 'abstract environment with soft glowing network patterns, neural pathways',
  'friend': 'warm abstract gradient background, soft magenta to purple tones',
  'learn': 'inspiring educational environment, soft light of discovery',
  'default': 'clean minimal environment with soft abstract shapes, soft daylight'
};

interface Lesson {
  day: number;
  title: string;
  lesson_id: string;
  topic?: string;
}

// === HELPER FUNCTIONS ===

function getMonthFromDay(day: number): { month: string; dayInMonth: number; monthIndex: number } {
  let remaining = day;
  for (let i = 0; i < MONTHS.length; i++) {
    if (remaining <= DAYS_PER_MONTH[i]) {
      return { month: MONTHS[i], dayInMonth: remaining, monthIndex: i };
    }
    remaining -= DAYS_PER_MONTH[i];
  }
  return { month: 'december', dayInMonth: 31, monthIndex: 11 };
}

function thumbnailExists(day: number): boolean {
  const { month, dayInMonth } = getMonthFromDay(day);
  
  // Check production webp
  const webpPath = path.join(OUTPUT_DIRS.production, month, `lesson-${dayInMonth}.webp`);
  if (fs.existsSync(webpPath)) return true;
  
  // Check raw png (any matching file)
  if (fs.existsSync(OUTPUT_DIRS.raw)) {
    const dayStr = String(day).padStart(3, '0');
    const files = fs.readdirSync(OUTPUT_DIRS.raw);
    if (files.some(f => f.startsWith(`lesson-${dayStr}-`))) return true;
  }
  
  return false;
}

function getEnvironment(title: string): string {
  const lower = title.toLowerCase();
  for (const [key, env] of Object.entries(ENVIRONMENTS)) {
    if (key !== 'default' && lower.includes(key)) return env;
  }
  return ENVIRONMENTS.default;
}

async function loadLessons(): Promise<Lesson[]> {
  // Try Supabase first
  if (SUPABASE_URL && SUPABASE_KEY) {
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
    const { data, error } = await supabase
      .from('core_lessons')
      .select('day_number, topic')
      .order('day_number');
    
    if (!error && data) {
      return data.map(d => ({
        day: d.day_number,
        title: d.topic,
        lesson_id: d.topic?.toLowerCase().replace(/[^a-z0-9]+/g, '-') || `day-${d.day_number}`,
        topic: d.topic
      }));
    }
  }
  
  // Fall back to local JSON
  const calendarPath = path.join(process.cwd(), "lessons", "365_day_calendar.json");
  if (fs.existsSync(calendarPath)) {
    const data = JSON.parse(fs.readFileSync(calendarPath, 'utf-8'));
    return data.lessons.map((l: any) => ({
      day: l.day,
      title: l.title,
      lesson_id: l.lesson_id,
      topic: l.title
    }));
  }
  
  throw new Error("Could not load lessons from Supabase or local file");
}

async function generateThumbnail(
  replicate: Replicate, 
  lesson: Lesson, 
  retries = 3
): Promise<{ success: boolean; url?: string }> {
  const dayStr = String(lesson.day).padStart(3, '0');
  const filename = `lesson-${dayStr}-${lesson.lesson_id}.png`;
  const filepath = path.join(OUTPUT_DIRS.raw, filename);
  
  // Skip if exists
  if (fs.existsSync(filepath)) {
    return { success: true, url: filepath };
  }
  
  const environment = getEnvironment(lesson.title);
  const prompt = `${KELLY_ANCHOR}, ${environment}, ${STYLE_LOCKS}`;
  
  console.log(`🎨 Day ${lesson.day}: "${lesson.title}"`);
  console.log(`   Env: ${environment.substring(0, 50)}...`);
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const output = await replicate.run(
        "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
        {
          input: {
            prompt,
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
      fs.mkdirSync(OUTPUT_DIRS.raw, { recursive: true });
      fs.writeFileSync(filepath, buffer);
      
      console.log(`   ✅ Saved`);
      return { success: true, url: filepath };
      
    } catch (error: any) {
      const msg = error.message || '';
      
      if (msg.includes('429') || msg.includes('rate limit')) {
        const waitTime = 15;
        console.log(`   ⏳ Rate limited, waiting ${waitTime}s...`);
        await new Promise(r => setTimeout(r, waitTime * 1000));
        continue;
      }
      
      if (msg.includes('credit') || msg.includes('payment')) {
        console.log(`   💳 Insufficient credits! Add funds at https://replicate.com/account/billing`);
        return { success: false };
      }
      
      console.log(`   ❌ Error (${attempt}/${retries}): ${msg.substring(0, 80)}`);
      
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, attempt * 5000));
      }
    }
  }
  
  return { success: false };
}

// === MAIN ===
async function main() {
  console.log("═".repeat(60));
  console.log("🖼️  GENERATE MISSING THUMBNAILS");
  console.log("═".repeat(60));
  
  // Parse args
  const args = process.argv.slice(2);
  const monthArg = args.find(a => a.startsWith('--month'));
  const dayArg = args.find(a => a.startsWith('--day'));
  
  let targetMonth: number | null = null;
  let targetDay: number | null = null;
  
  if (monthArg) {
    targetMonth = parseInt(args[args.indexOf(monthArg) + 1] || monthArg.split('=')[1]);
  }
  if (dayArg) {
    targetDay = parseInt(args[args.indexOf(dayArg) + 1] || dayArg.split('=')[1]);
  }
  
  // Check token
  if (!REPLICATE_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found in .env");
    console.error("Get one at https://replicate.com/account/api-tokens");
    process.exit(1);
  }
  
  const replicate = new Replicate({ auth: REPLICATE_TOKEN });
  
  // Load lessons
  console.log("\n📚 Loading lessons...");
  const allLessons = await loadLessons();
  console.log(`Loaded ${allLessons.length} lessons`);
  
  // Filter lessons
  let lessons = allLessons;
  
  if (targetDay) {
    lessons = allLessons.filter(l => l.day === targetDay);
    console.log(`\n🎯 Targeting day ${targetDay}`);
  } else if (targetMonth) {
    let startDay = 1;
    for (let i = 0; i < targetMonth - 1; i++) {
      startDay += DAYS_PER_MONTH[i];
    }
    const endDay = startDay + DAYS_PER_MONTH[targetMonth - 1] - 1;
    lessons = allLessons.filter(l => l.day >= startDay && l.day <= endDay);
    console.log(`\n🎯 Targeting ${MONTHS[targetMonth - 1]} (days ${startDay}-${endDay})`);
  }
  
  // Find missing
  const missing = lessons.filter(l => !thumbnailExists(l.day));
  console.log(`\n📊 Status:`);
  console.log(`   Total in scope: ${lessons.length}`);
  console.log(`   Already exist: ${lessons.length - missing.length}`);
  console.log(`   Missing: ${missing.length}`);
  
  if (missing.length === 0) {
    console.log("\n✅ All thumbnails exist!");
    return;
  }
  
  // Estimate cost and time
  const costPerImage = 0.10; // ~$0.10 per FLUX image
  const timePerImage = 15; // ~15 seconds per image with rate limiting
  console.log(`\n💰 Estimated cost: $${(missing.length * costPerImage).toFixed(2)}`);
  console.log(`⏱️  Estimated time: ${Math.ceil(missing.length * timePerImage / 60)} minutes`);
  
  // Generate
  console.log(`\n🚀 Starting generation...\n`);
  
  let generated = 0;
  let failed = 0;
  const startTime = Date.now();
  
  for (let i = 0; i < missing.length; i++) {
    const lesson = missing[i];
    const progress = `[${i + 1}/${missing.length}]`;
    
    console.log(`\n${progress} ─────────────────────────────────`);
    
    const result = await generateThumbnail(replicate, lesson);
    
    if (result.success) {
      generated++;
    } else {
      failed++;
    }
    
    // Rate limit delay (except for last item)
    if (i < missing.length - 1) {
      await new Promise(r => setTimeout(r, 12000)); // 12s between requests
    }
  }
  
  // Summary
  const elapsed = (Date.now() - startTime) / 1000 / 60;
  console.log("\n" + "═".repeat(60));
  console.log("✨ GENERATION COMPLETE");
  console.log("═".repeat(60));
  console.log(`Time: ${elapsed.toFixed(1)} minutes`);
  console.log(`Generated: ${generated}`);
  console.log(`Failed: ${failed}`);
  console.log(`Output: ${OUTPUT_DIRS.raw}`);
  
  if (generated > 0) {
    console.log(`\n📌 Next steps:`);
    console.log(`1. Review generated images in ${OUTPUT_DIRS.raw}`);
    console.log(`2. Run: npx tsx scripts/sync-thumbnails-to-supabase.ts`);
    console.log(`3. Optionally convert to WebP and move to production folder`);
  }
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});
