#!/usr/bin/env node
/**
 * February-March Visual Asset Generator
 * 
 * Generates assets for Days 32-64 using the detailed ULTRA prompts.
 * Uses Replicate's Flux model for high-quality image generation.
 * 
 * Usage:
 *   node scripts/generate-february-march-visuals.cjs --day=32
 *   node scripts/generate-february-march-visuals.cjs --range=32-41
 *   node scripts/generate-february-march-visuals.cjs --all
 *   node scripts/generate-february-march-visuals.cjs --priority1
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// Load environment
require('dotenv').config({ path: '.env.local' });
require('dotenv').config();

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  OUTPUT_DIR: path.join(__dirname, '../public/kelly/lessons'),
  MANIFEST_FILE: path.join(__dirname, 'kelly-visual-identity/visual-manifest-february-march.json'),
  ASSET_TYPES: ['background', 'hero', 'prop', 'guide_point', 'reaction'],
  REPLICATE_TOKEN: process.env.REPLICATE_API_TOKEN,
  // Rate limiting - Replicate allows ~10 requests/min on starter plans
  DELAY_BETWEEN_IMAGES: 8000, // 8 seconds
  DELAY_BETWEEN_LESSONS: 3000  // 3 seconds extra between lessons
};

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

class FluxGenerator {
  constructor() {
    this.Replicate = require('replicate');
    this.client = new this.Replicate({
      auth: CONFIG.REPLICATE_TOKEN
    });
    this.stats = { generated: 0, skipped: 0, failed: 0 };
  }

  async generate(prompt, outputPath, aspectRatio = '16:9') {
    // Skip if file already exists
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️  Skip (exists): ${path.basename(outputPath)}`);
      this.stats.skipped++;
      return true;
    }

    console.log(`  🎨 Generating: ${path.basename(outputPath)}`);
    
    try {
      // Use flux-schnell for speed (good quality, fast)
      // For higher quality, use flux-1.1-pro but it's slower and costs more
      const output = await this.client.run('black-forest-labs/flux-schnell', {
        input: {
          prompt: prompt,
          num_outputs: 1,
          aspect_ratio: aspectRatio,
          output_format: 'png',
          output_quality: 95
        }
      });

      // Handle the output
      if (output && output[0]) {
        if (output[0].getReader) {
          // Stream output
          const chunks = [];
          const reader = output[0].getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
          }
          const buffer = Buffer.concat(chunks);
          fs.writeFileSync(outputPath, buffer);
        } else if (typeof output[0] === 'string') {
          // URL output - download
          await this.downloadImage(output[0], outputPath);
        }
        
        console.log(`  ✅ Saved: ${path.basename(outputPath)}`);
        this.stats.generated++;
        return true;
      }
      
      throw new Error('No output received from API');
    } catch (error) {
      console.error(`  ❌ Failed: ${error.message}`);
      this.stats.failed++;
      return false;
    }
  }

  async downloadImage(url, outputPath) {
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(outputPath);
      https.get(url, (response) => {
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

  getStats() {
    return this.stats;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MANIFEST LOADER
// ═══════════════════════════════════════════════════════════════════════════

function loadManifest() {
  if (fs.existsSync(CONFIG.MANIFEST_FILE)) {
    return JSON.parse(fs.readFileSync(CONFIG.MANIFEST_FILE, 'utf8'));
  }
  console.error('❌ Manifest file not found:', CONFIG.MANIFEST_FILE);
  console.log('Creating fallback prompts from lesson data...');
  return createFallbackPrompts();
}

function createFallbackPrompts() {
  // Load lesson calendar and create basic prompts for days 32-64
  const calendarPath = path.join(__dirname, '../lessons/365_day_calendar.json');
  const calendar = JSON.parse(fs.readFileSync(calendarPath, 'utf8'));
  
  const KELLY_BASE = `kelly, young woman, brown wavy hair, light blue crewneck sweater, blue jeans, white sneakers`;
  
  return calendar.lessons
    .filter(l => l.day >= 32 && l.day <= 64)
    .map(lesson => ({
      lesson_id: lesson.day,
      title: lesson.title,
      date: lesson.date,
      assets: [
        {
          type: 'background',
          filename: `lesson-${lesson.day}-bg.png`,
          prompt: `4K photorealistic educational environment for ${lesson.title}, ${lesson.learning_objective}, cinematic lighting, no people, educational visualization`
        },
        {
          type: 'hero',
          filename: `lesson-${lesson.day}-hero.png`,
          prompt: `${KELLY_BASE}, standing in environment related to ${lesson.title}, full body shot facing camera with warm welcoming expression, professional photography, 4K`
        },
        {
          type: 'prop',
          filename: `lesson-${lesson.day}-prop.png`,
          prompt: `${KELLY_BASE}, holding visual aid related to ${lesson.title}, teaching expression, waist-up shot, professional photography, 4K`
        },
        {
          type: 'guide_point',
          filename: `lesson-${lesson.day}-guide-point.png`,
          prompt: `${KELLY_BASE}, pointing gesture explaining ${lesson.title}, upper body shot, teaching expression, professional photography, 4K, vertical 9:16`
        },
        {
          type: 'reaction',
          filename: `lesson-${lesson.day}-reaction.png`,
          prompt: `close-up portrait ${KELLY_BASE}, expression of wonder contemplating ${lesson.title}, soft bokeh background, square 1:1, 4K`
        }
      ]
    }));
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN PIPELINE
// ═══════════════════════════════════════════════════════════════════════════

async function generateForLesson(lessonData, generator) {
  const day = lessonData.lesson_id;
  const paddedDay = String(day).padStart(3, '0');
  const outputDir = path.join(CONFIG.OUTPUT_DIR, paddedDay);
  
  // Ensure directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📅 Day ${day}: ${lessonData.title}`);
  console.log(`${'═'.repeat(60)}`);
  
  for (const asset of lessonData.assets) {
    const outputPath = path.join(outputDir, asset.filename);
    
    // Determine aspect ratio
    let aspectRatio = '16:9';
    if (asset.type === 'reaction') aspectRatio = '1:1';
    if (asset.type === 'guide_point') aspectRatio = '9:16';
    
    await generator.generate(asset.prompt, outputPath, aspectRatio);
    
    // Rate limiting
    await delay(CONFIG.DELAY_BETWEEN_IMAGES);
  }
  
  console.log(`✅ Day ${day} complete`);
  await delay(CONFIG.DELAY_BETWEEN_LESSONS);
}

async function generateRange(startDay, endDay, manifest, generator) {
  const lessonsToGenerate = manifest.filter(
    l => l.lesson_id >= startDay && l.lesson_id <= endDay
  );
  
  console.log(`\n🚀 Generating Days ${startDay}-${endDay}`);
  console.log(`   Lessons: ${lessonsToGenerate.length}`);
  console.log(`   Assets: ${lessonsToGenerate.length * 5}`);
  
  for (const lesson of lessonsToGenerate) {
    await generateForLesson(lesson, generator);
  }
  
  const stats = generator.getStats();
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 GENERATION COMPLETE`);
  console.log(`   ✅ Generated: ${stats.generated}`);
  console.log(`   ⏭️  Skipped: ${stats.skipped}`);
  console.log(`   ❌ Failed: ${stats.failed}`);
  console.log(`${'═'.repeat(60)}`);
}

async function generatePriority1(manifest, generator) {
  // Priority 1: Days 32 (Moon/Tides), 33 (Gravity), 41 (Rainbows)
  const priority1Days = [32, 33, 41];
  const lessonsToGenerate = manifest.filter(l => priority1Days.includes(l.lesson_id));
  
  console.log(`\n🌟 PRIORITY 1 GENERATION`);
  console.log(`   Days: ${priority1Days.join(', ')}`);
  console.log(`   Topics: Moon/Tides, Gravity, Rainbows`);
  
  for (const lesson of lessonsToGenerate) {
    await generateForLesson(lesson, generator);
  }
  
  const stats = generator.getStats();
  console.log(`\n✨ Priority 1 Complete: ${stats.generated} generated, ${stats.skipped} skipped`);
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function showUsage() {
  console.log(`
╔═══════════════════════════════════════════════════════════════════════════╗
║        FEBRUARY-MARCH VISUAL ASSET GENERATOR (Days 32-64)                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Usage:                                                                  ║
║     node generate-february-march-visuals.cjs --day=32     Single day      ║
║     node generate-february-march-visuals.cjs --range=32-41 Day range      ║
║     node generate-february-march-visuals.cjs --all         All 32-64      ║
║     node generate-february-march-visuals.cjs --priority1   Days 32,33,41  ║
║                                                                           ║
║   Estimated time per lesson: ~1 minute (5 images)                         ║
║   Total for all 33 lessons: ~35-40 minutes                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
  `);
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  if (!CONFIG.REPLICATE_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found in environment');
    console.log('   Set it in .env.local: REPLICATE_API_TOKEN=r8_...');
    process.exit(1);
  }
  
  if (args.length === 0) {
    showUsage();
    return;
  }
  
  const manifest = loadManifest();
  const generator = new FluxGenerator();
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      const day = parseInt(arg.split('=')[1]);
      const lesson = manifest.find(l => l.lesson_id === day);
      if (lesson) {
        await generateForLesson(lesson, generator);
      } else {
        console.error(`❌ Day ${day} not found in manifest`);
      }
    }
    else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(Number);
      await generateRange(start, end, manifest, generator);
    }
    else if (arg === '--all') {
      await generateRange(32, 64, manifest, generator);
    }
    else if (arg === '--priority1') {
      await generatePriority1(manifest, generator);
    }
    else {
      showUsage();
    }
  }
}

main().catch(console.error);

