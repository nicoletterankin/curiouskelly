#!/usr/bin/env node
/**
 * Curious Kelly - Visual Asset Generation Pipeline
 * 
 * Generates 4K production-quality visual assets for all 365 lessons.
 * Uses AI image generation APIs (Midjourney/DALL-E/Flux) with pedagogically-crafted prompts.
 * 
 * Usage:
 *   node scripts/generate-visual-assets.js --day=1
 *   node scripts/generate-visual-assets.js --range=1-31
 *   node scripts/generate-visual-assets.js --all
 *   node scripts/generate-visual-assets.js --month=december
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  // Output directory
  OUTPUT_DIR: path.join(__dirname, '../public/kelly/lessons'),
  
  // Lesson data source
  LESSONS_JSON: path.join(__dirname, '../lessons/365_day_calendar.json'),
  
  // Asset types to generate
  ASSET_TYPES: ['bg', 'hero', 'prop', 'guide-point', 'reaction'],
  
  // Image specifications (Flux max is 1440x1440)
  IMAGE_SPECS: {
    width: 1344,   // 16:9 aspect ratio
    height: 768,   // 16:9 aspect ratio
    format: 'png',
    quality: 95
  },
  
  // API configuration (set via environment variables)
  API: {
    provider: process.env.IMAGE_API_PROVIDER || 'replicate', // 'replicate', 'openai', 'midjourney'
    apiKey: process.env.IMAGE_API_KEY,
    model: process.env.IMAGE_MODEL || 'black-forest-labs/flux-1.1-pro'
  },
  
  // Rate limiting
  RATE_LIMIT: {
    requestsPerMinute: 10,
    delayBetweenRequests: 6000 // ms
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// KELLY'S CONSISTENT APPEARANCE (for all images)
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_DESCRIPTION = `
A 28-year-old woman named Kelly with long, wavy brown hair with subtle 
caramel highlights, warm brown eyes, light-medium olive complexion, natural 
healthy skin. She wears a light blue crewneck sweater (her signature look), 
blue jeans, and white sneakers. Her expression is warm, approachable, curious, 
and intelligent. She looks like a favorite teacher - professional but 
relatable. She makes direct eye contact with genuine warmth.
`.trim().replace(/\n/g, ' ');

// ═══════════════════════════════════════════════════════════════════════════
// VISUAL THEME MAPPINGS
// ═══════════════════════════════════════════════════════════════════════════

const CATEGORY_THEMES = {
  // Nature & Earth (Days 1-66)
  nature: {
    dayRange: [1, 66],
    palette: 'deep greens, ocean blues, earth browns, golden sunlight',
    environments: ['forests', 'oceans', 'mountains', 'meadows', 'skies'],
    lighting: 'natural golden hour, soft diffused light, god rays through trees',
    mood: 'wonder, reverence, peaceful discovery'
  },
  
  // Physical Science (Days 67-100)
  physics: {
    dayRange: [67, 100],
    palette: 'deep blues, electric purples, bright whites, energy glows',
    environments: ['abstract scientific spaces', 'cosmic settings', 'clean labs'],
    lighting: 'dramatic rim lighting, neon accents, high contrast',
    mood: 'curiosity, experimentation, revelation'
  },
  
  // Human Body & Health (Days 71-90)
  health: {
    dayRange: [71, 90],
    palette: 'warm reds, soft pinks, organic curves, healthy glows',
    environments: ['anatomical visualizations', 'wellness spaces', 'medical-artistic'],
    lighting: 'warm internal glow, soft clinical, life-affirming',
    mood: 'self-awareness, appreciation, care'
  },
  
  // Emotions & Relationships (Days 91-130)
  social: {
    dayRange: [91, 130],
    palette: 'warm yellows, comfortable neutrals, soft blush pinks',
    environments: ['cozy living rooms', 'community spaces', 'parks'],
    lighting: 'warm afternoon light, golden hour, fireplace glow',
    mood: 'connection, empathy, belonging, warmth'
  },
  
  // History & Civilization (Days 101-160)
  history: {
    dayRange: [101, 160],
    palette: 'sepia tones, gold leaf, aged textures, rich burgundies',
    environments: ['ancient architecture', 'libraries', 'historical settings'],
    lighting: 'candlelit, dusty sunbeams, aged warm',
    mood: 'heritage, wisdom, continuity, reverence'
  },
  
  // Math & Logic (Days 96-105)
  math: {
    dayRange: [96, 105],
    palette: 'clean white, geometric blues, precise lines, subtle gradients',
    environments: ['abstract geometric spaces', 'ordered patterns', 'clean studios'],
    lighting: 'even studio lighting, soft shadows, clarity',
    mood: 'precision, elegance, order, satisfaction'
  },
  
  // Creativity & Arts (Days 161-200)
  creativity: {
    dayRange: [161, 200],
    palette: 'vibrant multicolor, expressive splashes, rainbow energy',
    environments: ['art studios', 'stages', 'imaginative dreamscapes'],
    lighting: 'dramatic theatrical, spotlight, colorful washes',
    mood: 'expression, play, freedom, inspiration'
  },
  
  // Society & Ethics (Days 201-280)
  society: {
    dayRange: [201, 280],
    palette: 'balanced neutrals, thoughtful grays, warm accents',
    environments: ['town halls', 'meeting rooms', 'global views'],
    lighting: 'balanced, democratic, no harsh shadows',
    mood: 'fairness, consideration, responsibility'
  },
  
  // Future & Technology (Days 281-330)
  future: {
    dayRange: [281, 330],
    palette: 'cool blues, silver metallics, neon accents, holographic',
    environments: ['futuristic cities', 'space stations', 'digital realms'],
    lighting: 'sci-fi ambiance, LED glow, clean futuristic',
    mood: 'innovation, possibility, wonder, progress'
  },
  
  // Philosophy & Wonder (Days 331-365)
  philosophy: {
    dayRange: [331, 365],
    palette: 'deep purples, cosmic blacks, starlight, ethereal glows',
    environments: ['cosmic vistas', 'abstract metaphysical', 'infinite spaces'],
    lighting: 'starlight, aurora, transcendent glow',
    mood: 'reflection, awe, meaning, gratitude'
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// PROMPT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

class PromptGenerator {
  constructor(lesson) {
    this.lesson = lesson;
    this.theme = this.getThemeForDay(lesson.day);
  }
  
  getThemeForDay(day) {
    for (const [category, theme] of Object.entries(CATEGORY_THEMES)) {
      if (day >= theme.dayRange[0] && day <= theme.dayRange[1]) {
        return { category, ...theme };
      }
    }
    return CATEGORY_THEMES.nature; // default
  }
  
  generateBackgroundPrompt() {
    const { title, objective } = this.lesson;
    const { palette, environments, lighting, mood } = this.theme;
    
    return `
4K photorealistic educational environment for lesson about "${title}". 
${objective}. 
Environment style: ${environments.join(' or ')}. 
Color palette: ${palette}. 
Lighting: ${lighting}. 
Mood: ${mood}. 
No people in frame. Cinematic composition with depth of field. 
The scene should visually teach the concept through environmental metaphor.
Ultra high quality, professional photography, educational visualization.
--ar 16:9 --q 2 --style raw
    `.trim();
  }
  
  generateHeroPrompt() {
    const { title } = this.lesson;
    const { palette, lighting, mood } = this.theme;
    
    return `
${KELLY_DESCRIPTION}

Kelly standing in an environment related to "${title}". Full body shot showing 
her complete figure from head to feet. She's facing the camera with a warm, 
welcoming expression that says "I'm excited to teach you this."

Background environment reflects the lesson theme with ${palette} colors.
Lighting: ${lighting}.
Mood: ${mood}.

Professional portrait photography, 4K ultra high definition, photorealistic.
--ar 16:9 --q 2
    `.trim();
  }
  
  generatePropPrompt() {
    const { title, objective } = this.lesson;
    const visualAid = this.generateVisualAid(title, objective);
    
    return `
${KELLY_DESCRIPTION}

Kelly holding or presenting ${visualAid}. She's demonstrating or teaching 
about "${title}". Her expression shows engaged curiosity and gentle wonder 
as she shares this knowledge. 

The ${visualAid} is clearly visible and catches light beautifully.
Background is softly blurred to focus attention on Kelly and the visual aid.
This is a teaching moment - she's making abstract knowledge concrete.

Professional photography, 4K ultra high definition, educational visual.
--ar 16:9 --q 2
    `.trim();
  }
  
  generateGuidePointPrompt() {
    const { title, objective } = this.lesson;
    
    return `
${KELLY_DESCRIPTION}

Kelly gesturing with her hand, pointing to or indicating something related 
to "${title}". She's in an explanatory pose, like a teacher directing 
attention to an important visual. Her expression shows engaged explanation.

${objective}

Her pointing gesture leads the viewer's eye to important educational content.
Dynamic composition showing teaching in action.

Professional photography, 4K ultra high definition, educational visual.
--ar 16:9 --q 2
    `.trim();
  }
  
  generateReactionPrompt() {
    const { title } = this.lesson;
    const { mood } = this.theme;
    const emotion = this.getEmotionForLesson(title);
    
    return `
${KELLY_DESCRIPTION}

Close-up portrait of Kelly from shoulders up. Her expression shows ${emotion} 
as she contemplates the wonder of "${title}". 

Eyes: ${this.getEyeDirection()} with intelligence and warmth.
Lighting: Soft, flattering light on her face with gentle shadows.
Background: Soft bokeh blur with colors suggesting the lesson theme.

This is an emotional moment of connection - Kelly sharing her genuine 
reaction to learning something beautiful.

Professional portrait photography, 4K ultra high definition.
--ar 1:1 --q 2
    `.trim();
  }
  
  generateVisualAid(title, objective) {
    // Generate a visual aid description based on lesson topic
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('water')) return 'a clear glass sphere containing water in multiple states - ice, liquid, and vapor';
    if (titleLower.includes('light')) return 'a glass prism splitting white light into a rainbow spectrum';
    if (titleLower.includes('sound')) return 'a tuning fork with visible sound wave visualizations emanating from it';
    if (titleLower.includes('seed')) return 'a small seedling in a transparent pot showing roots and soil';
    if (titleLower.includes('star')) return 'a holographic model of a star showing its layers';
    if (titleLower.includes('cloud')) return 'a miniature terrarium with a visible water cycle and tiny cloud';
    if (titleLower.includes('heart')) return 'an anatomical heart model that glows to show blood flow';
    if (titleLower.includes('brain')) return 'a translucent brain model with glowing neural pathways';
    if (titleLower.includes('earth')) return 'a detailed globe showing geological features';
    if (titleLower.includes('moon')) return 'a luminous moon model showing phases and surface features';
    if (titleLower.includes('sun')) return 'a warm glowing orb representing the sun with solar flares';
    if (titleLower.includes('atom')) return 'a glowing 3D model of an atom with orbiting electrons';
    if (titleLower.includes('cell')) return 'a magnified cell model showing organelles in beautiful detail';
    if (titleLower.includes('magnet')) return 'a magnet with visible magnetic field lines';
    if (titleLower.includes('electric')) return 'a glowing circuit or lightning visualization';
    if (titleLower.includes('music')) return 'a musical instrument with visible sound wave visualizations';
    if (titleLower.includes('color')) return 'a color wheel or prism with vibrant light effects';
    if (titleLower.includes('time')) return 'an elegant hourglass or mechanical clock';
    if (titleLower.includes('book')) return 'an ancient-looking book with glowing pages';
    if (titleLower.includes('map')) return 'an interactive globe or unfolding map';
    
    // Default: generate based on objective
    return `a visual aid that represents "${objective}"`;
  }
  
  getEmotionForLesson(title) {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('wonder') || titleLower.includes('star') || titleLower.includes('universe')) 
      return 'awe and wonder';
    if (titleLower.includes('love') || titleLower.includes('friend') || titleLower.includes('family')) 
      return 'warmth and affection';
    if (titleLower.includes('curious') || titleLower.includes('question')) 
      return 'engaged curiosity';
    if (titleLower.includes('courage') || titleLower.includes('brave')) 
      return 'determined courage';
    if (titleLower.includes('gratitude') || titleLower.includes('thank')) 
      return 'peaceful gratitude';
    if (titleLower.includes('joy') || titleLower.includes('happy') || titleLower.includes('celebrate')) 
      return 'joyful delight';
    
    return 'thoughtful contemplation and gentle wonder';
  }
  
  getEyeDirection() {
    const options = [
      'looking directly at camera with connection',
      'gazing slightly upward as if contemplating',
      'eyes bright with discovery',
      'soft gaze with wisdom'
    ];
    return options[this.lesson.day % options.length];
  }
  
  generateAllPrompts() {
    return {
      bg: this.generateBackgroundPrompt(),
      hero: this.generateHeroPrompt(),
      prop: this.generatePropPrompt(),
      'guide-point': this.generateGuidePointPrompt(),
      reaction: this.generateReactionPrompt()
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE GENERATOR (API Integration)
// ═══════════════════════════════════════════════════════════════════════════

class ImageGenerator {
  constructor() {
    this.provider = CONFIG.API.provider;
    this.apiKey = CONFIG.API.apiKey;
  }
  
  async generateImage(prompt, outputPath) {
    console.log(`  📸 Generating: ${path.basename(outputPath)}`);
    
    switch (this.provider) {
      case 'replicate':
        return this.generateWithReplicate(prompt, outputPath);
      case 'openai':
        return this.generateWithOpenAI(prompt, outputPath);
      case 'placeholder':
      default:
        return this.generatePlaceholder(prompt, outputPath);
    }
  }
  
  async generateWithReplicate(prompt, outputPath) {
    // Replicate API integration for Flux model
    const Replicate = require('replicate');
    const replicate = new Replicate({ auth: this.apiKey });
    
    try {
      // Use flux-schnell for faster generation (good quality, much faster)
      const output = await replicate.run('black-forest-labs/flux-schnell', {
        input: {
          prompt: prompt,
          num_outputs: 1,
          aspect_ratio: '16:9',
          output_format: 'png'
        }
      });
      
      // Handle ReadableStream output (new SDK behavior)
      if (output && output[0]) {
        if (output[0].getReader) {
          // It's a stream - read and save
          const chunks = [];
          const reader = output[0].getReader();
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
          }
          const buffer = Buffer.concat(chunks);
          fs.writeFileSync(outputPath, buffer);
        } else if (typeof output[0] === 'string') {
          // It's a URL - download
          await this.downloadImage(output[0], outputPath);
        }
        console.log(`  ✅ Saved: ${outputPath}`);
        return true;
      }
      throw new Error('No output received');
    } catch (error) {
      console.error(`  ❌ Failed: ${error.message}`);
      return false;
    }
  }
  
  async generateWithOpenAI(prompt, outputPath) {
    // OpenAI DALL-E integration
    const OpenAI = require('openai');
    const openai = new OpenAI({ apiKey: this.apiKey });
    
    try {
      const response = await openai.images.generate({
        model: 'dall-e-3',
        prompt: prompt,
        n: 1,
        size: '1792x1024', // Closest to 16:9
        quality: 'hd',
        style: 'natural'
      });
      
      const imageUrl = response.data[0].url;
      await this.downloadImage(imageUrl, outputPath);
      
      console.log(`  ✅ Saved: ${outputPath}`);
      return true;
    } catch (error) {
      console.error(`  ❌ Failed: ${error.message}`);
      return false;
    }
  }
  
  async generatePlaceholder(prompt, outputPath) {
    // Create a placeholder text file with the prompt for manual generation
    const promptPath = outputPath.replace('.png', '.prompt.txt');
    fs.writeFileSync(promptPath, prompt);
    console.log(`  📝 Prompt saved: ${promptPath}`);
    return true;
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
        fs.unlink(outputPath, () => {}); // Delete partial file
        reject(err);
      });
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════

class VisualAssetPipeline {
  constructor() {
    this.lessons = this.loadLessons();
    this.generator = new ImageGenerator();
  }
  
  loadLessons() {
    const data = JSON.parse(fs.readFileSync(CONFIG.LESSONS_JSON, 'utf8'));
    return data.lessons;
  }
  
  async generateForDay(dayNumber) {
    const lesson = this.lessons.find(l => l.day === dayNumber);
    if (!lesson) {
      console.error(`Lesson for day ${dayNumber} not found`);
      return;
    }
    
    console.log(`\n🎨 Day ${dayNumber}: ${lesson.title}`);
    console.log(`   ${lesson.objective}\n`);
    
    // Create output directory
    const paddedDay = String(dayNumber).padStart(3, '0');
    const outputDir = path.join(CONFIG.OUTPUT_DIR, paddedDay);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Generate prompts
    const promptGenerator = new PromptGenerator(lesson);
    const prompts = promptGenerator.generateAllPrompts();
    
    // Generate each asset
    for (const assetType of CONFIG.ASSET_TYPES) {
      const outputPath = path.join(outputDir, `lesson-${dayNumber}-${assetType}.png`);
      
      // Skip if already exists
      if (fs.existsSync(outputPath)) {
        console.log(`  ⏭️  Skipping ${assetType} (exists)`);
        continue;
      }
      
      await this.generator.generateImage(prompts[assetType], outputPath);
      
      // Rate limiting
      await this.delay(CONFIG.RATE_LIMIT.delayBetweenRequests);
    }
    
    console.log(`✅ Day ${dayNumber} complete\n`);
  }
  
  async generateRange(startDay, endDay) {
    console.log(`\n🚀 Generating assets for days ${startDay}-${endDay}`);
    console.log(`   Total lessons: ${endDay - startDay + 1}`);
    console.log(`   Total assets: ${(endDay - startDay + 1) * 5}\n`);
    
    for (let day = startDay; day <= endDay; day++) {
      await this.generateForDay(day);
    }
    
    console.log(`\n✨ Range ${startDay}-${endDay} complete!`);
  }
  
  async generateMonth(monthName) {
    const monthRanges = {
      january: [1, 31],
      february: [32, 59],
      march: [60, 90],
      april: [91, 120],
      may: [121, 151],
      june: [152, 181],
      july: [182, 212],
      august: [213, 243],
      september: [244, 273],
      october: [274, 304],
      november: [305, 334],
      december: [335, 365]
    };
    
    const range = monthRanges[monthName.toLowerCase()];
    if (!range) {
      console.error(`Unknown month: ${monthName}`);
      return;
    }
    
    await this.generateRange(range[0], range[1]);
  }
  
  async generateAll() {
    console.log('\n🌟 GENERATING ALL 365 DAYS OF VISUAL ASSETS 🌟');
    console.log('   Total assets: 1,825 images');
    console.log('   Estimated time: 30+ hours\n');
    
    await this.generateRange(1, 365);
  }
  
  async generatePromptsOnly(startDay, endDay) {
    console.log(`\n📝 Generating prompts only for days ${startDay}-${endDay}`);
    
    const allPrompts = [];
    
    for (let day = startDay; day <= endDay; day++) {
      const lesson = this.lessons.find(l => l.day === day);
      if (!lesson) continue;
      
      const promptGenerator = new PromptGenerator(lesson);
      const prompts = promptGenerator.generateAllPrompts();
      
      allPrompts.push({
        day: day,
        title: lesson.title,
        objective: lesson.objective,
        prompts: prompts
      });
    }
    
    // Save to JSON file
    const outputPath = path.join(__dirname, `../content/visual-prompts/prompts-${startDay}-${endDay}.json`);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(allPrompts, null, 2));
    
    console.log(`✅ Prompts saved to: ${outputPath}`);
  }
  
  getProgress() {
    let existing = 0;
    let missing = 0;
    
    for (let day = 1; day <= 365; day++) {
      const paddedDay = String(day).padStart(3, '0');
      const dir = path.join(CONFIG.OUTPUT_DIR, paddedDay);
      
      for (const type of CONFIG.ASSET_TYPES) {
        const filePath = path.join(dir, `lesson-${day}-${type}.png`);
        if (fs.existsSync(filePath)) {
          existing++;
        } else {
          missing++;
        }
      }
    }
    
    const total = existing + missing;
    const percentage = ((existing / total) * 100).toFixed(1);
    
    console.log(`\n📊 VISUAL ASSET PROGRESS`);
    console.log(`   ✅ Existing: ${existing} / ${total} (${percentage}%)`);
    console.log(`   ❌ Missing:  ${missing}`);
    console.log(`   📁 Location: ${CONFIG.OUTPUT_DIR}\n`);
  }
  
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  const pipeline = new VisualAssetPipeline();
  
  if (args.length === 0) {
    console.log(`
╔═══════════════════════════════════════════════════════════════════════════╗
║           CURIOUS KELLY - VISUAL ASSET GENERATION PIPELINE                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Usage:                                                                  ║
║     node generate-visual-assets.js --day=1           Single day           ║
║     node generate-visual-assets.js --range=1-31      Day range            ║
║     node generate-visual-assets.js --month=december  Full month           ║
║     node generate-visual-assets.js --all             All 365 days         ║
║     node generate-visual-assets.js --progress        Show progress        ║
║     node generate-visual-assets.js --prompts=1-31    Export prompts only  ║
║                                                                           ║
║   Environment:                                                            ║
║     IMAGE_API_PROVIDER=replicate|openai|placeholder                       ║
║     IMAGE_API_KEY=your-api-key                                            ║
║     IMAGE_MODEL=model-name                                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    `);
    pipeline.getProgress();
    return;
  }
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      const day = parseInt(arg.split('=')[1]);
      await pipeline.generateForDay(day);
    }
    else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(Number);
      await pipeline.generateRange(start, end);
    }
    else if (arg.startsWith('--month=')) {
      const month = arg.split('=')[1];
      await pipeline.generateMonth(month);
    }
    else if (arg === '--all') {
      await pipeline.generateAll();
    }
    else if (arg === '--progress') {
      pipeline.getProgress();
    }
    else if (arg.startsWith('--prompts=')) {
      const [start, end] = arg.split('=')[1].split('-').map(Number);
      await pipeline.generatePromptsOnly(start, end);
    }
  }
}

main().catch(console.error);

