#!/usr/bin/env npx tsx
/**
 * 🎨 DAY 351 VISUAL GENERATOR - "Practicing in Your Mind"
 * 
 * LAUNCH DAY: December 17, 2025
 * 
 * Generates ALL visuals for Day 351's incredible visualization lesson:
 * - Kelly phase images (topic-specific poses)
 * - Educational infographics (brain scans, piano study, elite athletes)
 * - Social media visuals (Instagram, Twitter, TikTok)
 * - Netflix-style thumbnail
 * 
 * Uses:
 * - Replicate Flux Pro 1.1 for backgrounds/infographics
 * - Replicate Flux-dev-lora for Kelly images
 * - Gemini for visual plan generation (optional)
 * 
 * Usage:
 *   npx tsx scripts/generate-day-351-visuals.ts
 *   npx tsx scripts/generate-day-351-visuals.ts --kelly-only
 *   npx tsx scripts/generate-day-351-visuals.ts --infographics-only
 *   npx tsx scripts/generate-day-351-visuals.ts --social-only
 *   npx tsx scripts/generate-day-351-visuals.ts --dry-run
 */

import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
dotenv.config();

import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  DAY: 351,
  TOPIC: "Practicing in Your Mind",
  EMOJI: "🔮",
  
  // Models
  FLUX_PRO: "black-forest-labs/flux-1.1-pro",
  KELLY_LORA: {
    model: "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    loraUrl: "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors",
    loraScale: 0.90,
    triggerWord: "kelly"
  },
  
  // Kelly's consistent appearance (matches LoRA training)
  KELLY_BASE: `kelly, woman late 20s, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown almond-shaped expressive eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater, blue jeans`,
  
  // Output directories
  OUTPUT_PHASES: path.join(process.cwd(), 'public', 'kelly', 'phases', '351'),
  OUTPUT_INFOGRAPHICS: path.join(process.cwd(), 'public', 'kelly', 'infographics', '351'),
  OUTPUT_SOCIAL: path.join(process.cwd(), 'public', 'kelly', 'social', '351'),
  
  // API
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN,
  
  // Rate limiting
  DELAY_MS: 3000
};

// ═══════════════════════════════════════════════════════════════════════════
// DAY 351 KELLY PHASE DEFINITIONS - VISUALIZATION THEME
// ═══════════════════════════════════════════════════════════════════════════

interface KellyPhase {
  phase: string;
  filename: string;
  pose: string;
  emotion: string;
  background?: string;
}

const DAY_351_KELLY_PHASES: KellyPhase[] = [
  {
    phase: "hook",
    filename: "hook.png",
    pose: "eyes gently closed, serene peaceful expression, hands positioned as if holding an invisible sphere of imagination in front of her chest, meditative visualization pose, slight ethereal glow around her",
    emotion: "peaceful concentration, wonder, mystery",
    background: "soft gradient from deep purple to midnight blue, subtle stars and neural pathway patterns floating around her, dreamy atmosphere"
  },
  {
    phase: "cliff", 
    filename: "cliff.png",
    pose: "leaning forward with curious engaged expression, one finger touching temple thoughtfully, other hand gesturing outward as if painting a vision in the air, teaching discovery pose",
    emotion: "curiosity, revelation, the moment before understanding",
    background: "abstract neural network patterns softly glowing in background, synapses lighting up, scientific wonder aesthetic"
  },
  {
    phase: "q1",
    filename: "q1.png",
    pose: "animated explaining gesture with both hands showing comparison, eyes bright with enthusiasm, body language showing 'mind = blown' energy, one hand near head one hand near heart",
    emotion: "excitement about discovery, sharing amazing knowledge",
    background: "split background showing brain scan imagery on one side and physical activity on other, representing the 90% overlap"
  },
  {
    phase: "q2",
    filename: "q2.png",
    pose: "storytelling pose, hands moving as if playing invisible piano keys in the air, captivating engaged expression, leaning in as if sharing a secret",
    emotion: "engaged storytelling, sharing research revelation",
    background: "soft focus piano keys in background, subtle musical notes and brain imagery blended together"
  },
  {
    phase: "q3",
    filename: "q3.png",
    pose: "powerful confident stance, one fist raised in triumph pose, other hand on heart, inspired passionate expression, gold medal champion energy",
    emotion: "inspiration, empowerment, elite performer energy",
    background: "stadium lights in background, Olympic rings subtly visible, champion visualization aesthetic"
  },
  {
    phase: "wisdom",
    filename: "wisdom.png",
    pose: "warm knowing smile, hands open in giving gesture, eyes connecting directly with viewer, wise mentor sharing life-changing truth pose",
    emotion: "wisdom, warmth, gift of knowledge",
    background: "cosmic consciousness background, stars forming neural pathways, peaceful profound atmosphere"
  },
  {
    phase: "outro",
    filename: "outro.png", 
    pose: "encouraging wave goodbye, warm proud smile, one hand touching heart in 'take this with you' gesture, supportive coach energy",
    emotion: "encouragement, confidence in the learner, warm farewell",
    background: "sunset gradient with stars appearing, transition from day to night representing going to practice visualization"
  }
];

// ═══════════════════════════════════════════════════════════════════════════
// INFOGRAPHIC DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

interface Infographic {
  id: string;
  filename: string;
  prompt: string;
  aspectRatio: string;
}

const DAY_351_INFOGRAPHICS: Infographic[] = [
  {
    id: "brain-scan-90-percent",
    filename: "infographic-brain-scan.png",
    prompt: `Educational split-scene infographic showing 90% neural overlap between physical action and visualization.

LEFT SIDE: Side profile of human head with transparent skull revealing brain, neural pathways lighting up in bright blue-white electrical pulses, hands at bottom physically playing piano keys, label "DOING" with activity indicator.

RIGHT SIDE: Identical head profile with SAME neural pathways lighting up almost identically, person's eyes closed in peaceful concentration, hands still, imagining playing piano, label "IMAGINING" with activity indicator.

CENTER: Large "90%" statistic connecting both sides, subtitle "Neural Overlap".

Style: Clean modern medical illustration, scientific accuracy, brain scan aesthetic, deep blue and white color palette with gold accents, photorealistic brain detail, educational infographic, 8K resolution, no watermarks, no logos, dramatic lighting`,
    aspectRatio: "16:9"
  },
  {
    id: "piano-study-comparison",
    filename: "infographic-piano-study.png",
    prompt: `Educational infographic visualizing the famous piano visualization research study.

THREE HORIZONTAL COMPARISON LANES like a scientific experiment chart:

TOP LANE - "Physical Practice Group": 
Person at piano actively practicing, musical notes flowing, brain icon showing neural growth with upward arrow, 5-day timeline, result showing significant improvement.

MIDDLE LANE - "Mental Practice Only Group":
Person sitting peacefully with eyes closed, thought bubble showing piano keys and notes, brain icon showing NEARLY EQUAL neural growth, 5-day timeline, result showing remarkable improvement.

BOTTOM LANE - "Control Group":
Person doing unrelated activity, brain icon showing minimal change, 5-day timeline, result showing no improvement.

RIGHT SIDE: Brain scan comparison showing physical and mental groups nearly identical.

Style: Clean research infographic design, scientific study aesthetic, warm educational colors, professional data visualization, comparison chart style, 8K resolution, no text errors`,
    aspectRatio: "16:9"
  },
  {
    id: "elite-athletes-50-percent",
    filename: "infographic-olympic-athletes.png",
    prompt: `Dramatic inspirational infographic about elite athlete mental training.

CENTRAL FIGURE: Olympic athlete silhouette (gymnast or diver) in dynamic mid-performance pose, body artistically split - one half solid representing physical training, other half ethereal/translucent with glowing neural pathways representing mental visualization.

LARGE STATISTIC: "50%" prominently displayed, subtitle "Mental Rehearsal Time".

BACKGROUND: Transitions from physical gym/stadium on solid side to abstract cosmic mind-space on ethereal side.

BOTTOM ROW: Three smaller icons showing other professions who visualize:
- Surgeon with glowing thought bubble of procedure
- Concert pianist with eyes closed, notes in mind  
- Basketball player visualizing free throw

Style: Cinematic sports photography meets scientific illustration, dramatic lighting, Olympic gold and deep blue color palette, inspirational aesthetic, 8K resolution, clean professional design`,
    aspectRatio: "16:9"
  },
  {
    id: "visualization-guide-steps",
    filename: "infographic-how-to-visualize.png",
    prompt: `Clean actionable step-by-step visualization guide infographic.

FLOWING PATH connecting 5 steps in elegant curved journey:

STEP 1 (Start): Person sitting comfortably in peaceful pose, icon showing relaxation waves, label "RELAX Your Body"

STEP 2: Detailed eye icon with magnifying detail symbols, label "SEE Every Detail"

STEP 3: Hand icon with tactile sensation waves, label "FEEL The Movements"

STEP 4: Ear icon with sound waves emanating, label "HEAR The Sounds"  

STEP 5 (End): Clock/calendar icon with repeat symbol, label "PRACTICE Daily"

CENTRAL IMAGE: Peaceful person with closed eyes, subtle golden aura of imagination around their head, representing successful visualization state.

Style: Modern process infographic, warm inviting colors (purple, gold, soft blue), step-by-step tutorial aesthetic, soft gradients, encouraging welcoming design, 8K resolution, clean typography areas`,
    aspectRatio: "16:9"
  },
  {
    id: "cosmic-mind-background",
    filename: "background-cosmic-mind.png",
    prompt: `Stunning cosmic visualization of human consciousness and imagination potential.

A translucent peaceful human profile silhouette with eyes closed, and INSIDE their head is an entire glowing universe - swirling galaxies, brilliant stars, colorful nebulae, neural pathways that look like constellations all interconnected with flowing golden light streams.

The mind-universe expands beyond the head boundaries, suggesting infinite potential.

Deep purple and midnight blue background with gold and white stars scattered throughout. Ethereal mist and stardust floating.

Represents the infinite creative and training potential of the visualizing mind.

Style: Cosmic digital art, consciousness visualization, deep space aesthetic, purple gold and midnight blue palette, ethereal bioluminescent lighting, 8K resolution, cinematic, awe-inspiring, dreamlike, no text`,
    aspectRatio: "16:9"
  }
];

// ═══════════════════════════════════════════════════════════════════════════
// SOCIAL MEDIA VISUALS
// ═══════════════════════════════════════════════════════════════════════════

const DAY_351_SOCIAL: Infographic[] = [
  {
    id: "instagram-carousel-hook",
    filename: "social-ig-carousel-1.png",
    prompt: `Instagram carousel opening slide for visualization lesson.

Olympic athlete (gymnast or swimmer) with eyes closed in focused meditative state, stadium or pool blurred in dramatic background with spotlights.

Large bold text area reserved at bottom third for overlay: "Olympic athletes spend 50% of training doing THIS..."

Dark cinematic gradient overlay from bottom for text readability. Dramatic sports photography lighting. Mystery and curiosity-inducing composition.

Style: Instagram carousel design 1080x1350, bold hook aesthetic, sports documentary photography, dramatic lighting, social media optimized, high contrast`,
    aspectRatio: "4:5"
  },
  {
    id: "instagram-carousel-brain",
    filename: "social-ig-carousel-2.png",
    prompt: `Instagram carousel educational slide showing brain science.

Split view of two brain scans side by side - left labeled "Doing" and right labeled "Imagining" - both showing nearly identical neural activation patterns lighting up.

Large "90%" in center connecting them.

Clean medical infographic style adapted for Instagram. Educational but visually striking. Space for text overlay at bottom.

Style: Instagram educational content 1080x1350, clean medical visualization, striking comparison graphic, social media optimized`,
    aspectRatio: "4:5"
  },
  {
    id: "wisdom-quote-card",
    filename: "social-quote-card.png",
    prompt: `Elegant Instagram quote card with cosmic brain background.

Subtle dark background showing cosmic neural pathways and stars, very understated and elegant.

Large center area reserved for quote text overlay: "The mind that rehearses builds pathways the passive mind never develops."

Subtle sparkle accents in corners. Deep purple gradient background with soft star field. Sophisticated and shareable.

Style: Instagram quote card 1080x1080, elegant typography space, dark cosmic aesthetic, minimal design, premium feel`,
    aspectRatio: "1:1"
  },
  {
    id: "twitter-thread-header",
    filename: "social-twitter-header.png",
    prompt: `Twitter thread header image for visualization science lesson.

Dramatic close-up of human eye with galaxy/neural network reflected in the iris. Represents seeing with the mind's eye.

Cinematic lighting with rim light effect. Mysterious and curiosity-inducing. Space on right side for thread intro text.

Style: Twitter header 1200x675, cinematic photography, dramatic eye closeup, galaxy reflection, social media optimized`,
    aspectRatio: "16:9"
  },
  {
    id: "tiktok-thumbnail",
    filename: "social-tiktok-thumb.png",
    prompt: `TikTok thumbnail for visualization science video.

Split face - one half showing person with eyes open doing physical activity, other half showing same person with eyes closed visualizing, brain glow effect on visualization side.

Bold, eye-catching, scroll-stopping composition. High contrast. Text area at bottom.

Style: TikTok thumbnail 1080x1920, bold high contrast, split face concept, scroll-stopping, vertical video thumbnail`,
    aspectRatio: "9:16"
  }
];

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function generateKellyImage(
  replicate: Replicate,
  phase: KellyPhase,
  outputDir: string
): Promise<boolean> {
  const outputPath = path.join(outputDir, phase.filename);
  
  if (fs.existsSync(outputPath)) {
    console.log(`  ⏭️  Skipping ${phase.phase} (exists)`);
    return true;
  }
  
  const prompt = `${CONFIG.KELLY_BASE}, ${phase.pose}, ${phase.emotion}, ${phase.background || 'clean professional studio background'}, professional photography, 8K resolution, highly detailed, warm lighting, educational teacher aesthetic`;
  
  console.log(`  🎭 Generating Kelly ${phase.phase}...`);
  
  try {
    const output = await replicate.run(
      CONFIG.KELLY_LORA.model as `${string}/${string}:${string}`,
      {
        input: {
          prompt: prompt,
          hf_lora: CONFIG.KELLY_LORA.loraUrl,
          lora_scale: CONFIG.KELLY_LORA.loraScale,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    let imageData: Buffer;
    if (imageUrl.getReader) {
      const reader = imageUrl.getReader();
      const chunks: Uint8Array[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      imageData = Buffer.concat(chunks);
    } else {
      const response = await fetch(imageUrl);
      imageData = Buffer.from(await response.arrayBuffer());
    }
    
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Saved: ${phase.filename}`);
    return true;
    
  } catch (error: any) {
    console.error(`  ❌ Error generating ${phase.phase}: ${error.message}`);
    return false;
  }
}

async function generateInfographic(
  replicate: Replicate,
  infographic: Infographic,
  outputDir: string
): Promise<boolean> {
  const outputPath = path.join(outputDir, infographic.filename);
  
  if (fs.existsSync(outputPath)) {
    console.log(`  ⏭️  Skipping ${infographic.id} (exists)`);
    return true;
  }
  
  console.log(`  📊 Generating ${infographic.id}...`);
  
  try {
    const output = await replicate.run(
      CONFIG.FLUX_PRO,
      {
        input: {
          prompt: infographic.prompt,
          aspect_ratio: infographic.aspectRatio,
          output_format: "png",
          output_quality: 100,
          safety_tolerance: 2
        }
      }
    ) as any;
    
    const imageUrl = typeof output === 'string' ? output : output.toString();
    const response = await fetch(imageUrl);
    const imageData = Buffer.from(await response.arrayBuffer());
    
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Saved: ${infographic.filename}`);
    return true;
    
  } catch (error: any) {
    console.error(`  ❌ Error generating ${infographic.id}: ${error.message}`);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  🎨 DAY 351 VISUAL GENERATOR                                     ║
║  "Practicing in Your Mind" — Visualization                       ║
║  LAUNCH DAY: December 17, 2025                                   ║
╚══════════════════════════════════════════════════════════════════╝
`);

  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const kellyOnly = args.includes('--kelly-only');
  const infographicsOnly = args.includes('--infographics-only');
  const socialOnly = args.includes('--social-only');
  const all = !kellyOnly && !infographicsOnly && !socialOnly;
  
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set!');
    console.log('   Set in .env.local or environment');
    process.exit(1);
  }
  
  if (dryRun) {
    console.log('🔍 DRY RUN MODE - showing what would be generated:\n');
    
    if (all || kellyOnly) {
      console.log('📸 KELLY PHASE IMAGES:');
      DAY_351_KELLY_PHASES.forEach(p => {
        console.log(`   • ${p.phase}: ${p.emotion}`);
      });
    }
    
    if (all || infographicsOnly) {
      console.log('\n📊 INFOGRAPHICS:');
      DAY_351_INFOGRAPHICS.forEach(i => {
        console.log(`   • ${i.id}: ${i.filename}`);
      });
    }
    
    if (all || socialOnly) {
      console.log('\n📱 SOCIAL MEDIA:');
      DAY_351_SOCIAL.forEach(s => {
        console.log(`   • ${s.id}: ${s.filename}`);
      });
    }
    
    console.log('\n✅ Dry run complete. Remove --dry-run to generate.');
    return;
  }
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  let generated = 0;
  let failed = 0;
  
  // Generate Kelly phases
  if (all || kellyOnly) {
    console.log('\n📸 GENERATING KELLY PHASE IMAGES...');
    console.log(`   Output: ${CONFIG.OUTPUT_PHASES}\n`);
    
    for (const phase of DAY_351_KELLY_PHASES) {
      const success = await generateKellyImage(replicate, phase, CONFIG.OUTPUT_PHASES);
      success ? generated++ : failed++;
      await new Promise(r => setTimeout(r, CONFIG.DELAY_MS));
    }
  }
  
  // Generate infographics
  if (all || infographicsOnly) {
    console.log('\n📊 GENERATING EDUCATIONAL INFOGRAPHICS...');
    console.log(`   Output: ${CONFIG.OUTPUT_INFOGRAPHICS}\n`);
    
    for (const infographic of DAY_351_INFOGRAPHICS) {
      const success = await generateInfographic(replicate, infographic, CONFIG.OUTPUT_INFOGRAPHICS);
      success ? generated++ : failed++;
      await new Promise(r => setTimeout(r, CONFIG.DELAY_MS));
    }
  }
  
  // Generate social media visuals
  if (all || socialOnly) {
    console.log('\n📱 GENERATING SOCIAL MEDIA VISUALS...');
    console.log(`   Output: ${CONFIG.OUTPUT_SOCIAL}\n`);
    
    for (const social of DAY_351_SOCIAL) {
      const success = await generateInfographic(replicate, social, CONFIG.OUTPUT_SOCIAL);
      success ? generated++ : failed++;
      await new Promise(r => setTimeout(r, CONFIG.DELAY_MS));
    }
  }
  
  // Summary
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  ✅ DAY 351 VISUAL GENERATION COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Generated: ${String(generated).padEnd(3)} assets                                       ║
║  Failed:    ${String(failed).padEnd(3)} assets                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  📁 Kelly Phases:    ${CONFIG.OUTPUT_PHASES}
║  📁 Infographics:    ${CONFIG.OUTPUT_INFOGRAPHICS}
║  📁 Social Media:    ${CONFIG.OUTPUT_SOCIAL}
╚══════════════════════════════════════════════════════════════════╝

🚀 Day 351 visuals ready for launch!
"The mind that rehearses grows stronger than the mind that merely waits."
`);
}

main().catch(console.error);
