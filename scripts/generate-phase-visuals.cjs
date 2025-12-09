/**
 * PHASE VISUAL GENERATOR - Production Version
 * 
 * Generates 5 phase visuals per lesson using Kelly LoRA for character consistency.
 * 
 * Usage:
 *   node generate-phase-visuals.cjs --day=1
 *   node generate-phase-visuals.cjs --range=1-7
 *   node generate-phase-visuals.cjs --day=1 --phases=hook,q1
 * 
 * Environment:
 *   REPLICATE_API_TOKEN - Required for image generation
 */

const Replicate = require('replicate');
const fs = require('fs');
const path = require('path');

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  // Kelly LoRA
  LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  LORA_SCALE: 0.85,
  
  // Kelly's consistent appearance
  KELLY_BASE: 'kelly, woman late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, soft natural features, wearing soft powder blue cashmere sweater',
  
  // Image settings
  ASPECT_RATIO: '16:9',
  OUTPUT_FORMAT: 'png',
  GUIDANCE_SCALE: 3.5,
  INFERENCE_STEPS: 28,
  
  // Output directory
  OUTPUT_DIR: path.join(process.cwd(), 'public', 'kelly', 'phases'),
  
  // Rate limiting (ms between requests)
  RATE_LIMIT: 2000
};

// ═══════════════════════════════════════════════════════════════════
// LESSON DATABASE - 365 Days
// Each lesson has a topic, icon, and visual context
// ═══════════════════════════════════════════════════════════════════

const LESSONS = {
  1: {
    title: "Starting Fresh",
    topic: "Leaves and photosynthesis",
    icon: "🌱",
    context: {
      environment: "sunlit forest clearing, green leaves, morning light",
      props: ["large green leaf", "tree trunk", "plant with visible roots", "majestic oak tree"],
      mood: "fresh, natural, growth"
    }
  },
  2: {
    title: "The Three Lives of Water",
    topic: "States of matter - solid, liquid, gas",
    icon: "💧",
    context: {
      environment: "lakeside with mountains, misty morning, ice crystals",
      props: ["glass of water", "ice cubes", "steam rising", "rain clouds"],
      mood: "fluid, transformative, wonder"
    }
  },
  3: {
    title: "Where Clouds Come From",
    topic: "Water cycle and cloud formation",
    icon: "☁️",
    context: {
      environment: "open sky with dramatic clouds, sun rays, hilltop",
      props: ["fluffy clouds", "rain drops", "sun beams", "earth and sky"],
      mood: "airy, expansive, magical"
    }
  },
  4: {
    title: "How Light Travels",
    topic: "Light and optics basics",
    icon: "💡",
    context: {
      environment: "sunny room with prisms, rainbow light, laboratory",
      props: ["prism with rainbow", "flashlight beam", "mirror reflection", "sunbeam through window"],
      mood: "bright, scientific, discovery"
    }
  },
  5: {
    title: "How Sound Moves",
    topic: "Sound waves and vibration",
    icon: "🔊",
    context: {
      environment: "music room with instruments, sound waves visible",
      props: ["tuning fork", "guitar strings", "speaker cone", "ripples in water"],
      mood: "dynamic, rhythmic, vibrant"
    }
  },
  6: {
    title: "What's Inside a Seed",
    topic: "Plant biology and germination",
    icon: "🌰",
    context: {
      environment: "garden setting, rich soil, spring sunshine",
      props: ["cross-section of seed", "sprouting seedling", "gardening tools", "blooming flower"],
      mood: "nurturing, hopeful, growth"
    }
  },
  7: {
    title: "What Stars Are Made Of",
    topic: "Stellar composition and fusion",
    icon: "⭐",
    context: {
      environment: "night sky observatory, constellation backdrop, telescope",
      props: ["bright star close-up", "telescope", "constellation map", "galaxy backdrop"],
      mood: "cosmic, awe-inspiring, vast"
    }
  }
  // More lessons will be added from curriculum database
};

// ═══════════════════════════════════════════════════════════════════
// PHASE TEMPLATES
// These create prompts for each lesson phase
// ═══════════════════════════════════════════════════════════════════

function getPhasePrompts(lesson) {
  const { context } = lesson;
  const env = context.environment;
  const props = context.props;
  
  return {
    hook: {
      name: 'hook',
      prompt: `${CONFIG.KELLY_BASE}, standing in ${env}, welcoming open stance, arms slightly open in invitation, warm genuine smile, looking at viewer, full body visible, cinematic photography, 8K, shallow depth of field`,
      filename: 'hook.png'
    },
    q1: {
      name: 'q1',
      prompt: `${CONFIG.KELLY_BASE}, in ${env}, holding and examining ${props[0]}, curious fascinated expression, pointing at or gesturing toward the object, teaching moment, cinematic photography, 8K`,
      filename: 'q1.png'
    },
    q2: {
      name: 'q2',
      prompt: `${CONFIG.KELLY_BASE}, in ${env}, thoughtful expression, chin resting on hand, looking contemplatively at ${props[1]}, pondering a deeper question, soft lighting, shallow depth of field, 8K`,
      filename: 'q2.png'
    },
    q3: {
      name: 'q3',
      prompt: `${CONFIG.KELLY_BASE}, in ${env}, encouraging teaching expression, gesturing toward ${props[2]}, leaning forward with engagement, explaining an important concept, educational moment, warm lighting, 8K`,
      filename: 'q3.png'
    },
    wisdom: {
      name: 'wisdom',
      prompt: `${CONFIG.KELLY_BASE}, in ${env} at golden hour, standing proudly with hand on heart, satisfied accomplished smile, ${props[3]} in background, sense of completion and mastery, cinematic wide shot, inspirational, 8K`,
      filename: 'wisdom.png'
    }
  };
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATOR
// ═══════════════════════════════════════════════════════════════════

async function downloadImage(urlOrStream) {
  if (urlOrStream.getReader) {
    const reader = urlOrStream.getReader();
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    return Buffer.concat(chunks);
  } else {
    const response = await fetch(urlOrStream);
    return Buffer.from(await response.arrayBuffer());
  }
}

async function generateImage(replicate, prompt, outputPath) {
  try {
    const output = await replicate.run(CONFIG.LORA_MODEL, {
      input: {
        prompt: prompt,
        hf_lora: CONFIG.LORA_URL,
        lora_scale: CONFIG.LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: CONFIG.ASPECT_RATIO,
        output_format: CONFIG.OUTPUT_FORMAT,
        guidance_scale: CONFIG.GUIDANCE_SCALE,
        num_inference_steps: CONFIG.INFERENCE_STEPS
      }
    });
    
    const url = Array.isArray(output) ? output[0] : output;
    const buffer = await downloadImage(url);
    
    fs.writeFileSync(outputPath, buffer);
    return { success: true, size: buffer.length };
    
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN GENERATOR
// ═══════════════════════════════════════════════════════════════════

async function generateLessonVisuals(dayNumber, phases = ['hook', 'q1', 'q2', 'q3', 'wisdom']) {
  const lesson = LESSONS[dayNumber];
  
  if (!lesson) {
    console.error(`❌ Lesson ${dayNumber} not found in database`);
    return false;
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📚 Day ${dayNumber}: ${lesson.icon} ${lesson.title}`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`Topic: ${lesson.topic}`);
  console.log(`Phases: ${phases.join(', ')}`);
  
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  
  // Create output directory
  const paddedDay = String(dayNumber).padStart(3, '0');
  const dayDir = path.join(CONFIG.OUTPUT_DIR, paddedDay);
  fs.mkdirSync(dayDir, { recursive: true });
  
  const phasePrompts = getPhasePrompts(lesson);
  let successCount = 0;
  
  for (const phaseName of phases) {
    const phase = phasePrompts[phaseName];
    if (!phase) {
      console.log(`  ⚠️ Unknown phase: ${phaseName}`);
      continue;
    }
    
    const outputPath = path.join(dayDir, phase.filename);
    
    // Skip if already exists
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️ ${phaseName}: Already exists`);
      successCount++;
      continue;
    }
    
    console.log(`\n  🎬 ${phaseName.toUpperCase()}`);
    console.log(`     Prompt: ${phase.prompt.substring(0, 80)}...`);
    
    const result = await generateImage(replicate, phase.prompt, outputPath);
    
    if (result.success) {
      console.log(`     ✅ Saved: ${phase.filename} (${(result.size / 1024).toFixed(1)} KB)`);
      successCount++;
    } else {
      console.log(`     ❌ Failed: ${result.error}`);
    }
    
    // Rate limiting
    await new Promise(r => setTimeout(r, CONFIG.RATE_LIMIT));
  }
  
  console.log(`\n📊 Day ${dayNumber} Result: ${successCount}/${phases.length} phases generated`);
  return successCount === phases.length;
}

async function generateRange(startDay, endDay) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  PHASE VISUAL GENERATOR - Days ${startDay} to ${endDay}`);
  console.log(`${'█'.repeat(60)}`);
  
  const results = { success: 0, failed: 0, days: [] };
  
  for (let day = startDay; day <= endDay; day++) {
    const success = await generateLessonVisuals(day);
    if (success) {
      results.success++;
      results.days.push({ day, status: 'success' });
    } else {
      results.failed++;
      results.days.push({ day, status: 'failed' });
    }
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 FINAL RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`✅ Success: ${results.success} days`);
  console.log(`❌ Failed: ${results.failed} days`);
  
  return results;
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  const options = {};
  
  for (const arg of args) {
    const [key, value] = arg.replace('--', '').split('=');
    options[key] = value;
  }
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN environment variable required');
    process.exit(1);
  }
  
  if (options.range) {
    const [start, end] = options.range.split('-').map(Number);
    await generateRange(start, end);
  } else if (options.day) {
    const phases = options.phases ? options.phases.split(',') : undefined;
    await generateLessonVisuals(Number(options.day), phases);
  } else {
    console.log(`
Phase Visual Generator
═══════════════════════════════════════════════════════════

Usage:
  node generate-phase-visuals.cjs --day=1
  node generate-phase-visuals.cjs --range=1-7
  node generate-phase-visuals.cjs --day=1 --phases=hook,q1

Options:
  --day=N         Generate visuals for day N
  --range=A-B     Generate visuals for days A through B
  --phases=LIST   Comma-separated list of phases (hook,q1,q2,q3,wisdom)

Environment:
  REPLICATE_API_TOKEN    Your Replicate API token
    `);
  }
}

main().catch(console.error);



