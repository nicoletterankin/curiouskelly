import Replicate from 'replicate';
import fs from 'fs';
import https from 'https';
import path from 'path';

const replicate = new Replicate();

const KELLY_LORA = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';
const OUTPUT_BASE = 'public/kelly/phases';

// Visual contexts for different topics
const VISUAL_CONTEXTS = {
  friend: { env: 'cozy living room with warm lighting, comfortable atmosphere, family photos on wall', mood: 'warm, connected, trusting' },
  kindness: { env: 'sunny community park, people helping each other, garden setting', mood: 'generous, rippling outward, hopeful' },
  listen: { env: 'quiet library with comfortable reading chairs, warm wood', mood: 'attentive, present, understanding' },
  patience: { env: 'zen garden with raked sand, peaceful space', mood: 'calm, enduring, rewarding' },
  gratitude: { env: 'golden hour field, abundance, harvest setting', mood: 'thankful, abundant, glowing' },
  courage: { env: 'mountain peak at sunrise, triumphant setting', mood: 'brave, determined, victorious' },
  curious: { env: 'explorer study room with maps, discoveries, globe', mood: 'inquisitive, excited, discovering' },
  body: { env: 'clean wellness space, healthy lifestyle imagery', mood: 'appreciative, aware, balanced' },
  breath: { env: 'mountain air setting, fresh morning, yoga space', mood: 'refreshing, vital, centering' },
  move: { env: 'active outdoors, playground, sports field', mood: 'energetic, joyful, alive' },
  rest: { env: 'cozy bedroom, peaceful night sky through window', mood: 'peaceful, restorative, calm' },
  energy: { env: 'energy transformation lab, power visualization', mood: 'powerful, transforming, electric' },
  sense: { env: 'sensory garden with diverse textures and colors', mood: 'aware, experiencing, discovering' },
  water: { env: 'lakeside with mountains, misty morning', mood: 'fluid, transformative, wonder' },
  cloud: { env: 'open sky with dramatic clouds, sun rays', mood: 'airy, expansive, magical' },
  light: { env: 'bright laboratory with prisms, rainbow spectrum', mood: 'illuminating, scientific, discovery' },
  sound: { env: 'music room with instruments, acoustic panels', mood: 'dynamic, rhythmic, vibrant' },
  seed: { env: 'garden setting with rich soil, spring sunshine', mood: 'nurturing, hopeful, growth' },
  plant: { env: 'garden setting with rich soil, spring sunshine', mood: 'nurturing, hopeful, growth' },
  star: { env: 'night sky observatory, constellation backdrop', mood: 'cosmic, awe-inspiring, vast' },
  space: { env: 'night sky observatory, constellation backdrop', mood: 'cosmic, awe-inspiring, vast' },
  default: { env: 'bright modern learning studio with clean background', mood: 'curious, educational, engaging' }
};

// Lesson topics (will fetch from DB in production, hardcoded subset for now)
const LESSONS = {
  8: 'What Makes a Real Friend',
  9: 'How Kindness Spreads',
  10: 'The Art of Really Listening',
  11: 'Why Patience Pays Off',
  12: 'How Gratitude Changes You',
  13: 'What Courage Really Means',
  14: 'Why Curious People Learn More',
  15: 'How Your Body Stays Balanced',
  16: 'Why Breathing Matters',
  17: 'Why Bodies Need to Move',
  18: 'What Happens When You Rest',
  19: 'How Energy Changes Form',
  20: 'Your Five Senses (And More)',
};

function getContextForTopic(topic) {
  const topicLower = topic.toLowerCase();
  for (const [keyword, context] of Object.entries(VISUAL_CONTEXTS)) {
    if (keyword !== 'default' && topicLower.includes(keyword)) {
      return context;
    }
  }
  return VISUAL_CONTEXTS.default;
}

const PHASES = [
  { name: 'hook', pose: 'welcoming open stance with arms slightly open in invitation, warm genuine smile showing excitement' },
  { name: 'q1', pose: 'holding and examining an educational object with genuine fascination, curious engaged expression' },
  { name: 'q2', pose: 'thoughtful contemplative expression, chin resting gently on hand, pondering deeply' },
  { name: 'q3', pose: 'encouraging supportive expression with warm smile, gesturing with open hand, leaning forward' },
  { name: 'wisdom', pose: 'standing proudly with hand placed gently on heart, satisfied accomplished smile' },
];

async function downloadImage(url, outputPath) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(outputPath);
    fs.mkdirSync(dir, { recursive: true });
    
    const file = fs.createWriteStream(outputPath);
    const handleResponse = (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        https.get(response.headers.location, handleResponse).on('error', reject);
      } else {
        response.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
        file.on('error', reject);
      }
    };
    https.get(url, handleResponse).on('error', reject);
  });
}

async function generateImage(prompt, outputPath) {
  try {
    const prediction = await replicate.predictions.create({
      model: 'lucataco/flux-dev-lora',
      version: 'a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
      input: {
        prompt: prompt,
        hf_lora: KELLY_LORA,
        lora_scale: 0.85,
        num_outputs: 1,
        aspect_ratio: '16:9',
        output_format: 'png',
        guidance_scale: 3.5,
        num_inference_steps: 28
      }
    });
    
    let result = prediction;
    while (result.status !== 'succeeded' && result.status !== 'failed') {
      await new Promise(r => setTimeout(r, 2000));
      result = await replicate.predictions.get(prediction.id);
    }
    
    if (result.status === 'failed') {
      return { success: false, error: result.error };
    }
    
    const imageUrl = result.output?.[0];
    if (imageUrl) {
      await downloadImage(imageUrl, outputPath);
      return { success: true, cost: 0.04 };
    }
    return { success: false, error: 'No image URL' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function generateLessonPhases(dayNumber, topic) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const dayDir = path.join(OUTPUT_BASE, paddedDay);
  const context = getContextForTopic(topic);
  
  console.log(`\n📚 Day ${dayNumber}: ${topic}`);
  console.log(`   Environment: ${context.env.substring(0, 40)}...`);
  
  let success = 0;
  let cost = 0;
  
  for (const phase of PHASES) {
    const outputPath = path.join(dayDir, `${phase.name}.png`);
    
    if (fs.existsSync(outputPath)) {
      console.log(`   ⏭️ ${phase.name}: exists`);
      success++;
      continue;
    }
    
    const prompt = `kelly, photorealistic woman named Kelly, late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown almond-shaped eyes, wearing soft powder blue cashmere sweater, in ${context.env}, ${phase.pose}, looking at viewer, mood: ${context.mood}, cinematic photography, natural lighting, 8K, shallow depth of field`;
    
    console.log(`   🎨 ${phase.name}...`);
    const result = await generateImage(prompt, outputPath);
    
    if (result.success) {
      console.log(`   ✅ ${phase.name} saved`);
      success++;
      cost += result.cost;
    } else {
      console.log(`   ❌ ${phase.name}: ${result.error}`);
    }
    
    // Small delay between images
    await new Promise(r => setTimeout(r, 1000));
  }
  
  return { success, total: PHASES.length, cost };
}

async function main() {
  console.log('🚀 BATCH PHASE VISUAL GENERATOR');
  console.log('================================\n');
  
  const startDay = parseInt(process.argv[2]) || 8;
  const endDay = parseInt(process.argv[3]) || 20;
  
  console.log(`Generating Days ${startDay} to ${endDay}`);
  console.log(`Estimated images: ${(endDay - startDay + 1) * 5}`);
  console.log(`Estimated cost: ~$${((endDay - startDay + 1) * 5 * 0.04).toFixed(2)}\n`);
  
  let totalSuccess = 0;
  let totalCost = 0;
  const startTime = Date.now();
  
  for (let day = startDay; day <= endDay; day++) {
    const topic = LESSONS[day] || `Lesson Day ${day}`;
    const result = await generateLessonPhases(day, topic);
    totalSuccess += result.success;
    totalCost += result.cost;
    
    // Progress report
    const progress = ((day - startDay + 1) / (endDay - startDay + 1) * 100).toFixed(1);
    const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    console.log(`   📊 Progress: ${progress}% | Elapsed: ${elapsed}min | Cost: $${totalCost.toFixed(2)}`);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🎉 BATCH COMPLETE!');
  console.log(`✅ Images generated: ${totalSuccess}`);
  console.log(`💰 Total cost: $${totalCost.toFixed(2)}`);
  console.log(`⏱️ Time: ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
}

main().catch(console.error);


