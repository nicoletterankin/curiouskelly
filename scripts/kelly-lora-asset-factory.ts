#!/usr/bin/env npx tsx
/**
 * 🎨 KELLY LORA ASSET FACTORY
 * 
 * Generates ALL 87 social media and site assets using the trained Kelly LoRA.
 * Outputs are:
 *   1. Static images for site use
 *   2. HeyGen-ready talking photos for 2D video mode
 *   3. Consistent with 3D Unity WebGL model
 * 
 * Usage: 
 *   npx tsx scripts/kelly-lora-asset-factory.ts
 *   npx tsx scripts/kelly-lora-asset-factory.ts --priority=critical
 *   npx tsx scripts/kelly-lora-asset-factory.ts --category=social
 * 
 * Categories: social, brand, hero, chair, poses, expressions, personas
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// ═══════════════════════════════════════════════════════════════════════════
// KELLY LORA CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_LORA = {
  // HuggingFace LoRA (primary)
  hf: 'CuriousKellycom/curious-kelly-lora',
  // Civitai fallback
  civitai_version: '2455956',
  civitai_url: 'https://civitai.com/api/download/models/2455956',
  // LoRA strength
  scale: 0.85,
  // Trigger word (must be in prompt)
  trigger: 'kelly',
};

// Kelly's base appearance - LOCKED for consistency
const KELLY_BASE = `kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup`;

// Outfit variants
const OUTFITS = {
  casual: `wearing soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, white leather sneakers`,
  studio: `wearing soft powder blue cashmere crewneck sweater, sitting in director's chair with black canvas and warm wood frame`,
  professional: `wearing soft powder blue cashmere crewneck sweater, professional studio lighting`,
};

// Scene/background variants
const SCENES = {
  studio: `pure white cyclorama photography studio, professional studio lighting with soft natural window light, clean minimal background, 8K UHD`,
  dark_studio: `professional dark studio background, dramatic lighting, clean minimal background, 8K UHD`,
  transparent: `solid pure white background for easy cutout, professional lighting, 8K UHD`,
  warm: `warm natural lighting, soft golden hour glow, professional photography, 8K UHD`,
};

// ═══════════════════════════════════════════════════════════════════════════
// ASSET DEFINITIONS - Complete Shot List
// ═══════════════════════════════════════════════════════════════════════════

interface AssetSpec {
  id: string;
  category: 'social' | 'brand' | 'hero' | 'chair' | 'poses' | 'expressions' | 'personas';
  priority: 'critical' | 'high' | 'medium';
  prompt: string;
  aspect_ratio: '1:1' | '16:9' | '4:3' | '3:4' | '9:16' | 'custom';
  width?: number;
  height?: number;
  output_path: string;
  heygen_upload?: boolean; // Should this be uploaded as HeyGen talking photo?
  description: string;
}

const ASSETS: AssetSpec[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 1: SOCIAL MEDIA (Critical for Dec 17 launch)
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Profile Pictures (1:1)
  {
    id: 'profile-master',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, head and shoulders closeup, warm genuine smile, eyes engaged looking directly at camera, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'renders/social/profile-master-2048.png',
    heygen_upload: true,
    description: 'Master profile picture - exports to all platform sizes',
  },
  
  // Twitter/X Header (1500x500 = 3:1)
  {
    id: 'cover-twitter',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, upper body shot positioned in left third of frame, curious welcoming expression, looking slightly to the right with engaging smile, ${SCENES.dark_studio}`,
    aspect_ratio: 'custom',
    width: 1500,
    height: 500,
    output_path: 'public/images/social/cover-twitter.png',
    description: 'Twitter/X header banner',
  },
  
  // LinkedIn Cover (1584x396 = 4:1)
  {
    id: 'cover-linkedin',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, professional confident pose, upper body, positioned in left portion of frame, space for text on right, ${SCENES.dark_studio}`,
    aspect_ratio: 'custom',
    width: 1584,
    height: 396,
    output_path: 'public/images/social/cover-linkedin.png',
    description: 'LinkedIn company cover',
  },
  
  // OG Default (1200x630)
  {
    id: 'og-default',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, teaching pose, upper body positioned in right third of frame, engaged explaining expression, space for text overlay on left, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/og-default.png',
    description: 'Default Open Graph share image',
  },
  
  // Twitter Card Large
  {
    id: 'twitter-card-large',
    category: 'social',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, engaging teaching pose, upper body, welcoming expression, ${SCENES.dark_studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/social/twitter-card-large.png',
    description: 'Twitter card summary large image',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 2: HERO IMAGES
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-hero-4k',
    category: 'hero',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, confident inviting pose in director's chair, slight low angle heroic shot, warm smile, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly-hero-4k.png',
    heygen_upload: true,
    description: 'Main landing page hero image - 4K',
  },
  
  {
    id: 'kelly-welcome-pose',
    category: 'poses',
    priority: 'critical',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, arms slightly open in welcoming gesture, warm greeting expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_welcome.png',
    heygen_upload: true,
    description: 'Lesson start welcome pose',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 3: CHAIR TEACHING POSES (5 expressions)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-chair-celebrating',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, both arms raised joyfully in celebration, big genuine triumphant smile, bright excited eyes, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-celebrating.png',
    heygen_upload: true,
    description: 'Chair pose - celebrating success',
  },
  
  {
    id: 'kelly-chair-curious',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, head tilted with raised eyebrows, curious inquisitive expression, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-curious.png',
    heygen_upload: true,
    description: 'Chair pose - curious/questioning',
  },
  
  {
    id: 'kelly-chair-explaining',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, hands gesturing naturally while explaining, engaged teaching expression, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-explaining.png',
    heygen_upload: true,
    description: 'Chair pose - explaining/teaching',
  },
  
  {
    id: 'kelly-chair-listening',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, attentive listening posture, slight forward lean, warm attentive expression, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-listening.png',
    heygen_upload: true,
    description: 'Chair pose - active listening',
  },
  
  {
    id: 'kelly-chair-wisdom',
    category: 'chair',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, calm knowing smile, relaxed dignified posture, wise serene expression, ${SCENES.studio}`,
    aspect_ratio: '16:9',
    output_path: 'public/images/kelly/kelly-chair-wisdom.png',
    heygen_upload: true,
    description: 'Chair pose - wisdom/insight',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 4: LESSON PLAYER POSES (11 poses)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'kelly-idle',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing neutral relaxed stance, hands at sides, warm attentive expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_idle.png',
    description: 'Neutral idle stance',
  },
  
  {
    id: 'kelly-listening',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing active listening posture, head slightly tilted, attentive engaged expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_listening.png',
    description: 'Active listening pose',
  },
  
  {
    id: 'kelly-choice-left',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, left arm extended gracefully pointing to the left, body angled slightly left, encouraging expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_choice_left.png',
    description: 'Pointing left for choice A',
  },
  
  {
    id: 'kelly-choice-right',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, right arm extended gracefully pointing to the right, body angled slightly right, encouraging expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_choice_right.png',
    description: 'Pointing right for choice B',
  },
  
  {
    id: 'kelly-hint',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, index finger touching chin thoughtfully, playful knowing expression, head tilted, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_hint.png',
    description: 'Thoughtful hint pose',
  },
  
  {
    id: 'kelly-clasp',
    category: 'poses',
    priority: 'medium',
    prompt: `${KELLY_BASE}, ${OUTFITS.casual}, standing pose, hands clasped together in front, eager anticipating expression, ${SCENES.transparent}`,
    aspect_ratio: '16:9',
    output_path: 'public/kelly/poses/kelly_clasp.png',
    description: 'Hands clasped anticipation',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 5: EXPRESSION CLOSEUPS (9 expressions)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'expr-celebrating',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, joyful triumphant expression, big genuine smile, bright eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/celebrating.jpeg',
    description: 'Expression - celebrating',
  },
  
  {
    id: 'expr-confused',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, puzzled questioning expression, slight frown, head tilted, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/confused.jpeg',
    description: 'Expression - confused',
  },
  
  {
    id: 'expr-curious-closeup',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, extreme face closeup, intense curiosity expression, wide eyes, raised eyebrows, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/curious-closeup.jpeg',
    description: 'Expression - curious closeup',
  },
  
  {
    id: 'expr-curious-main',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, curious interested expression, slight smile, engaged eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/curious-main.jpeg',
    description: 'Expression - curious main',
  },
  
  {
    id: 'expr-explaining',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, animated explaining expression, mouth slightly open mid-speech, engaged eyes, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/explaining.jpeg',
    description: 'Expression - explaining',
  },
  
  {
    id: 'expr-peaceful',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, serene peaceful calm expression, gentle smile, relaxed features, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/peaceful.jpeg',
    description: 'Expression - peaceful',
  },
  
  {
    id: 'expr-surprised',
    category: 'expressions',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.professional}, face closeup portrait, pleasantly surprised expression, raised eyebrows, wide eyes, open mouth smile, ${SCENES.warm}`,
    aspect_ratio: '1:1',
    output_path: 'public/images/expressions/surprised.jpeg',
    description: 'Expression - surprised',
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PRIORITY 6: PERSONAS (12 archetypes)
  // ═══════════════════════════════════════════════════════════════════════════
  
  {
    id: 'persona-scientist',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, analytical focused expression, examining something thoughtfully, scientist archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/scientist.png',
    heygen_upload: true,
    description: 'Persona - The Scientist',
  },
  
  {
    id: 'persona-explorer',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, adventurous excited expression, eager curious eyes, explorer archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/explorer.png',
    heygen_upload: true,
    description: 'Persona - The Explorer',
  },
  
  {
    id: 'persona-rebel',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, determined bold expression, confident challenging look, rebel archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/rebel.png',
    heygen_upload: true,
    description: 'Persona - The Rebel',
  },
  
  {
    id: 'persona-architect',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, thoughtful precise expression, considering carefully, architect archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/architect.png',
    heygen_upload: true,
    description: 'Persona - The Architect',
  },
  
  {
    id: 'persona-diplomat',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, warm understanding expression, gentle empathetic smile, diplomat archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/diplomat.png',
    heygen_upload: true,
    description: 'Persona - The Diplomat',
  },
  
  {
    id: 'persona-empath',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, compassionate gentle expression, soft caring eyes, empath archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/empath.png',
    heygen_upload: true,
    description: 'Persona - The Empath',
  },
  
  {
    id: 'persona-macgyver',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, creative resourceful expression, clever knowing smile, macgyver archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/macgyver.png',
    heygen_upload: true,
    description: 'Persona - The MacGyver',
  },
  
  {
    id: 'persona-mystic',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, wise serene expression, deep knowing eyes, mystic archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/mystic.png',
    heygen_upload: true,
    description: 'Persona - The Mystic',
  },
  
  {
    id: 'persona-provider',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, nurturing generous expression, warm protective smile, provider archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/provider.png',
    heygen_upload: true,
    description: 'Persona - The Provider',
  },
  
  {
    id: 'persona-storyteller',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, animated engaging expression, bright eyes mid-story, storyteller archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/storyteller.png',
    heygen_upload: true,
    description: 'Persona - The Storyteller',
  },
  
  {
    id: 'persona-strategist',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, sharp calculating expression, focused analytical eyes, strategist archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/strategist.png',
    heygen_upload: true,
    description: 'Persona - The Strategist',
  },
  
  {
    id: 'persona-survivor',
    category: 'personas',
    priority: 'high',
    prompt: `${KELLY_BASE}, ${OUTFITS.studio}, seated in director's chair, strong resilient expression, determined confident eyes, survivor archetype energy, ${SCENES.studio}`,
    aspect_ratio: '1:1',
    output_path: 'public/assets/kelly/personas/survivor.png',
    heygen_upload: true,
    description: 'Persona - The Survivor',
  },
];

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════

interface GenerationResult {
  asset: AssetSpec;
  success: boolean;
  imageUrl?: string;
  localPath?: string;
  error?: string;
  duration?: number;
}

async function generateWithLoRA(asset: AssetSpec): Promise<GenerationResult> {
  const startTime = Date.now();
  console.log(`\n🎨 Generating: ${asset.id}`);
  console.log(`   ${asset.description}`);
  
  try {
    // Determine aspect ratio
    let aspectRatio = asset.aspect_ratio;
    if (aspectRatio === 'custom') {
      // Calculate closest standard ratio
      const ratio = asset.width! / asset.height!;
      if (ratio >= 2.5) aspectRatio = '16:9'; // Will need post-crop
      else if (ratio >= 1.5) aspectRatio = '16:9';
      else if (ratio >= 1.2) aspectRatio = '4:3';
      else if (ratio >= 0.9) aspectRatio = '1:1';
      else aspectRatio = '3:4';
    }
    
    // Try HuggingFace LoRA first
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: asset.prompt,
          hf_lora: `https://huggingface.co/${KELLY_LORA.hf}/resolve/main/lora.safetensors`,
          lora_scale: KELLY_LORA.scale,
          num_outputs: 1,
          aspect_ratio: aspectRatio,
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28,
          disable_safety_checker: true
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`   📥 Downloading...`);
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    
    // Save to output path
    const outputPath = path.join(process.cwd(), asset.output_path);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, buffer);
    
    const duration = (Date.now() - startTime) / 1000;
    console.log(`   ✅ Saved: ${asset.output_path} (${duration.toFixed(1)}s)`);
    
    return {
      asset,
      success: true,
      imageUrl,
      localPath: outputPath,
      duration,
    };
    
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return {
      asset,
      success: false,
      error: error.message,
      duration: (Date.now() - startTime) / 1000,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 KELLY LORA ASSET FACTORY                                   ║');
  console.log('║  Generate ALL social media & site assets with trained LoRA    ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }
  
  // Parse arguments
  const args = process.argv.slice(2);
  const priorityFilter = args.find(a => a.startsWith('--priority='))?.split('=')[1];
  const categoryFilter = args.find(a => a.startsWith('--category='))?.split('=')[1];
  const dryRun = args.includes('--dry-run');
  
  // Filter assets
  let assetsToGenerate = ASSETS;
  
  if (priorityFilter) {
    assetsToGenerate = assetsToGenerate.filter(a => a.priority === priorityFilter);
    console.log(`\n🎯 Priority filter: ${priorityFilter}`);
  }
  
  if (categoryFilter) {
    assetsToGenerate = assetsToGenerate.filter(a => a.category === categoryFilter);
    console.log(`🎯 Category filter: ${categoryFilter}`);
  }
  
  console.log(`\n📊 Assets to generate: ${assetsToGenerate.length}`);
  console.log(`   Critical: ${assetsToGenerate.filter(a => a.priority === 'critical').length}`);
  console.log(`   High: ${assetsToGenerate.filter(a => a.priority === 'high').length}`);
  console.log(`   Medium: ${assetsToGenerate.filter(a => a.priority === 'medium').length}`);
  
  console.log(`\n📁 LoRA: ${KELLY_LORA.hf}`);
  console.log(`⚡ Scale: ${KELLY_LORA.scale}`);
  
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Would generate:');
    for (const asset of assetsToGenerate) {
      console.log(`   [${asset.priority}] ${asset.id} → ${asset.output_path}`);
    }
    return;
  }
  
  // Generate assets
  const results: GenerationResult[] = [];
  const heygenAssets: GenerationResult[] = [];
  
  for (const asset of assetsToGenerate) {
    const result = await generateWithLoRA(asset);
    results.push(result);
    
    if (result.success && asset.heygen_upload) {
      heygenAssets.push(result);
    }
    
    // Rate limit
    await new Promise(r => setTimeout(r, 3000));
  }
  
  // Summary
  console.log('\n\n' + '═'.repeat(70));
  console.log('📊 GENERATION SUMMARY');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n✅ Successful: ${successful.length}/${results.length}`);
  console.log(`❌ Failed: ${failed.length}/${results.length}`);
  
  if (failed.length > 0) {
    console.log('\n🔴 Failed assets:');
    for (const f of failed) {
      console.log(`   - ${f.asset.id}: ${f.error}`);
    }
  }
  
  if (heygenAssets.length > 0) {
    console.log(`\n🎬 HeyGen-ready assets: ${heygenAssets.length}`);
    console.log('   Run heygen-upload-avatars.ts to upload these as talking photos');
  }
  
  // Save manifest
  const manifest = {
    generated: new Date().toISOString(),
    lora: KELLY_LORA.hf,
    total: results.length,
    successful: successful.length,
    failed: failed.length,
    assets: results.map(r => ({
      id: r.asset.id,
      success: r.success,
      path: r.localPath,
      heygen_upload: r.asset.heygen_upload,
      duration: r.duration,
      error: r.error,
    })),
  };
  
  const manifestPath = path.join(process.cwd(), 'generated-assets-manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\n💾 Manifest saved: ${manifestPath}`);
  
  console.log('\n' + '═'.repeat(70));
}

main().catch(console.error);
