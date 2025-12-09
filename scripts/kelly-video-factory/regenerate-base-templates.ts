/**
 * 🎬 Base Video Template Batch Regenerator
 * 
 * Regenerates ALL 10 base video templates using the production-grade
 * specifications from BASE_VIDEO_TEMPLATES.json and the SOTA pipeline.
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/regenerate-base-templates.ts
 *   npx tsx scripts/kelly-video-factory/regenerate-base-templates.ts --template T01
 *   npx tsx scripts/kelly-video-factory/regenerate-base-templates.ts --tier sync --upscale
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import {
  runSOTAPipeline,
  generateKellyImage,
  CONFIG,
  KELLY,
} from './sota-video-pipeline';

// =============================================================================
// CONFIGURATION
// =============================================================================

const TEMPLATES_PATH = path.join(process.cwd(), 'template-forge', 'BASE_VIDEO_TEMPLATES.json');
const OUTPUT_DIR = path.join(process.cwd(), 'template-forge', 'regenerated-templates');

// Mapping from template categories to SOTA poses
const TEMPLATE_TO_POSE: Record<string, string> = {
  'T01': 'welcome',      // Welcome Walk → welcoming pose
  'T02': 'explaining',   // Present & Explain → explaining pose
  'T03': 'curious',      // Curious Examine → curious pose
  'T04': 'heartfelt',    // Heartfelt Share → heartfelt pose
  'T05': 'excited',      // Excited Discovery → excited pose
  'T06': 'thoughtful',   // Thoughtful Pause → thoughtful pose
  'T07': 'celebrating',  // Celebrating Success → celebrating pose
  'T08': 'curious',      // Active Listening → attentive/curious
  'T09': 'heartfelt',    // Warm Reassurance → warm/heartfelt
  'T10': 'welcome',      // Closing Gratitude → warm closing
};

// Representative scripts for each template
const TEMPLATE_SCRIPTS: Record<string, string> = {
  'T01': "Hello, curious learner! I'm so excited to see you today. Welcome to a new adventure in learning - let's discover something amazing together!",
  'T02': "Here's the fascinating thing - when we break big ideas into smaller pieces, they become so much clearer. Let me show you step by step.",
  'T03': "Hmm, that's a really interesting question. What do you think happens when we look at it from this angle?",
  'T04': "You know, learning can feel hard sometimes. But I believe in you, and I know you have everything you need inside to understand this.",
  'T05': "Oh wow! Did you see that? This is so incredible! When you put these pieces together, something magical happens!",
  'T06': "Let's take a moment to really think about this. Sometimes the best insights come when we pause and let our minds wander.",
  'T07': "Yes! You did it! I knew you could! This is exactly the kind of thinking that makes great learners great.",
  'T08': "I'm listening. Take your time - there's no rush. Your thoughts and questions matter to me.",
  'T09': "It's okay if this feels challenging. Every great learner faces moments like this. Let's work through it together, one small step at a time.",
  'T10': "Thank you so much for learning with me today. Remember, every question you ask makes you wiser. See you next time!",
};

// =============================================================================
// TEMPLATE LOADER
// =============================================================================

interface Template {
  id: string;
  name: string;
  internal_name: string;
  category: string;
  purpose: string;
  duration: {
    total_sec: number;
    loop_start_sec: number;
    loop_end_sec: number;
    seamless_loop: boolean;
  };
  camera: {
    shot_type: string;
    framing: string;
  };
  prompt_guidance: {
    minimax: string;
    runway?: string;
    luma?: string;
  };
}

interface TemplatesConfig {
  version: string;
  templates: Template[];
}

function loadTemplates(): TemplatesConfig {
  if (!fs.existsSync(TEMPLATES_PATH)) {
    throw new Error(`Templates file not found: ${TEMPLATES_PATH}`);
  }
  
  const raw = fs.readFileSync(TEMPLATES_PATH, 'utf-8');
  return JSON.parse(raw);
}

// =============================================================================
// BATCH GENERATION
// =============================================================================

interface GenerationResult {
  templateId: string;
  templateName: string;
  success: boolean;
  imageUrl?: string;
  videoUrl?: string;
  audioUrl?: string;
  duration?: number;
  error?: string;
  outputPath?: string;
}

async function regenerateTemplate(
  template: Template,
  tier: string,
  upscale: boolean
): Promise<GenerationResult> {
  const startTime = Date.now();
  const pose = TEMPLATE_TO_POSE[template.id] || 'excited';
  const script = TEMPLATE_SCRIPTS[template.id] || "Hello! Let's learn together.";
  
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`🎬 REGENERATING: ${template.id} - ${template.name}`);
  console.log(`${'═'.repeat(70)}`);
  console.log(`   Category: ${template.category}`);
  console.log(`   Purpose: ${template.purpose}`);
  console.log(`   Pose: ${pose}`);
  console.log(`   Duration: ${template.duration.total_sec}s`);
  console.log(`   Camera: ${template.camera.shot_type} (${template.camera.framing})`);
  console.log(`   Tier: ${tier}`);
  
  try {
    // Run the SOTA pipeline
    const result = await runSOTAPipeline({
      tier: tier as any,
      text: script,
      pose: pose,
      upscale: upscale,
    });
    
    if (!result.success) {
      throw new Error(result.error || 'Pipeline failed');
    }
    
    // Save locally
    const templateDir = path.join(OUTPUT_DIR, template.id);
    fs.mkdirSync(templateDir, { recursive: true });
    
    // Download and save video
    let outputPath: string | undefined;
    if (result.finalVideoUrl || result.videoUrl) {
      const videoUrl = result.finalVideoUrl || result.videoUrl!;
      const videoFilename = `${template.internal_name}_${Date.now()}.mp4`;
      outputPath = path.join(templateDir, videoFilename);
      
      console.log(`\n   📥 Downloading video to ${outputPath}...`);
      const response = await fetch(videoUrl);
      const buffer = Buffer.from(await response.arrayBuffer());
      fs.writeFileSync(outputPath, buffer);
      console.log(`   ✅ Saved (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`);
    }
    
    // Save metadata
    const metadata = {
      template: template,
      generation: {
        timestamp: new Date().toISOString(),
        tier: result.tier,
        pose: pose,
        script: script,
        duration_sec: result.duration,
        imageUrl: result.imageUrl,
        audioUrl: result.audioUrl,
        videoUrl: result.videoUrl,
        finalVideoUrl: result.finalVideoUrl,
        localPath: outputPath,
      }
    };
    
    fs.writeFileSync(
      path.join(templateDir, `${template.internal_name}_metadata.json`),
      JSON.stringify(metadata, null, 2)
    );
    
    return {
      templateId: template.id,
      templateName: template.name,
      success: true,
      imageUrl: result.imageUrl,
      videoUrl: result.finalVideoUrl || result.videoUrl,
      audioUrl: result.audioUrl,
      duration: result.duration,
      outputPath,
    };
    
  } catch (error: any) {
    console.error(`\n   ❌ Failed: ${error.message}`);
    return {
      templateId: template.id,
      templateName: template.name,
      success: false,
      error: error.message,
      duration: (Date.now() - startTime) / 1000,
    };
  }
}

async function regenerateAllTemplates(
  tier: string,
  upscale: boolean,
  specificTemplate?: string
): Promise<GenerationResult[]> {
  const config = loadTemplates();
  const results: GenerationResult[] = [];
  
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 BASE VIDEO TEMPLATE BATCH REGENERATOR                            ║');
  console.log('║  Making Kelly the best digital human teacher on the planet           ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  Templates Version: ${config.version}`);
  console.log(`  Total Templates: ${config.templates.length}`);
  console.log(`  Tier: ${tier}`);
  console.log(`  Upscale: ${upscale ? 'Yes (4K)' : 'No'}`);
  console.log(`  Output Dir: ${OUTPUT_DIR}`);
  
  // Create output directory
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  // Filter to specific template if requested
  let templatesToProcess = config.templates;
  if (specificTemplate) {
    templatesToProcess = config.templates.filter(t => t.id === specificTemplate);
    if (templatesToProcess.length === 0) {
      throw new Error(`Template ${specificTemplate} not found`);
    }
    console.log(`\n  📌 Processing only: ${specificTemplate}`);
  }
  
  const startTime = Date.now();
  
  for (let i = 0; i < templatesToProcess.length; i++) {
    const template = templatesToProcess[i];
    console.log(`\n\n  [${ i + 1 }/${templatesToProcess.length}] Processing ${template.id}...`);
    
    const result = await regenerateTemplate(template, tier, upscale);
    results.push(result);
    
    // Rate limiting between templates
    if (i < templatesToProcess.length - 1) {
      console.log(`\n   ⏳ Cooling down 5s before next template...`);
      await new Promise(r => setTimeout(r, 5000));
    }
  }
  
  const totalDuration = (Date.now() - startTime) / 1000;
  
  // Print summary
  console.log('\n\n');
  console.log('═'.repeat(70));
  console.log('📊 BATCH REGENERATION COMPLETE');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n  Total Time: ${(totalDuration / 60).toFixed(1)} minutes`);
  console.log(`  Success: ${successful.length}/${results.length}`);
  console.log(`  Failed: ${failed.length}/${results.length}`);
  
  if (successful.length > 0) {
    console.log('\n  ✅ Successful:');
    for (const r of successful) {
      console.log(`     ${r.templateId}: ${r.templateName} (${r.duration?.toFixed(1)}s)`);
      if (r.outputPath) {
        console.log(`        → ${r.outputPath}`);
      }
    }
  }
  
  if (failed.length > 0) {
    console.log('\n  ❌ Failed:');
    for (const r of failed) {
      console.log(`     ${r.templateId}: ${r.templateName}`);
      console.log(`        Error: ${r.error}`);
    }
  }
  
  // Save batch results
  const batchReport = {
    timestamp: new Date().toISOString(),
    config: {
      tier,
      upscale,
      templatesVersion: config.version,
    },
    summary: {
      total: results.length,
      successful: successful.length,
      failed: failed.length,
      totalDurationSec: totalDuration,
    },
    results: results,
  };
  
  const reportPath = path.join(OUTPUT_DIR, `batch_report_${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(batchReport, null, 2));
  console.log(`\n  📄 Report saved: ${reportPath}`);
  
  console.log('═'.repeat(70));
  
  return results;
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  let tier = 'best-available';
  let upscale = false;
  let specificTemplate: string | undefined;
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--tier':
        tier = args[++i];
        break;
      case '--upscale':
        upscale = true;
        break;
      case '--template':
        specificTemplate = args[++i];
        break;
      case '--help':
        console.log(`
Base Video Template Batch Regenerator

Regenerates all 10 base video templates using the SOTA pipeline
and production-grade specifications from BASE_VIDEO_TEMPLATES.json.

Usage:
  npx tsx regenerate-base-templates.ts [options]

Options:
  --tier <tier>       Pipeline tier: sync, hedra, omnihuman, best-available (default)
  --upscale           Upscale all outputs to 4K
  --template <id>     Only regenerate specific template (e.g., T01, T02)
  --help              Show this help

Examples:
  npx tsx regenerate-base-templates.ts
  npx tsx regenerate-base-templates.ts --tier sync --upscale
  npx tsx regenerate-base-templates.ts --template T01
  npx tsx regenerate-base-templates.ts --tier sync --template T05

Templates:
  T01 - Welcome Walk (entrance)
  T02 - Present & Explain (teaching)
  T03 - Curious Examine (inquiry)
  T04 - Heartfelt Share (emotional)
  T05 - Excited Discovery (energy)
  T06 - Thoughtful Pause (contemplation)
  T07 - Celebrating Success (celebration)
  T08 - Active Listening (engagement)
  T09 - Warm Reassurance (support)
  T10 - Closing Gratitude (closing)
        `);
        process.exit(0);
    }
  }
  
  // Check API keys
  console.log('\n🔑 Checking API Keys...');
  console.log(`   REPLICATE: ${process.env.REPLICATE_API_TOKEN ? '✅' : '❌'}`);
  console.log(`   ELEVENLABS: ${process.env.ELEVENLABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   SYNC_LABS: ${process.env.SYNC_LABS_API_KEY ? '✅' : '⚪'}`);
  console.log(`   HEDRA: ${process.env.HEDRA_API_KEY ? '✅' : '⚪'}`);
  console.log(`   FAL: ${process.env.FAL_KEY ? '✅' : '⚪'}`);
  
  if (!process.env.REPLICATE_API_TOKEN || !process.env.ELEVENLABS_API_KEY) {
    console.error('\n❌ Missing required API keys (REPLICATE_API_TOKEN, ELEVENLABS_API_KEY)');
    process.exit(1);
  }
  
  // Run batch regeneration
  const results = await regenerateAllTemplates(tier, upscale, specificTemplate);
  
  // Exit with error if any failed
  const failed = results.filter(r => !r.success);
  if (failed.length > 0) {
    process.exit(1);
  }
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

export { regenerateAllTemplates, regenerateTemplate };


