/**
 * Kelly Lipsync Pipeline - Complete Runner
 * 
 * Orchestrates the full pipeline:
 * 1. Generate audio files using ElevenLabs
 * 2. Generate phoneme alignments
 * 3. Store everything in Supabase
 * 
 * Usage:
 *   npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --days 1-30
 *   npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --day 1 --dry-run
 */

import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCRIPT_DIR = __dirname;

// Load both .env and .env.local
dotenv.config({ path: path.join(__dirname, '../../.env') });
dotenv.config({ path: path.join(__dirname, '../../.env.local') });

interface PipelineConfig {
  startDay: number;
  endDay: number;
  ageBuckets: string[];
  language: string;
  dryRun: boolean;
  skipAudio: boolean;
  skipAlignment: boolean;
}

async function runCommand(
  command: string,
  args: string[],
  dryRun: boolean
): Promise<void> {
  console.log(`\n📌 Running: ${command} ${args.join(' ')}`);
  
  if (dryRun) {
    console.log('   (DRY RUN - skipped)');
    return;
  }
  
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      stdio: 'inherit',
      shell: true,
      // Pass environment variables to child process
      env: { ...process.env },
      cwd: path.join(__dirname, '../..'),
    });
    
    proc.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error(`Command failed with code ${code}`));
    });
    
    proc.on('error', reject);
  });
}

async function runPipeline(config: PipelineConfig): Promise<void> {
  console.log('\n🚀 KELLY LIPSYNC PIPELINE');
  console.log('='.repeat(60));
  console.log(`Days: ${config.startDay}-${config.endDay}`);
  console.log(`Ages: ${config.ageBuckets.join(', ')}`);
  console.log(`Language: ${config.language}`);
  console.log(`Dry Run: ${config.dryRun}`);
  console.log('='.repeat(60));
  
  // Check prerequisites
  console.log('\n📋 Checking prerequisites...');
  
  if (!process.env.ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set');
    process.exit(1);
  }
  console.log('   ✓ ElevenLabs API key');
  
  if (!process.env.PUBLIC_SUPABASE_URL && !process.env.NEXT_PUBLIC_SUPABASE_URL) {
    console.error('❌ Supabase URL not set');
    process.exit(1);
  }
  console.log('   ✓ Supabase URL');
  
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.error('❌ SUPABASE_SERVICE_ROLE_KEY not set');
    process.exit(1);
  }
  console.log('   ✓ Supabase service key');
  
  // Estimate costs
  const totalFiles = (config.endDay - config.startDay + 1) * config.ageBuckets.length * 4; // script + 3 responses
  const estimatedCost = totalFiles * 0.015; // ~$0.015 per short audio
  console.log(`\n💰 Estimated audio generation cost: $${estimatedCost.toFixed(2)} (${totalFiles} files)`);
  
  if (!config.dryRun) {
    console.log('\nStarting in 5 seconds... (Ctrl+C to cancel)');
    await sleep(5000);
  }
  
  // Step 1: Generate Audio
  if (!config.skipAudio) {
    console.log('\n' + '='.repeat(60));
    console.log('STEP 1: AUDIO GENERATION');
    console.log('='.repeat(60));
    
    const audioArgs = [
      path.join(SCRIPT_DIR, 'generate-lesson-audio.ts'),
      '--days', `${config.startDay}-${config.endDay}`,
      '--ages', config.ageBuckets.join(','),
      '--lang', config.language,
    ];
    
    await runCommand('npx', ['ts-node', ...audioArgs], config.dryRun);
  }
  
  // Step 2: Generate Alignments
  if (!config.skipAlignment) {
    console.log('\n' + '='.repeat(60));
    console.log('STEP 2: ALIGNMENT GENERATION');
    console.log('='.repeat(60));
    
    const alignArgs = [
      path.join(SCRIPT_DIR, 'generate-alignments.ts'),
    ];
    
    await runCommand('npx', ['ts-node', ...alignArgs], config.dryRun);
  }
  
  // Done
  console.log('\n' + '='.repeat(60));
  console.log('✅ PIPELINE COMPLETE');
  console.log('='.repeat(60));
  
  console.log(`
Next steps:
1. Upload audio files to Supabase Storage or CDN
2. Update lesson_assets table with audio URLs
3. Test lipsync playback in the lesson player
4. Verify blendshape data in lipsync_alignments table

Files created:
- ./generated-audio/          - Audio files by day
- ./generated-audio/manifest.json - Audio manifest
- ./generated-alignments/     - Alignment JSON files
  `);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function parseArgs(): PipelineConfig {
  const args = process.argv.slice(2);
  const config: PipelineConfig = {
    startDay: 1,
    endDay: 30,
    ageBuckets: ['2-5', '6-12', '13-17', '18-35', '36-60', '61+'],
    language: 'en',
    dryRun: false,
    skipAudio: false,
    skipAlignment: false,
  };
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--days':
        const days = args[++i];
        if (days.includes('-')) {
          const [start, end] = days.split('-').map(Number);
          config.startDay = start;
          config.endDay = end;
        } else {
          config.startDay = config.endDay = Number(days);
        }
        break;
        
      case '--day':
        config.startDay = config.endDay = Number(args[++i]);
        break;
        
      case '--ages':
        config.ageBuckets = args[++i].split(',');
        break;
        
      case '--lang':
        config.language = args[++i];
        break;
        
      case '--dry-run':
        config.dryRun = true;
        break;
        
      case '--skip-audio':
        config.skipAudio = true;
        break;
        
      case '--skip-alignment':
        config.skipAlignment = true;
        break;
        
      case '--help':
        console.log(`
Kelly Lipsync Pipeline Runner

Usage:
  npx ts-node run-pipeline.ts [options]

Options:
  --days <range>      Day range (e.g., "1-30" or "1-5")
  --day <number>      Single day
  --ages <list>       Age buckets (default: all 6)
  --lang <code>       Language code (default: "en")
  --dry-run           Show what would be done without doing it
  --skip-audio        Skip audio generation step
  --skip-alignment    Skip alignment generation step
  --help              Show this help

Examples:
  npx ts-node run-pipeline.ts --days 1-5 --dry-run
  npx ts-node run-pipeline.ts --days 1-30 --ages 6-12,18-35
  npx ts-node run-pipeline.ts --skip-audio  # Only regenerate alignments
        `);
        process.exit(0);
    }
  }
  
  return config;
}

// Main
const config = parseArgs();
runPipeline(config).catch(error => {
  console.error('Pipeline failed:', error);
  process.exit(1);
});

