#!/usr/bin/env npx tsx
/**
 * Kelly Factory Runner
 * 
 * Orchestrates the full pipeline: scripts → audio → video
 * Runs workers in sequence and reports progress
 * 
 * Usage:
 *   npx tsx scripts/kelly-factory/factory-runner.ts             # Run full pipeline
 *   npx tsx scripts/kelly-factory/factory-runner.ts --day=39    # Process specific day
 *   npx tsx scripts/kelly-factory/factory-runner.ts --status    # Show status only
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { spawn } from 'child_process';
import path from 'path';

const CONFIG = {
  supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '',
  supabaseKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '',
};

const supabase = createClient(CONFIG.supabaseUrl, CONFIG.supabaseKey);

// ═══════════════════════════════════════════════════════════════════════════
// STATUS & DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

async function showStatus() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║                    🏭 KELLY CONTENT FACTORY                              ║
╚══════════════════════════════════════════════════════════════════════════╝
`);

  // Get overall stats
  const { data: stats } = await supabase.rpc('get_factory_stats');
  
  if (stats?.[0]) {
    const s = stats[0];
    const barWidth = 40;
    const completeBar = Math.round((s.complete / s.total_assets) * barWidth);
    const audioBar = Math.round((s.audio_ready / s.total_assets) * barWidth);
    const scriptBar = Math.round((s.script_ready / s.total_assets) * barWidth);
    
    console.log(`Overall Progress: ${s.progress_pct}%`);
    console.log(`${'█'.repeat(completeBar)}${'░'.repeat(barWidth - completeBar)} ${s.complete}/${s.total_assets}`);
    console.log(`
┌────────────────────────────────────────────────────────────────────┐
│ Status        │ Count     │ Progress                               │
├───────────────┼───────────┼────────────────────────────────────────┤
│ ✅ Complete   │ ${String(s.complete).padStart(9)} │ ${'█'.repeat(completeBar)}${'░'.repeat(20 - completeBar)} │
│ 🎵 Audio Ready│ ${String(s.audio_ready).padStart(9)} │ ${'█'.repeat(Math.min(audioBar, 20))}${'░'.repeat(20 - Math.min(audioBar, 20))} │
│ 📝 Script     │ ${String(s.script_ready).padStart(9)} │ ${'█'.repeat(Math.min(scriptBar, 20))}${'░'.repeat(20 - Math.min(scriptBar, 20))} │
│ ⏳ Pending    │ ${String(s.pending).padStart(9)} │                                        │
└───────────────┴───────────┴────────────────────────────────────────┘
`);
  }

  // Get per-day breakdown for non-empty days
  const { data: dashboard } = await supabase
    .from('kelly_factory_dashboard')
    .select('*')
    .or('complete.gt.0,audio_ready.gt.0,script_ready.gt.0')
    .order('day_number')
    .limit(20);

  if (dashboard && dashboard.length > 0) {
    console.log('Active Days:');
    console.log('┌─────────┬──────────┬────────────┬─────────────┬─────────┐');
    console.log('│ Day     │ Complete │ Audio Ready│ Script Ready│ Pending │');
    console.log('├─────────┼──────────┼────────────┼─────────────┼─────────┤');
    
    for (const day of dashboard) {
      console.log(`│ ${String(day.day_number).padStart(7)} │ ${String(day.complete).padStart(8)} │ ${String(day.audio_ready).padStart(10)} │ ${String(day.script_ready).padStart(11)} │ ${String(day.pending).padStart(7)} │`);
    }
    console.log('└─────────┴──────────┴────────────┴─────────────┴─────────┘');
  }

  // Check available backends
  console.log(`
Available Backends:
  ${process.env.FAL_KEY ? '✅' : '❌'} Fal.ai (FAL_KEY)
  ${process.env.HEYGEN_API_KEY ? '✅' : '❌'} HeyGen (HEYGEN_API_KEY)
  ${process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_LABS_API_KEY ? '✅' : '❌'} ElevenLabs (ELEVENLABS_API_KEY)
  ${process.env.REPLICATE_API_TOKEN ? '✅' : '❌'} Replicate (REPLICATE_API_TOKEN)
`);
}

// ═══════════════════════════════════════════════════════════════════════════
// WORKER RUNNERS
// ═══════════════════════════════════════════════════════════════════════════

function runWorker(script: string, args: string[]): Promise<number> {
  return new Promise((resolve) => {
    const scriptPath = path.join(__dirname, script);
    const child = spawn('npx', ['tsx', scriptPath, ...args], {
      stdio: 'inherit',
      env: process.env,
    });
    
    child.on('close', (code) => resolve(code || 0));
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  // Status only mode
  if (args.includes('--status')) {
    await showStatus();
    return;
  }

  // Parse options
  const dayArg = args.find(a => a.startsWith('--day='));
  const limitArg = args.find(a => a.startsWith('--limit='));
  const skipAudio = args.includes('--skip-audio');
  const skipVideo = args.includes('--skip-video');

  const workerArgs: string[] = [];
  if (dayArg) workerArgs.push(dayArg);
  if (limitArg) workerArgs.push(limitArg);

  // Show initial status
  await showStatus();

  // Run audio worker (script_ready → audio_ready)
  if (!skipAudio) {
    console.log('\n🎤 Running Audio Worker...\n');
    const audioCode = await runWorker('audio-worker.ts', workerArgs);
    if (audioCode !== 0) {
      console.log('⚠️  Audio worker had issues');
    }
  }

  // Run video worker (audio_ready → complete)
  if (!skipVideo) {
    console.log('\n🎬 Running Video Worker...\n');
    const videoCode = await runWorker('video-worker.ts', workerArgs);
    if (videoCode !== 0) {
      console.log('⚠️  Video worker had issues');
    }
  }

  // Show final status
  console.log('\n');
  await showStatus();
}

main().catch(console.error);
