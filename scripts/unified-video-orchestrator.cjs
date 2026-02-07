#!/usr/bin/env node
/**
 * UNIFIED VIDEO ORCHESTRATOR
 * 
 * Generates lip-synced Kelly videos using multiple providers:
 * 1. HeyGen (best quality) - Primary
 * 2. Sync Labs (good quality) - Secondary
 * 3. Fal.ai MuseTalk (acceptable) - Fallback
 * 
 * All results written to heygen_videos table for unified playback.
 * 
 * Usage:
 *   node scripts/unified-video-orchestrator.cjs --days=1-7
 *   node scripts/unified-video-orchestrator.cjs --days=1-30 --phases=hook
 *   node scripts/unified-video-orchestrator.cjs --provider=heygen --days=1-7
 */

require('dotenv').config();
const fs = require('fs');
const { Pool } = require('pg');

// Configuration
const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY,
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY,
  FAL_API_KEY: process.env.FAL_KEY || process.env.FAL_API_KEY,
  DATABASE_URL: process.env.DATABASE_URL || process.env.NEON_DATABASE_URL,
};

const pool = new Pool({
  connectionString: CONFIG.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Kelly avatar IDs for HeyGen
const HEYGEN_AVATARS = {
  adult: {
    storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
    scientist: '7bb18cddacd44333813cc90ffa44f766',
    explorer: '45e5ef8b651846e0b62b7477e552e87b',
  }
};

// Default scripts for each phase
const PHASE_SCRIPTS = {
  hook: "Welcome to today's lesson! Let's spark your curiosity with something fascinating.",
  story: "Let me tell you an incredible story that will change how you see the world.",
  wonder: "Now let's explore the deeper questions this brings up. What do you wonder about?",
  action: "Time to put this into practice! Here's something you can try right now.",
  wisdom: "Before we go, let me share one important insight to carry with you today."
};

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Progress tracking
const progress = {
  started: new Date().toISOString(),
  total: 0,
  completed: 0,
  failed: 0,
  byProvider: { heygen: 0, sync: 0, fal: 0 },
  jobs: []
};

// ============================================================================
// PROVIDER: HEYGEN
// ============================================================================
async function generateWithHeyGen(job) {
  if (!CONFIG.HEYGEN_API_KEY) {
    return { success: false, error: 'No HeyGen API key' };
  }
  
  const avatarId = HEYGEN_AVATARS[job.age]?.[job.archetype] || HEYGEN_AVATARS.adult.storyteller;
  const script = job.script || PHASE_SCRIPTS[job.phase];
  
  try {
    // Submit video generation
    const submitRes = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': CONFIG.HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        video_inputs: [{
          character: {
            type: 'talking_photo',
            talking_photo_id: avatarId
          },
          voice: {
            type: 'text',
            input_text: script,
            voice_id: '1bd001e7e50f421d891986aad5158bc8'
          }
        }],
        dimension: { width: 1920, height: 1080 },
        test: false,
        title: `Day ${job.day} - ${job.phase.toUpperCase()}`
      })
    });
    
    const submitData = await submitRes.json();
    
    if (!submitData.data?.video_id) {
      return { success: false, error: submitData.error?.message || 'No video_id returned' };
    }
    
    // Return the video_id - we'll poll for completion separately
    return {
      success: true,
      provider: 'heygen',
      videoId: submitData.data.video_id,
      status: 'processing'
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ============================================================================
// PROVIDER: SYNC LABS
// ============================================================================
async function generateWithSyncLabs(job) {
  if (!CONFIG.SYNC_LABS_API_KEY) {
    return { success: false, error: 'No Sync Labs API key' };
  }
  
  // Need audio URL and base video URL for Sync Labs
  const audioUrl = job.audioUrl;
  const baseVideoUrl = 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/064dbfab193c461fbb2869f27d663c7b.mp4';
  
  if (!audioUrl) {
    return { success: false, error: 'No audio URL for Sync Labs' };
  }
  
  try {
    const response = await fetch('https://api.sync.so/v2/generate', {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.SYNC_LABS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'lipsync-2-pro',
        input: [
          { type: 'video', url: baseVideoUrl },
          { type: 'audio', url: audioUrl }
        ],
        options: { output_format: 'mp4' }
      })
    });
    
    const data = await response.json();
    
    if (data.id) {
      return {
        success: true,
        provider: 'sync',
        videoId: data.id,
        status: 'processing'
      };
    }
    
    return { success: false, error: data.error || 'Unknown error' };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ============================================================================
// PROVIDER: FAL.AI (MUSETALK)
// ============================================================================
async function generateWithFal(job) {
  if (!CONFIG.FAL_API_KEY) {
    return { success: false, error: 'No Fal API key' };
  }
  
  const audioUrl = job.audioUrl;
  const imageUrl = '/kelly/archetypes/storyteller.png'; // Need full URL
  
  if (!audioUrl) {
    return { success: false, error: 'No audio URL for Fal' };
  }
  
  try {
    const response = await fetch('https://queue.fal.run/fal-ai/musetalk', {
      method: 'POST',
      headers: {
        'Authorization': `Key ${CONFIG.FAL_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        source_image: imageUrl,
        audio_input: { audio_url: audioUrl }
      })
    });
    
    const data = await response.json();
    
    if (data.request_id) {
      return {
        success: true,
        provider: 'fal',
        videoId: data.request_id,
        status: 'processing'
      };
    }
    
    return { success: false, error: data.detail || 'Unknown error' };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ============================================================================
// DATABASE
// ============================================================================
async function saveJobToDatabase(job, result) {
  try {
    await pool.query(`
      INSERT INTO heygen_videos (
        day_of_year, phase, age_category, archetype, language,
        heygen_video_id, status, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
      ON CONFLICT (day_of_year, phase, age_category, archetype, language)
      DO UPDATE SET 
        heygen_video_id = $6,
        status = $7,
        updated_at = NOW()
    `, [job.day, job.phase, job.age, job.archetype, job.language, result.videoId, 'processing']);
    
    return true;
  } catch (err) {
    console.error('  DB Error:', err.message);
    return false;
  }
}

async function getAudioUrl(day, phase, age, language) {
  try {
    const result = await pool.query(`
      SELECT audio_url FROM kelly_lesson_assets
      WHERE day_number = $1 AND phase = $2 AND language = $3
        AND audio_url IS NOT NULL
      LIMIT 1
    `, [day, phase, language]);
    
    return result.rows[0]?.audio_url || null;
  } catch {
    return null;
  }
}

// ============================================================================
// MAIN ORCHESTRATOR
// ============================================================================
async function generateVideo(job, preferredProvider = 'heygen') {
  const providers = [
    { name: 'heygen', fn: generateWithHeyGen },
    { name: 'sync', fn: generateWithSyncLabs },
    { name: 'fal', fn: generateWithFal },
  ];
  
  // Reorder based on preference
  if (preferredProvider !== 'heygen') {
    const idx = providers.findIndex(p => p.name === preferredProvider);
    if (idx > 0) {
      const preferred = providers.splice(idx, 1)[0];
      providers.unshift(preferred);
    }
  }
  
  // Get audio URL for non-HeyGen providers
  job.audioUrl = await getAudioUrl(job.day, job.phase, job.age, job.language);
  
  for (const provider of providers) {
    process.stdout.write(`  Trying ${provider.name}... `);
    
    const result = await provider.fn(job);
    
    if (result.success) {
      console.log(`✅ ${result.videoId}`);
      await saveJobToDatabase(job, result);
      progress.byProvider[provider.name]++;
      progress.completed++;
      progress.jobs.push({ ...job, result });
      return result;
    } else {
      console.log(`❌ ${result.error}`);
    }
  }
  
  progress.failed++;
  return { success: false, error: 'All providers failed' };
}

// ============================================================================
// CLI
// ============================================================================
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    days: [1, 2, 3, 4, 5, 6, 7],
    phases: PHASES,
    age: 'adult',
    archetype: 'storyteller',
    language: 'en',
    provider: 'heygen'
  };
  
  for (const arg of args) {
    if (arg.startsWith('--days=')) {
      const range = arg.split('=')[1];
      if (range.includes('-')) {
        const [start, end] = range.split('-').map(Number);
        config.days = Array.from({ length: end - start + 1 }, (_, i) => start + i);
      } else {
        config.days = range.split(',').map(Number);
      }
    }
    if (arg.startsWith('--phases=')) {
      config.phases = arg.split('=')[1].split(',');
    }
    if (arg.startsWith('--provider=')) {
      config.provider = arg.split('=')[1];
    }
  }
  
  return config;
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║         UNIFIED VIDEO ORCHESTRATOR                         ║');
  console.log('║         Kelly Lip-Sync Generation Pipeline                 ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  const config = parseArgs();
  
  console.log('Configuration:');
  console.log(`  Days: ${config.days[0]}-${config.days[config.days.length - 1]} (${config.days.length} days)`);
  console.log(`  Phases: ${config.phases.join(', ')}`);
  console.log(`  Provider: ${config.provider}`);
  console.log(`  Age: ${config.age}`);
  console.log(`  Archetype: ${config.archetype}`);
  console.log('');
  
  // Check API keys
  console.log('API Keys:');
  console.log(`  HeyGen: ${CONFIG.HEYGEN_API_KEY ? '✅' : '❌'}`);
  console.log(`  Sync Labs: ${CONFIG.SYNC_LABS_API_KEY ? '✅' : '❌'}`);
  console.log(`  Fal.ai: ${CONFIG.FAL_API_KEY ? '✅' : '❌'}`);
  console.log('');
  
  const totalJobs = config.days.length * config.phases.length;
  progress.total = totalJobs;
  
  console.log(`Starting ${totalJobs} video generation jobs...\n`);
  
  let jobNum = 0;
  for (const day of config.days) {
    console.log(`\n📅 Day ${day}:`);
    
    for (const phase of config.phases) {
      jobNum++;
      console.log(`\n[${jobNum}/${totalJobs}] Day ${day} / ${phase}`);
      
      const job = {
        day,
        phase,
        age: config.age,
        archetype: config.archetype,
        language: config.language,
        script: PHASE_SCRIPTS[phase]
      };
      
      await generateVideo(job, config.provider);
      
      // Rate limit
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  
  // Summary
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║                    GENERATION COMPLETE                      ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log(`\n📊 Summary:`);
  console.log(`  Total jobs: ${progress.total}`);
  console.log(`  Completed: ${progress.completed}`);
  console.log(`  Failed: ${progress.failed}`);
  console.log(`\n📦 By Provider:`);
  console.log(`  HeyGen: ${progress.byProvider.heygen}`);
  console.log(`  Sync Labs: ${progress.byProvider.sync}`);
  console.log(`  Fal.ai: ${progress.byProvider.fal}`);
  
  // Save progress
  fs.writeFileSync('orchestrator-progress.json', JSON.stringify(progress, null, 2));
  console.log('\n💾 Progress saved to orchestrator-progress.json');
  
  await pool.end();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
