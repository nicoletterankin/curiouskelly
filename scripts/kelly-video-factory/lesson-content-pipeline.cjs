#!/usr/bin/env node
/**
 * Kelly Lesson Content Pipeline
 * 
 * THE MISSING PIECE: Connects actual lesson content to video generation
 * 
 * Structure:
 *   - 365 days × 5 phases × 12 archetypes = 21,900 unique scripts
 *   - We reuse: 1 image per day/phase, 1 animation per image
 *   - We generate: unique audio per script, lipsync per audio
 * 
 * Run: node lesson-content-pipeline.cjs --day 1 [--archetype "The Explorer"]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/lesson-videos');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Phase mapping: lesson_atoms phase → our template
const PHASE_TO_TEMPLATE = {
  'Hook': 'excited',
  'Fact1': 'curious',
  'Fact2': 'explain',
  'Fact3': 'thoughtful',
  'Wisdom': 'heartfelt'
};

// Phase mapping for our system
const PHASE_TO_KEY = {
  'Hook': 'hook',
  'Fact1': 'q1',
  'Fact2': 'q2',
  'Fact3': 'q3',
  'Wisdom': 'wisdom'
};

async function fetchLessonContent(dayNumber, archetype = null) {
  let query = supabase
    .from('lesson_atoms')
    .select(`
      id,
      phase,
      archetype,
      content,
      core_lessons!inner(day_number, topic)
    `)
    .eq('core_lessons.day_number', dayNumber)
    .in('phase', ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom']);
  
  if (archetype) {
    query = query.eq('archetype', archetype);
  }
  
  const { data, error } = await query.order('phase');
  
  if (error) throw new Error(`Failed to fetch lesson content: ${error.message}`);
  return data;
}

async function generateAudio(text, voiceId = process.env.ELEVENLABS_KELLY_VOICE_ID) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify({
      text,
      model_id: 'eleven_turbo_v2_5',
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.85,
        use_speaker_boost: true
      }
    });
    
    const options = {
      hostname: 'api.elevenlabs.io',
      path: `/v1/text-to-speech/${voiceId}`,
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': process.env.ELEVENLABS_API_KEY
      }
    };
    
    const req = https.request(options, (res) => {
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`ElevenLabs error: ${res.statusCode}`));
        } else {
          resolve(Buffer.concat(chunks));
        }
      });
    });
    
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

async function applyLipsync(animationUrl, audioBuffer) {
  const replicate = require('replicate');
  const client = new replicate.default({ auth: process.env.REPLICATE_API_TOKEN });
  
  // Convert audio to data URL
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const output = await client.run(
    'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    {
      input: {
        face: animationUrl,
        audio: audioBase64,
        fps: 25,
        smooth: true,
        resize_factor: 1
      }
    }
  );
  
  return output;
}

async function processLessonDay(dayNumber, options = {}) {
  const archetypeFilter = options.archetype;
  const dryRun = options.dryRun || false;
  
  console.log('═'.repeat(70));
  console.log('📚 LESSON CONTENT PIPELINE');
  console.log('   Generating Kelly videos from actual lesson content');
  console.log('═'.repeat(70));
  console.log(`\n  Day: ${dayNumber}`);
  if (archetypeFilter) console.log(`  Archetype: ${archetypeFilter}`);
  
  // Fetch lesson content
  const content = await fetchLessonContent(dayNumber, archetypeFilter);
  console.log(`  Found ${content.length} lesson atoms`);
  
  if (content.length === 0) {
    console.log('\n  ❌ No content found for this day/archetype');
    return;
  }
  
  // Group by phase
  const byPhase = {};
  for (const atom of content) {
    const phaseKey = PHASE_TO_KEY[atom.phase];
    if (!byPhase[phaseKey]) byPhase[phaseKey] = [];
    byPhase[phaseKey].push(atom);
  }
  
  console.log('\n  Content structure:');
  for (const [phase, atoms] of Object.entries(byPhase)) {
    console.log(`    ${phase}: ${atoms.length} archetypes`);
  }
  
  if (dryRun) {
    console.log('\n  [DRY RUN - showing first script per phase]\n');
    for (const [phase, atoms] of Object.entries(byPhase)) {
      const script = atoms[0]?.content?.script;
      if (script) {
        console.log(`  ${phase.toUpperCase()}:`);
        console.log(`    "${script.substring(0, 100)}..."\n`);
      }
    }
    return;
  }
  
  // Check for existing animations
  const { data: animations } = await supabase
    .from('kelly_video_assets')
    .select('phase, public_url')
    .eq('day_number', dayNumber)
    .eq('asset_type', 'animation');
  
  const animationMap = {};
  animations?.forEach(a => { animationMap[a.phase] = a.public_url; });
  
  console.log(`\n  Available animations: ${Object.keys(animationMap).length}/5`);
  
  if (Object.keys(animationMap).length === 0) {
    console.log('\n  ⚠️ No animations found. Generate animations first:');
    console.log(`    node batch-animation-generator.cjs --days ${dayNumber}`);
    return;
  }
  
  // Process each atom
  console.log('\n  Processing...\n');
  let processed = 0;
  let failed = 0;
  
  for (const [phase, atoms] of Object.entries(byPhase)) {
    const animationUrl = animationMap[phase];
    if (!animationUrl) {
      console.log(`  ⏭️ Skipping ${phase} - no animation`);
      continue;
    }
    
    for (const atom of atoms) {
      const script = atom.content?.script;
      if (!script) continue;
      
      const filename = `day_${String(dayNumber).padStart(3, '0')}_${phase}_${atom.archetype.replace(/\s+/g, '_')}.mp4`;
      
      process.stdout.write(`  🎬 ${filename}...`);
      
      try {
        // Generate audio
        const audioBuffer = await generateAudio(script);
        
        // Apply lipsync
        const videoUrl = await applyLipsync(animationUrl, audioBuffer);
        
        // Save locally
        const localPath = path.join(OUTPUT_DIR, filename);
        // Download and save...
        
        // Register in database
        await supabase.from('kelly_video_assets').insert({
          day_number: dayNumber,
          phase,
          template: PHASE_TO_TEMPLATE[atom.phase],
          asset_type: 'video',
          age_bucket: atom.archetype,
          language: 'en',
          storage_path: `lesson-videos/${filename}`,
          public_url: videoUrl,
          quality_tier: 'standard',
          status: 'generated'
        });
        
        processed++;
        console.log(' ✅');
        
      } catch (error) {
        failed++;
        console.log(` ❌ ${error.message}`);
      }
    }
  }
  
  console.log('\n' + '═'.repeat(70));
  console.log(`📊 COMPLETE: ${processed} videos, ${failed} failed`);
  console.log('═'.repeat(70));
}

async function showStats() {
  console.log('═'.repeat(70));
  console.log('📊 LESSON CONTENT STATS');
  console.log('═'.repeat(70));
  
  // Count atoms by day
  const { data: atomCounts } = await supabase
    .from('lesson_atoms')
    .select('core_lessons!inner(day_number)')
    .then(({ data }) => {
      const counts = {};
      data?.forEach(d => {
        const day = d.core_lessons.day_number;
        counts[day] = (counts[day] || 0) + 1;
      });
      return { data: counts };
    });
  
  const days = Object.keys(atomCounts || {}).length;
  const totalAtoms = Object.values(atomCounts || {}).reduce((a, b) => a + b, 0);
  
  console.log(`\n  Total days with content: ${days}`);
  console.log(`  Total lesson atoms: ${totalAtoms.toLocaleString()}`);
  console.log(`  Atoms per day: ~${Math.round(totalAtoms / days)}`);
  
  // Count generated videos
  const { count: videoCount } = await supabase
    .from('kelly_video_assets')
    .select('*', { count: 'exact', head: true })
    .eq('asset_type', 'video');
  
  const { count: animCount } = await supabase
    .from('kelly_video_assets')
    .select('*', { count: 'exact', head: true })
    .eq('asset_type', 'animation');
  
  console.log(`\n  Generated animations: ${animCount || 0}`);
  console.log(`  Generated videos: ${videoCount || 0}`);
  
  // Calculate what's needed
  const neededVideos = totalAtoms;
  const neededAnimations = days * 5;
  
  console.log(`\n  NEEDED:`);
  console.log(`    Animations: ${neededAnimations} (${animCount || 0} done = ${(((animCount || 0) / neededAnimations) * 100).toFixed(1)}%)`);
  console.log(`    Videos: ${neededVideos.toLocaleString()} (${videoCount || 0} done = ${(((videoCount || 0) / neededVideos) * 100).toFixed(3)}%)`);
}

// Main
async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--stats')) {
    return showStats();
  }
  
  const dayIndex = args.indexOf('--day');
  const day = dayIndex > -1 ? parseInt(args[dayIndex + 1]) : null;
  
  const archetypeIndex = args.indexOf('--archetype');
  const archetype = archetypeIndex > -1 ? args[archetypeIndex + 1] : null;
  
  const dryRun = args.includes('--dry-run');
  
  if (!day) {
    console.log(`
Lesson Content Pipeline

Usage:
  node lesson-content-pipeline.cjs --stats                    Show content statistics
  node lesson-content-pipeline.cjs --day 1 --dry-run         Preview Day 1 content
  node lesson-content-pipeline.cjs --day 1                   Generate all Day 1 videos
  node lesson-content-pipeline.cjs --day 1 --archetype "The Explorer"  Single archetype

This connects actual lesson content to Kelly video generation.
`);
    return;
  }
  
  await processLessonDay(day, { archetype, dryRun });
}

main().catch(console.error);

