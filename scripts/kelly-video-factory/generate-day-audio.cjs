#!/usr/bin/env node
/**
 * Generate audio for a single day's lesson content
 * 
 * Run: node generate-day-audio.cjs --day 1
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

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/lesson-audio');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const PHASE_MAP = {
  'Hook': 'hook',
  'Fact1': 'q1', 
  'Fact2': 'q2',
  'Fact3': 'q3',
  'Wisdom': 'wisdom'
};

async function generateAudio(text) {
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
    
    const voiceId = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
    
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

async function uploadToSupabase(buffer, storagePath) {
  await supabase.storage.from('kelly-templates').upload(storagePath, buffer, {
    contentType: 'audio/mpeg',
    upsert: true
  });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(storagePath);
  return data.publicUrl;
}

async function main() {
  const args = process.argv.slice(2);
  const dayIndex = args.indexOf('--day');
  const day = dayIndex > -1 ? parseInt(args[dayIndex + 1]) : 1;
  const limit = args.includes('--limit') ? parseInt(args[args.indexOf('--limit') + 1]) : null;
  
  console.log('═'.repeat(70));
  console.log('🎙️ AUDIO GENERATION');
  console.log(`   Day ${day} lesson scripts → ElevenLabs audio`);
  console.log('═'.repeat(70));
  
  // Fetch lesson atoms for this day
  const { data: atoms, error } = await supabase
    .from('lesson_atoms')
    .select(`
      id,
      phase,
      archetype,
      content,
      core_lessons!inner(day_number, topic)
    `)
    .eq('core_lessons.day_number', day)
    .in('phase', ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'])
    .order('phase')
    .order('archetype');
  
  if (error) {
    console.log('Error:', error.message);
    return;
  }
  
  const toProcess = limit ? atoms.slice(0, limit) : atoms;
  console.log(`\n  Found ${atoms.length} atoms, processing ${toProcess.length}`);
  console.log(`  Topic: ${atoms[0]?.core_lessons?.topic || 'Unknown'}`);
  
  let generated = 0;
  let failed = 0;
  let cost = 0;
  const startTime = Date.now();
  
  console.log('\n');
  
  for (const atom of toProcess) {
    const script = atom.content?.script;
    if (!script) continue;
    
    const phase = PHASE_MAP[atom.phase] || atom.phase.toLowerCase();
    const archetype = atom.archetype.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
    const filename = `day_${String(day).padStart(3, '0')}_${phase}_${archetype}.mp3`;
    
    process.stdout.write(`  🎙️ ${filename}...`);
    
    try {
      // Generate audio
      const audioBuffer = await generateAudio(script);
      
      // Save locally
      const localPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(localPath, audioBuffer);
      
      // Upload to Supabase
      const storagePath = `production/audio/${filename}`;
      const publicUrl = await uploadToSupabase(audioBuffer, storagePath);
      
      // Register in database
      await supabase.from('kelly_video_assets').insert({
        day_number: day,
        phase,
        template: phase,
        asset_type: 'audio',
        age_bucket: atom.archetype,
        language: 'en',
        storage_bucket: 'kelly-templates',
        storage_path: storagePath,
        public_url: publicUrl,
        file_size_bytes: audioBuffer.length,
        quality_tier: 'standard',
        status: 'generated'
      });
      
      generated++;
      cost += script.length * 0.00003; // Approximate ElevenLabs cost
      console.log(` ✅ ${(audioBuffer.length / 1024).toFixed(1)}KB`);
      
    } catch (err) {
      failed++;
      console.log(` ❌ ${err.message}`);
    }
    
    // Small delay to avoid rate limits
    await new Promise(r => setTimeout(r, 500));
  }
  
  const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 AUDIO GENERATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Generated: ${generated}`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Duration: ${duration} minutes`);
  console.log(`  Est. Cost: $${cost.toFixed(3)}`);
  console.log(`  Output: ${OUTPUT_DIR}`);
}

main().catch(console.error);

