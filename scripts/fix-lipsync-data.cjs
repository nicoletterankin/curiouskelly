/**
 * Sprint E: Fix lip-sync data pipeline
 * 1. Add viseme_data column to kelly_lesson_assets if missing
 * 2. Populate from kellyos_lessons alignment_json
 */
require('dotenv').config();
const { Client } = require('pg');

async function fixLipsyncData() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== Fix Lip-Sync Data Pipeline ===\n');
  
  // Step 1: Add viseme_data column if it doesn't exist
  console.log('Step 1: Ensure viseme_data column exists on kelly_lesson_assets');
  try {
    await client.query(`
      ALTER TABLE kelly_lesson_assets 
      ADD COLUMN IF NOT EXISTS viseme_data JSONB
    `);
    console.log('  Column added/verified');
  } catch (e) {
    if (e.message.includes('already exists')) {
      console.log('  Column already exists');
    } else {
      console.log('  Note:', e.message);
    }
  }
  
  // Step 2: Populate from kellyos_lessons
  console.log('\nStep 2: Populate viseme_data from kellyos_lessons');
  
  // Get kellyos_lessons with alignment data
  const alignments = await client.query(`
    SELECT day_number, phase, alignment_json
    FROM kellyos_lessons
    WHERE alignment_json IS NOT NULL
  `);
  console.log(`  Found ${alignments.rows.length} alignment records in kellyos_lessons`);
  
  // Map phase numbers to phase names
  const phaseMap = {
    1: 'hook', 2: 'teach', 3: 'example', 4: 'practice',
    5: 'reflect', 6: 'apply', 7: 'close',
    // Also map old names
    'hook': 'hook', 'cliff': 'cliff', 'q1': 'q1', 'q2': 'q2',
    'q3': 'q3', 'wisdom': 'wisdom', 'outro': 'outro',
    'story': 'story', 'wonder': 'wonder', 'action': 'action'
  };
  
  let updated = 0;
  let notFound = 0;
  
  for (const row of alignments.rows) {
    // Try to match with kelly_lesson_assets by day_number + phase
    // Phase in kellyos_lessons is 1-5 (per the 5-slot structure)
    const phaseNames = [];
    const phase = row.phase;
    
    // Map numeric phase to possible column matches
    if (phase === 1) phaseNames.push('hook', 'cliff');
    else if (phase === 2) phaseNames.push('story', 'teach', 'q1');
    else if (phase === 3) phaseNames.push('wonder', 'example', 'q2');
    else if (phase === 4) phaseNames.push('action', 'practice', 'q3');
    else if (phase === 5) phaseNames.push('wisdom', 'close', 'outro');
    else phaseNames.push(String(phase));
    
    // Try updating kelly_lesson_assets
    for (const phaseName of phaseNames) {
      const res = await client.query(`
        UPDATE kelly_lesson_assets 
        SET viseme_data = $1
        WHERE day_number = $2 AND phase = $3 AND viseme_data IS NULL
      `, [JSON.stringify(row.alignment_json), parseInt(row.day_number), phaseName]);
      
      if (res.rowCount > 0) {
        updated += res.rowCount;
        break;
      }
    }
  }
  
  console.log(`  Updated ${updated} kelly_lesson_assets records`);
  console.log(`  ${notFound} records had no matching asset`);
  
  // Step 3: Verify
  console.log('\nStep 3: Verification');
  const verify = await client.query(`
    SELECT 
      COUNT(*) as total,
      COUNT(viseme_data) as with_viseme,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
  `);
  const v = verify.rows[0];
  console.log(`  Total assets: ${v.total}`);
  console.log(`  With viseme_data: ${v.with_viseme} (${Math.round(v.with_viseme / v.total * 100)}%)`);
  console.log(`  With audio: ${v.with_audio}`);
  console.log(`  With video: ${v.with_video}`);
  
  // Also update lesson_atoms with viseme info
  console.log('\nStep 4: Cross-populate to lesson_atoms (new schema)');
  try {
    const atomUpdate = await client.query(`
      UPDATE lesson_atoms la
      SET script = COALESCE(la.script, kl.text)
      FROM kellyos_lessons kl
      JOIN core_lessons_v2 cl ON cl.day_number = kl.day_number
      WHERE la.lesson_id = cl.id 
        AND la.phase = kl.phase
        AND la.script IS NULL
        AND kl.text IS NOT NULL
    `);
    console.log(`  Updated ${atomUpdate.rowCount} lesson_atoms with scripts from kellyos_lessons`);
  } catch (e) {
    console.log('  Step 4 skipped (type mismatch):', e.message.substring(0, 80));
  }
  
  await client.end();
  console.log('\nDone.');
}

fixLipsyncData().catch(e => { console.error(e); process.exit(1); });
