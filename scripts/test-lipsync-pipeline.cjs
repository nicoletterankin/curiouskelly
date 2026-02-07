/**
 * Sprint E: Test Lip-Sync Pipeline End-to-End
 * Tests viseme data flow from audio to mouth shapes
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function testPipeline() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== Lip-Sync Pipeline Test ===\n');
  const results = { tests: [], summary: {} };
  
  // TEST 1: Check if viseme data exists in DB
  console.log('TEST 1: Viseme data in kelly_lesson_assets');
  try {
    const visemeCount = await client.query(`
      SELECT 
        COUNT(*) as total,
        COUNT(viseme_data) as with_viseme,
        COUNT(audio_url) as with_audio,
        COUNT(video_url) as with_video
      FROM kelly_lesson_assets
    `);
    const v = visemeCount.rows[0];
    console.log(`  Total assets: ${v.total}`);
    console.log(`  With viseme_data: ${v.with_viseme} (${Math.round(v.with_viseme/v.total*100)}%)`);
    console.log(`  With audio_url: ${v.with_audio}`);
    console.log(`  With video_url: ${v.with_video}`);
    results.tests.push({
      name: 'viseme_data_in_db',
      status: parseInt(v.with_viseme) > 0 ? 'pass' : 'fail',
      details: v
    });
  } catch (e) {
    console.log('  ERROR:', e.message);
    results.tests.push({ name: 'viseme_data_in_db', status: 'error', error: e.message });
  }
  
  // TEST 2: Check pre-computed viseme JSON files
  console.log('\nTEST 2: Pre-computed viseme JSON files');
  const motionDir = path.join(__dirname, '..', 'public', 'kelly-motion', 'personas');
  if (fs.existsSync(motionDir)) {
    let visemeFiles = 0;
    let totalVisemes = 0;
    function walk(dir) {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (entry.name.endsWith('_unity.json')) {
          visemeFiles++;
          try {
            const data = JSON.parse(fs.readFileSync(full, 'utf-8'));
            if (data.visemes) totalVisemes += data.visemes.length;
          } catch (e) {}
        }
      }
    }
    walk(motionDir);
    console.log(`  Found ${visemeFiles} viseme JSON files`);
    console.log(`  Total viseme events: ${totalVisemes}`);
    results.tests.push({
      name: 'precomputed_viseme_files',
      status: visemeFiles > 0 ? 'pass' : 'fail',
      details: { files: visemeFiles, events: totalVisemes }
    });
  } else {
    console.log('  Directory not found:', motionDir);
    results.tests.push({ name: 'precomputed_viseme_files', status: 'missing', details: { path: motionDir } });
  }
  
  // TEST 3: Check kelly-lipsync.js exists and has expected functions
  console.log('\nTEST 3: Kelly LipSync Engine');
  const lipsyncPath = path.join(__dirname, '..', 'public', 'js', 'kelly-lipsync.js');
  if (fs.existsSync(lipsyncPath)) {
    const content = fs.readFileSync(lipsyncPath, 'utf-8');
    const hasOnBlendshapes = content.includes('onBlendshapesUpdate');
    const hasAudioContext = content.includes('AudioContext');
    const hasAnalyser = content.includes('AnalyserNode') || content.includes('createAnalyser');
    const hasVisemeMapping = content.includes('viseme') || content.includes('VISEME');
    
    console.log(`  File size: ${content.length} chars`);
    console.log(`  Has onBlendshapesUpdate callback: ${hasOnBlendshapes ? 'YES' : 'NO'}`);
    console.log(`  Has AudioContext: ${hasAudioContext ? 'YES' : 'NO'}`);
    console.log(`  Has AnalyserNode: ${hasAnalyser ? 'YES' : 'NO'}`);
    console.log(`  Has viseme mapping: ${hasVisemeMapping ? 'YES' : 'NO'}`);
    
    results.tests.push({
      name: 'lipsync_engine',
      status: hasOnBlendshapes && hasAudioContext ? 'pass' : 'partial',
      details: { onBlendshapes: hasOnBlendshapes, audioContext: hasAudioContext, analyser: hasAnalyser, viseme: hasVisemeMapping }
    });
  } else {
    console.log('  File not found:', lipsyncPath);
    results.tests.push({ name: 'lipsync_engine', status: 'missing' });
  }
  
  // TEST 4: Check PixiJS compositor
  console.log('\nTEST 4: PixiJS Mouth Compositor');
  const pixiPath = path.join(__dirname, '..', 'public', 'js', 'kelly-pixi-compositor.js');
  if (fs.existsSync(pixiPath)) {
    const content = fs.readFileSync(pixiPath, 'utf-8');
    const hasSetBlendshapes = content.includes('setBlendshapes');
    const hasMouth = content.includes('mouth') || content.includes('Mouth');
    const hasJawOpen = content.includes('jawOpen');
    
    console.log(`  File size: ${content.length} chars`);
    console.log(`  Has setBlendshapes: ${hasSetBlendshapes ? 'YES' : 'NO'}`);
    console.log(`  Has mouth rendering: ${hasMouth ? 'YES' : 'NO'}`);
    console.log(`  Has jawOpen blendshape: ${hasJawOpen ? 'YES' : 'NO'}`);
    
    results.tests.push({
      name: 'pixi_compositor',
      status: hasSetBlendshapes && hasMouth ? 'pass' : 'partial',
      details: { setBlendshapes: hasSetBlendshapes, mouth: hasMouth, jawOpen: hasJawOpen }
    });
  } else {
    console.log('  File not found:', pixiPath);
    results.tests.push({ name: 'pixi_compositor', status: 'missing' });
  }
  
  // TEST 5: Check learn.html integration
  console.log('\nTEST 5: learn.html integration wiring');
  const learnPath = path.join(__dirname, '..', 'public', 'learn.html');
  if (fs.existsSync(learnPath)) {
    const content = fs.readFileSync(learnPath, 'utf-8');
    const hasLipSyncScript = content.includes('kelly-lipsync');
    const hasPixiScript = content.includes('kelly-pixi-compositor');
    const hasCallback = content.includes('onBlendshapesUpdate');
    const hasSetBlendshapes = content.includes('setBlendshapes');
    
    console.log(`  Includes kelly-lipsync.js: ${hasLipSyncScript ? 'YES' : 'NO'}`);
    console.log(`  Includes kelly-pixi-compositor.js: ${hasPixiScript ? 'YES' : 'NO'}`);
    console.log(`  Wires onBlendshapesUpdate: ${hasCallback ? 'YES' : 'NO'}`);
    console.log(`  Calls setBlendshapes: ${hasSetBlendshapes ? 'YES' : 'NO'}`);
    
    results.tests.push({
      name: 'learn_html_wiring',
      status: hasCallback && hasSetBlendshapes ? 'pass' : 'broken',
      details: { lipsync: hasLipSyncScript, pixi: hasPixiScript, callback: hasCallback, setBlendshapes: hasSetBlendshapes }
    });
  } else {
    console.log('  File not found:', learnPath);
    results.tests.push({ name: 'learn_html_wiring', status: 'missing' });
  }
  
  // TEST 6: Test kelly-lipsync Cloudflare Worker
  console.log('\nTEST 6: Kelly LipSync Cloudflare Worker');
  const workerUrl = 'https://kelly-lipsync.nicoletterankin.workers.dev';
  try {
    const res = await fetch(workerUrl, { method: 'GET', signal: AbortSignal.timeout(10000) });
    console.log(`  Status: ${res.status}`);
    const body = await res.text();
    console.log(`  Response: ${body.substring(0, 200)}`);
    results.tests.push({
      name: 'cloudflare_worker',
      status: res.status === 200 ? 'pass' : 'partial',
      details: { status: res.status, body: body.substring(0, 200) }
    });
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    results.tests.push({ name: 'cloudflare_worker', status: 'error', error: e.message });
  }
  
  // TEST 7: Check KellyOS viseme timelines from Sprint 4
  console.log('\nTEST 7: KellyOS viseme timelines in database');
  try {
    const vt = await client.query(`
      SELECT COUNT(*) as total,
             COUNT(alignment_json) as with_alignment
      FROM kellyos_lessons
    `);
    const r = vt.rows[0];
    console.log(`  Total kellyos_lessons: ${r.total}`);
    console.log(`  With alignment_json: ${r.with_alignment}`);
    results.tests.push({
      name: 'kellyos_viseme_timelines',
      status: parseInt(r.with_alignment) > 0 ? 'pass' : 'fail',
      details: r
    });
  } catch (e) {
    console.log('  ERROR:', e.message);
    results.tests.push({ name: 'kellyos_viseme_timelines', status: 'error', error: e.message });
  }
  
  // TEST 8: Check Sync Labs / MuseTalk provider configs
  console.log('\nTEST 8: Video generation providers');
  const hasSyncLabs = !!process.env.SYNC_LABS_API_KEY;
  const hasFalKey = !!process.env.FAL_KEY;
  console.log(`  SYNC_LABS_API_KEY: ${hasSyncLabs ? 'SET' : 'NOT SET'}`);
  console.log(`  FAL_KEY: ${hasFalKey ? 'SET' : 'NOT SET'}`);
  results.tests.push({
    name: 'provider_api_keys',
    status: hasSyncLabs || hasFalKey ? 'pass' : 'fail',
    details: { sync_labs: hasSyncLabs, fal: hasFalKey }
  });
  
  // SUMMARY
  const passed = results.tests.filter(t => t.status === 'pass').length;
  const partial = results.tests.filter(t => t.status === 'partial').length;
  const failed = results.tests.filter(t => t.status === 'fail' || t.status === 'error' || t.status === 'missing' || t.status === 'broken').length;
  
  results.summary = {
    total: results.tests.length,
    passed,
    partial,
    failed,
    verdict: failed === 0 ? 'PIPELINE OPERATIONAL' : `${failed} ISSUES FOUND`
  };
  
  console.log(`\n=== SUMMARY ===`);
  console.log(`${passed} passed, ${partial} partial, ${failed} failed`);
  console.log(`Verdict: ${results.summary.verdict}`);
  
  // Write results
  const outPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'lipsync-test-results.json');
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to: ${outPath}`);
  
  await client.end();
}

testPipeline().catch(e => { console.error(e); process.exit(1); });
