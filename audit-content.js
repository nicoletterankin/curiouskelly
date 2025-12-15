/**
 * Full content audit - what exists vs what's needed
 */
const https = require('https');

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

function supabaseQuery(endpoint) {
  return new Promise((resolve, reject) => {
    const url = `${SUPABASE_URL}/rest/v1/${endpoint}`;
    const options = {
      headers: {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json',
        'Prefer': 'count=exact'
      }
    };
    https.get(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const count = res.headers['content-range']?.split('/')[1] || '?';
        try {
          resolve({ data: JSON.parse(data), count, status: res.statusCode });
        } catch (e) {
          resolve({ error: data, count, status: res.statusCode });
        }
      });
    }).on('error', reject);
  });
}

async function main() {
  console.log('=== FULL CONTENT AUDIT ===\n');

  // 1. Core Lessons - check what fields are populated
  console.log('1. CORE LESSONS');
  const lessons = await supabaseQuery('core_lessons?select=*&limit=5&order=day_number');
  console.log(`   Total: ${lessons.count} lessons`);
  if (lessons.data?.length) {
    const sample = lessons.data[0];
    console.log('\n   Sample lesson (Day 1):');
    Object.entries(sample).forEach(([key, val]) => {
      const display = val === null ? '❌ NULL' : 
                     val === '' ? '⚠️ EMPTY' :
                     typeof val === 'string' && val.length > 50 ? `✅ "${val.substring(0, 50)}..."` :
                     Array.isArray(val) ? `✅ [${val.length} items]` :
                     `✅ ${JSON.stringify(val)}`;
      console.log(`     ${key}: ${display}`);
    });
  }

  // 2. Lesson Atoms - check video URLs
  console.log('\n\n2. LESSON ATOMS (Dialog content)');
  const atoms = await supabaseQuery('lesson_atoms?select=id,core_lesson_id,archetype,phase,hd_video_url,visual_url&limit=20');
  console.log(`   Total: ${atoms.count} atoms`);
  
  let atomsWithVideo = 0;
  let atomsWithVisual = 0;
  if (atoms.data) {
    atoms.data.forEach(a => {
      if (a.hd_video_url) atomsWithVideo++;
      if (a.visual_url) atomsWithVisual++;
    });
    console.log(`   Sample of ${atoms.data.length}: ${atomsWithVideo} have hd_video_url, ${atomsWithVisual} have visual_url`);
    
    // Show a sample
    const sampleAtom = atoms.data[0];
    console.log('\n   Sample atom:');
    console.log(`     phase: ${sampleAtom.phase}`);
    console.log(`     archetype: ${sampleAtom.archetype}`);
    console.log(`     hd_video_url: ${sampleAtom.hd_video_url || '❌ NULL'}`);
    console.log(`     visual_url: ${sampleAtom.visual_url || '❌ NULL'}`);
  }

  // 3. Kelly Video Assets
  console.log('\n\n3. KELLY VIDEO ASSETS');
  const videoAssets = await supabaseQuery('kelly_video_assets?select=*&limit=20');
  console.log(`   Total: ${videoAssets.count} assets`);
  
  // Count by type
  const byType = {};
  const byStatus = {};
  if (videoAssets.data) {
    videoAssets.data.forEach(v => {
      byType[v.asset_type] = (byType[v.asset_type] || 0) + 1;
      byStatus[v.status] = (byStatus[v.status] || 0) + 1;
    });
    console.log('   By type:', byType);
    console.log('   By status:', byStatus);
    
    // Sample
    const sample = videoAssets.data[0];
    console.log('\n   Sample asset:');
    Object.entries(sample).forEach(([key, val]) => {
      const display = val === null ? '❌ NULL' : 
                     typeof val === 'string' && val.length > 60 ? `"${val.substring(0, 60)}..."` :
                     JSON.stringify(val);
      console.log(`     ${key}: ${display}`);
    });
  }

  // 4. Kelly Motion Library
  console.log('\n\n4. KELLY MOTION LIBRARY');
  const motions = await supabaseQuery('kelly_motion_library?select=*&limit=20');
  console.log(`   Total: ${motions.count} motion clips`);
  
  const byPhase = {};
  const byAvatar = {};
  if (motions.data) {
    motions.data.forEach(m => {
      byPhase[m.phase] = (byPhase[m.phase] || 0) + 1;
      const avatar = m.avatar_key?.split('_')[0] || 'unknown';
      byAvatar[avatar] = (byAvatar[avatar] || 0) + 1;
    });
    console.log('   By phase:', byPhase);
    console.log('   By avatar (first part):', byAvatar);
    
    // Sample with video URL
    const withUrl = motions.data.find(m => m.video_url);
    if (withUrl) {
      console.log('\n   Sample with video:');
      console.log(`     avatar_key: ${withUrl.avatar_key}`);
      console.log(`     phase: ${withUrl.phase}`);
      console.log(`     video_url: ${withUrl.video_url?.substring(0, 80)}...`);
      console.log(`     status: ${withUrl.status}`);
    }
  }

  // 5. Check lesson_visuals table (if exists)
  console.log('\n\n5. LESSON VISUALS TABLE');
  const visuals = await supabaseQuery('lesson_visuals?select=*&limit=5');
  if (visuals.status === 200) {
    console.log(`   Total: ${visuals.count} visuals`);
    if (visuals.data?.length) {
      console.log('   Sample:', JSON.stringify(visuals.data[0], null, 2).substring(0, 200));
    }
  } else {
    console.log(`   ❌ Table does not exist or access denied (${visuals.status})`);
  }

  // 6. Check thumbnails in core_lessons
  console.log('\n\n6. THUMBNAILS IN CORE_LESSONS');
  const thumbs = await supabaseQuery('core_lessons?select=day_number,thumbnail_url,hero_image_url&limit=10&order=day_number');
  if (thumbs.data) {
    let withThumb = 0;
    let withHero = 0;
    thumbs.data.forEach(l => {
      if (l.thumbnail_url) withThumb++;
      if (l.hero_image_url) withHero++;
    });
    console.log(`   Sample of ${thumbs.data.length}: ${withThumb} have thumbnail_url, ${withHero} have hero_image_url`);
    
    thumbs.data.slice(0, 3).forEach(l => {
      console.log(`\n   Day ${l.day_number}:`);
      console.log(`     thumbnail_url: ${l.thumbnail_url || '❌ NULL'}`);
      console.log(`     hero_image_url: ${l.hero_image_url || '❌ NULL'}`);
    });
  }

  // 7. Storage buckets check
  console.log('\n\n7. STORAGE BUCKETS');
  console.log('   Checking if assets exist in storage...');
  
  const testUrls = [
    'kelly-videos/motion/scientist/adult/hook.mp4',
    'kelly-templates/heygen/archetypes-head-only/kelly_scientist_head.png',
    'lesson-thumbnails/day-001.png',
    'lesson-images/day-001-hero.png'
  ];
  
  for (const path of testUrls) {
    const url = `${SUPABASE_URL}/storage/v1/object/public/${path}`;
    try {
      const response = await new Promise((resolve, reject) => {
        https.get(url, { method: 'HEAD' }, res => {
          resolve({ status: res.statusCode, url: path });
        }).on('error', reject);
      });
      console.log(`   ${response.status === 200 ? '✅' : '❌'} ${path} (${response.status})`);
    } catch (e) {
      console.log(`   ❌ ${path} (error)`);
    }
  }

  console.log('\n\n=== SUMMARY ===');
  console.log('What EXISTS:');
  console.log(`  - ${lessons.count} lesson records (text content)`);
  console.log(`  - ${atoms.count} dialog atoms (phases like Hook, Fact1, etc.)`);
  console.log(`  - ${motions.count} motion clips (generic persona videos)`);
  console.log(`  - ${videoAssets.count} video assets (mostly images, some audio)`);
  
  console.log('\nWhat may be MISSING:');
  console.log('  - Per-lesson video URLs in lesson_atoms.hd_video_url');
  console.log('  - Thumbnail images for each lesson');
  console.log('  - Hero images for lesson cards');
  console.log('  - Per-lesson/per-phase HeyGen videos');
}

main().catch(console.error);
