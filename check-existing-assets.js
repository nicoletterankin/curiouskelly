/**
 * CHECK WHAT ASSETS ACTUALLY EXIST
 */

const { createClient } = require('@supabase/supabase-js');
const https = require('https');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function checkUrl(url) {
  return new Promise((resolve) => {
    https.get(url, { method: 'HEAD' }, (res) => {
      resolve(res.statusCode === 200);
    }).on('error', () => resolve(false));
  });
}

async function main() {
  console.log('═'.repeat(70));
  console.log('CHECKING EXISTING ASSETS');
  console.log('═'.repeat(70));
  
  // 1. Get assets from kelly_video_assets where type=image
  console.log('\n1. IMAGES IN KELLY_VIDEO_ASSETS:');
  const { data: images } = await supabase
    .from('kelly_video_assets')
    .select('day_number, phase, public_url, status')
    .eq('asset_type', 'image')
    .order('day_number')
    .order('phase');
  
  console.log(`   Total image records: ${images?.length || 0}`);
  
  // Group by day
  const byDay = {};
  images?.forEach(img => {
    if (!byDay[img.day_number]) byDay[img.day_number] = [];
    byDay[img.day_number].push(img.phase);
  });
  
  console.log(`   Days with images: ${Object.keys(byDay).length}`);
  console.log(`   Sample: Day 1 has phases: ${byDay[1]?.join(', ') || 'none'}`);
  
  // 2. Check if URLs are actually accessible
  console.log('\n2. CHECKING IF ASSETS ARE ACCESSIBLE:');
  const sample = images?.slice(0, 5) || [];
  for (const img of sample) {
    const accessible = await checkUrl(img.public_url);
    console.log(`   Day ${img.day_number} ${img.phase}: ${accessible ? '✅ EXISTS' : '❌ 404'}`);
    console.log(`      ${img.public_url.substring(0, 80)}...`);
  }
  
  // 3. Check lesson_atoms with visual_url
  console.log('\n3. LESSON_ATOMS WITH VISUAL_URL:');
  const { data: atomsWithVisuals } = await supabase
    .from('lesson_atoms')
    .select('core_lesson_id, phase, visual_url')
    .not('visual_url', 'is', null);
  
  console.log(`   Total atoms with visual_url: ${atomsWithVisuals?.length || 0}`);
  
  // 4. Get core_lesson details for those atoms
  if (atomsWithVisuals?.length) {
    const lessonIds = [...new Set(atomsWithVisuals.map(a => a.core_lesson_id))];
    const { data: lessons } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic')
      .in('id', lessonIds);
    
    const lessonMap = {};
    lessons?.forEach(l => lessonMap[l.id] = l);
    
    console.log('\n   Lessons with visual URLs:');
    lessonIds.forEach(id => {
      const lesson = lessonMap[id];
      const atoms = atomsWithVisuals.filter(a => a.core_lesson_id === id);
      console.log(`     Day ${lesson?.day_number}: ${lesson?.topic}`);
      console.log(`       Phases: ${atoms.map(a => a.phase).join(', ')}`);
    });
  }
  
  // 5. Summary
  console.log('\n' + '═'.repeat(70));
  console.log('SUMMARY');
  console.log('═'.repeat(70));
  console.log(`
ASSETS THAT EXIST:
- kelly_video_assets (type=image): ${images?.length || 0} records
- Days covered: ${Object.keys(byDay).length}
- lesson_atoms.visual_url populated: ${atomsWithVisuals?.length || 0}

WHAT NEEDS TO HAPPEN:
1. Generate infographic images for each phase (hook, q1, q2, q3, wisdom)
2. Upload to Supabase storage (kelly-templates bucket)
3. Update lesson_atoms.visual_url with the URLs
4. OR register in kelly_video_assets and link via join

EXISTING PATTERN:
- Storage path: production/images/day_XXX_PHASE.png
- Public URL: https://...supabase.co/storage/v1/object/public/kelly-templates/...
  `);
}

main().catch(console.error);
