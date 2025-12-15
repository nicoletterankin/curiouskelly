/**
 * UNDERSTAND THE VISUAL SYSTEM
 * What are visuals? How do they work?
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function main() {
  console.log('═'.repeat(70));
  console.log('UNDERSTANDING THE VISUAL SYSTEM');
  console.log('═'.repeat(70));
  
  // 1. Check lesson_atoms schema
  console.log('\n1. LESSON_ATOMS SCHEMA:');
  const { data: sampleAtom } = await supabase
    .from('lesson_atoms')
    .select('*')
    .limit(1);
  
  if (sampleAtom?.[0]) {
    console.log('   Columns:');
    Object.keys(sampleAtom[0]).forEach(key => {
      const val = sampleAtom[0][key];
      let preview;
      if (val === null) preview = 'NULL';
      else if (typeof val === 'string') preview = val.substring(0, 50) + (val.length > 50 ? '...' : '');
      else if (typeof val === 'object') preview = JSON.stringify(val).substring(0, 50) + '...';
      else preview = val;
      console.log(`     ${key}: ${preview}`);
    });
  }
  
  // 2. Check if visual_url exists and has data
  console.log('\n2. VISUAL_URL USAGE:');
  const { data: withVisuals } = await supabase
    .from('lesson_atoms')
    .select('id, phase, visual_url')
    .not('visual_url', 'is', null)
    .limit(10);
  
  console.log(`   Atoms with visual_url: ${withVisuals?.length || 0}`);
  if (withVisuals?.length) {
    console.log('   Sample:');
    withVisuals.slice(0, 3).forEach(a => {
      console.log(`     Phase: ${a.phase}, URL: ${a.visual_url?.substring(0, 60)}...`);
    });
  }
  
  // 3. Check kelly_video_assets for asset types
  console.log('\n3. KELLY_VIDEO_ASSETS:');
  const { data: assets } = await supabase
    .from('kelly_video_assets')
    .select('asset_type, storage_path, public_url')
    .limit(20);
  
  console.log(`   Sample assets (${assets?.length || 0}):`);
  assets?.slice(0, 5).forEach(a => {
    console.log(`     Type: ${a.asset_type}`);
    console.log(`     Path: ${a.storage_path}`);
    console.log(`     URL: ${a.public_url?.substring(0, 70)}...`);
    console.log('');
  });
  
  // Count by asset type
  const { data: allAssets } = await supabase
    .from('kelly_video_assets')
    .select('asset_type');
  
  const counts = {};
  allAssets?.forEach(a => counts[a.asset_type] = (counts[a.asset_type] || 0) + 1);
  
  console.log('   Counts by asset_type:');
  Object.entries(counts).forEach(([type, count]) => {
    console.log(`     ${type}: ${count}`);
  });
  
  // 4. Check if hd_video_url is used
  console.log('\n4. HD_VIDEO_URL USAGE:');
  const { data: withVideos } = await supabase
    .from('lesson_atoms')
    .select('id, phase, hd_video_url')
    .not('hd_video_url', 'is', null)
    .limit(5);
  
  console.log(`   Atoms with hd_video_url: ${withVideos?.length || 0}`);
  
  // 5. Summary
  console.log('\n' + '═'.repeat(70));
  console.log('SUMMARY: WHAT ARE VISUALS?');
  console.log('═'.repeat(70));
  console.log(`
Based on the code analysis:

1. WHAT ARE VISUALS?
   - Infographics/images that appear in a popup overlay
   - NOT backgrounds or wallpapers
   - Educational visual aids for each phase

2. WHEN DO THEY APPEAR?
   - When user clicks the 📊 button (btn-infographic)
   - Opens an overlay (overlay-infographic)
   - Shows the current phase's visual

3. WHAT TRIGGERS THEM?
   - User action: clicking the infographic button
   - Code: document.getElementById('btn-infographic').addEventListener('click')

4. WHAT FORMAT?
   - Images (PNG/JPG)
   - Stored in lesson_atoms.visual_url
   - Displayed as: <img src="{visualUrl}" alt="infographic">

5. DATABASE STORAGE:
   - Column: lesson_atoms.visual_url (currently NULL for most)
   - Also: lesson_atoms.hd_video_url (for videos)
   - Asset registry: kelly_video_assets table

6. CURRENT STATE:
   - visual_url: ${withVisuals?.length || 0} populated
   - hd_video_url: ${withVideos?.length || 0} populated
   - kelly_video_assets: ${allAssets?.length || 0} total assets

7. UI BEHAVIOR:
   - If visualUrl exists: Shows image in popup
   - If visualUrl is null: Shows "Infographic Coming Soon"
  `);
}

main().catch(console.error);
