/**
 * Quick test to verify Supabase connection and lesson data
 * Run with: node test-supabase-connection.js
 */

const https = require('https');

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

function supabaseQuery(table, select, filters = '') {
  return new Promise((resolve, reject) => {
    const url = new URL(`${SUPABASE_URL}/rest/v1/${table}`);
    url.searchParams.set('select', select);
    if (filters) {
      filters.split('&').forEach(f => {
        const [key, val] = f.split('=');
        url.searchParams.set(key, val);
      });
    }

    const options = {
      headers: {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json',
        'Prefer': 'count=exact'
      }
    };

    https.get(url.toString(), options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const count = res.headers['content-range']?.split('/')[1] || 'unknown';
          resolve({ data: JSON.parse(data), count, status: res.statusCode });
        } catch (e) {
          resolve({ error: data, status: res.statusCode });
        }
      });
    }).on('error', reject);
  });
}

async function main() {
  console.log('=== SUPABASE CONNECTION TEST ===\n');

  // 1. Test core_lessons table
  console.log('1. Checking core_lessons table...');
  const lessons = await supabaseQuery('core_lessons', 'id,day_number,topic', 'limit=10&order=day_number');
  console.log(`   Status: ${lessons.status}`);
  console.log(`   Total count: ${lessons.count}`);
  if (lessons.data?.length) {
    console.log(`   Sample lessons:`);
    lessons.data.slice(0, 5).forEach(l => console.log(`     Day ${l.day_number}: ${l.topic}`));
  } else if (lessons.error) {
    console.log(`   ERROR: ${lessons.error}`);
  }

  // 2. Test lesson_atoms table
  console.log('\n2. Checking lesson_atoms table...');
  const atoms = await supabaseQuery('lesson_atoms', 'id,core_lesson_id,archetype,phase,hd_video_url', 'limit=10');
  console.log(`   Status: ${atoms.status}`);
  console.log(`   Total count: ${atoms.count}`);
  if (atoms.data?.length) {
    console.log(`   Sample atoms:`);
    atoms.data.slice(0, 3).forEach(a => console.log(`     Phase: ${a.phase}, Archetype: ${a.archetype}, Video: ${a.hd_video_url ? 'YES' : 'NO'}`));
  } else if (atoms.error) {
    console.log(`   ERROR: ${atoms.error}`);
  }

  // 3. Test kelly_video_assets table
  console.log('\n3. Checking kelly_video_assets table...');
  const videos = await supabaseQuery('kelly_video_assets', 'id,day_number,phase,asset_type,status,public_url', 'limit=10');
  console.log(`   Status: ${videos.status}`);
  console.log(`   Total count: ${videos.count}`);
  if (videos.data?.length) {
    console.log(`   Sample video assets:`);
    videos.data.slice(0, 5).forEach(v => console.log(`     Day ${v.day_number}, Phase: ${v.phase}, Type: ${v.asset_type}, Status: ${v.status}`));
  } else if (videos.error) {
    console.log(`   ERROR: ${videos.error}`);
  } else {
    console.log('   ⚠️ NO VIDEO ASSETS FOUND!');
  }

  // 4. Test kelly_motion_library table
  console.log('\n4. Checking kelly_motion_library table...');
  const motions = await supabaseQuery('kelly_motion_library', 'id,avatar_key,phase,status,video_url', 'limit=10');
  console.log(`   Status: ${motions.status}`);
  console.log(`   Total count: ${motions.count}`);
  if (motions.data?.length) {
    console.log(`   Sample motion clips:`);
    motions.data.slice(0, 5).forEach(m => console.log(`     Avatar: ${m.avatar_key}, Phase: ${m.phase}, Status: ${m.status}, Has URL: ${!!m.video_url}`));
  } else if (motions.error) {
    console.log(`   ERROR: ${motions.error}`);
  } else {
    console.log('   ⚠️ NO MOTION LIBRARY ENTRIES FOUND!');
  }

  // 5. Check RLS policies
  console.log('\n5. Checking if RLS might be blocking...');
  const rlsTest = await supabaseQuery('core_lessons', 'count', 'limit=1');
  if (rlsTest.status === 200) {
    console.log('   ✅ Anonymous read access works for core_lessons');
  } else {
    console.log(`   ❌ RLS may be blocking: ${rlsTest.status}`);
  }

  console.log('\n=== SUMMARY ===');
  console.log(`core_lessons: ${lessons.count || 'ERROR'} rows`);
  console.log(`lesson_atoms: ${atoms.count || 'ERROR'} rows`);
  console.log(`kelly_video_assets: ${videos.count || 'ERROR'} rows`);
  console.log(`kelly_motion_library: ${motions.count || 'ERROR'} rows`);
}

main().catch(console.error);
