/**
 * Test if RLS is blocking specific queries the frontend makes
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
        'Content-Type': 'application/json'
      }
    };

    https.get(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ data: JSON.parse(data), status: res.statusCode });
        } catch (e) {
          resolve({ error: data, status: res.statusCode });
        }
      });
    }).on('error', reject);
  });
}

async function main() {
  console.log('=== TESTING EXACT FRONTEND QUERIES ===\n');

  // Test 1: Get today's lesson (Day 349 for Dec 14)
  const today = new Date();
  const start = new Date(today.getFullYear(), 0, 0);
  const diff = today - start;
  const oneDay = 1000 * 60 * 60 * 24;
  const dayNumber = Math.floor(diff / oneDay);
  
  console.log(`Today is Day ${dayNumber}\n`);

  // Test the exact query the frontend makes
  console.log(`1. Fetching core_lessons for day ${dayNumber}...`);
  const lesson = await supabaseQuery(`core_lessons?day_number=eq.${dayNumber}&select=*`);
  console.log(`   Status: ${lesson.status}`);
  if (lesson.data?.length) {
    console.log(`   ✅ Found lesson: "${lesson.data[0].topic}"`);
  } else {
    console.log(`   ❌ No lesson found for day ${dayNumber}!`);
    console.log(`   Response: ${JSON.stringify(lesson.data || lesson.error)}`);
  }

  // Test 2: Fetch atoms for that lesson
  if (lesson.data?.length) {
    const lessonId = lesson.data[0].id;
    console.log(`\n2. Fetching lesson_atoms for lesson ID ${lessonId}...`);
    const atoms = await supabaseQuery(`lesson_atoms?core_lesson_id=eq.${lessonId}&archetype=eq.The%20Scientist&select=*`);
    console.log(`   Status: ${atoms.status}`);
    if (atoms.data?.length) {
      console.log(`   ✅ Found ${atoms.data.length} atoms`);
      atoms.data.forEach(a => console.log(`      - Phase: ${a.phase}`));
    } else {
      console.log(`   ❌ No atoms found!`);
    }
  }

  // Test 3: Check kelly_video_assets for video
  console.log(`\n3. Fetching kelly_video_assets for day ${dayNumber}...`);
  const videos = await supabaseQuery(`kelly_video_assets?day_number=eq.${dayNumber}&asset_type=eq.video&select=*`);
  console.log(`   Status: ${videos.status}`);
  if (videos.data?.length) {
    console.log(`   ✅ Found ${videos.data.length} video assets`);
    videos.data.forEach(v => console.log(`      - Phase: ${v.phase}, URL: ${v.public_url?.substring(0, 50)}...`));
  } else {
    console.log(`   ⚠️ No VIDEO assets for day ${dayNumber}`);
    
    // Check for any assets for this day
    const anyAssets = await supabaseQuery(`kelly_video_assets?day_number=eq.${dayNumber}&select=*`);
    if (anyAssets.data?.length) {
      console.log(`   Found ${anyAssets.data.length} OTHER assets (not video):`);
      anyAssets.data.slice(0, 3).forEach(a => console.log(`      - Type: ${a.asset_type}, Phase: ${a.phase}`));
    }
  }

  // Test 4: Check motion library
  console.log(`\n4. Fetching kelly_motion_library for scientist_adult...`);
  const motions = await supabaseQuery(`kelly_motion_library?avatar_key=eq.scientist_adult&status=eq.completed&select=*`);
  console.log(`   Status: ${motions.status}`);
  if (motions.data?.length) {
    console.log(`   ✅ Found ${motions.data.length} motion clips`);
    motions.data.slice(0, 5).forEach(m => console.log(`      - Phase: ${m.phase}, URL: ${m.video_url?.substring(0, 50)}...`));
  } else {
    console.log(`   ❌ No motion clips found!`);
  }

  console.log('\n=== DIAGNOSIS ===');
  if (!lesson.data?.length) {
    console.log('❌ PROBLEM: core_lessons not returning data - check RLS policies');
  } else {
    console.log('✅ core_lessons is accessible');
  }
}

main().catch(console.error);
