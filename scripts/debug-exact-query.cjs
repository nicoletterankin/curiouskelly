require('dotenv').config();
const { neon } = require('@neondatabase/serverless');
const sql = neon(process.env.DATABASE_URL);

async function test() {
  const dayNumber = 34;
  const phase = 'hook';
  const ageGroup = 'adult'; // This is what getAgeGroup(30) returns
  const archetype = 'storyteller';
  
  console.log('Query params:', { dayNumber, phase, ageGroup, archetype });
  
  // Exact query from v0's route.ts
  const heygenData = await sql`
    SELECT video_url, audio_url, script, thumbnail_url, age_category, archetype, day_of_year, status
    FROM heygen_videos
    WHERE day_of_year = ${dayNumber}
      AND phase = ${phase}
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY 
      CASE WHEN age_category = ${ageGroup} THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
      CASE WHEN archetype = ${archetype} THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
      updated_at DESC NULLS LAST,
      created_at DESC
    LIMIT 1
  `;
  
  console.log('Result:', heygenData.length, 'rows');
  if (heygenData.length > 0) {
    const row = heygenData[0];
    console.log('video_url:', row.video_url?.substring(0, 60));
    console.log('age_category:', row.age_category);
    console.log('archetype:', row.archetype);
    console.log('status:', row.status);
    console.log('Is HeyGen URL:', row.video_url?.includes('heygen'));
  }
}

test().catch(e => console.error('Error:', e.message));
