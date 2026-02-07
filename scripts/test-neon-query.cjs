require('dotenv').config();
const { neon } = require('@neondatabase/serverless');

async function main() {
  const sql = neon(process.env.DATABASE_URL);
  
  const dayNumber = 34;
  const phase = 'hook';
  const ageGroup = 'adult';
  const archetype = 'storyteller';
  
  console.log('Testing with Neon serverless driver (same as production)...');
  console.log(`dayNumber=${dayNumber}, phase=${phase}, ageGroup=${ageGroup}, archetype=${archetype}`);
  
  try {
    const heygenData = await sql`
      SELECT video_url, audio_url, script, thumbnail_url, age_category, archetype, day_of_year
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
    
    console.log('Query returned:', heygenData.length, 'rows');
    if (heygenData.length > 0) {
      const r = heygenData[0];
      console.log('age_category:', r.age_category);
      console.log('archetype:', r.archetype);
      console.log('video_url:', r.video_url?.substring(0, 60));
      console.log('Is HeyGen:', r.video_url?.includes('files2.heygen'));
    } else {
      console.log('NO ROWS RETURNED!');
    }
  } catch (err) {
    console.log('Query FAILED:', err.message);
  }
}

main().catch(console.error);
