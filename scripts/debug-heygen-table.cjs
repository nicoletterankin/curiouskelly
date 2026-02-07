require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || process.env.NEON_DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

async function main() {
  // Show all rows with video URLs
  console.log('=== ALL ROWS WITH VIDEO URLs ===\n');
  
  const result = await pool.query(`
    SELECT day_of_year, phase, age_category, archetype, language, 
           LEFT(video_url, 60) as url_start,
           heygen_video_id,
           updated_at
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
    ORDER BY updated_at DESC
    LIMIT 50
  `);
  
  for (const row of result.rows) {
    console.log(`Day ${row.day_of_year.toString().padStart(2)} | ${row.phase.padEnd(6)} | ${row.age_category} | ${row.archetype || 'null'} | ${row.language || 'null'}`);
    console.log(`         URL: ${row.url_start}...`);
    console.log(`         Updated: ${row.updated_at}`);
    console.log('');
  }
  
  // Check what the production API query looks like
  console.log('\n=== CHECKING DAY 1 HOOK SPECIFICALLY ===\n');
  
  const day1 = await pool.query(`
    SELECT day_of_year, phase, age_category, archetype, language, video_url
    FROM heygen_videos 
    WHERE day_of_year = 1 AND phase = 'hook'
  `);
  
  console.log(`Found ${day1.rowCount} rows for Day 1 hook:`);
  for (const row of day1.rows) {
    console.log(`  age=${row.age_category}, arch=${row.archetype}, lang=${row.language}, url=${row.video_url ? 'YES' : 'NO'}`);
  }
  
  await pool.end();
}

main().catch(console.error);
