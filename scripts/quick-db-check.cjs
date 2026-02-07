require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  const result = await pool.query(`
    SELECT video_url, updated_at, created_at
    FROM heygen_videos 
    WHERE day_of_year = 1 AND phase = 'hook' AND age_category = 'adult' AND archetype = 'storyteller'
    ORDER BY updated_at DESC NULLS LAST, created_at DESC
    LIMIT 3
  `);
  
  console.log('Day 1 hook (adult/storyteller) - ORDER BY updated_at DESC:');
  result.rows.forEach((row, i) => {
    const isHeyGen = row.video_url?.includes('files2.heygen');
    console.log(`${i+1}. ${isHeyGen ? '✅ HeyGen' : '❌ Blob'}: ${row.video_url?.substring(0, 55)}...`);
    console.log(`   Updated: ${row.updated_at}`);
  });
  
  await pool.end();
}

main().catch(console.error);
