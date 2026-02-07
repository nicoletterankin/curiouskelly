require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  console.log('DAY 34 IN DATABASE:');
  console.log('='.repeat(50));
  
  const result = await pool.query(`
    SELECT phase, video_url, status, updated_at
    FROM heygen_videos 
    WHERE day_of_year = 34 AND age_category = 'adult'
    ORDER BY phase
  `);
  
  result.rows.forEach(row => {
    const isHeyGen = row.video_url?.includes('heygen');
    console.log(`${row.phase}: ${isHeyGen ? '✅ HeyGen' : '❌ Other'}`);
    console.log(`  URL: ${row.video_url?.substring(0, 70)}...`);
    console.log(`  Status: ${row.status}`);
    console.log(`  Updated: ${row.updated_at}`);
  });
  
  await pool.end();
}

main().catch(console.error);
