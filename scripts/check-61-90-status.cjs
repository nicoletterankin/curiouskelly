require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  const result = await pool.query(`
    SELECT status, COUNT(*) as count 
    FROM heygen_videos 
    WHERE day_of_year BETWEEN 61 AND 90 
    GROUP BY status
  `);
  
  console.log('Days 61-90 by status:');
  result.rows.forEach(r => console.log(`  ${r.status}: ${r.count}`));
  
  // Check if heygen_video_id exists
  const withId = await pool.query(`
    SELECT COUNT(*) as count 
    FROM heygen_videos 
    WHERE day_of_year BETWEEN 61 AND 90 AND heygen_video_id IS NOT NULL
  `);
  console.log('\nWith heygen_video_id:', withId.rows[0].count);
  
  await pool.end();
}

main().catch(console.error);
