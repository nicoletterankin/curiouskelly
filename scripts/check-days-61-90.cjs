require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  const result = await pool.query(`
    SELECT day_of_year, COUNT(*) as count, 
           SUM(CASE WHEN video_url IS NOT NULL THEN 1 ELSE 0 END) as with_video
    FROM heygen_videos 
    WHERE day_of_year BETWEEN 61 AND 90
    GROUP BY day_of_year
    ORDER BY day_of_year
  `);
  
  console.log('Days 61-90:');
  result.rows.forEach(r => console.log(`Day ${r.day_of_year}: ${r.count} total, ${r.with_video} with video`));
  
  if (result.rows.length === 0) {
    console.log('NO ROWS for days 61-90!');
  }
  
  await pool.end();
}

main().catch(console.error);
