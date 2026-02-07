require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check distinct age_category values for Day 31
  const result = await pool.query(`
    SELECT DISTINCT age_category, LENGTH(age_category) as len
    FROM heygen_videos 
    WHERE day_of_year = 31 AND phase = 'hook'
  `);
  
  console.log('Distinct age_category values for Day 31:');
  result.rows.forEach(r => {
    console.log(`"${r.age_category}" (length: ${r.len})`);
    // Show hex to detect hidden characters
    const hex = Buffer.from(r.age_category).toString('hex');
    console.log(`  hex: ${hex}`);
  });
  
  await pool.end();
}

main().catch(console.error);
