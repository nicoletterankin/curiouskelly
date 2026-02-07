require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check generated_assets for Day 1
  const result = await pool.query(`
    SELECT lesson_id, phase, asset_type, url, status
    FROM generated_assets
    WHERE lesson_id LIKE 'day-001%'
    LIMIT 10
  `);
  
  console.log('generated_assets for Day 1:', result.rows.length, 'rows');
  result.rows.forEach(r => {
    console.log(`${r.lesson_id} | ${r.phase} | ${r.asset_type} | ${r.status}`);
    console.log(`  URL: ${r.url?.substring(0, 60)}...`);
  });
  
  await pool.end();
}

main().catch(console.error);
