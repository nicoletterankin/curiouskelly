require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  console.log('=== ACTUAL DATABASE SCHEMA ===\n');
  
  // Check kelly_lesson_assets columns
  const klaColumns = await pool.query(`
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'kelly_lesson_assets'
    ORDER BY ordinal_position
  `);
  console.log('kelly_lesson_assets columns:');
  klaColumns.rows.forEach(r => console.log(`  - ${r.column_name} (${r.data_type})`));
  
  // Check heygen_videos columns
  const hvColumns = await pool.query(`
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'heygen_videos'
    ORDER BY ordinal_position
  `);
  console.log('\nheygen_videos columns:');
  hvColumns.rows.forEach(r => console.log(`  - ${r.column_name} (${r.data_type})`));
  
  // Check lesson_perspectives columns
  const lpColumns = await pool.query(`
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'lesson_perspectives'
    ORDER BY ordinal_position
  `);
  console.log('\nlesson_perspectives columns:');
  lpColumns.rows.forEach(r => console.log(`  - ${r.column_name} (${r.data_type})`));
  
  // Quick data check for Day 34
  console.log('\n=== DAY 34 DATA CHECK ===');
  
  const hv = await pool.query(`
    SELECT video_url, status FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook' AND status = 'completed' AND video_url IS NOT NULL
    LIMIT 1
  `);
  console.log('heygen_videos Day 34 hook:', hv.rows.length > 0 ? hv.rows[0].video_url?.substring(0, 50) : 'NONE');
  
  await pool.end();
}

main().catch(console.error);
