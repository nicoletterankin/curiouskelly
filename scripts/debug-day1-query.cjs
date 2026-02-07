require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || process.env.NEON_DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

async function main() {
  // Reproduce the exact query from the API
  const result = await pool.query(`
    SELECT video_url, status, age_category, archetype, created_at,
           CASE WHEN age_category = 'adult' THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END as age_priority,
           CASE WHEN archetype = 'storyteller' THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END as arch_priority
    FROM heygen_videos
    WHERE day_of_year = 1
      AND phase = 'hook'
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY 
      CASE WHEN age_category = 'adult' THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
      CASE WHEN archetype = 'storyteller' THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
      created_at DESC
    LIMIT 5
  `);
  
  console.log('Day 1 hook rows that would be queried:');
  for (const row of result.rows) {
    console.log('  status:', row.status);
    console.log('  age:', row.age_category, '(priority:', row.age_priority + ')');
    console.log('  arch:', row.archetype, '(priority:', row.arch_priority + ')');
    console.log('  url:', row.video_url?.substring(0, 60) + '...');
    console.log('  created:', row.created_at);
    console.log('');
  }
  
  await pool.end();
}

main().catch(console.error);
