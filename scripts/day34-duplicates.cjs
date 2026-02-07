require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check for ALL Day 34 hook entries - even without video_url
  const all = await pool.query(`
    SELECT id, age_category, archetype, status, video_url IS NOT NULL as has_video, 
           created_at, updated_at
    FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook'
    ORDER BY age_category, archetype, created_at DESC
  `);
  
  console.log('All Day 34 hook entries (' + all.rows.length + ' total):');
  console.log('---');
  
  // Group by age/archetype to find duplicates
  const groups = {};
  all.rows.forEach(r => {
    const key = `${r.age_category}/${r.archetype}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(r);
  });
  
  Object.entries(groups).forEach(([key, rows]) => {
    const marker = rows.length > 1 ? ' [DUPLICATES!]' : '';
    console.log(`${key}${marker}: ${rows.length} entries`);
    rows.forEach(r => {
      console.log(`  - ${r.status} | video: ${r.has_video} | updated: ${r.updated_at}`);
    });
  });
  
  // Specifically check storyteller
  console.log('\n=== STORYTELLER SPECIFICALLY ===');
  const st = await pool.query(`
    SELECT * FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook' AND archetype = 'storyteller'
    ORDER BY updated_at DESC
  `);
  st.rows.forEach(r => {
    console.log(JSON.stringify({
      id: r.id?.substring(0, 8),
      status: r.status,
      video_url: r.video_url?.substring(0, 40),
      age_category: r.age_category
    }));
  });
  
  await pool.end();
}

main().catch(console.error);
