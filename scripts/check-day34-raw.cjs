require('dotenv').config();
const { neon } = require('@neondatabase/serverless');
const sql = neon(process.env.DATABASE_URL);

async function test() {
  // Raw query without filters
  const all = await sql`
    SELECT day_of_year, phase, status, video_url IS NOT NULL as has_video
    FROM heygen_videos
    WHERE day_of_year = 34
    LIMIT 10
  `;
  
  console.log('All Day 34 rows (no filters):');
  all.forEach(r => console.log(`  ${r.phase} | status: ${r.status} | has_video: ${r.has_video}`));
  
  // Check what statuses exist
  const statuses = await sql`
    SELECT DISTINCT status FROM heygen_videos WHERE day_of_year = 34
  `;
  console.log('\nDistinct statuses for Day 34:', statuses.map(r => r.status));
  
  // Check with just status filter
  const withStatus = await sql`
    SELECT phase, status, video_url
    FROM heygen_videos
    WHERE day_of_year = 34 AND status IN ('completed', 'placeholder', 'ready')
    LIMIT 5
  `;
  console.log('\nWith status filter:', withStatus.length, 'rows');
  withStatus.forEach(r => console.log(`  ${r.phase} | ${r.status} | ${r.video_url?.substring(0, 40)}`));
}

test().catch(e => console.error('Error:', e.message));
