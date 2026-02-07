const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function addConstraint() {
  console.log('Adding unique constraint to heygen_videos...');
  try {
    await sql`
      ALTER TABLE heygen_videos 
      ADD CONSTRAINT heygen_videos_unique_combo 
      UNIQUE (day_of_year, phase, age_category, archetype)
    `;
    console.log('✅ Constraint added successfully');
  } catch (e) {
    if (e.message.includes('already exists')) {
      console.log('✓ Constraint already exists');
    } else if (e.message.includes('duplicate key')) {
      console.log('⚠️ Duplicates exist - need to clean up first');
      
      // Find and show duplicates
      const dupes = await sql`
        SELECT day_of_year, phase, age_category, archetype, COUNT(*) as count
        FROM heygen_videos
        GROUP BY day_of_year, phase, age_category, archetype
        HAVING COUNT(*) > 1
        LIMIT 10
      `;
      console.log('Duplicates found:', dupes);
    } else {
      console.log('Error:', e.message);
    }
  }
}

addConstraint();
