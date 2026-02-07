/**
 * Sync completed HeyGen videos to kelly_lesson_assets table
 */
const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function sync() {
  console.log('Syncing 414 completed HeyGen videos to kelly_lesson_assets...\n');
  
  // Get all completed videos with URLs
  const completed = await sql`
    SELECT day_of_year, phase, age_category, archetype, video_url, heygen_video_id, script
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL
  `;
  
  console.log(`Found ${completed.length} completed videos to sync\n`);
  
  let synced = 0;
  let errors = 0;
  
  for (const v of completed) {
    try {
      // Check if exists in kelly_lesson_assets
      const existing = await sql`
        SELECT id FROM kelly_lesson_assets 
        WHERE day_number = ${v.day_of_year} 
          AND phase = ${v.phase} 
          AND age_group = ${v.age_category}
        LIMIT 1
      `;
      
      if (existing.length > 0) {
        // Update existing
        await sql`
          UPDATE kelly_lesson_assets SET
            video_url = ${v.video_url},
            video_id = ${v.heygen_video_id},
            video_source = 'heygen',
            status = 'completed',
            updated_at = NOW()
          WHERE id = ${existing[0].id}
        `;
      } else {
        // Insert new
        await sql`
          INSERT INTO kelly_lesson_assets (
            id, day_number, phase, age_group, archetype, 
            video_url, video_id, video_source, status, 
            script_text, language, created_at, updated_at
          ) VALUES (
            gen_random_uuid(), ${v.day_of_year}, ${v.phase}, ${v.age_category}, ${v.archetype || 'explorer'},
            ${v.video_url}, ${v.heygen_video_id}, 'heygen', 'completed',
            ${v.script}, 'en', NOW(), NOW()
          )
        `;
      }
      synced++;
    } catch (e) {
      console.log(`Error syncing Day ${v.day_of_year} ${v.phase} ${v.age_category}: ${e.message}`);
      errors++;
    }
  }
  
  console.log(`\n✅ Synced: ${synced}`);
  console.log(`❌ Errors: ${errors}`);
}

sync().catch(console.error);
