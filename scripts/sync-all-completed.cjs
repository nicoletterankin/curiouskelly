const { neon } = require('@neondatabase/serverless');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function syncAll() {
  // Get all completed HeyGen videos
  const completed = await sql`
    SELECT day_of_year, phase, age_category, archetype, video_url, heygen_video_id, script
    FROM heygen_videos
    WHERE status = 'completed' AND video_url IS NOT NULL
  `;
  
  console.log(`Found ${completed.length} completed HeyGen videos to sync\n`);
  
  let updated = 0;
  let inserted = 0;
  let skipped = 0;
  let errors = 0;
  
  for (const v of completed) {
    try {
      // First try to update existing record
      const updateResult = await sql`
        UPDATE kelly_lesson_assets
        SET 
          video_url = ${v.video_url}, 
          video_source = 'heygen', 
          video_id = ${v.heygen_video_id},
          updated_at = NOW()
        WHERE day_number = ${v.day_of_year} 
        AND phase = ${v.phase} 
        AND age_group = ${v.age_category}
        AND (video_url IS NULL OR video_url != ${v.video_url})
        RETURNING id
      `;
      
      if (updateResult.length > 0) {
        updated++;
      } else {
        // Check if already exists with same video_url
        const exists = await sql`
          SELECT id FROM kelly_lesson_assets
          WHERE day_number = ${v.day_of_year} 
          AND phase = ${v.phase} 
          AND age_group = ${v.age_category}
          AND video_url = ${v.video_url}
          LIMIT 1
        `;
        
        if (exists.length > 0) {
          skipped++;
        } else {
          // Check if there's ANY record for this combo
          const anyExists = await sql`
            SELECT id FROM kelly_lesson_assets
            WHERE day_number = ${v.day_of_year} 
            AND phase = ${v.phase} 
            AND age_group = ${v.age_category}
            LIMIT 1
          `;
          
          if (anyExists.length === 0) {
            // Insert new record (without video_id to avoid constraint)
            await sql`
              INSERT INTO kelly_lesson_assets (day_number, phase, age_group, video_url, video_source, script_text)
              VALUES (${v.day_of_year}, ${v.phase}, ${v.age_category}, ${v.video_url}, 'heygen', ${v.script})
            `;
            inserted++;
          } else {
            // Update the existing one even if video_url is same
            await sql`
              UPDATE kelly_lesson_assets
              SET 
                video_url = ${v.video_url}, 
                video_source = 'heygen',
                updated_at = NOW()
              WHERE day_number = ${v.day_of_year} 
              AND phase = ${v.phase} 
              AND age_group = ${v.age_category}
            `;
            updated++;
          }
        }
      }
    } catch (e) {
      errors++;
      if (errors <= 5) {
        console.log(`Error syncing Day ${v.day_of_year} ${v.phase} ${v.age_category}: ${e.message}`);
      }
    }
  }
  
  console.log('\n=== SYNC COMPLETE ===');
  console.log(`Updated: ${updated}`);
  console.log(`Inserted: ${inserted}`);
  console.log(`Skipped (already synced): ${skipped}`);
  console.log(`Errors: ${errors}`);
  
  // Final count
  const finalCount = await sql`
    SELECT COUNT(*) as count FROM kelly_lesson_assets WHERE video_url IS NOT NULL
  `;
  console.log(`\nTotal videos in kelly_lesson_assets: ${finalCount[0].count}`);
}

syncAll().catch(console.error);
