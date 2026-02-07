const { neon } = require('@neondatabase/serverless');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function fullStatus() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║          CURIOUS KELLY VIDEO PIPELINE STATUS                 ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log(`Time: ${new Date().toISOString()}\n`);

  // HeyGen status
  const heygen = await sql`
    SELECT 
      status,
      COUNT(*) as count
    FROM heygen_videos
    GROUP BY status
    ORDER BY count DESC
  `;
  
  console.log('┌─────────────────────────────────────────────────────────────┐');
  console.log('│ HEYGEN PIPELINE (Cursor)                                    │');
  console.log('├─────────────────────────────────────────────────────────────┤');
  
  let heygenTotal = 0;
  let heygenCompleted = 0;
  for (const h of heygen) {
    const icon = h.status === 'completed' ? '✅' : h.status === 'processing' ? '⏳' : '❌';
    console.log(`│ ${icon} ${h.status.padEnd(12)}: ${h.count.toString().padStart(5)} videos                          │`);
    heygenTotal += parseInt(h.count);
    if (h.status === 'completed') heygenCompleted = parseInt(h.count);
  }
  console.log(`│                                                             │`);
  console.log(`│ Progress: ${((heygenCompleted/heygenTotal)*100).toFixed(1)}% (${heygenCompleted}/${heygenTotal})                           │`);
  console.log('└─────────────────────────────────────────────────────────────┘\n');

  // Kelly lesson assets
  const assets = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as has_video,
      COUNT(CASE WHEN audio_url IS NOT NULL AND video_url IS NULL THEN 1 END) as audio_only,
      COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as has_audio
    FROM kelly_lesson_assets
  `;
  
  console.log('┌─────────────────────────────────────────────────────────────┐');
  console.log('│ KELLY_LESSON_ASSETS                                         │');
  console.log('├─────────────────────────────────────────────────────────────┤');
  const a = assets[0];
  console.log(`│ 📊 Total records:        ${a.total.toString().padStart(6)}                            │`);
  console.log(`│ 🎬 With video_url:       ${a.has_video.toString().padStart(6)}                            │`);
  console.log(`│ 🔊 With audio_url:       ${a.has_audio.toString().padStart(6)}                            │`);
  console.log(`│ ⏳ Ready for Fal:        ${a.audio_only.toString().padStart(6)} (audio, no video)        │`);
  console.log('└─────────────────────────────────────────────────────────────┘\n');

  // Day coverage
  const dayCoverage = await sql`
    SELECT 
      COUNT(DISTINCT day_of_year) as days_submitted,
      COUNT(DISTINCT CASE WHEN status = 'completed' THEN day_of_year END) as days_with_completions
    FROM heygen_videos
  `;
  
  console.log('┌─────────────────────────────────────────────────────────────┐');
  console.log('│ COVERAGE                                                    │');
  console.log('├─────────────────────────────────────────────────────────────┤');
  const dc = dayCoverage[0];
  console.log(`│ 📅 Days submitted:       ${dc.days_submitted}/365                            │`);
  console.log(`│ ✅ Days with completions: ${dc.days_with_completions.toString().padStart(3)}/365                            │`);
  console.log('└─────────────────────────────────────────────────────────────┘\n');

  // Today's lesson (Day 34)
  const day34 = await sql`
    SELECT 
      status,
      COUNT(*) as count
    FROM heygen_videos
    WHERE day_of_year = 34
    GROUP BY status
  `;
  
  console.log('┌─────────────────────────────────────────────────────────────┐');
  console.log('│ TODAY (Day 34 - How Magnets Work)                           │');
  console.log('├─────────────────────────────────────────────────────────────┤');
  for (const d of day34) {
    const icon = d.status === 'completed' ? '✅' : '⏳';
    console.log(`│ ${icon} ${d.status.padEnd(12)}: ${d.count.toString().padStart(2)} videos                               │`);
  }
  console.log('└─────────────────────────────────────────────────────────────┘\n');

  // Video sources in kelly_lesson_assets
  const sources = await sql`
    SELECT 
      video_source,
      COUNT(*) as count
    FROM kelly_lesson_assets
    WHERE video_url IS NOT NULL
    GROUP BY video_source
  `;
  
  if (sources.length > 0) {
    console.log('┌─────────────────────────────────────────────────────────────┐');
    console.log('│ VIDEO SOURCES                                               │');
    console.log('├─────────────────────────────────────────────────────────────┤');
    for (const s of sources) {
      console.log(`│ ${(s.video_source || 'unknown').padEnd(15)}: ${s.count.toString().padStart(5)} videos                          │`);
    }
    console.log('└─────────────────────────────────────────────────────────────┘\n');
  }

  // Summary
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║ SUMMARY                                                      ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log(`║ HeyGen: ${heygenCompleted} completed, ${heygenTotal - heygenCompleted} processing               ║`);
  console.log(`║ Synced to assets: ${a.has_video} videos                             ║`);
  console.log(`║ Ready for Fal: ${a.audio_only} (v0's task)                         ║`);
  console.log('╚══════════════════════════════════════════════════════════════╝');
}

fullStatus().catch(console.error);
