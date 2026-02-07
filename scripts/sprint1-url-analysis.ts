/**
 * SPRINT 1 - URL ANALYSIS
 * Check where existing video URLs are hosted (HeyGen CDN? Supabase? Blob?)
 * and whether kelly_lesson_assets videos overlap with heygen_videos.
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';

const sql = neon(process.env.DATABASE_URL!);

async function main() {
  console.log('=== URL ANALYSIS ===\n');

  // 1. heygen_videos URL patterns
  console.log('1. heygen_videos URL patterns:');
  const heygenUrls = await sql`
    SELECT 
      CASE 
        WHEN video_url LIKE '%heygen%' THEN 'heygen-cdn'
        WHEN video_url LIKE '%supabase%' THEN 'supabase'
        WHEN video_url LIKE '%blob.vercel%' THEN 'vercel-blob'
        WHEN video_url LIKE '%cloudflare%' THEN 'cloudflare'
        WHEN video_url LIKE '%r2.dev%' THEN 'cloudflare-r2'
        ELSE 'other'
      END as host_type,
      COUNT(*)::int as count,
      MIN(video_url) as sample_url
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
    GROUP BY host_type
    ORDER BY count DESC
  `;
  for (const row of heygenUrls) {
    console.log(`   ${row.host_type}: ${row.count} videos`);
    console.log(`     Sample: ${(row.sample_url as string).substring(0, 100)}`);
  }

  // 2. kelly_lesson_assets video URL patterns
  console.log('\n2. kelly_lesson_assets video URL patterns:');
  const klaUrls = await sql`
    SELECT 
      CASE 
        WHEN video_url LIKE '%heygen%' THEN 'heygen-cdn'
        WHEN video_url LIKE '%supabase%' THEN 'supabase'
        WHEN video_url LIKE '%blob.vercel%' THEN 'vercel-blob'
        WHEN video_url LIKE '%cloudflare%' THEN 'cloudflare'
        WHEN video_url LIKE '%r2.dev%' THEN 'cloudflare-r2'
        WHEN video_url LIKE '%kelly-videos%' THEN 'kelly-videos-worker'
        ELSE 'other'
      END as host_type,
      COUNT(*)::int as count,
      MIN(video_url) as sample_url
    FROM kelly_lesson_assets 
    WHERE video_url IS NOT NULL AND video_url != ''
    GROUP BY host_type
    ORDER BY count DESC
  `;
  for (const row of klaUrls) {
    console.log(`   ${row.host_type}: ${row.count} videos`);
    console.log(`     Sample: ${(row.sample_url as string).substring(0, 120)}`);
  }

  // 3. kelly_lesson_assets audio URL patterns
  console.log('\n3. kelly_lesson_assets audio URL patterns:');
  const klaAudioUrls = await sql`
    SELECT 
      CASE 
        WHEN audio_url LIKE '%r2.dev%' THEN 'cloudflare-r2'
        WHEN audio_url LIKE '%cloudflare%' THEN 'cloudflare'
        WHEN audio_url LIKE '%elevenlabs%' THEN 'elevenlabs'
        WHEN audio_url LIKE '%supabase%' THEN 'supabase'
        WHEN audio_url LIKE '%blob.vercel%' THEN 'vercel-blob'
        WHEN audio_url LIKE '%workers.dev%' THEN 'cloudflare-worker'
        ELSE 'other'
      END as host_type,
      COUNT(*)::int as count,
      MIN(audio_url) as sample_url
    FROM kelly_lesson_assets 
    WHERE audio_url IS NOT NULL AND audio_url != ''
    GROUP BY host_type
    ORDER BY count DESC
  `;
  for (const row of klaAudioUrls) {
    console.log(`   ${row.host_type}: ${row.count} audio files`);
    console.log(`     Sample: ${(row.sample_url as string).substring(0, 120)}`);
  }

  // 4. Check overlap between heygen_videos and kelly_lesson_assets
  console.log('\n4. Overlap check (heygen vs kelly_lesson_assets):');
  // heygen uses day_of_year, kla uses day_number
  const overlap = await sql`
    SELECT COUNT(DISTINCT (h.day_of_year, h.phase))::int as count
    FROM heygen_videos h
    JOIN kelly_lesson_assets kla ON h.day_of_year = kla.day_number AND h.phase = kla.phase
    WHERE h.video_url IS NOT NULL AND kla.video_url IS NOT NULL AND kla.video_url != ''
  `;
  console.log(`   Overlapping day/phase combos: ${overlap[0]?.count}`);

  // 5. kelly_lesson_assets videos NOT in heygen_videos
  const klaOnly = await sql`
    SELECT DISTINCT kla.day_number, kla.phase
    FROM kelly_lesson_assets kla
    LEFT JOIN heygen_videos h ON kla.day_number = h.day_of_year AND kla.phase = h.phase AND h.video_url IS NOT NULL
    WHERE kla.video_url IS NOT NULL AND kla.video_url != ''
      AND h.day_of_year IS NULL
    ORDER BY kla.day_number, kla.phase
  `;
  console.log(`   KLA-only video slots (NOT in heygen_videos): ${klaOnly.length}`);
  if (klaOnly.length > 0 && klaOnly.length <= 30) {
    for (const row of klaOnly) {
      console.log(`     Day ${row.day_number} - ${row.phase}`);
    }
  } else if (klaOnly.length > 30) {
    // Group by day
    const days = new Set(klaOnly.map((r: any) => r.day_number));
    console.log(`     Across ${days.size} unique days`);
    console.log(`     First 10 days: ${[...days].slice(0, 10).join(', ')}`);
  }

  // 6. Check a few heygen_video URLs for liveness
  console.log('\n5. Liveness check for heygen_videos URLs (random 3)...');
  const randomUrls = await sql`
    SELECT day_of_year, phase, video_url 
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 3
  `;
  for (const row of randomUrls) {
    try {
      const resp = await fetch(row.video_url as string, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
      console.log(`   Day ${row.day_of_year} ${row.phase}: ${resp.status} (${resp.headers.get('content-type')}, ${Math.round(parseInt(resp.headers.get('content-length') || '0') / 1024)}KB)`);
    } catch (err) {
      console.log(`   Day ${row.day_of_year} ${row.phase}: ERROR - ${(err as Error).message}`);
    }
  }

  // 7. Check a few kelly_lesson_assets video URLs for liveness
  console.log('\n6. Liveness check for kelly_lesson_assets video URLs (random 3)...');
  const klaRandomUrls = await sql`
    SELECT day_number, phase, video_url 
    FROM kelly_lesson_assets 
    WHERE video_url IS NOT NULL AND video_url != ''
    ORDER BY RANDOM()
    LIMIT 3
  `;
  for (const row of klaRandomUrls) {
    try {
      const resp = await fetch(row.video_url as string, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
      console.log(`   Day ${row.day_number} ${row.phase}: ${resp.status} (${resp.headers.get('content-type')}, ${Math.round(parseInt(resp.headers.get('content-length') || '0') / 1024)}KB)`);
    } catch (err) {
      console.log(`   Day ${row.day_number} ${row.phase}: ERROR - ${(err as Error).message}`);
    }
  }

  // 8. Which days 1-30 have full heygen coverage?
  console.log('\n7. Days 1-30 heygen_videos coverage:');
  const first30 = await sql`
    SELECT day_of_year, 
           array_agg(DISTINCT phase ORDER BY phase) as phases,
           COUNT(DISTINCT phase)::int as phase_count
    FROM heygen_videos 
    WHERE day_of_year <= 30 AND video_url IS NOT NULL
    GROUP BY day_of_year
    ORDER BY day_of_year
  `;
  for (const row of first30) {
    const complete = (row.phase_count as number) >= 5 ? '✅' : `❌ (${row.phase_count}/5)`;
    console.log(`   Day ${String(row.day_of_year).padStart(3)}: ${complete} [${(row.phases as string[]).join(', ')}]`);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
