/**
 * SPRINT 1 - INVENTORY SCRIPT
 * Cross-references ALL video-related tables in Neon.
 * Uses tagged template literals as required by @neondatabase/serverless.
 * Output: sprint1-inventory-report.json
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';
import * as fs from 'fs';

if (!process.env.DATABASE_URL) {
  console.error('FATAL: DATABASE_URL not found. Check .env file.');
  process.exit(1);
}

const sql = neon(process.env.DATABASE_URL);

async function main() {
  console.log('=== SPRINT 1: DATABASE INVENTORY ===\n');
  const report: Record<string, unknown> = {};

  try {
    // 1. heygen_videos (Priority 0 - THE target table)
    console.log('1. Querying heygen_videos...');
    const heygenTotal = await sql`SELECT COUNT(*)::int as count FROM heygen_videos`;
    const heygenWithVideo = await sql`SELECT COUNT(*)::int as count FROM heygen_videos WHERE video_url IS NOT NULL AND video_url != ''`;
    const heygenStatuses = await sql`SELECT status, COUNT(*)::int as count FROM heygen_videos GROUP BY status ORDER BY count DESC`;
    const heygenPhases = await sql`SELECT phase, COUNT(*)::int as count FROM heygen_videos GROUP BY phase ORDER BY count DESC`;
    const heygenDayRange = await sql`SELECT MIN(day_of_year)::int as min_day, MAX(day_of_year)::int as max_day FROM heygen_videos`;
    const heygenDayCoverage = await sql`SELECT COUNT(DISTINCT day_of_year)::int as unique_days FROM heygen_videos WHERE video_url IS NOT NULL`;
    const heygenSample = await sql`SELECT day_of_year, phase, status, video_url, age_category, archetype FROM heygen_videos WHERE video_url IS NOT NULL ORDER BY day_of_year LIMIT 5`;
    const heygenUrlPrefixes = await sql`SELECT SUBSTRING(video_url FROM 1 FOR 60) as url_prefix, COUNT(*)::int as count FROM heygen_videos WHERE video_url IS NOT NULL GROUP BY SUBSTRING(video_url FROM 1 FOR 60) ORDER BY count DESC LIMIT 10`;
    const heygenAgeCategories = await sql`SELECT age_category, COUNT(*)::int as count FROM heygen_videos GROUP BY age_category ORDER BY count DESC`;

    report.heygen_videos = {
      total: heygenTotal[0]?.count,
      withVideoUrl: heygenWithVideo[0]?.count,
      statuses: Object.fromEntries(heygenStatuses.map((r: any) => [r.status, r.count])),
      phases: Object.fromEntries(heygenPhases.map((r: any) => [r.phase, r.count])),
      dayRange: heygenDayRange[0],
      uniqueDaysWithVideo: heygenDayCoverage[0]?.unique_days,
      ageCategories: Object.fromEntries(heygenAgeCategories.map((r: any) => [r.age_category, r.count])),
      urlPrefixes: heygenUrlPrefixes,
      sample: heygenSample,
    };
    console.log(`   Total: ${heygenTotal[0]?.count}, With video_url: ${heygenWithVideo[0]?.count}`);
    console.log(`   Day range: ${heygenDayRange[0]?.min_day} - ${heygenDayRange[0]?.max_day}`);
    console.log(`   Unique days with video: ${heygenDayCoverage[0]?.unique_days}`);
    console.log(`   Statuses:`, Object.fromEntries(heygenStatuses.map((r: any) => [r.status, r.count])));
    console.log(`   Phases:`, Object.fromEntries(heygenPhases.map((r: any) => [r.phase, r.count])));
    console.log(`   Age categories:`, Object.fromEntries(heygenAgeCategories.map((r: any) => [r.age_category, r.count])));
  } catch (err) {
    console.error('heygen_videos query failed:', (err as Error).message);
    report.heygen_videos = { error: (err as Error).message };
  }

  try {
    // 2. kelly_video_assets
    console.log('\n2. Querying kelly_video_assets...');
    const kvaTotal = await sql`SELECT COUNT(*)::int as count FROM kelly_video_assets`;
    const kvaColumns = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'kelly_video_assets' ORDER BY ordinal_position`;
    console.log(`   Total: ${kvaTotal[0]?.count}`);
    console.log(`   Columns:`, kvaColumns.map((r: any) => r.column_name).join(', '));
    
    // Check different URL column names
    const kvaStatuses = await sql`SELECT status, COUNT(*)::int as count FROM kelly_video_assets GROUP BY status ORDER BY count DESC`;
    console.log(`   Statuses:`, Object.fromEntries(kvaStatuses.map((r: any) => [r.status, r.count])));
    
    report.kelly_video_assets = {
      total: kvaTotal[0]?.count,
      columns: kvaColumns.map((r: any) => r.column_name),
      statuses: Object.fromEntries(kvaStatuses.map((r: any) => [r.status, r.count])),
    };
  } catch (err) {
    console.error('kelly_video_assets query failed:', (err as Error).message);
    report.kelly_video_assets = { error: (err as Error).message };
  }

  try {
    // 3. generated_assets
    console.log('\n3. Querying generated_assets...');
    const gaTotal = await sql`SELECT COUNT(*)::int as count FROM generated_assets`;
    const gaByType = await sql`SELECT asset_type, status, COUNT(*)::int as count FROM generated_assets GROUP BY asset_type, status ORDER BY count DESC`;
    console.log(`   Total: ${gaTotal[0]?.count}`);
    console.log(`   By type/status:`, gaByType);

    report.generated_assets = {
      total: gaTotal[0]?.count,
      byTypeStatus: gaByType,
    };
  } catch (err) {
    console.error('generated_assets query failed:', (err as Error).message);
    report.generated_assets = { error: (err as Error).message };
  }

  try {
    // 4. kelly_lesson_assets
    console.log('\n4. Querying kelly_lesson_assets...');
    const klaTotal = await sql`SELECT COUNT(*)::int as count FROM kelly_lesson_assets`;
    const klaWithVideo = await sql`SELECT COUNT(*)::int as count FROM kelly_lesson_assets WHERE video_url IS NOT NULL AND video_url != ''`;
    const klaWithAudio = await sql`SELECT COUNT(*)::int as count FROM kelly_lesson_assets WHERE audio_url IS NOT NULL AND audio_url != ''`;
    const klaDayRange = await sql`SELECT MIN(day_number)::int as min_day, MAX(day_number)::int as max_day, COUNT(DISTINCT day_number)::int as unique_days FROM kelly_lesson_assets WHERE audio_url IS NOT NULL`;
    const klaPhases = await sql`SELECT phase, COUNT(*)::int as count FROM kelly_lesson_assets GROUP BY phase ORDER BY count DESC`;

    report.kelly_lesson_assets = {
      total: klaTotal[0]?.count,
      withVideoUrl: klaWithVideo[0]?.count,
      withAudioUrl: klaWithAudio[0]?.count,
      dayRange: klaDayRange[0],
      phases: Object.fromEntries(klaPhases.map((r: any) => [r.phase, r.count])),
    };
    console.log(`   Total: ${klaTotal[0]?.count}, With video: ${klaWithVideo[0]?.count}, With audio: ${klaWithAudio[0]?.count}`);
    console.log(`   Day range: ${klaDayRange[0]?.min_day} - ${klaDayRange[0]?.max_day}`);
    console.log(`   Unique days with audio: ${klaDayRange[0]?.unique_days}`);
  } catch (err) {
    console.error('kelly_lesson_assets query failed:', (err as Error).message);
    report.kelly_lesson_assets = { error: (err as Error).message };
  }

  try {
    // 5. video_jobs
    console.log('\n5. Querying video_jobs...');
    const vjTotal = await sql`SELECT COUNT(*)::int as count FROM video_jobs`;
    const vjCompleted = await sql`SELECT COUNT(*)::int as count FROM video_jobs WHERE status = 'completed'`;
    const vjWithUrl = await sql`SELECT COUNT(*)::int as count FROM video_jobs WHERE output_url IS NOT NULL AND output_url != ''`;
    const vjEngines = await sql`SELECT engine, status, COUNT(*)::int as count FROM video_jobs GROUP BY engine, status ORDER BY count DESC`;

    report.video_jobs = {
      total: vjTotal[0]?.count,
      completed: vjCompleted[0]?.count,
      withOutputUrl: vjWithUrl[0]?.count,
      engines: vjEngines,
    };
    console.log(`   Total: ${vjTotal[0]?.count}, Completed: ${vjCompleted[0]?.count}, With output_url: ${vjWithUrl[0]?.count}`);
    console.log(`   Engines:`, vjEngines);
  } catch (err) {
    console.error('video_jobs query failed:', (err as Error).message);
    report.video_jobs = { error: (err as Error).message };
  }

  try {
    // 6. Content tables
    console.log('\n6. Querying content tables...');
    const coreLessons = await sql`SELECT COUNT(*)::int as count FROM core_lessons`;
    const lessonAtoms = await sql`SELECT COUNT(*)::int as count FROM lesson_atoms`;
    const lessons = await sql`SELECT COUNT(*)::int as count FROM lessons`;

    report.content = {
      core_lessons: coreLessons[0]?.count,
      lesson_atoms: lessonAtoms[0]?.count,
      lessons: lessons[0]?.count,
    };
    console.log(`   core_lessons: ${coreLessons[0]?.count}, lesson_atoms: ${lessonAtoms[0]?.count}, lessons: ${lessons[0]?.count}`);
  } catch (err) {
    console.error('Content tables query failed:', (err as Error).message);
    report.content = { error: (err as Error).message };
  }

  try {
    // 7. Coverage analysis — which days have videos in heygen_videos?
    console.log('\n7. Coverage analysis (heygen_videos)...');
    const coveredDays = await sql`
      SELECT day_of_year, 
             COUNT(*)::int as total_rows,
             COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END)::int as with_video,
             array_agg(DISTINCT phase) as phases
      FROM heygen_videos 
      WHERE video_url IS NOT NULL
      GROUP BY day_of_year 
      ORDER BY day_of_year
    `;

    report.coverage = {
      daysWithAnyVideo: coveredDays.length,
      daysWithAllPhases: coveredDays.filter((d: any) => d.with_video >= 5).length,
      daysList: coveredDays.map((d: any) => ({
        day: d.day_of_year,
        withVideo: d.with_video,
        phases: d.phases,
      })),
    };
    console.log(`   Days with any video: ${coveredDays.length}`);
    console.log(`   Days with all 5 phases: ${coveredDays.filter((d: any) => d.with_video >= 5).length}`);
    
    // Show first 10 days
    if (coveredDays.length > 0) {
      console.log('   First 10 covered days:');
      coveredDays.slice(0, 10).forEach((d: any) => {
        console.log(`     Day ${d.day_of_year}: ${d.with_video} phases [${d.phases?.join(', ')}]`);
      });
    }
  } catch (err) {
    console.error('Coverage query failed:', (err as Error).message);
    report.coverage = { error: (err as Error).message };
  }

  try {
    // 8. URL liveness check (sample 5 URLs)
    console.log('\n8. URL liveness check (5 samples)...');
    const urlSample = await sql`SELECT day_of_year, phase, video_url FROM heygen_videos WHERE video_url IS NOT NULL ORDER BY day_of_year LIMIT 5`;
    
    const urlChecks = [];
    for (const row of urlSample) {
      const url = row.video_url as string;
      try {
        const resp = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
        const info = {
          day: row.day_of_year,
          phase: row.phase,
          url: url.substring(0, 80),
          status: resp.status,
          contentType: resp.headers.get('content-type'),
          contentLength: resp.headers.get('content-length'),
        };
        urlChecks.push(info);
        console.log(`   Day ${row.day_of_year} ${row.phase}: ${resp.status} (${resp.headers.get('content-type')}, ${resp.headers.get('content-length')} bytes)`);
      } catch (err) {
        urlChecks.push({
          day: row.day_of_year,
          phase: row.phase,
          url: url.substring(0, 80),
          status: 'ERROR',
          error: (err as Error).message,
        });
        console.log(`   Day ${row.day_of_year} ${row.phase}: ERROR - ${(err as Error).message}`);
      }
    }
    report.urlChecks = urlChecks;
  } catch (err) {
    console.error('URL check failed:', (err as Error).message);
    report.urlChecks = { error: (err as Error).message };
  }

  // Write report
  const reportPath = 'C:\\Users\\user\\kelly-pipeline\\sprint1-inventory-report.json';
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n=== REPORT SAVED: ${reportPath} ===`);

  // Summary
  console.log('\n══════════════════════════════════════════');
  console.log('        SPRINT 1 INVENTORY SUMMARY');
  console.log('══════════════════════════════════════════');
  const hv = report.heygen_videos as any;
  const kva = report.kelly_video_assets as any;
  const ga = report.generated_assets as any;
  const kla = report.kelly_lesson_assets as any;
  const vj = report.video_jobs as any;
  const content = report.content as any;
  const coverage = report.coverage as any;
  
  console.log(`heygen_videos:       ${hv?.total ?? 'ERR'} rows (${hv?.withVideoUrl ?? '?'} with video_url)`);
  console.log(`kelly_video_assets:  ${kva?.total ?? 'ERR'} rows`);
  console.log(`generated_assets:    ${ga?.total ?? 'ERR'} rows`);
  console.log(`kelly_lesson_assets: ${kla?.total ?? 'ERR'} rows (${kla?.withVideoUrl ?? '?'} video, ${kla?.withAudioUrl ?? '?'} audio)`);
  console.log(`video_jobs:          ${vj?.total ?? 'ERR'} rows (${vj?.completed ?? '?'} completed)`);
  console.log(`core_lessons:        ${content?.core_lessons ?? 'ERR'}`);
  console.log(`lesson_atoms:        ${content?.lesson_atoms ?? 'ERR'}`);
  console.log(`lessons:             ${content?.lessons ?? 'ERR'}`);
  console.log('──────────────────────────────────────────');
  console.log(`Days with video:     ${coverage?.daysWithAnyVideo ?? '?'} / 365`);
  console.log(`Full 5-phase days:   ${coverage?.daysWithAllPhases ?? '?'}`);
  console.log('══════════════════════════════════════════');
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
