/**
 * SPRINT 1 - FULL GAP ANALYSIS
 * Identifies exactly which days/phases are covered vs missing.
 * Also checks kelly_lesson_assets for additional videos.
 * Output: gap-analysis.json
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';
import * as fs from 'fs';

const sql = neon(process.env.DATABASE_URL!);

async function main() {
  console.log('=== FULL GAP ANALYSIS ===\n');

  // 1. Get all covered days/phases from heygen_videos (Priority 0)
  console.log('1. Querying heygen_videos coverage...');
  const heygenCoverage = await sql`
    SELECT day_of_year, phase, video_url, age_category, archetype
    FROM heygen_videos 
    WHERE video_url IS NOT NULL AND status = 'completed'
    ORDER BY day_of_year, phase
  `;
  console.log(`   Found ${heygenCoverage.length} video records in heygen_videos`);

  // 2. Get all covered days/phases from kelly_lesson_assets (Priority 2)
  console.log('2. Querying kelly_lesson_assets video coverage...');
  const klaCoverage = await sql`
    SELECT DISTINCT day_number, phase, video_url, age_group
    FROM kelly_lesson_assets 
    WHERE video_url IS NOT NULL AND video_url != ''
    ORDER BY day_number, phase
  `;
  console.log(`   Found ${klaCoverage.length} video records in kelly_lesson_assets`);

  // 3. Get audio coverage from kelly_lesson_assets
  console.log('3. Querying kelly_lesson_assets audio coverage...');
  const klaAudioCoverage = await sql`
    SELECT DISTINCT day_number, phase
    FROM kelly_lesson_assets 
    WHERE audio_url IS NOT NULL AND audio_url != ''
    ORDER BY day_number, phase
  `;
  console.log(`   Found ${klaAudioCoverage.length} unique day/phase audio records`);

  // 4. Get audio coverage from generated_assets
  console.log('4. Querying generated_assets audio coverage...');
  const gaCoverage = await sql`
    SELECT lesson_id, phase, url, status
    FROM generated_assets 
    WHERE asset_type = 'audio' AND status = 'completed'
    ORDER BY lesson_id, phase
  `;
  console.log(`   Found ${gaCoverage.length} audio records in generated_assets`);

  // 5. Check lessons table (scripts needed for TTS/video generation)
  console.log('5. Querying lessons table for scripts...');
  let lessonsCount = 0;
  try {
    const lessons = await sql`SELECT COUNT(*)::int as count FROM lessons`;
    lessonsCount = lessons[0]?.count || 0;
  } catch {
    // Try alternative table names
    try {
      const lessons = await sql`SELECT COUNT(*)::int as count FROM lesson_atoms`;
      lessonsCount = lessons[0]?.count || 0;
      console.log(`   (Using lesson_atoms instead)`);
    } catch {
      console.log('   Neither lessons nor lesson_atoms found');
    }
  }
  console.log(`   Lessons/scripts: ${lessonsCount}`);

  // 6. Build the coverage map
  console.log('\n6. Building coverage map...');
  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'];
  
  // Track coverage per day
  type DayCoverage = {
    day: number;
    heygen: string[]; // phases with heygen video
    klaVideo: string[]; // phases with kelly_lesson_assets video
    klaAudio: string[]; // phases with kelly_lesson_assets audio
    gaAudio: string[]; // phases with generated_assets audio
    totalVideoPhases: number;
    totalAudioPhases: number;
    missingVideoPhases: string[];
    missingAudioPhases: string[];
  };

  const coverageMap: DayCoverage[] = [];

  // Index heygen coverage
  const heygenByDay = new Map<number, Set<string>>();
  for (const row of heygenCoverage) {
    const day = row.day_of_year as number;
    if (!heygenByDay.has(day)) heygenByDay.set(day, new Set());
    heygenByDay.get(day)!.add(row.phase as string);
  }

  // Index kla video coverage
  const klaVideoByDay = new Map<number, Set<string>>();
  for (const row of klaCoverage) {
    const day = row.day_number as number;
    if (!klaVideoByDay.has(day)) klaVideoByDay.set(day, new Set());
    klaVideoByDay.get(day)!.add(row.phase as string);
  }

  // Index kla audio coverage
  const klaAudioByDay = new Map<number, Set<string>>();
  for (const row of klaAudioCoverage) {
    const day = row.day_number as number;
    if (!klaAudioByDay.has(day)) klaAudioByDay.set(day, new Set());
    klaAudioByDay.get(day)!.add(row.phase as string);
  }

  // Index generated_assets audio coverage  
  const gaAudioByDay = new Map<number, Set<string>>();
  for (const row of gaCoverage) {
    // lesson_id format: 'day-019-2026' or 'day-019'
    const match = (row.lesson_id as string).match(/day-(\d+)/);
    if (match) {
      const day = parseInt(match[1], 10);
      if (!gaAudioByDay.has(day)) gaAudioByDay.set(day, new Set());
      gaAudioByDay.get(day)!.add(row.phase as string);
    }
  }

  // Build per-day coverage
  for (let day = 1; day <= 365; day++) {
    const heygen = heygenByDay.get(day) || new Set();
    const klaVideo = klaVideoByDay.get(day) || new Set();
    const klaAudio = klaAudioByDay.get(day) || new Set();
    const gaAudio = gaAudioByDay.get(day) || new Set();

    // Video phases = union of heygen + kla video
    const videoPhases = new Set([...heygen, ...klaVideo]);
    // Audio phases = union of kla audio + ga audio
    const audioPhases = new Set([...klaAudio, ...gaAudio]);

    const missingVideoPhases = phases.filter(p => !videoPhases.has(p));
    const missingAudioPhases = phases.filter(p => !audioPhases.has(p));

    coverageMap.push({
      day,
      heygen: [...heygen],
      klaVideo: [...klaVideo],
      klaAudio: [...klaAudio],
      gaAudio: [...gaAudio],
      totalVideoPhases: videoPhases.size,
      totalAudioPhases: audioPhases.size,
      missingVideoPhases,
      missingAudioPhases,
    });
  }

  // Summary stats
  const daysWithFullVideo = coverageMap.filter(d => d.totalVideoPhases === 5).length;
  const daysWithAnyVideo = coverageMap.filter(d => d.totalVideoPhases > 0).length;
  const daysWithNoVideo = coverageMap.filter(d => d.totalVideoPhases === 0).length;
  const daysWithFullAudio = coverageMap.filter(d => d.totalAudioPhases === 5).length;
  const daysWithAnyAudio = coverageMap.filter(d => d.totalAudioPhases > 0).length;
  
  const totalVideoSlots = 365 * 5;
  const filledVideoSlots = coverageMap.reduce((sum, d) => sum + d.totalVideoPhases, 0);
  const filledAudioSlots = coverageMap.reduce((sum, d) => sum + d.totalAudioPhases, 0);

  const report = {
    summary: {
      totalDays: 365,
      totalSlots: totalVideoSlots,
      video: {
        filledSlots: filledVideoSlots,
        emptySlots: totalVideoSlots - filledVideoSlots,
        coveragePercent: ((filledVideoSlots / totalVideoSlots) * 100).toFixed(1),
        daysWithFullCoverage: daysWithFullVideo,
        daysWithPartialCoverage: daysWithAnyVideo - daysWithFullVideo,
        daysWithNoCoverage: daysWithNoVideo,
      },
      audio: {
        filledSlots: filledAudioSlots,
        emptySlots: totalVideoSlots - filledAudioSlots,
        coveragePercent: ((filledAudioSlots / totalVideoSlots) * 100).toFixed(1),
        daysWithFullCoverage: daysWithFullAudio,
        daysWithAnyAudio,
      },
      sources: {
        heygen_videos: heygenCoverage.length,
        kelly_lesson_assets_video: klaCoverage.length,
        kelly_lesson_assets_audio: klaAudioCoverage.length,
        generated_assets_audio: gaCoverage.length,
      },
    },
    // All days missing video (sorted)
    missingVideoDays: coverageMap
      .filter(d => d.totalVideoPhases < 5)
      .map(d => ({
        day: d.day,
        missingPhases: d.missingVideoPhases,
        hasAudio: d.totalAudioPhases > 0,
        audioPhases: d.totalAudioPhases,
      })),
    // Full coverage map
    coverageMap,
  };

  // Write report
  const reportPath = 'C:\\Users\\user\\kelly-pipeline\\gap-analysis.json';
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\nReport saved: ${reportPath}`);

  // Print summary
  console.log('\n═══════════════════════════════════════════');
  console.log('         GAP ANALYSIS RESULTS');
  console.log('═══════════════════════════════════════════');
  console.log(`Total lesson slots:     ${totalVideoSlots} (365 days × 5 phases)`);
  console.log('');
  console.log('VIDEO COVERAGE:');
  console.log(`  Filled slots:         ${filledVideoSlots} / ${totalVideoSlots} (${report.summary.video.coveragePercent}%)`);
  console.log(`  Empty slots:          ${totalVideoSlots - filledVideoSlots}`);
  console.log(`  Full days (5/5):      ${daysWithFullVideo}`);
  console.log(`  Partial days:         ${daysWithAnyVideo - daysWithFullVideo}`);
  console.log(`  Zero-video days:      ${daysWithNoVideo}`);
  console.log('');
  console.log('AUDIO COVERAGE:');
  console.log(`  Filled slots:         ${filledAudioSlots} / ${totalVideoSlots} (${report.summary.audio.coveragePercent}%)`);
  console.log(`  Full days (5/5):      ${daysWithFullAudio}`);
  console.log(`  Days with any audio:  ${daysWithAnyAudio}`);
  console.log('');
  console.log('SOURCES:');
  console.log(`  heygen_videos:        ${heygenCoverage.length} video records`);
  console.log(`  kelly_lesson_assets:  ${klaCoverage.length} video, ${klaAudioCoverage.length} audio`);
  console.log(`  generated_assets:     ${gaCoverage.length} audio`);
  console.log('');
  
  // Show which days have NO video at all
  const zeroDays = coverageMap.filter(d => d.totalVideoPhases === 0);
  if (zeroDays.length > 0) {
    console.log(`DAYS WITH ZERO VIDEO (${zeroDays.length}):`);
    // Group into ranges for readability
    const ranges: string[] = [];
    let start = zeroDays[0].day;
    let end = start;
    for (let i = 1; i < zeroDays.length; i++) {
      if (zeroDays[i].day === end + 1) {
        end = zeroDays[i].day;
      } else {
        ranges.push(start === end ? `${start}` : `${start}-${end}`);
        start = zeroDays[i].day;
        end = start;
      }
    }
    ranges.push(start === end ? `${start}` : `${start}-${end}`);
    console.log(`  ${ranges.join(', ')}`);
  }
  
  console.log('═══════════════════════════════════════════');
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
