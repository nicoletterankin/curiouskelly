/**
 * Sprint D: Map HeyGen videos to lessons
 * Cross-reference filenames with core_lessons and heygen_videos tables
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function mapVideos() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== HeyGen to Lesson Mapping ===\n');
  
  // Load audit results
  const auditPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'heygen-video-audit.json');
  if (!fs.existsSync(auditPath)) {
    console.error('Run audit-heygen-videos.cjs first');
    process.exit(1);
  }
  const audit = JSON.parse(fs.readFileSync(auditPath, 'utf-8'));
  
  // Get heygen_videos from database
  let dbVideos = [];
  try {
    const res = await client.query(`
      SELECT id, day_number, phase, age_group, video_url, blob_url, status, heygen_id
      FROM heygen_videos
      ORDER BY day_number, phase
    `);
    dbVideos = res.rows;
    console.log(`Database heygen_videos: ${dbVideos.length} records`);
  } catch (e) {
    console.log('heygen_videos table not accessible:', e.message);
  }
  
  // Also check kelly_lesson_assets for video URLs
  let lessonAssets = [];
  try {
    const res = await client.query(`
      SELECT id, day_number, phase, age_group, video_url, status
      FROM kelly_lesson_assets
      WHERE video_url IS NOT NULL AND video_url != ''
      ORDER BY day_number, phase
    `);
    lessonAssets = res.rows;
    console.log(`kelly_lesson_assets with video: ${lessonAssets.length} records`);
  } catch (e) {
    console.log('kelly_lesson_assets not accessible:', e.message);
  }
  
  // Map files to days by directory structure
  const dayMapping = {}; // day -> { files, db_records, phases }
  
  // From file audit - parse directory names like day-001/
  for (const video of audit.videos) {
    if (video.status !== 'valid') continue;
    
    const dayMatch = video.relativePath?.match(/day[_-]?0*(\d+)/i);
    if (dayMatch) {
      const day = parseInt(dayMatch[1]);
      if (!dayMapping[day]) dayMapping[day] = { files: [], db_records: [], phases_covered: new Set() };
      dayMapping[day].files.push(video);
      
      // Determine phase from filename
      const phaseMatch = video.filename?.match(/^(hook|teach|example|practice|reflect|apply|close|story|wonder|action|wisdom|cliff|q[123]|outro)/i);
      if (phaseMatch) {
        dayMapping[day].phases_covered.add(phaseMatch[1].toLowerCase());
      }
    }
  }
  
  // From database records
  for (const rec of dbVideos) {
    const day = rec.day_number;
    if (!dayMapping[day]) dayMapping[day] = { files: [], db_records: [], phases_covered: new Set() };
    dayMapping[day].db_records.push(rec);
    if (rec.phase) dayMapping[day].phases_covered.add(rec.phase.toLowerCase());
  }
  
  for (const rec of lessonAssets) {
    const day = rec.day_number;
    if (!dayMapping[day]) dayMapping[day] = { files: [], db_records: [], phases_covered: new Set() };
    dayMapping[day].db_records.push(rec);
    if (rec.phase) dayMapping[day].phases_covered.add(rec.phase.toLowerCase());
  }
  
  // Build mapping report
  const expectedPhases = ['hook', 'teach', 'example', 'practice', 'reflect', 'apply', 'close'];
  const mapping = {
    summary: {
      days_with_videos: Object.keys(dayMapping).length,
      days_without_videos: 365 - Object.keys(dayMapping).length,
      total_video_files: audit.videos.filter(v => v.status === 'valid').length,
      total_db_records: dbVideos.length + lessonAssets.length,
    },
    days: {},
    gaps: [],
  };
  
  for (let day = 1; day <= 365; day++) {
    const dm = dayMapping[day];
    if (dm) {
      const phasesArr = Array.from(dm.phases_covered);
      const missing = expectedPhases.filter(p => !dm.phases_covered.has(p));
      mapping.days[day] = {
        file_count: dm.files.length,
        db_record_count: dm.db_records.length,
        phases_covered: phasesArr,
        phases_missing: missing,
        coverage: `${phasesArr.length}/7`,
      };
      
      if (missing.length > 0) {
        mapping.gaps.push({ day, missing, covered: phasesArr.length });
      }
    } else {
      mapping.days[day] = { file_count: 0, db_record_count: 0, phases_covered: [], phases_missing: expectedPhases, coverage: '0/7' };
      mapping.gaps.push({ day, missing: expectedPhases, covered: 0 });
    }
  }
  
  // Save mapping
  const mappingPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'heygen-lesson-mapping.json');
  fs.writeFileSync(mappingPath, JSON.stringify(mapping, null, 2));
  
  // Coverage report
  const coveragePath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'heygen-coverage-report.json');
  const fullCoverage = Object.entries(mapping.days).filter(([, d]) => d.phases_covered.length === 7).length;
  const partialCoverage = Object.entries(mapping.days).filter(([, d]) => d.phases_covered.length > 0 && d.phases_covered.length < 7).length;
  const noCoverage = Object.entries(mapping.days).filter(([, d]) => d.phases_covered.length === 0).length;
  
  const coverageReport = {
    summary: {
      full_coverage_7_of_7: fullCoverage,
      partial_coverage: partialCoverage,
      no_coverage: noCoverage,
      total_days: 365,
    },
    gaps_by_day: mapping.gaps.slice(0, 50), // first 50 gaps
    human_readable: mapping.gaps.slice(0, 20).map(g => 
      `Day ${g.day}: ${g.covered}/7 phases covered, missing: ${g.missing.join(', ')}`
    ),
  };
  fs.writeFileSync(coveragePath, JSON.stringify(coverageReport, null, 2));
  
  console.log(`\n=== Coverage Summary ===`);
  console.log(`Days with at least 1 video: ${mapping.summary.days_with_videos} of 365`);
  console.log(`Total video files mapped: ${mapping.summary.total_video_files}`);
  console.log(`Full coverage (7/7): ${fullCoverage}`);
  console.log(`Partial coverage: ${partialCoverage}`);
  console.log(`No coverage: ${noCoverage}`);
  console.log(`\nFirst 10 gaps:`);
  coverageReport.human_readable.slice(0, 10).forEach(l => console.log(`  ${l}`));
  console.log(`\nSaved to:`);
  console.log(`  ${mappingPath}`);
  console.log(`  ${coveragePath}`);
  
  await client.end();
}

mapVideos().catch(e => { console.error(e); process.exit(1); });
