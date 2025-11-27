/**
 * Auto-Fix Lesson Date Formats
 *
 * Converts "Day X" → "January X, 2025" format
 * Converts "Week X" → proper month names
 *
 * Run with --dry-run to preview changes
 * Run without flag to apply changes
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DRY_RUN = process.argv.includes('--dry-run');

// Generate day number to date mapping for 2025
function generateDayToDateMap(year) {
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  const map = {};
  let dayCount = 1;

  for (let month = 0; month < 12; month++) {
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    for (let day = 1; day <= daysInMonth && dayCount <= 365; day++) {
      map[dayCount] = {
        date: `${months[month]} ${day}, ${year}`,
        shortDate: `${months[month]} ${day}`,
        month: months[month]
      };
      dayCount++;
    }
  }
  return map;
}

const DAY_TO_DATE = generateDayToDateMap(2025);

// Fix a single lesson file
function fixLessonFile(filePath) {
  const fileName = path.basename(filePath);
  let modified = false;
  const changes = [];

  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const lesson = JSON.parse(content);

    // Get day number
    const dayNumber = lesson.calendar?.day || lesson.day_number || null;

    if (!dayNumber) {
      return { modified: false, changes: [], error: 'No day number found' };
    }

    const correctDate = DAY_TO_DATE[dayNumber];
    if (!correctDate) {
      return { modified: false, changes: [], error: `Invalid day number: ${dayNumber}` };
    }

    // Fix date field
    if (lesson.calendar?.date) {
      const dayPattern = /^Day\s+\d+$/i;
      if (dayPattern.test(lesson.calendar.date)) {
        changes.push({
          field: 'calendar.date',
          from: lesson.calendar.date,
          to: correctDate.date
        });
        lesson.calendar.date = correctDate.date;
        modified = true;
      }
    }

    // Fix month field
    if (lesson.calendar?.month) {
      const weekPattern = /^Week\s+\d+$/i;
      if (weekPattern.test(lesson.calendar.month)) {
        changes.push({
          field: 'calendar.month',
          from: lesson.calendar.month,
          to: correctDate.month
        });
        lesson.calendar.month = correctDate.month;
        modified = true;
      }
    }

    // Write changes if not dry run
    if (modified && !DRY_RUN) {
      fs.writeFileSync(filePath, JSON.stringify(lesson, null, 2) + '\n', 'utf8');
    }

    return { modified, changes, error: null };

  } catch (error) {
    return { modified: false, changes: [], error: error.message };
  }
}

// Main execution
async function main() {
  console.log('╔════════════════════════════════════════════════════════════════════════════╗');
  console.log('║          CURIOUS KELLY - AUTO-FIX LESSON DATES                            ║');
  console.log('╚════════════════════════════════════════════════════════════════════════════╝\n');

  if (DRY_RUN) {
    console.log('🔍 DRY RUN MODE - No files will be modified\n');
  } else {
    console.log('⚠️  LIVE MODE - Files will be modified\n');
  }

  // Directories to scan
  const lessonDirs = [
    path.join(__dirname, '..', 'lessons'),
    path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons')
  ];

  let totalFiles = 0;
  let modifiedFiles = 0;
  let totalChanges = 0;

  for (const dir of lessonDirs) {
    if (!fs.existsSync(dir)) {
      console.log(`⚠️  Directory not found: ${dir}`);
      continue;
    }

    console.log(`📁 Processing: ${dir}\n`);

    const files = fs.readdirSync(dir).filter(f =>
      f.endsWith('-dna.json') || f.endsWith('_dna.json')
    );

    for (const file of files) {
      const filePath = path.join(dir, file);
      const result = fixLessonFile(filePath);
      totalFiles++;

      if (result.error) {
        console.log(`  ❌ ${file}: ${result.error}`);
        continue;
      }

      if (result.modified) {
        modifiedFiles++;
        totalChanges += result.changes.length;
        console.log(`  ✅ ${file}:`);
        for (const change of result.changes) {
          console.log(`      ${change.field}: "${change.from}" → "${change.to}"`);
        }
      }
    }
    console.log('');
  }

  // Summary
  console.log('='.repeat(80));
  console.log('📊 SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total files scanned: ${totalFiles}`);
  console.log(`Files ${DRY_RUN ? 'to be ' : ''}modified: ${modifiedFiles}`);
  console.log(`Total changes ${DRY_RUN ? 'to be ' : ''}applied: ${totalChanges}`);

  if (DRY_RUN && modifiedFiles > 0) {
    console.log('\n💡 Run without --dry-run to apply these changes');
  }
}

main().catch(console.error);
