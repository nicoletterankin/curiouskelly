/**
 * Lesson Standards Audit Script
 *
 * Audits all lesson content (local JSON files and Supabase) for:
 * - Completeness (required fields present)
 * - Standards adherence (date format, content quality, etc.)
 *
 * STANDARDS ENFORCED:
 * 1. Dates must be calendar format (e.g., "January 1, 2025") NOT "Day 1"
 * 2. Month must be actual month name (e.g., "January") NOT "Week 1"
 * 3. Required fields must be present
 * 4. Multilingual content must be precomputed (EN + ES/FR)
 * 5. All phases must be defined
 */

import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

// Supabase config
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = process.env.PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

// Calendar mapping for day numbers to dates
const DAY_TO_DATE = generateDayToDateMap(2025);

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

// Standards definitions
const STANDARDS = {
  // Date format violations
  DATE_FORMAT: {
    id: 'DATE_FORMAT',
    name: 'Date Format Standard',
    description: 'Dates should be calendar format (e.g., "January 1, 2025") not "Day X"',
    severity: 'high',
    check: (value, context) => {
      if (!value) return null;
      const dayPattern = /^Day\s+\d+$/i;
      if (dayPattern.test(value)) {
        const dayNum = parseInt(value.match(/\d+/)[0]);
        const correctDate = DAY_TO_DATE[dayNum];
        return {
          violation: true,
          current: value,
          expected: correctDate ? correctDate.date : `Calendar date for day ${dayNum}`,
          suggestion: correctDate ? correctDate.date : null
        };
      }
      return null;
    }
  },

  // Month format violations
  MONTH_FORMAT: {
    id: 'MONTH_FORMAT',
    name: 'Month Format Standard',
    description: 'Month should be actual month name (e.g., "January") not "Week X"',
    severity: 'high',
    check: (value, context) => {
      if (!value) return null;
      const weekPattern = /^Week\s+\d+$/i;
      if (weekPattern.test(value)) {
        const dayNum = context?.dayNumber || 1;
        const correctMonth = DAY_TO_DATE[dayNum];
        return {
          violation: true,
          current: value,
          expected: correctMonth ? correctMonth.month : 'Month name',
          suggestion: correctMonth ? correctMonth.month : null
        };
      }
      return null;
    }
  },

  // Required fields check
  REQUIRED_FIELDS: {
    id: 'REQUIRED_FIELDS',
    name: 'Required Fields Present',
    description: 'Core required fields must be present',
    severity: 'high',
    requiredFields: ['id', 'title', 'description', 'category']
  },

  // Multilingual content check
  MULTILINGUAL: {
    id: 'MULTILINGUAL',
    name: 'Multilingual Content',
    description: 'Content must have EN + ES/FR precomputed',
    severity: 'medium',
    requiredLanguages: ['en', 'es', 'fr']
  },

  // Placeholder content check
  PLACEHOLDER_CONTENT: {
    id: 'PLACEHOLDER_CONTENT',
    name: 'No Placeholder Content',
    description: 'Content should not contain placeholder text',
    severity: 'medium',
    patterns: [
      /Requiere traducción completa/i,
      /Nécessite traduction complète/i,
      /TODO/i,
      /PLACEHOLDER/i,
      /Lorem ipsum/i
    ]
  },

  // Phase completeness
  PHASES_COMPLETE: {
    id: 'PHASES_COMPLETE',
    name: 'Lesson Phases Complete',
    description: 'All required phases (welcome, teaching, practice, reflection, wisdom) must be defined',
    severity: 'high',
    requiredPhases: ['welcome', 'teaching', 'practice', 'reflection', 'wisdom']
  }
};

// Audit results collector
class AuditReport {
  constructor() {
    this.violations = [];
    this.warnings = [];
    this.passed = [];
    this.summary = {
      totalFiles: 0,
      totalViolations: 0,
      totalWarnings: 0,
      totalPassed: 0,
      byStandard: {}
    };
  }

  addViolation(file, standard, details) {
    this.violations.push({ file, standard, details, severity: STANDARDS[standard]?.severity || 'medium' });
    this.summary.totalViolations++;
    this.summary.byStandard[standard] = (this.summary.byStandard[standard] || 0) + 1;
  }

  addWarning(file, standard, details) {
    this.warnings.push({ file, standard, details });
    this.summary.totalWarnings++;
  }

  addPassed(file, standard) {
    this.passed.push({ file, standard });
    this.summary.totalPassed++;
  }

  generateReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📋 LESSON STANDARDS AUDIT REPORT');
    console.log('='.repeat(80));
    console.log(`Generated: ${new Date().toISOString()}\n`);

    // Summary
    console.log('📊 SUMMARY');
    console.log('-'.repeat(40));
    console.log(`Total Files Audited: ${this.summary.totalFiles}`);
    console.log(`Total Violations: ${this.summary.totalViolations}`);
    console.log(`Total Warnings: ${this.summary.totalWarnings}`);
    console.log(`Total Checks Passed: ${this.summary.totalPassed}\n`);

    // Violations by standard
    if (Object.keys(this.summary.byStandard).length > 0) {
      console.log('📊 VIOLATIONS BY STANDARD');
      console.log('-'.repeat(40));
      for (const [standard, count] of Object.entries(this.summary.byStandard)) {
        const stdInfo = STANDARDS[standard];
        console.log(`  ${stdInfo?.name || standard}: ${count} violations`);
      }
      console.log('');
    }

    // High severity violations
    const highSeverity = this.violations.filter(v => v.severity === 'high');
    if (highSeverity.length > 0) {
      console.log('🚨 HIGH SEVERITY VIOLATIONS');
      console.log('-'.repeat(40));
      for (const v of highSeverity) {
        console.log(`\n  File: ${v.file}`);
        console.log(`  Standard: ${STANDARDS[v.standard]?.name || v.standard}`);
        console.log(`  Details:`);
        if (v.details.current) console.log(`    Current: "${v.details.current}"`);
        if (v.details.expected) console.log(`    Expected: "${v.details.expected}"`);
        if (v.details.suggestion) console.log(`    Suggestion: "${v.details.suggestion}"`);
        if (v.details.field) console.log(`    Field: ${v.details.field}`);
        if (v.details.path) console.log(`    Path: ${v.details.path}`);
      }
      console.log('');
    }

    // Medium severity violations
    const mediumSeverity = this.violations.filter(v => v.severity === 'medium');
    if (mediumSeverity.length > 0) {
      console.log('⚠️  MEDIUM SEVERITY VIOLATIONS');
      console.log('-'.repeat(40));
      for (const v of mediumSeverity) {
        console.log(`\n  File: ${v.file}`);
        console.log(`  Standard: ${STANDARDS[v.standard]?.name || v.standard}`);
        if (v.details.field) console.log(`    Field: ${v.details.field}`);
        if (v.details.pattern) console.log(`    Found placeholder: "${v.details.pattern}"`);
      }
      console.log('');
    }

    console.log('='.repeat(80));
    console.log('END OF AUDIT REPORT');
    console.log('='.repeat(80));

    return {
      violations: this.violations,
      warnings: this.warnings,
      summary: this.summary
    };
  }
}

// Audit a single lesson JSON file
function auditLessonFile(filePath, report) {
  const fileName = path.basename(filePath);

  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const lesson = JSON.parse(content);

    report.summary.totalFiles++;

    // Get day number from lesson for context
    const dayNumber = lesson.calendar?.day || lesson.day_number || 1;
    const context = { dayNumber, filePath };

    // Check date format
    if (lesson.calendar?.date) {
      const dateCheck = STANDARDS.DATE_FORMAT.check(lesson.calendar.date, context);
      if (dateCheck?.violation) {
        report.addViolation(fileName, 'DATE_FORMAT', {
          ...dateCheck,
          path: 'calendar.date'
        });
      } else {
        report.addPassed(fileName, 'DATE_FORMAT');
      }
    }

    // Check month format
    if (lesson.calendar?.month) {
      const monthCheck = STANDARDS.MONTH_FORMAT.check(lesson.calendar.month, context);
      if (monthCheck?.violation) {
        report.addViolation(fileName, 'MONTH_FORMAT', {
          ...monthCheck,
          path: 'calendar.month'
        });
      } else {
        report.addPassed(fileName, 'MONTH_FORMAT');
      }
    }

    // Check required fields
    for (const field of STANDARDS.REQUIRED_FIELDS.requiredFields) {
      if (!lesson[field]) {
        report.addViolation(fileName, 'REQUIRED_FIELDS', {
          field,
          message: `Missing required field: ${field}`
        });
      }
    }

    // Check multilingual content
    if (lesson.ageVariants) {
      for (const [ageGroup, variant] of Object.entries(lesson.ageVariants)) {
        if (variant.language) {
          const languages = Object.keys(variant.language);
          for (const reqLang of STANDARDS.MULTILINGUAL.requiredLanguages) {
            if (!languages.includes(reqLang)) {
              report.addViolation(fileName, 'MULTILINGUAL', {
                field: `ageVariants.${ageGroup}.language`,
                message: `Missing language: ${reqLang}`
              });
            }
          }

          // Check for placeholder content in translations
          for (const [lang, langContent] of Object.entries(variant.language)) {
            for (const pattern of STANDARDS.PLACEHOLDER_CONTENT.patterns) {
              const mainContent = langContent.mainContent || '';
              if (pattern.test(mainContent)) {
                report.addViolation(fileName, 'PLACEHOLDER_CONTENT', {
                  field: `ageVariants.${ageGroup}.language.${lang}.mainContent`,
                  pattern: mainContent.match(pattern)?.[0]
                });
              }
            }
          }
        }

        // Check phases completeness
        if (variant.phases) {
          const phaseIds = variant.phases.map(p => p.id || p.type);
          for (const reqPhase of STANDARDS.PHASES_COMPLETE.requiredPhases) {
            if (!phaseIds.includes(reqPhase)) {
              report.addViolation(fileName, 'PHASES_COMPLETE', {
                field: `ageVariants.${ageGroup}.phases`,
                message: `Missing phase: ${reqPhase}`
              });
            }
          }
        }
      }
    }

  } catch (error) {
    console.error(`Error parsing ${fileName}:`, error.message);
  }
}

// Audit all local lesson files
async function auditLocalLessons() {
  const report = new AuditReport();

  console.log('🔍 AUDITING LOCAL LESSON FILES\n');

  // Directories to scan
  const lessonDirs = [
    path.join(__dirname, '..', 'lessons'),
    path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
    path.join(__dirname, '..', '_archive', 'curious-kellly', 'lesson-player-v2', 'lessons')
  ];

  for (const dir of lessonDirs) {
    if (fs.existsSync(dir)) {
      console.log(`📁 Scanning: ${dir}`);
      const files = fs.readdirSync(dir).filter(f => f.endsWith('-dna.json') || f.endsWith('_dna.json'));

      for (const file of files) {
        auditLessonFile(path.join(dir, file), report);
      }
    }
  }

  return report.generateReport();
}

// Audit Supabase lessons
async function auditSupabaseLessons() {
  console.log('\n🔍 AUDITING SUPABASE LESSONS\n');

  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

    // Fetch all core lessons
    const { data: lessons, error } = await supabase
      .from('core_lessons')
      .select('*')
      .order('day_number', { ascending: true });

    if (error) {
      console.error('❌ Error fetching from Supabase:', error.message);
      console.log('ℹ️  Skipping Supabase audit (network may be unavailable)');
      return null;
    }

    console.log(`Found ${lessons.length} lessons in Supabase core_lessons table\n`);

    const report = new AuditReport();
    report.summary.totalFiles = lessons.length;

    for (const lesson of lessons) {
      const context = { dayNumber: lesson.day_number };
      const identifier = `Day ${lesson.day_number}: ${lesson.topic || 'Unknown'}`;

      // Check if date field uses "Day X" format
      if (lesson.date) {
        const dateCheck = STANDARDS.DATE_FORMAT.check(lesson.date, context);
        if (dateCheck?.violation) {
          report.addViolation(identifier, 'DATE_FORMAT', {
            ...dateCheck,
            field: 'date'
          });
        }
      }

      // Check required fields
      const requiredDbFields = ['topic', 'universal_truth', 'day_number'];
      for (const field of requiredDbFields) {
        if (!lesson[field]) {
          report.addViolation(identifier, 'REQUIRED_FIELDS', {
            field,
            message: `Missing required field: ${field}`
          });
        }
      }
    }

    return report.generateReport();

  } catch (error) {
    console.error('❌ Supabase connection failed:', error.message);
    return null;
  }
}

// Generate fix script
function generateFixScript(violations) {
  console.log('\n' + '='.repeat(80));
  console.log('🔧 AUTO-FIX SUGGESTIONS');
  console.log('='.repeat(80));

  const dateViolations = violations.filter(v => v.standard === 'DATE_FORMAT');
  const monthViolations = violations.filter(v => v.standard === 'MONTH_FORMAT');

  if (dateViolations.length > 0 || monthViolations.length > 0) {
    console.log('\nTo fix date format violations, update the calendar fields:');
    console.log('```');
    console.log('"calendar": {');
    console.log('  "day": 1,');
    console.log('  "date": "January 1, 2025",  // NOT "Day 1"');
    console.log('  "month": "January"          // NOT "Week 1"');
    console.log('}');
    console.log('```\n');

    console.log('Day number to date mapping (2025):');
    console.log('-'.repeat(40));
    for (let i = 1; i <= 31; i++) {
      console.log(`  Day ${i.toString().padStart(3)} → ${DAY_TO_DATE[i].date}`);
    }
    console.log('  ... (use generateDayToDateMap() for full mapping)');
  }
}

// Main execution
async function main() {
  console.log('╔════════════════════════════════════════════════════════════════════════════╗');
  console.log('║          CURIOUS KELLY - LESSON STANDARDS AUDIT                           ║');
  console.log('╚════════════════════════════════════════════════════════════════════════════╝\n');

  // Audit local files
  const localResults = await auditLocalLessons();

  // Audit Supabase
  const supabaseResults = await auditSupabaseLessons();

  // Generate fix suggestions
  if (localResults?.violations) {
    generateFixScript(localResults.violations);
  }

  // Final summary
  console.log('\n' + '='.repeat(80));
  console.log('📋 FINAL SUMMARY');
  console.log('='.repeat(80));
  console.log(`Local Files: ${localResults?.summary?.totalViolations || 0} violations`);
  console.log(`Supabase: ${supabaseResults?.summary?.totalViolations || 'N/A (network unavailable)'} violations`);
  console.log('\nRun with --fix flag to auto-apply corrections (when implemented)');
}

main().catch(console.error);
