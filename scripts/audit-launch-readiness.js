/**
 * Launch Readiness Audit for Curious Kelly Lessons
 *
 * Validates that lessons are complete and ready for production:
 * 1. Phase content completeness (welcome, teaching, practice, wisdom)
 * 2. Interaction structure (2 choices per question, with responses)
 * 3. Age variant coverage (6 buckets)
 * 4. Language completeness (EN required, ES/FR optional but flagged)
 * 5. Expression cues for Kelly avatar
 * 6. Text quality (no placeholders, speakable content)
 *
 * Generates CSV report for tracking remediation
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// CONFIGURATION
// ============================================================================

const REQUIRED_PHASES = ['welcome', 'teaching', 'practice', 'wisdom'];
const OPTIONAL_PHASES = ['reflection']; // Nice to have

const AGE_BUCKETS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
const REQUIRED_LANGUAGES = ['en'];
const OPTIONAL_LANGUAGES = ['es', 'fr'];

// Minimum text length for "real" content (not placeholder)
const MIN_CONTENT_LENGTH = 20;

// Patterns that indicate placeholder content
const PLACEHOLDER_PATTERNS = [
  /TODO/i,
  /PLACEHOLDER/i,
  /Lorem ipsum/i,
  /Requiere traducción/i,
  /Nécessite traduction/i,
  /\[INSERT\]/i,
  /\[FILL IN\]/i,
  /^\s*\.\.\.\s*$/,
  /^TBD$/i,
];

// ============================================================================
// AUDIT CHECKS
// ============================================================================

class LessonAuditor {
  constructor() {
    this.results = [];
    this.summary = {
      totalLessons: 0,
      launchReady: 0,
      needsWork: 0,
      critical: 0,
      warnings: 0,
    };
  }

  /**
   * Audit a single lesson DNA file
   */
  auditLesson(filePath) {
    const fileName = path.basename(filePath);
    const lessonResult = {
      file: fileName,
      path: filePath,
      dayNumber: null,
      topic: null,
      launchReady: true,
      issues: [],
      warnings: [],
      scores: {
        phaseCompleteness: 0,
        ageVariantCoverage: 0,
        languageCoverage: 0,
        interactionQuality: 0,
        expressionCues: 0,
        overall: 0,
      },
    };

    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lesson = JSON.parse(content);

      // Extract metadata
      lessonResult.dayNumber = lesson.calendar?.day || lesson.day_number;
      lessonResult.topic = lesson.title || lesson.topic || fileName;

      // Run all checks
      this.checkMetadata(lesson, lessonResult);
      this.checkPhaseCompleteness(lesson, lessonResult);
      this.checkAgeVariants(lesson, lessonResult);
      this.checkInteractions(lesson, lessonResult);
      this.checkExpressionCues(lesson, lessonResult);
      this.checkLanguages(lesson, lessonResult);
      this.checkTextQuality(lesson, lessonResult);

      // Calculate overall score
      this.calculateOverallScore(lessonResult);

    } catch (error) {
      lessonResult.launchReady = false;
      lessonResult.issues.push({
        severity: 'critical',
        check: 'file_parse',
        message: `Failed to parse file: ${error.message}`,
      });
    }

    this.results.push(lessonResult);
    this.summary.totalLessons++;

    if (lessonResult.launchReady) {
      this.summary.launchReady++;
    } else {
      this.summary.needsWork++;
    }

    return lessonResult;
  }

  /**
   * Check basic metadata is present
   */
  checkMetadata(lesson, result) {
    const requiredFields = ['title', 'description', 'category'];

    for (const field of requiredFields) {
      if (!lesson[field]) {
        result.issues.push({
          severity: 'warning',
          check: 'metadata',
          message: `Missing metadata field: ${field}`,
        });
        result.warnings.push(`Missing: ${field}`);
      }
    }

    // Check calendar/date format
    if (lesson.calendar?.date) {
      const dayPattern = /^Day\s+\d+$/i;
      if (dayPattern.test(lesson.calendar.date)) {
        result.issues.push({
          severity: 'error',
          check: 'date_format',
          message: `Date uses "Day X" format instead of calendar date`,
          current: lesson.calendar.date,
        });
        result.launchReady = false;
      }
    }
  }

  /**
   * Check all required phases are present with content
   */
  checkPhaseCompleteness(lesson, result) {
    const ageVariants = lesson.ageVariants || {};
    let totalPhaseScore = 0;
    let maxPhaseScore = 0;

    for (const ageBucket of AGE_BUCKETS) {
      const variant = ageVariants[ageBucket];
      if (!variant) continue;

      const phases = variant.phases || [];
      const phaseMap = new Map(phases.map(p => [p.id || p.type, p]));

      for (const requiredPhase of REQUIRED_PHASES) {
        maxPhaseScore++;
        const phase = phaseMap.get(requiredPhase);

        if (!phase) {
          result.issues.push({
            severity: 'error',
            check: 'phase_missing',
            message: `Missing required phase: ${requiredPhase}`,
            ageBucket,
          });
          result.launchReady = false;
        } else if (!phase.content || phase.content.length < MIN_CONTENT_LENGTH) {
          result.issues.push({
            severity: 'error',
            check: 'phase_empty',
            message: `Phase "${requiredPhase}" has insufficient content`,
            ageBucket,
            contentLength: phase.content?.length || 0,
          });
          result.launchReady = false;
        } else {
          totalPhaseScore++;
        }
      }
    }

    result.scores.phaseCompleteness = maxPhaseScore > 0
      ? Math.round((totalPhaseScore / maxPhaseScore) * 100)
      : 0;
  }

  /**
   * Check age variant coverage
   */
  checkAgeVariants(lesson, result) {
    const ageVariants = lesson.ageVariants || {};
    const presentBuckets = Object.keys(ageVariants);
    const missingBuckets = AGE_BUCKETS.filter(b => !presentBuckets.includes(b));

    result.scores.ageVariantCoverage = Math.round(
      (presentBuckets.length / AGE_BUCKETS.length) * 100
    );

    if (missingBuckets.length > 0) {
      // At minimum, need 18-35 (default adult bucket)
      if (missingBuckets.includes('18-35')) {
        result.issues.push({
          severity: 'critical',
          check: 'age_variants',
          message: 'Missing default adult age variant (18-35)',
        });
        result.launchReady = false;
        this.summary.critical++;
      } else {
        result.issues.push({
          severity: 'warning',
          check: 'age_variants',
          message: `Missing age buckets: ${missingBuckets.join(', ')}`,
        });
        this.summary.warnings++;
      }
    }

    // Check each variant has required content
    for (const [bucket, variant] of Object.entries(ageVariants)) {
      if (!variant.title) {
        result.issues.push({
          severity: 'warning',
          check: 'age_variant_title',
          message: `Age variant ${bucket} missing title`,
        });
      }
      if (!variant.description) {
        result.issues.push({
          severity: 'warning',
          check: 'age_variant_desc',
          message: `Age variant ${bucket} missing description`,
        });
      }
    }
  }

  /**
   * Check interactions have proper structure (2 choices with responses)
   */
  checkInteractions(lesson, result) {
    const ageVariants = lesson.ageVariants || {};
    let totalInteractionScore = 0;
    let maxInteractionScore = 0;

    for (const [bucket, variant] of Object.entries(ageVariants)) {
      const phases = variant.phases || [];

      for (const phase of phases) {
        // Teaching and practice phases should have interactions
        if (phase.id === 'teaching' || phase.id === 'practice' ||
            phase.type === 'teaching' || phase.type === 'practice') {
          maxInteractionScore++;

          // Check for interaction/choices structure
          const hasInteraction = this.checkPhaseInteraction(phase, bucket, result);
          if (hasInteraction) {
            totalInteractionScore++;
          }
        }
      }
    }

    result.scores.interactionQuality = maxInteractionScore > 0
      ? Math.round((totalInteractionScore / maxInteractionScore) * 100)
      : 0;
  }

  /**
   * Check a single phase has proper interaction structure
   */
  checkPhaseInteraction(phase, ageBucket, result) {
    // Look for various interaction structures
    const choices = phase.choices || phase.options || phase.interactions;

    if (!choices || !Array.isArray(choices)) {
      // No explicit choices - check if it's a teaching moment without choice
      if (phase.teachingMoments && phase.teachingMoments.length > 0) {
        // Has teaching content, but no interactive choice
        result.issues.push({
          severity: 'warning',
          check: 'interaction_missing',
          message: `Phase "${phase.id}" has teaching but no interactive choices`,
          ageBucket,
          phase: phase.id,
        });
        return false;
      }
      return false;
    }

    if (choices.length < 2) {
      result.issues.push({
        severity: 'error',
        check: 'interaction_choices',
        message: `Phase "${phase.id}" needs 2 choices, has ${choices.length}`,
        ageBucket,
        phase: phase.id,
      });
      result.launchReady = false;
      return false;
    }

    // Check each choice has text and response
    let validChoices = 0;
    for (let i = 0; i < choices.length; i++) {
      const choice = choices[i];
      if (choice.text && choice.text.length >= 5) {
        if (choice.response || choice.feedback || choice.nextStep) {
          validChoices++;
        } else {
          result.issues.push({
            severity: 'warning',
            check: 'choice_response',
            message: `Choice ${i + 1} in "${phase.id}" missing Kelly's response`,
            ageBucket,
            phase: phase.id,
          });
        }
      }
    }

    return validChoices >= 2;
  }

  /**
   * Check expression cues exist for avatar animation
   */
  checkExpressionCues(lesson, result) {
    const ageVariants = lesson.ageVariants || {};
    let totalCues = 0;
    let phasesWithCues = 0;
    let totalPhases = 0;

    for (const [bucket, variant] of Object.entries(ageVariants)) {
      const phases = variant.phases || [];

      for (const phase of phases) {
        totalPhases++;
        const cues = phase.expressionCues || [];

        if (cues.length > 0) {
          phasesWithCues++;
          totalCues += cues.length;
        }
      }
    }

    result.scores.expressionCues = totalPhases > 0
      ? Math.round((phasesWithCues / totalPhases) * 100)
      : 0;

    if (phasesWithCues === 0 && totalPhases > 0) {
      result.issues.push({
        severity: 'warning',
        check: 'expression_cues',
        message: 'No expression cues found for Kelly avatar animation',
      });
    }
  }

  /**
   * Check language coverage
   */
  checkLanguages(lesson, result) {
    const ageVariants = lesson.ageVariants || {};
    let hasEnglish = false;
    let hasSpanish = false;
    let hasFrench = false;

    for (const [bucket, variant] of Object.entries(ageVariants)) {
      const languages = variant.language || {};

      if (languages.en) hasEnglish = true;
      if (languages.es && !this.isPlaceholder(languages.es)) hasSpanish = true;
      if (languages.fr && !this.isPlaceholder(languages.fr)) hasFrench = true;
    }

    const langCount = [hasEnglish, hasSpanish, hasFrench].filter(Boolean).length;
    result.scores.languageCoverage = Math.round((langCount / 3) * 100);

    if (!hasEnglish) {
      result.issues.push({
        severity: 'critical',
        check: 'language_en',
        message: 'Missing English content (required)',
      });
      result.launchReady = false;
      this.summary.critical++;
    }

    if (!hasSpanish) {
      result.issues.push({
        severity: 'info',
        check: 'language_es',
        message: 'Missing Spanish translation',
      });
    }

    if (!hasFrench) {
      result.issues.push({
        severity: 'info',
        check: 'language_fr',
        message: 'Missing French translation',
      });
    }
  }

  /**
   * Check text quality (no placeholders, speakable content)
   */
  checkTextQuality(lesson, result) {
    const textFields = this.extractAllText(lesson);

    for (const { path: fieldPath, text } of textFields) {
      // Check for placeholder patterns
      for (const pattern of PLACEHOLDER_PATTERNS) {
        if (pattern.test(text)) {
          result.issues.push({
            severity: 'error',
            check: 'placeholder_content',
            message: `Placeholder content found`,
            path: fieldPath,
            match: text.match(pattern)?.[0],
          });
          result.launchReady = false;
        }
      }

      // Check for very short content (might be incomplete)
      if (text.length < MIN_CONTENT_LENGTH && text.length > 0) {
        result.issues.push({
          severity: 'warning',
          check: 'short_content',
          message: `Content might be incomplete (${text.length} chars)`,
          path: fieldPath,
        });
      }
    }
  }

  /**
   * Extract all text content from lesson for quality checking
   */
  extractAllText(obj, prefix = '') {
    const texts = [];

    if (typeof obj === 'string' && obj.length > 0) {
      texts.push({ path: prefix, text: obj });
    } else if (Array.isArray(obj)) {
      obj.forEach((item, i) => {
        texts.push(...this.extractAllText(item, `${prefix}[${i}]`));
      });
    } else if (obj && typeof obj === 'object') {
      for (const [key, value] of Object.entries(obj)) {
        // Skip non-content fields
        if (['id', 'type', 'duration', 'timestamp', 'intensity'].includes(key)) continue;
        texts.push(...this.extractAllText(value, prefix ? `${prefix}.${key}` : key));
      }
    }

    return texts;
  }

  /**
   * Check if language content is placeholder
   */
  isPlaceholder(langContent) {
    if (!langContent) return true;

    const text = typeof langContent === 'string'
      ? langContent
      : JSON.stringify(langContent);

    return PLACEHOLDER_PATTERNS.some(p => p.test(text));
  }

  /**
   * Calculate overall launch readiness score
   */
  calculateOverallScore(result) {
    const weights = {
      phaseCompleteness: 0.30,
      ageVariantCoverage: 0.20,
      languageCoverage: 0.15,
      interactionQuality: 0.25,
      expressionCues: 0.10,
    };

    let overall = 0;
    for (const [key, weight] of Object.entries(weights)) {
      overall += (result.scores[key] || 0) * weight;
    }

    result.scores.overall = Math.round(overall);

    // Determine launch readiness threshold
    if (result.scores.overall < 70) {
      result.launchReady = false;
    }
  }

  /**
   * Generate CSV report
   */
  generateCSVReport() {
    const headers = [
      'File',
      'Day',
      'Topic',
      'Launch Ready',
      'Overall Score',
      'Phase Score',
      'Age Variant Score',
      'Language Score',
      'Interaction Score',
      'Expression Score',
      'Critical Issues',
      'Warnings',
      'Issues Summary'
    ];

    const rows = this.results.map(r => [
      r.file,
      r.dayNumber || '',
      r.topic || '',
      r.launchReady ? 'YES' : 'NO',
      r.scores.overall,
      r.scores.phaseCompleteness,
      r.scores.ageVariantCoverage,
      r.scores.languageCoverage,
      r.scores.interactionQuality,
      r.scores.expressionCues,
      r.issues.filter(i => i.severity === 'critical').length,
      r.issues.filter(i => i.severity === 'warning').length,
      r.issues.map(i => `[${i.severity}] ${i.message}`).join('; ')
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell =>
        typeof cell === 'string' && (cell.includes(',') || cell.includes('"'))
          ? `"${cell.replace(/"/g, '""')}"`
          : cell
      ).join(','))
    ].join('\n');

    return csv;
  }

  /**
   * Print summary report
   */
  printSummary() {
    console.log('\n' + '='.repeat(80));
    console.log('LAUNCH READINESS AUDIT SUMMARY');
    console.log('='.repeat(80));
    console.log(`Total Lessons Audited: ${this.summary.totalLessons}`);
    console.log(`Launch Ready: ${this.summary.launchReady} (${Math.round(this.summary.launchReady / this.summary.totalLessons * 100)}%)`);
    console.log(`Needs Work: ${this.summary.needsWork}`);
    console.log(`Critical Issues: ${this.summary.critical}`);
    console.log(`Warnings: ${this.summary.warnings}`);

    // Top issues
    const issueCounts = {};
    for (const result of this.results) {
      for (const issue of result.issues) {
        const key = issue.check;
        issueCounts[key] = (issueCounts[key] || 0) + 1;
      }
    }

    console.log('\nMost Common Issues:');
    Object.entries(issueCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([check, count]) => {
        console.log(`  ${check}: ${count}`);
      });

    // Lessons needing most work
    console.log('\nLessons Needing Most Work:');
    this.results
      .filter(r => !r.launchReady)
      .sort((a, b) => a.scores.overall - b.scores.overall)
      .slice(0, 10)
      .forEach(r => {
        console.log(`  ${r.file}: ${r.scores.overall}% (${r.issues.length} issues)`);
      });
  }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
  console.log('=' .repeat(80));
  console.log('CURIOUS KELLY - LAUNCH READINESS AUDIT');
  console.log('='.repeat(80));
  console.log(`Started: ${new Date().toISOString()}\n`);

  const auditor = new LessonAuditor();

  // Directories to scan
  const lessonDirs = [
    path.join(__dirname, '..', 'lessons'),
    path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
  ];

  for (const dir of lessonDirs) {
    if (!fs.existsSync(dir)) {
      console.log(`Directory not found: ${dir}`);
      continue;
    }

    console.log(`Scanning: ${dir}`);
    const files = fs.readdirSync(dir).filter(f =>
      f.endsWith('-dna.json') || f.endsWith('_dna.json')
    );

    for (const file of files) {
      const filePath = path.join(dir, file);
      const result = auditor.auditLesson(filePath);

      const status = result.launchReady ? '✅' : '❌';
      console.log(`  ${status} ${file}: ${result.scores.overall}%`);
    }
    console.log('');
  }

  // Print summary
  auditor.printSummary();

  // Generate CSV report
  const csv = auditor.generateCSVReport();
  const csvPath = path.join(__dirname, '..', 'reports', 'launch-readiness-audit.csv');

  // Ensure reports directory exists
  const reportsDir = path.dirname(csvPath);
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  fs.writeFileSync(csvPath, csv, 'utf8');
  console.log(`\nCSV report saved to: ${csvPath}`);

  // Also save detailed JSON report
  const jsonPath = path.join(__dirname, '..', 'reports', 'launch-readiness-audit.json');
  fs.writeFileSync(jsonPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    summary: auditor.summary,
    results: auditor.results,
  }, null, 2), 'utf8');
  console.log(`JSON report saved to: ${jsonPath}`);
}

main().catch(console.error);
