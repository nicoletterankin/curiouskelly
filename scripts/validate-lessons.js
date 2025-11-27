#!/usr/bin/env node
/**
 * Lesson Validator
 *
 * Validates lesson DNA files against the canonical schema.
 * Used by pre-commit hook and CI pipeline.
 *
 * Exit codes:
 *   0 = All lessons valid
 *   1 = Validation errors found
 *   2 = Script error
 *
 * Usage:
 *   node validate-lessons.js                    # Validate all
 *   node validate-lessons.js --staged           # Only staged files
 *   node validate-lessons.js path/to/lesson.json # Specific file
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// VALIDATION RULES (Inline - no external dependencies)
// ============================================================================

const REQUIRED_PHASES = ['welcome', 'teaching', 'practice', 'wisdom'];
const VALID_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'];
const DATE_PATTERN = /^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$/;
const PLACEHOLDER_PATTERNS = [/TODO/i, /PLACEHOLDER/i, /Lorem ipsum/i, /\[INSERT\]/i, /TBD/i];

class LessonValidator {
  constructor() {
    this.errors = [];
    this.warnings = [];
  }

  validate(lesson, filePath) {
    this.errors = [];
    this.warnings = [];
    const fileName = path.basename(filePath);

    // Top-level required fields
    this.requireField(lesson, 'id', fileName);
    this.requireField(lesson, 'title', fileName);
    this.requireField(lesson, 'category', fileName);

    // Calendar validation
    if (lesson.calendar) {
      this.validateCalendar(lesson.calendar, fileName);
    } else {
      this.errors.push(`${fileName}: Missing calendar`);
    }

    // Age variants validation
    if (lesson.ageVariants) {
      this.validateAgeVariants(lesson.ageVariants, fileName);
    } else {
      this.errors.push(`${fileName}: Missing ageVariants`);
    }

    return {
      valid: this.errors.length === 0,
      errors: this.errors,
      warnings: this.warnings,
      file: fileName,
    };
  }

  requireField(obj, field, context) {
    if (!obj[field]) {
      this.errors.push(`${context}: Missing required field '${field}'`);
      return false;
    }
    return true;
  }

  validateCalendar(calendar, context) {
    if (!calendar.day || typeof calendar.day !== 'number') {
      this.errors.push(`${context}: calendar.day must be a number`);
    }

    if (!calendar.date) {
      this.errors.push(`${context}: calendar.date is required`);
    } else if (!DATE_PATTERN.test(calendar.date)) {
      this.errors.push(`${context}: calendar.date must be format "January 1, 2025" not "${calendar.date}"`);
    }

    if (!calendar.month || !VALID_MONTHS.includes(calendar.month)) {
      this.errors.push(`${context}: calendar.month must be a valid month name`);
    }
  }

  validateAgeVariants(ageVariants, context) {
    // Must have at least 18-35 bucket
    if (!ageVariants['18-35']) {
      this.errors.push(`${context}: Missing required age variant '18-35'`);
    }

    for (const [bucket, variant] of Object.entries(ageVariants)) {
      this.validateVariant(variant, `${context}[${bucket}]`);
    }
  }

  validateVariant(variant, context) {
    // Must have title
    if (!variant.title) {
      this.errors.push(`${context}: Missing title`);
    }

    // Must have phases
    if (!variant.phases || !Array.isArray(variant.phases)) {
      this.errors.push(`${context}: Missing phases array`);
      return;
    }

    // Check required phases exist
    const phaseIds = variant.phases.map(p => p.id || p.type);
    for (const required of REQUIRED_PHASES) {
      if (!phaseIds.includes(required)) {
        this.errors.push(`${context}: Missing required phase '${required}'`);
      }
    }

    // Validate each phase
    for (const phase of variant.phases) {
      this.validatePhase(phase, context);
    }

    // Validate language content
    if (variant.language) {
      this.validateLanguage(variant.language, context);
    }
  }

  validatePhase(phase, context) {
    const phaseId = phase.id || phase.type || 'unknown';
    const phaseContext = `${context}.phases[${phaseId}]`;

    // Must have content
    if (!phase.content || phase.content.length < 10) {
      this.errors.push(`${phaseContext}: Content too short or missing`);
    }

    // Check for placeholder content
    if (phase.content) {
      for (const pattern of PLACEHOLDER_PATTERNS) {
        if (pattern.test(phase.content)) {
          this.errors.push(`${phaseContext}: Contains placeholder text`);
          break;
        }
      }
    }

    // Teaching and practice phases MUST have choices
    if (phaseId === 'teaching' || phaseId === 'practice') {
      if (!phase.choices || !Array.isArray(phase.choices) || phase.choices.length < 2) {
        this.errors.push(`${phaseContext}: Must have exactly 2 choices`);
      } else {
        this.validateChoices(phase.choices, phaseContext);
      }
    }
  }

  validateChoices(choices, context) {
    if (choices.length !== 2) {
      this.errors.push(`${context}: Must have exactly 2 choices, has ${choices.length}`);
      return;
    }

    for (let i = 0; i < choices.length; i++) {
      const choice = choices[i];
      const choiceContext = `${context}.choices[${i}]`;

      if (!choice.text || choice.text.length < 5) {
        this.errors.push(`${choiceContext}: Choice text too short or missing`);
      }

      if (!choice.response || choice.response.length < 10) {
        this.errors.push(`${choiceContext}: Kelly's response too short or missing`);
      }
    }
  }

  validateLanguage(language, context) {
    // English is required
    if (!language.en) {
      this.errors.push(`${context}.language: Missing English (en) content`);
    } else {
      this.validateLanguageContent(language.en, `${context}.language.en`);
    }

    // Spanish and French are optional but flag if placeholder
    if (language.es) {
      this.checkPlaceholderLanguage(language.es, `${context}.language.es`);
    }
    if (language.fr) {
      this.checkPlaceholderLanguage(language.fr, `${context}.language.fr`);
    }
  }

  validateLanguageContent(content, context) {
    const required = ['welcome', 'mainContent', 'wisdomMoment'];
    for (const field of required) {
      if (!content[field] || content[field].length < 20) {
        this.errors.push(`${context}: Missing or too short '${field}'`);
      }
    }
  }

  checkPlaceholderLanguage(content, context) {
    const text = JSON.stringify(content);
    for (const pattern of PLACEHOLDER_PATTERNS) {
      if (pattern.test(text)) {
        this.warnings.push(`${context}: Contains placeholder translation`);
        break;
      }
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  const stagedOnly = args.includes('--staged');
  const specificFile = args.find(a => a.endsWith('.json'));

  console.log('=' .repeat(60));
  console.log('LESSON VALIDATOR');
  console.log('=' .repeat(60));

  let filesToValidate = [];

  if (specificFile) {
    filesToValidate = [specificFile];
  } else if (stagedOnly) {
    // Get staged files from git
    try {
      const staged = execSync('git diff --cached --name-only --diff-filter=ACM', { encoding: 'utf8' });
      filesToValidate = staged.split('\n')
        .filter(f => f.endsWith('-dna.json') || f.endsWith('_dna.json'))
        .map(f => path.resolve(f));
    } catch (e) {
      console.error('Failed to get staged files:', e.message);
      process.exit(2);
    }
  } else {
    // Validate all lesson files
    const dirs = [
      path.join(__dirname, '..', 'lessons'),
      path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
    ];

    for (const dir of dirs) {
      if (fs.existsSync(dir)) {
        const files = fs.readdirSync(dir)
          .filter(f => f.endsWith('-dna.json') || f.endsWith('_dna.json'))
          .map(f => path.join(dir, f));
        filesToValidate.push(...files);
      }
    }

    // Remove duplicates based on filename (prefer first occurrence = main lessons dir)
    const seen = new Set();
    filesToValidate = filesToValidate.filter(f => {
      const name = path.basename(f);
      if (seen.has(name)) return false;
      seen.add(name);
      return true;
    });
  }

  if (filesToValidate.length === 0) {
    console.log('No lesson files to validate.');
    process.exit(0);
  }

  console.log(`Validating ${filesToValidate.length} files...\n`);

  const validator = new LessonValidator();
  let totalErrors = 0;
  let totalWarnings = 0;
  const results = [];

  for (const file of filesToValidate) {
    if (!fs.existsSync(file)) continue;

    try {
      const content = fs.readFileSync(file, 'utf8');
      const lesson = JSON.parse(content);
      const result = validator.validate(lesson, file);
      results.push(result);

      const status = result.valid ? '✅' : '❌';
      console.log(`${status} ${result.file}: ${result.errors.length} errors, ${result.warnings.length} warnings`);

      if (!result.valid) {
        result.errors.slice(0, 5).forEach(e => console.log(`   ⛔ ${e}`));
        if (result.errors.length > 5) {
          console.log(`   ... and ${result.errors.length - 5} more errors`);
        }
      }

      totalErrors += result.errors.length;
      totalWarnings += result.warnings.length;
    } catch (e) {
      console.log(`❌ ${path.basename(file)}: Parse error - ${e.message}`);
      totalErrors++;
    }
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('VALIDATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Files validated: ${filesToValidate.length}`);
  console.log(`Total errors: ${totalErrors}`);
  console.log(`Total warnings: ${totalWarnings}`);
  console.log(`Pass rate: ${Math.round(results.filter(r => r.valid).length / results.length * 100)}%`);

  process.exit(totalErrors > 0 ? 1 : 0);
}

main().catch(e => {
  console.error('Validator error:', e);
  process.exit(2);
});
