#!/usr/bin/env node

/**
 * Batch Lesson Validator
 * Validates all lessons in the canonical directory
 */

const fs = require('fs');
const path = require('path');
const { validateLesson } = require('./validate-lesson-v2');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');

/**
 * Validate all lessons
 */
function validateAllLessons() {
  const results = {
    timestamp: new Date().toISOString(),
    total: 0,
    valid: 0,
    invalid: 0,
    warnings: 0,
    lessons: []
  };
  
  if (!fs.existsSync(LESSONS_DIR)) {
    console.error(`Error: Lessons directory not found: ${LESSONS_DIR}`);
    return results;
  }
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.')
  );
  
  results.total = files.length;
  
  console.log(`\n🔍 Validating ${files.length} lessons...\n`);
  
  files.forEach((file, index) => {
    const filePath = path.join(LESSONS_DIR, file);
    console.log(`[${index + 1}/${files.length}] ${file}`);
    
    const result = validateLesson(filePath);
    
    const lessonResult = {
      filename: file,
      path: filePath,
      valid: result.valid,
      errorCount: result.errors.length,
      warningCount: result.warnings.length,
      infoCount: result.info.length,
      errors: result.errors,
      warnings: result.warnings,
      info: result.info
    };
    
    results.lessons.push(lessonResult);
    
    if (result.valid) {
      results.valid++;
    } else {
      results.invalid++;
    }
    
    if (result.warnings.length > 0) {
      results.warnings += result.warnings.length;
    }
  });
  
  return results;
}

/**
 * Generate validation report
 */
function generateValidationReport(results) {
  const report = {
    ...results,
    summary: {
      total: results.total,
      valid: results.valid,
      invalid: results.invalid,
      passRate: results.total > 0 ? (results.valid / results.total * 100).toFixed(1) + '%' : '0%',
      totalWarnings: results.warnings,
      totalErrors: results.lessons.reduce((sum, l) => sum + l.errorCount, 0)
    },
    byStatus: {
      valid: results.lessons.filter(l => l.valid),
      invalid: results.lessons.filter(l => !l.valid)
    },
    errorBreakdown: {},
    warningBreakdown: {}
  };
  
  // Analyze error patterns
  results.lessons.forEach(lesson => {
    lesson.errors.forEach(error => {
      const category = error.split(':')[0] || 'other';
      report.errorBreakdown[category] = (report.errorBreakdown[category] || 0) + 1;
    });
  });
  
  // Analyze warning patterns
  results.lessons.forEach(lesson => {
    lesson.warnings.forEach(warning => {
      const category = warning.split(':')[0] || 'other';
      report.warningBreakdown[category] = (report.warningBreakdown[category] || 0) + 1;
    });
  });
  
  return report;
}

// CLI
if (require.main === module) {
  const results = validateAllLessons();
  const report = generateValidationReport(results);
  
  console.log('\n' + '='.repeat(60));
  console.log('📊 VALIDATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total lessons: ${report.summary.total}`);
  console.log(`✅ Valid: ${report.summary.valid}`);
  console.log(`❌ Invalid: ${report.summary.invalid}`);
  console.log(`⚠️  Total warnings: ${report.summary.totalWarnings}`);
  console.log(`📈 Pass rate: ${report.summary.passRate}`);
  
  if (Object.keys(report.errorBreakdown).length > 0) {
    console.log('\n🔴 Error Breakdown:');
    Object.entries(report.errorBreakdown)
      .sort((a, b) => b[1] - a[1])
      .forEach(([category, count]) => {
        console.log(`  ${category}: ${count}`);
      });
  }
  
  if (Object.keys(report.warningBreakdown).length > 0) {
    console.log('\n⚠️  Warning Breakdown:');
    Object.entries(report.warningBreakdown)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([category, count]) => {
        console.log(`  ${category}: ${count}`);
      });
  }
  
  if (report.byStatus.invalid.length > 0) {
    console.log('\n❌ Invalid Lessons:');
    report.byStatus.invalid.forEach(lesson => {
      console.log(`  ${lesson.filename} (${lesson.errorCount} errors)`);
      if (lesson.errors.length > 0) {
        lesson.errors.slice(0, 3).forEach(err => {
          console.log(`    - ${err}`);
        });
        if (lesson.errors.length > 3) {
          console.log(`    ... and ${lesson.errors.length - 3} more`);
        }
      }
    });
  }
  
  // Save report
  const reportPath = path.join(WORKSPACE_ROOT, 'docs', 'phasedna', 'validation-report.json');
  const reportDir = path.dirname(reportPath);
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n💾 Validation report saved to: ${reportPath}\n`);
  
  process.exit(report.summary.invalid > 0 ? 1 : 0);
}

module.exports = {
  validateAllLessons,
  generateValidationReport
};

