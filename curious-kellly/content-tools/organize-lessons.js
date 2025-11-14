#!/usr/bin/env node

/**
 * Lesson Organization Script
 * Discovers, categorizes, and organizes all lesson files
 */

const fs = require('fs');
const path = require('path');

// Resolve paths relative to workspace root
// From curious-kellly/content-tools, go up 2 levels to workspace root
const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const CANONICAL_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const LEGACY_DIR = path.join(WORKSPACE_ROOT, 'lessons');
const ARCHIVE_DIR = path.join(WORKSPACE_ROOT, 'lessons', 'archive');

// Required age groups
const REQUIRED_AGE_GROUPS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
const REQUIRED_LANGUAGES = ['en', 'es', 'fr'];

/**
 * Detect schema version of a lesson file
 */
function detectSchemaVersion(lessonData) {
  // Check for PhaseDNA v2 indicators
  if (lessonData.universal_concept || lessonData.calendar || lessonData.language_adaptation_framework) {
    return 'v2';
  }
  
  // Check for PhaseDNA v1 structure
  if (lessonData.ageVariants && lessonData.metadata && lessonData.interactions) {
    return 'v1';
  }
  
  // Check for old format (has age_expressions, lesson_id, etc.)
  if (lessonData.age_expressions || lessonData.lesson_id) {
    return 'old';
  }
  
  return 'unknown';
}

/**
 * Check multilingual completeness
 */
function checkMultilingualCompleteness(lessonData) {
  const result = {
    hasEn: false,
    hasEs: false,
    hasFr: false,
    missingSections: [],
    ageGroups: {}
  };
  
  if (!lessonData.ageVariants) {
    return result;
  }
  
  REQUIRED_AGE_GROUPS.forEach(ageGroup => {
    const variant = lessonData.ageVariants[ageGroup];
    if (!variant) {
      result.ageGroups[ageGroup] = { status: 'missing', languages: {} };
      return;
    }
    
    const langStatus = {
      en: false,
      es: false,
      fr: false,
      missingSections: []
    };
    
    if (variant.language) {
      ['en', 'es', 'fr'].forEach(lang => {
        if (variant.language[lang]) {
          langStatus[lang] = true;
          result[`has${lang.toUpperCase()}`] = true;
          
          // Check required sections
          const requiredSections = ['welcome', 'mainContent', 'keyPoints', 'interactionPrompts', 'wisdomMoment'];
          requiredSections.forEach(section => {
            if (!variant.language[lang][section]) {
              langStatus.missingSections.push(`${lang}.${section}`);
            }
          });
        }
      });
    }
    
    result.ageGroups[ageGroup] = {
      status: 'present',
      languages: langStatus
    };
  });
  
  return result;
}

/**
 * Check age variant completeness
 */
function checkAgeVariantCompleteness(lessonData) {
  const present = [];
  const missing = [];
  
  REQUIRED_AGE_GROUPS.forEach(ageGroup => {
    if (lessonData.ageVariants && lessonData.ageVariants[ageGroup]) {
      present.push(ageGroup);
    } else {
      missing.push(ageGroup);
    }
  });
  
  return { present, missing, completeness: present.length / REQUIRED_AGE_GROUPS.length };
}

/**
 * Standardize filename to kebab-case
 */
function standardizeFilename(filename) {
  return filename
    .replace(/_/g, '-')
    .replace(/\s+/g, '-')
    .replace(/--+/g, '-')
    .replace(/^dna-/, '')
    .replace(/-dna$/, '')
    .replace(/\.json$/, '')
    .toLowerCase() + '.json';
}

/**
 * Extract lesson ID from filename or data
 */
function extractLessonId(filename, lessonData) {
  if (lessonData.id) {
    return lessonData.id;
  }
  if (lessonData.lesson_id) {
    return lessonData.lesson_id;
  }
  return standardizeFilename(filename).replace('.json', '');
}

/**
 * Scan and categorize all lesson files
 */
function discoverLessons() {
  const inventory = {
    canonical: [],
    legacy: [],
    duplicates: []
  };
  
  // Scan canonical directory
  if (fs.existsSync(CANONICAL_DIR)) {
    const files = fs.readdirSync(CANONICAL_DIR).filter(f => f.endsWith('.json') && !f.startsWith('.'));
    files.forEach(file => {
      const filePath = path.join(CANONICAL_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        const schemaVersion = detectSchemaVersion(data);
        const multilingual = checkMultilingualCompleteness(data);
        const ageVariants = checkAgeVariantCompleteness(data);
        
        inventory.canonical.push({
          filename: file,
          path: filePath,
          id: extractLessonId(file, data),
          schemaVersion,
          multilingual,
          ageVariants,
          size: fs.statSync(filePath).size
        });
      } catch (e) {
        inventory.canonical.push({
          filename: file,
          path: filePath,
          error: e.message
        });
      }
    });
  }
  
  // Scan legacy directory
  if (fs.existsSync(LEGACY_DIR)) {
    const allFiles = fs.readdirSync(LEGACY_DIR);
    const files = allFiles.filter(f => 
      f.endsWith('.json') && 
      !f.startsWith('.') && 
      f !== 'archive' &&
      !f.includes('curriculum') &&
      !f.includes('schema') &&
      !f.includes('migrations')
    );
    
    files.forEach(file => {
      const filePath = path.join(LEGACY_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        const schemaVersion = detectSchemaVersion(data);
        const multilingual = checkMultilingualCompleteness(data);
        const ageVariants = checkAgeVariantCompleteness(data);
        
        const standardizedName = standardizeFilename(file);
        const lessonId = extractLessonId(file, data);
        
        // Check for duplicates
        const duplicate = inventory.canonical.find(l => l.id === lessonId);
        
        inventory.legacy.push({
          filename: file,
          standardizedName,
          path: filePath,
          id: lessonId,
          schemaVersion,
          multilingual,
          ageVariants,
          size: fs.statSync(filePath).size,
          isDuplicate: !!duplicate
        });
        
        if (duplicate) {
          inventory.duplicates.push({
            canonical: duplicate,
            legacy: {
              filename: file,
              standardizedName,
              path: filePath,
              id: lessonId,
              schemaVersion,
              multilingual,
              ageVariants,
              size: fs.statSync(filePath).size
            }
          });
        }
      } catch (e) {
        inventory.legacy.push({
          filename: file,
          path: filePath,
          error: e.message
        });
      }
    });
  }
  
  return inventory;
}

/**
 * Generate inventory report
 */
function generateInventoryReport(inventory) {
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      canonical: inventory.canonical.length,
      legacy: inventory.legacy.length,
      duplicates: inventory.duplicates.length,
      bySchemaVersion: {
        v1: 0,
        v2: 0,
        old: 0,
        unknown: 0
      },
      multilingualStatus: {
        complete: 0,
        partial: 0,
        missing: 0
      },
      ageVariantStatus: {
        complete: 0,
        partial: 0,
        missing: 0
      }
    },
    details: inventory
  };
  
  // Count by schema version
  [...inventory.canonical, ...inventory.legacy].forEach(lesson => {
    if (lesson.schemaVersion) {
      report.summary.bySchemaVersion[lesson.schemaVersion] = 
        (report.summary.bySchemaVersion[lesson.schemaVersion] || 0) + 1;
    } else {
      report.summary.bySchemaVersion.unknown++;
    }
    
    // Count multilingual status
    if (lesson.multilingual) {
      const hasAll = lesson.multilingual.hasEn && lesson.multilingual.hasEs && lesson.multilingual.hasFr;
      if (hasAll) {
        report.summary.multilingualStatus.complete++;
      } else if (lesson.multilingual.hasEn) {
        report.summary.multilingualStatus.partial++;
      } else {
        report.summary.multilingualStatus.missing++;
      }
    }
    
    // Count age variant status
    if (lesson.ageVariants) {
      if (lesson.ageVariants.completeness === 1) {
        report.summary.ageVariantStatus.complete++;
      } else if (lesson.ageVariants.completeness > 0) {
        report.summary.ageVariantStatus.partial++;
      } else {
        report.summary.ageVariantStatus.missing++;
      }
    }
  });
  
  return report;
}

/**
 * Organize lessons (move, rename, archive)
 */
function organizeLessons(inventory, dryRun = true) {
  const operations = {
    moved: [],
    renamed: [],
    archived: [],
    errors: []
  };
  
  // Ensure archive directory exists
  if (!dryRun && !fs.existsSync(ARCHIVE_DIR)) {
    fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
  }
  
  // Process legacy files
  inventory.legacy.forEach(lesson => {
    if (lesson.error) {
      operations.errors.push({
        file: lesson.filename,
        error: `Parse error: ${lesson.error}`
      });
      return;
    }
    
    const targetPath = path.join(CANONICAL_DIR, lesson.standardizedName);
    const archivePath = path.join(ARCHIVE_DIR, lesson.filename);
    
    // Skip duplicates (keep canonical version)
    if (lesson.isDuplicate) {
      if (!dryRun) {
        // Archive the legacy duplicate
        fs.copyFileSync(lesson.path, archivePath);
        fs.unlinkSync(lesson.path);
        operations.archived.push({
          from: lesson.path,
          to: archivePath,
          reason: 'duplicate'
        });
      } else {
        operations.archived.push({
          from: lesson.path,
          to: archivePath,
          reason: 'duplicate (dry run)'
        });
      }
      return;
    }
    
    // Move and rename legacy files
    if (!dryRun) {
      try {
        // Ensure canonical directory exists
        if (!fs.existsSync(CANONICAL_DIR)) {
          fs.mkdirSync(CANONICAL_DIR, { recursive: true });
        }
        
        // Copy to canonical location
        fs.copyFileSync(lesson.path, targetPath);
        
        // Archive original
        fs.copyFileSync(lesson.path, archivePath);
        
        // Remove from legacy location
        fs.unlinkSync(lesson.path);
        
        operations.moved.push({
          from: lesson.path,
          to: targetPath,
          renamed: lesson.filename !== lesson.standardizedName
        });
      } catch (e) {
        operations.errors.push({
          file: lesson.filename,
          error: e.message
        });
      }
    } else {
      operations.moved.push({
        from: lesson.path,
        to: targetPath,
        renamed: lesson.filename !== lesson.standardizedName,
        dryRun: true
      });
    }
  });
  
  return operations;
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  const dryRun = !args.includes('--execute');
  
  console.log('\n🔍 Discovering lessons...\n');
  
  const inventory = discoverLessons();
  const report = generateInventoryReport(inventory);
  
  console.log('📊 Inventory Summary:');
  console.log(`  Canonical location: ${report.summary.canonical} lessons`);
  console.log(`  Legacy location: ${report.summary.legacy} lessons`);
  console.log(`  Duplicates: ${report.summary.duplicates}`);
  console.log('\n📋 Schema Versions:');
  Object.keys(report.summary.bySchemaVersion).forEach(version => {
    console.log(`  ${version}: ${report.summary.bySchemaVersion[version]}`);
  });
  console.log('\n🌍 Multilingual Status:');
  console.log(`  Complete (EN+ES+FR): ${report.summary.multilingualStatus.complete}`);
  console.log(`  Partial (EN only): ${report.summary.multilingualStatus.partial}`);
  console.log(`  Missing: ${report.summary.multilingualStatus.missing}`);
  console.log('\n👥 Age Variant Status:');
  console.log(`  Complete (6/6): ${report.summary.ageVariantStatus.complete}`);
  console.log(`  Partial: ${report.summary.ageVariantStatus.partial}`);
  console.log(`  Missing: ${report.summary.ageVariantStatus.missing}`);
  
  if (dryRun) {
    console.log('\n⚠️  DRY RUN MODE - No files will be modified');
    console.log('   Use --execute to perform actual operations\n');
  }
  
  const operations = organizeLessons(inventory, dryRun);
  
  console.log('\n📦 Organization Operations:');
  console.log(`  Files to move: ${operations.moved.length}`);
  console.log(`  Files to archive: ${operations.archived.length}`);
  console.log(`  Errors: ${operations.errors.length}`);
  
  if (operations.errors.length > 0) {
    console.log('\n❌ Errors:');
    operations.errors.forEach(err => {
      console.log(`  ${err.file}: ${err.error}`);
    });
  }
  
  // Save report
  const reportPath = path.join(WORKSPACE_ROOT, 'docs', 'phasedna', 'lesson-inventory.json');
  const reportDir = path.dirname(reportPath);
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n💾 Inventory report saved to: ${reportPath}`);
  
  if (!dryRun) {
    console.log('\n✅ Organization complete!\n');
  }
}

module.exports = {
  discoverLessons,
  generateInventoryReport,
  organizeLessons,
  detectSchemaVersion,
  checkMultilingualCompleteness,
  checkAgeVariantCompleteness,
  standardizeFilename
};

