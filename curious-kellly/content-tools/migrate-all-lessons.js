#!/usr/bin/env node

/**
 * Batch Migration Script
 * Migrates all old-format lessons to PhaseDNA v2
 */

const fs = require('fs');
const path = require('path');
const { migrateLesson } = require('./migrate-to-phasedna-v2');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const { detectSchemaVersion } = require('./organize-lessons');

/**
 * Migrate all old-format lessons
 */
function migrateAllLessons() {
  const results = {
    timestamp: new Date().toISOString(),
    total: 0,
    migrated: 0,
    skipped: 0,
    errors: 0,
    details: []
  };
  
  if (!fs.existsSync(LESSONS_DIR)) {
    console.error(`Error: Lessons directory not found: ${LESSONS_DIR}`);
    return results;
  }
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.')
  );
  
  results.total = files.length;
  
  console.log(`\n🔄 Migrating old-format lessons...\n`);
  
  files.forEach((file, index) => {
    const filePath = path.join(LESSONS_DIR, file);
    console.log(`[${index + 1}/${files.length}] ${file}`);
    
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const schemaVersion = detectSchemaVersion(data);
      
      // Check if it has age_expressions (old format) even if detected as v2
      const hasOldFormat = data.age_expressions && !data.ageVariants;
      
      if (schemaVersion === 'old' || hasOldFormat) {
        // Create backup
        const backupPath = filePath + '.backup';
        fs.copyFileSync(filePath, backupPath);
        
        // Migrate
        migrateLesson(filePath, filePath);
        
        results.migrated++;
        results.details.push({
          filename: file,
          status: 'migrated',
          backup: backupPath
        });
        console.log(`  ✅ Migrated (backup: ${path.basename(backupPath)})`);
      } else if (schemaVersion === 'v1' || schemaVersion === 'v2') {
        results.skipped++;
        results.details.push({
          filename: file,
          status: 'skipped',
          reason: `Already ${schemaVersion}`
        });
        console.log(`  ⏭️  Skipped (already ${schemaVersion})`);
      } else {
        results.skipped++;
        results.details.push({
          filename: file,
          status: 'skipped',
          reason: `Unknown schema version: ${schemaVersion}`
        });
        console.log(`  ⚠️  Skipped (unknown schema: ${schemaVersion})`);
      }
    } catch (e) {
      results.errors++;
      results.details.push({
        filename: file,
        status: 'error',
        error: e.message
      });
      console.log(`  ❌ Error: ${e.message}`);
    }
  });
  
  return results;
}

// CLI
if (require.main === module) {
  const results = migrateAllLessons();
  
  console.log('\n' + '='.repeat(60));
  console.log('📊 MIGRATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total lessons: ${results.total}`);
  console.log(`✅ Migrated: ${results.migrated}`);
  console.log(`⏭️  Skipped: ${results.skipped}`);
  console.log(`❌ Errors: ${results.errors}`);
  
  // Save report
  const reportPath = path.join(WORKSPACE_ROOT, 'docs', 'phasedna', 'migration-report.json');
  const reportDir = path.dirname(reportPath);
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`\n💾 Migration report saved to: ${reportPath}\n`);
}

module.exports = { migrateAllLessons };

