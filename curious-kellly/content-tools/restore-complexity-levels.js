#!/usr/bin/env node

/**
 * Restore Complexity Levels Script
 * Restores complexity_level values from backup files
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');

function restoreComplexityLevels() {
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.') && !f.includes('backup') && !f.includes('index')
  );
  
  console.log(`\n🔄 Restoring complexity_level values from backups...\n`);
  
  let restoredCount = 0;
  
  files.forEach(file => {
    const filePath = path.join(LESSONS_DIR, file);
    const backupPath = filePath + '.backup';
    
    if (!fs.existsSync(backupPath)) {
      return; // No backup to restore from
    }
    
    try {
      const current = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const backup = JSON.parse(fs.readFileSync(backupPath, 'utf8'));
      
      let fixed = false;
      
      if (current.ageVariants && backup.age_expressions) {
        // Map old age keys to new ones
        const ageMap = {
          'early_childhood': '2-5',
          'youth': '6-12',
          'young_adult': '13-17',
          'midlife': '36-60',
          'wisdom_years': '61-102'
        };
        
        Object.keys(ageMap).forEach(oldKey => {
          const newKey = ageMap[oldKey];
          if (current.ageVariants[newKey] && backup.age_expressions[oldKey]) {
            const backupComplexity = backup.age_expressions[oldKey].complexity_level;
            const currentComplexity = current.ageVariants[newKey].complexity_level;
            
            // Restore if it was changed to invalid value
            if (backupComplexity && 
                (currentComplexity === 'simple' || currentComplexity === 'moderate' || currentComplexity === 'complex')) {
              current.ageVariants[newKey].complexity_level = backupComplexity;
              console.log(`  ✅ ${file} ${newKey}: restored ${backupComplexity}`);
              fixed = true;
            }
          }
        });
      }
      
      if (fixed) {
        fs.writeFileSync(filePath, JSON.stringify(current, null, 2));
        restoredCount++;
      }
    } catch (e) {
      console.log(`  ❌ Error restoring ${file}: ${e.message}`);
    }
  });
  
  console.log(`\n✅ Restored complexity_level in ${restoredCount} files\n`);
}

if (require.main === module) {
  restoreComplexityLevels();
}

module.exports = { restoreComplexityLevels };


