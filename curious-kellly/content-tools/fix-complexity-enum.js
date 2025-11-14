#!/usr/bin/env node

/**
 * Fix Complexity Level Enum Script
 * Removes invalid complexity_level values (since it's optional)
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');

const VALID_COMPLEXITY = [
  'concrete_observable_actions',
  'systems_thinking_with_concrete_examples',
  'mechanistic_understanding_with_applications',
  'systems_optimization_and_prevention',
  'philosophical_integration_with_scientific_understanding'
];

function fixComplexityLevels() {
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.') && !f.includes('backup') && !f.includes('index')
  );
  
  console.log(`\n🔧 Fixing invalid complexity_level values...\n`);
  
  let fixedCount = 0;
  
  files.forEach(file => {
    const filePath = path.join(LESSONS_DIR, file);
    
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      let fixed = false;
      
      if (data.ageVariants) {
        Object.keys(data.ageVariants).forEach(ageKey => {
          const variant = data.ageVariants[ageKey];
          if (variant.complexity_level && !VALID_COMPLEXITY.includes(variant.complexity_level)) {
            console.log(`  ✅ ${file} ${ageKey}: removed invalid "${variant.complexity_level}"`);
            delete variant.complexity_level;
            fixed = true;
          }
        });
      }
      
      if (fixed) {
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        fixedCount++;
      }
    } catch (e) {
      console.log(`  ❌ Error fixing ${file}: ${e.message}`);
    }
  });
  
  console.log(`\n✅ Fixed ${fixedCount} files\n`);
}

if (require.main === module) {
  fixComplexityLevels();
}

module.exports = { fixComplexityLevels };


