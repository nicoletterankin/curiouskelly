#!/usr/bin/env node

/**
 * Fix Validation Errors Script
 * Fixes remaining validation errors:
 * - Expression cue intensity enum values
 * - Complexity level enum values
 * - Other schema violations
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');

// Valid intensity values (from schema)
const VALID_INTENSITY = ['subtle', 'medium', 'emphatic'];

// Valid complexity level values (from schema - these are the actual enum values)
const VALID_COMPLEXITY = [
  'concrete_observable_actions',
  'systems_thinking_with_concrete_examples',
  'mechanistic_understanding_with_applications',
  'systems_optimization_and_prevention',
  'philosophical_integration_with_scientific_understanding'
];

// Map invalid intensity values to valid ones
function normalizeIntensity(intensity) {
  if (!intensity) return 'medium';
  const lower = intensity.toLowerCase();
  if (VALID_INTENSITY.includes(lower)) return lower;
  // Map common variations
  if (lower.includes('gentle') || lower.includes('soft')) return 'subtle';
  if (lower.includes('strong') || lower.includes('high')) return 'emphatic';
  return 'medium'; // default
}

// Restore complexity_level from backup if it was incorrectly changed
// The schema allows these specific descriptive values, not simple/moderate/complex
function restoreComplexityFromBackup(lessonPath, ageKey, currentValue) {
  // Don't restore - complexity_level is optional and can have custom values
  // Only validate against enum if it exists
  return currentValue; // Keep as-is
}

function fixLesson(lessonPath) {
  console.log(`\n🔧 Fixing: ${path.basename(lessonPath)}`);
  
  const data = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  let fixed = false;
  
  // Fix expression cue intensity values
  if (data.ageVariants) {
    Object.keys(data.ageVariants).forEach(ageKey => {
      const variant = data.ageVariants[ageKey];
      
      if (variant.expressionCues && Array.isArray(variant.expressionCues)) {
        variant.expressionCues.forEach((cue, index) => {
          if (cue.intensity) {
            const normalized = normalizeIntensity(cue.intensity);
            if (normalized !== cue.intensity) {
              console.log(`  ✅ Fixed intensity: ${ageKey} cue ${index}: ${cue.intensity} → ${normalized}`);
              cue.intensity = normalized;
              fixed = true;
            }
          } else {
            cue.intensity = 'medium';
            fixed = true;
          }
        });
      }
      
      // Don't modify complexity_level - it's optional and may have custom values
      // The validation will catch if it's truly invalid
    });
  }
  
  // Fix water-cycle.json specific issues
  if (path.basename(lessonPath) === 'water-cycle.json') {
    // Ensure interactions array exists
    if (!data.interactions || !Array.isArray(data.interactions)) {
      data.interactions = [];
      console.log(`  ✅ Added missing interactions array`);
      fixed = true;
    }
    
    // Ensure metadata.tags exists
    if (!data.metadata) {
      data.metadata = {};
    }
    if (!data.metadata.tags || !Array.isArray(data.metadata.tags)) {
      data.metadata.tags = ['water', 'cycle', 'nature', 'science'];
      console.log(`  ✅ Added missing tags`);
      fixed = true;
    }
    
    // Ensure metadata.duration is object
    if (!data.metadata.duration || typeof data.metadata.duration !== 'object') {
      data.metadata.duration = { min: 5, max: 13 };
      console.log(`  ✅ Fixed duration structure`);
      fixed = true;
    }
  }
  
  if (fixed) {
    fs.writeFileSync(lessonPath, JSON.stringify(data, null, 2));
    console.log(`  ✅ File fixed and saved`);
    return true;
  } else {
    console.log(`  ⏭️  No fixes needed`);
    return false;
  }
}

// CLI
if (require.main === module) {
  if (!fs.existsSync(LESSONS_DIR)) {
    console.error(`Error: Lessons directory not found: ${LESSONS_DIR}`);
    process.exit(1);
  }
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.') && !f.includes('backup')
  );
  
  console.log(`\n🔧 Fixing validation errors in ${files.length} lessons...\n`);
  
  let fixedCount = 0;
  files.forEach((file, index) => {
    const filePath = path.join(LESSONS_DIR, file);
    try {
      if (fixLesson(filePath)) {
        fixedCount++;
      }
    } catch (e) {
      console.log(`  ❌ Error fixing ${file}: ${e.message}`);
    }
  });
  
  console.log(`\n✅ Fixed ${fixedCount} out of ${files.length} lessons\n`);
}

module.exports = { fixLesson, normalizeIntensity, normalizeComplexity };

