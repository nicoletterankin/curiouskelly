#!/usr/bin/env node

/**
 * Fix Water Cycle Teaching Moments Script
 * Converts timing to timestamp and fixes type enum values
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const WATER_CYCLE_FILE = path.join(LESSONS_DIR, 'water-cycle.json');

// Valid teaching moment types
const VALID_TYPES = ['explanation', 'question', 'demonstration', 'story', 'wisdom'];

// Map invalid types to valid ones
function normalizeType(type) {
  if (!type) return 'explanation';
  const lower = type.toLowerCase();
  if (VALID_TYPES.includes(lower)) return lower;
  // Map common variations
  if (lower.includes('visual') || lower.includes('show')) return 'demonstration';
  if (lower.includes('inquiry') || lower.includes('ask')) return 'question';
  if (lower.includes('reflection') || lower.includes('wisdom') || lower.includes('legacy')) return 'wisdom';
  if (lower.includes('application') || lower.includes('experiment')) return 'demonstration';
  return 'explanation'; // default
}

// Convert timing string (MM:SS) to timestamp (seconds)
function timingToTimestamp(timing) {
  if (!timing) return 15; // default
  if (typeof timing === 'number') return timing;
  const parts = timing.split(':');
  if (parts.length === 2) {
    return parseInt(parts[0]) * 60 + parseInt(parts[1]);
  }
  return 15; // default
}

function fixTeachingMoments() {
  console.log(`\n🔧 Fixing teaching moments in water-cycle.json...\n`);
  
  const data = JSON.parse(fs.readFileSync(WATER_CYCLE_FILE, 'utf8'));
  let fixed = false;
  
  Object.keys(data.ageVariants).forEach(ageKey => {
    const variant = data.ageVariants[ageKey];
    
    if (variant.teachingMoments && Array.isArray(variant.teachingMoments)) {
      variant.teachingMoments.forEach((moment, index) => {
        // Convert timing to timestamp
        if (moment.timing && !moment.timestamp) {
          moment.timestamp = timingToTimestamp(moment.timing);
          delete moment.timing;
          console.log(`  ✅ ${ageKey} moment ${index}: converted timing to timestamp`);
          fixed = true;
        }
        
        // Ensure timestamp exists
        if (!moment.timestamp) {
          moment.timestamp = 15 + (index * 60); // Default spacing
          console.log(`  ✅ ${ageKey} moment ${index}: added timestamp`);
          fixed = true;
        }
        
        // Fix type enum
        if (moment.type) {
          const normalized = normalizeType(moment.type);
          if (normalized !== moment.type) {
            console.log(`  ✅ ${ageKey} moment ${index}: fixed type ${moment.type} → ${normalized}`);
            moment.type = normalized;
            fixed = true;
          }
        } else {
          moment.type = 'explanation';
          fixed = true;
        }
        
        // Ensure id follows pattern
        if (!moment.id || !moment.id.includes(ageKey)) {
          moment.id = `tm${index + 1}-${ageKey}`;
          fixed = true;
        }
      });
      
      // Update expressionCues to reference correct moment IDs
      if (variant.expressionCues && Array.isArray(variant.expressionCues)) {
        variant.expressionCues.forEach((cue, index) => {
          if (cue.momentRef && !cue.momentRef.includes(ageKey)) {
            // Find matching moment
            const momentIndex = variant.teachingMoments.findIndex(m => 
              m.id === cue.momentRef || m.id === `tm${index + 1}`
            );
            if (momentIndex >= 0 && variant.teachingMoments[momentIndex]) {
              cue.momentRef = variant.teachingMoments[momentIndex].id;
              fixed = true;
            } else {
              cue.momentRef = variant.teachingMoments[0]?.id || `tm1-${ageKey}`;
              fixed = true;
            }
          }
        });
      }
    }
  });
  
  if (fixed) {
    fs.writeFileSync(WATER_CYCLE_FILE, JSON.stringify(data, null, 2));
    console.log(`\n✅ Teaching moments fixed and saved\n`);
  } else {
    console.log(`\n⏭️  No fixes needed\n`);
  }
}

if (require.main === module) {
  fixTeachingMoments();
}

module.exports = { fixTeachingMoments };


