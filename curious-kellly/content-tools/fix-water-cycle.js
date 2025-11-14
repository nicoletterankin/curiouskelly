#!/usr/bin/env node

/**
 * Fix Water Cycle Lesson Script
 * Adds missing required fields to all age variants
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const WATER_CYCLE_FILE = path.join(LESSONS_DIR, 'water-cycle.json');

const KELLY_AGES = {
  '2-5': 3,
  '6-12': 9,
  '13-17': 15,
  '18-35': 27,
  '36-60': 48,
  '61-102': 82
};

const KELLY_PERSONAS = {
  '2-5': 'playful-toddler',
  '6-12': 'curious-kid',
  '13-17': 'enthusiastic-teen',
  '18-35': 'knowledgeable-adult',
  '36-60': 'wise-mentor',
  '61-102': 'reflective-elder'
};

const DEFAULT_VOICE_PROFILE = {
  provider: 'elevenlabs',
  voiceId: 'wAdymQH5YucAkXwmrdL0',
  speechRate: 1.0,
  pitch: 0,
  energy: 'warm',
  language: 'en-US'
};

const VOICE_PROFILE_ADJUSTMENTS = {
  '2-5': { speechRate: 0.85, pitch: 2, energy: 'bright' },
  '6-12': { speechRate: 1.0, pitch: 0, energy: 'bright' },
  '13-17': { speechRate: 1.05, pitch: -1, energy: 'dynamic' },
  '18-35': { speechRate: 1.1, pitch: -2, energy: 'warm' },
  '36-60': { speechRate: 1.0, pitch: -3, energy: 'warm' },
  '61-102': { speechRate: 0.95, pitch: -4, energy: 'gentle' }
};

const DEFAULT_PACING = {
  '2-5': { speechRate: 'slow', pauseFrequency: 'frequent', interactionLevel: 'high' },
  '6-12': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '13-17': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '18-35': { speechRate: 'moderate', pauseFrequency: 'minimal', interactionLevel: 'low' },
  '36-60': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '61-102': { speechRate: 'slow', pauseFrequency: 'frequent', interactionLevel: 'low' }
};

function fixWaterCycle() {
  console.log(`\n🔧 Fixing water-cycle.json...\n`);
  
  const data = JSON.parse(fs.readFileSync(WATER_CYCLE_FILE, 'utf8'));
  let fixed = false;
  
  const ageTitles = {
    '2-5': 'Water Goes Round and Round!',
    '6-12': 'The Amazing Water Cycle',
    '13-17': 'Understanding the Water Cycle',
    '18-35': 'The Water Cycle: Science and Systems',
    '36-60': 'The Water Cycle: Patterns and Wisdom',
    '61-102': 'The Water Cycle: Eternal Renewal'
  };
  
  const ageDescriptions = {
    '2-5': 'Let\'s learn about how water moves and changes!',
    '6-12': 'Discover how water travels through the sky and back!',
    '13-17': 'Explore the science behind Earth\'s water cycle',
    '18-35': 'Understand the water cycle and its global importance',
    '36-60': 'Reflect on the water cycle\'s patterns and lessons',
    '61-102': 'Contemplate water\'s eternal journey and renewal'
  };
  
  Object.keys(data.ageVariants).forEach(ageKey => {
    const variant = data.ageVariants[ageKey];
    let ageFixed = false;
    
    // Add missing title
    if (!variant.title) {
      variant.title = ageTitles[ageKey] || data.title;
      ageFixed = true;
    }
    
    // Add missing description
    if (!variant.description) {
      variant.description = ageDescriptions[ageKey] || data.description;
      ageFixed = true;
    }
    
    // Add missing video
    if (!variant.video) {
      variant.video = `kelly_water-cycle_${ageKey}.mp4`;
      ageFixed = true;
    }
    
    // Add missing script (use welcome from language.en)
    if (!variant.script && variant.language && variant.language.en && variant.language.en.welcome) {
      variant.script = variant.language.en.welcome;
      ageFixed = true;
    }
    
    // Fix kellyPersona
    if (variant.kellyPersona && variant.kellyPersona !== KELLY_PERSONAS[ageKey]) {
      variant.kellyPersona = KELLY_PERSONAS[ageKey];
      ageFixed = true;
    }
    
    // Add missing voiceProfile
    if (!variant.voiceProfile) {
      variant.voiceProfile = {
        ...DEFAULT_VOICE_PROFILE,
        ...VOICE_PROFILE_ADJUSTMENTS[ageKey]
      };
      ageFixed = true;
    }
    
    // Add missing objectives
    if (!variant.objectives || !Array.isArray(variant.objectives)) {
      variant.objectives = [
        'Understand the water cycle',
        'Learn about evaporation and condensation',
        'Appreciate water\'s journey'
      ];
      ageFixed = true;
    }
    
    // Add missing vocabulary
    if (!variant.vocabulary) {
      variant.vocabulary = {
        keyTerms: ['water', 'cycle', 'evaporation', 'condensation', 'precipitation'],
        complexity: ageKey === '2-5' ? 'simple' : 'moderate',
        explanations: {}
      };
      ageFixed = true;
    }
    
    // Fix pacing structure
    if (!variant.pacing || typeof variant.pacing !== 'object' || !variant.pacing.speechRate) {
      variant.pacing = DEFAULT_PACING[ageKey];
      ageFixed = true;
    }
    
    // Add missing teachingMoments
    if (!variant.teachingMoments || !Array.isArray(variant.teachingMoments) || variant.teachingMoments.length === 0) {
      variant.teachingMoments = [
        {
          id: `tm1-${ageKey}`,
          timestamp: 15,
          type: 'explanation',
          content: 'Water changes form through the cycle'
        }
      ];
      ageFixed = true;
    }
    
    // Add missing expressionCues
    if (!variant.expressionCues || !Array.isArray(variant.expressionCues) || variant.expressionCues.length === 0) {
      variant.expressionCues = [
        {
          id: `ec1-${ageKey}`,
          momentRef: `tm1-${ageKey}`,
          type: 'micro-smile',
          offset: 0,
          duration: 2,
          intensity: 'medium',
          gazeTarget: 'camera'
        }
      ];
      ageFixed = true;
    }
    
    // Ensure language structure has title
    if (variant.language) {
      ['en', 'es', 'fr'].forEach(lang => {
        if (variant.language[lang] && !variant.language[lang].title) {
          variant.language[lang].title = variant.title || data.title;
          ageFixed = true;
        }
      });
    }
    
    if (ageFixed) {
      console.log(`  ✅ Fixed age variant: ${ageKey}`);
      fixed = true;
    }
  });
  
  if (fixed) {
    fs.writeFileSync(WATER_CYCLE_FILE, JSON.stringify(data, null, 2));
    console.log(`\n✅ Water cycle lesson fixed and saved\n`);
  } else {
    console.log(`\n⏭️  No fixes needed\n`);
  }
}

if (require.main === module) {
  fixWaterCycle();
}

module.exports = { fixWaterCycle };


