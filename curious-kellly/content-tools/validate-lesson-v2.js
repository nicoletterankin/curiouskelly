#!/usr/bin/env node

/**
 * Lesson Validator v2
 * Validates lesson JSON against PhaseDNA v2 schema and content quality rules
 */

const fs = require('fs');
const path = require('path');
const Ajv = require('ajv');

// Load schema (try v2 first, fallback to v1)
let schemaPath = path.join(__dirname, '../../lesson-player/lesson-dna-schema-v2.json');
if (!fs.existsSync(schemaPath)) {
  schemaPath = path.join(__dirname, '../backend/config/lesson-dna-schema.json');
}
const lessonSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));

// Initialize validator
const ajv = new Ajv({ 
  allErrors: true,
  validateFormats: false // Disable format validation to avoid date-time issues
});
const validate = ajv.compile(lessonSchema);

// Content quality rules
const QUALITY_RULES = {
  minWelcomeWords: 10,
  maxWelcomeWords: 50,
  minMainContentWords: 50,
  maxMainContentWords: 500,
  minKeyPoints: 2,
  maxKeyPoints: 5,
  minInteractionPrompts: 2,
  maxInteractionPrompts: 6,
  minWisdomWords: 5,
  maxWisdomWords: 100,
  requiredAgeGroups: ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'],
  requiredLanguages: ['en'],
  requiredPhases: ['welcome', 'teaching', 'practice', 'reflection', 'wisdom'],
};

// Kelly age mapping
const KELLY_AGES = {
  '2-5': 3,
  '6-12': 9,
  '13-17': 15,
  '18-35': 27,
  '36-60': 48,
  '61-102': 82,
};

// Kelly persona mapping
const KELLY_PERSONAS = {
  '2-5': 'playful-toddler',
  '6-12': 'curious-kid',
  '13-17': 'enthusiastic-teen',
  '18-35': 'knowledgeable-adult',
  '36-60': 'wise-mentor',
  '61-102': 'reflective-elder',
};

function validateLesson(lessonPath) {
  console.log(`\n🔍 Validating lesson (v2): ${lessonPath}\n`);
  
  const errors = [];
  const warnings = [];
  const info = [];
  
  // 1. Check file exists
  if (!fs.existsSync(lessonPath)) {
    errors.push(`File not found: ${lessonPath}`);
    return { valid: false, errors, warnings, info };
  }
  
  // 2. Parse JSON
  let lessonData;
  try {
    lessonData = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  } catch (e) {
    errors.push(`Invalid JSON: ${e.message}`);
    return { valid: false, errors, warnings, info };
  }
  
  // 3. Check schema version
  if (lessonData.version) {
    info.push(`Schema version: ${lessonData.version}`);
  } else {
    warnings.push('No version specified - assuming PhaseDNA v1');
  }
  
  // 4. Validate against schema
  const isValid = validate(lessonData);
  if (!isValid) {
    validate.errors.forEach(err => {
      errors.push(`Schema error: ${err.instancePath || '/'} ${err.message}`);
    });
  }
  
  // 5. Validate content quality
  
  // Check all age groups present
  QUALITY_RULES.requiredAgeGroups.forEach(ageGroup => {
    if (!lessonData.ageVariants || !lessonData.ageVariants[ageGroup]) {
      errors.push(`Missing age group: ${ageGroup}`);
      return;
    }
    
    const variant = lessonData.ageVariants[ageGroup];
    
    // Check Kelly age
    if (variant.kellyAge !== KELLY_AGES[ageGroup]) {
      errors.push(`${ageGroup}: Kelly age should be ${KELLY_AGES[ageGroup]}, got ${variant.kellyAge}`);
    }
    
    // Check Kelly persona
    if (variant.kellyPersona !== KELLY_PERSONAS[ageGroup]) {
      errors.push(`${ageGroup}: Kelly persona should be ${KELLY_PERSONAS[ageGroup]}, got ${variant.kellyPersona}`);
    }
    
    // Check languages
    QUALITY_RULES.requiredLanguages.forEach(lang => {
      if (!variant.language || !variant.language[lang]) {
        errors.push(`${ageGroup}: Missing language: ${lang}`);
        return;
      }
      
      const content = variant.language[lang];
      
      // Check welcome
      if (content.welcome) {
        const wordCount = content.welcome.split(/\s+/).length;
        if (wordCount < QUALITY_RULES.minWelcomeWords) {
          warnings.push(`${ageGroup}: Welcome too short (${wordCount} words, min ${QUALITY_RULES.minWelcomeWords})`);
        }
        if (wordCount > QUALITY_RULES.maxWelcomeWords) {
          warnings.push(`${ageGroup}: Welcome too long (${wordCount} words, max ${QUALITY_RULES.maxWelcomeWords})`);
        }
      } else {
        errors.push(`${ageGroup}: Missing welcome`);
      }
      
      // Check main content
      if (content.mainContent) {
        const wordCount = content.mainContent.split(/\s+/).length;
        if (wordCount < QUALITY_RULES.minMainContentWords) {
          warnings.push(`${ageGroup}: Main content too short (${wordCount} words, min ${QUALITY_RULES.minMainContentWords})`);
        }
        if (wordCount > QUALITY_RULES.maxMainContentWords) {
          warnings.push(`${ageGroup}: Main content too long (${wordCount} words, max ${QUALITY_RULES.maxMainContentWords})`);
        }
      } else {
        errors.push(`${ageGroup}: Missing mainContent`);
      }
      
      // Check key points
      if (content.keyPoints) {
        if (content.keyPoints.length < QUALITY_RULES.minKeyPoints) {
          warnings.push(`${ageGroup}: Too few key points (${content.keyPoints.length}, min ${QUALITY_RULES.minKeyPoints})`);
        }
        if (content.keyPoints.length > QUALITY_RULES.maxKeyPoints) {
          warnings.push(`${ageGroup}: Too many key points (${content.keyPoints.length}, max ${QUALITY_RULES.maxKeyPoints})`);
        }
      } else {
        errors.push(`${ageGroup}: Missing keyPoints`);
      }
      
      // Check interaction prompts
      if (content.interactionPrompts) {
        if (content.interactionPrompts.length < QUALITY_RULES.minInteractionPrompts) {
          warnings.push(`${ageGroup}: Too few interaction prompts (${content.interactionPrompts.length}, min ${QUALITY_RULES.minInteractionPrompts})`);
        }
        if (content.interactionPrompts.length > QUALITY_RULES.maxInteractionPrompts) {
          warnings.push(`${ageGroup}: Too many interaction prompts (${content.interactionPrompts.length}, max ${QUALITY_RULES.maxInteractionPrompts})`);
        }
      } else {
        errors.push(`${ageGroup}: Missing interactionPrompts`);
      }
      
      // Check wisdom moment
      if (content.wisdomMoment) {
        const wordCount = content.wisdomMoment.split(/\s+/).length;
        if (wordCount < QUALITY_RULES.minWisdomWords) {
          warnings.push(`${ageGroup}: Wisdom too short (${wordCount} words, min ${QUALITY_RULES.minWisdomWords})`);
        }
        if (wordCount > QUALITY_RULES.maxWisdomWords) {
          warnings.push(`${ageGroup}: Wisdom too long (${wordCount} words, max ${QUALITY_RULES.maxWisdomWords})`);
        }
      } else {
        errors.push(`${ageGroup}: Missing wisdomMoment`);
      }
    });
    
    // Check pacing
    if (!variant.pacing) {
      errors.push(`${ageGroup}: Missing pacing`);
    }
    
    // Check teaching moments
    if (!variant.teachingMoments || variant.teachingMoments.length === 0) {
      warnings.push(`${ageGroup}: No teaching moments defined`);
    }
    
    // Check expression cues
    if (!variant.expressionCues || variant.expressionCues.length === 0) {
      warnings.push(`${ageGroup}: No expression cues defined`);
    }
    
    // Check v2 optional fields (informational)
    if (variant.core_metaphor) {
      info.push(`${ageGroup}: Has core_metaphor (v2 feature)`);
    }
    if (variant.complexity_level) {
      info.push(`${ageGroup}: Has complexity_level (v2 feature)`);
    }
    if (variant.tone) {
      info.push(`${ageGroup}: Has tone patterns (v2 feature)`);
    }
  });
  
  // Check v2 top-level optional fields
  if (lessonData.universal_concept) {
    info.push('Has universal_concept (v2 feature)');
  }
  if (lessonData.calendar) {
    info.push('Has calendar integration (v2 feature)');
  }
  if (lessonData.language_adaptation_framework) {
    info.push('Has language_adaptation_framework (v2 feature)');
  }
  if (lessonData.quality_validation_targets) {
    info.push('Has quality_validation_targets (v2 feature)');
  }
  
  // 6. Summary
  const valid = errors.length === 0;
  
  if (valid && warnings.length === 0 && info.length === 0) {
    console.log('✅ Lesson is valid and high quality!\n');
  } else if (valid) {
    console.log(`✅ Lesson is valid`);
    if (info.length > 0) {
      console.log(`\n📋 PhaseDNA v2 features detected:\n`);
      info.forEach(i => console.log(`  ℹ️  ${i}`));
    }
    if (warnings.length > 0) {
      console.log(`\n⚠️  ${warnings.length} warning(s):\n`);
      warnings.forEach(w => console.log(`  ⚠️  ${w}`));
    }
    console.log();
  } else {
    console.log(`❌ Lesson validation failed with ${errors.length} error(s):\n`);
    errors.forEach(e => console.log(`  ❌ ${e}`));
    if (warnings.length > 0) {
      console.log(`\nAlso ${warnings.length} warning(s):`);
      warnings.forEach(w => console.log(`  ⚠️  ${w}`));
    }
    if (info.length > 0) {
      console.log(`\nPhaseDNA v2 features detected:`);
      info.forEach(i => console.log(`  ℹ️  ${i}`));
    }
    console.log();
  }
  
  return { valid, errors, warnings, info };
}

// CLI
if (require.main === module) {
  const lessonPath = process.argv[2];
  
  if (!lessonPath) {
    console.error('Usage: node validate-lesson-v2.js <lesson.json>');
    process.exit(1);
  }
  
  const result = validateLesson(lessonPath);
  process.exit(result.valid ? 0 : 1);
}

module.exports = { validateLesson };

