#!/usr/bin/env node

/**
 * Migration Script: Alternative Schema → PhaseDNA v2
 * Converts lessons from alternative schema format to PhaseDNA v2 format
 */

const fs = require('fs');
const path = require('path');

// Age bucket mapping
const AGE_BUCKET_MAP = {
  'early_childhood': '2-5',
  'youth': '6-12',
  'young_adult': '13-17',
  'midlife': '36-60',
  'wisdom_years': '61-102'
};

// All required age buckets for PhaseDNA v2
const ALL_AGE_BUCKETS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];

// Kelly age mapping
const KELLY_AGES = {
  '2-5': 3,
  '6-12': 9,
  '13-17': 15,
  '18-35': 27,
  '36-60': 48,
  '61-102': 82
};

// Kelly persona mapping
const KELLY_PERSONAS = {
  '2-5': 'playful-toddler',
  '6-12': 'curious-kid',
  '13-17': 'enthusiastic-teen',
  '18-35': 'knowledgeable-adult',
  '36-60': 'wise-mentor',
  '61-102': 'reflective-elder'
};

// Default voice profile (can be customized)
const DEFAULT_VOICE_PROFILE = {
  provider: 'elevenlabs',
  voiceId: 'wAdymQH5YucAkXwmrdL0',
  speechRate: 1.0,
  pitch: 0,
  energy: 'warm',
  language: 'en-US'
};

// Age-specific voice profile adjustments
const VOICE_PROFILE_ADJUSTMENTS = {
  '2-5': { speechRate: 0.85, pitch: 2, energy: 'bright' },
  '6-12': { speechRate: 1.0, pitch: 0, energy: 'bright' },
  '13-17': { speechRate: 1.05, pitch: -1, energy: 'dynamic' },
  '18-35': { speechRate: 1.1, pitch: -2, energy: 'warm' },
  '36-60': { speechRate: 1.0, pitch: -3, energy: 'warm' },
  '61-102': { speechRate: 0.95, pitch: -4, energy: 'gentle' }
};

// Default pacing by age
const DEFAULT_PACING = {
  '2-5': { speechRate: 'slow', pauseFrequency: 'frequent', interactionLevel: 'high' },
  '6-12': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '13-17': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '18-35': { speechRate: 'moderate', pauseFrequency: 'minimal', interactionLevel: 'low' },
  '36-60': { speechRate: 'moderate', pauseFrequency: 'moderate', interactionLevel: 'moderate' },
  '61-102': { speechRate: 'slow', pauseFrequency: 'frequent', interactionLevel: 'low' }
};

function migrateLesson(sourcePath, outputPath) {
  console.log(`\n🔄 Migrating lesson: ${sourcePath}\n`);
  
  // Read source file
  const sourceData = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
  
  // Build PhaseDNA v2 structure
  // Fix ID format: convert underscores to hyphens, ensure kebab-case
  let lessonId = sourceData.lesson_id || sourceData.id || 'migrated-lesson';
  lessonId = lessonId.replace(/_/g, '-').replace(/[^a-z0-9-]/g, '').toLowerCase();
  
  const migrated = {
    id: lessonId,
    title: sourceData.universal_concept_translations?.en || 'Migrated Lesson',
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    author: 'Migration Script',
    description: (sourceData.learning_essence_translations?.en || sourceData.learning_essence || 'Migrated lesson').substring(0, 200),
    
    // Calendar integration
    calendar: sourceData.day ? {
      day: sourceData.day,
      date: sourceData.date
    } : undefined,
    
    // Universal concepts
    universal_concept: sourceData.universal_concept,
    universal_concept_translations: sourceData.universal_concept_translations,
    core_principle: sourceData.core_principle,
    core_principle_translations: sourceData.core_principle_translations,
    learning_essence: sourceData.learning_essence,
    learning_essence_translations: sourceData.learning_essence_translations,
    
    // Metadata (basic - may need enhancement)
    metadata: {
      category: 'science', // Default - should be extracted from source
      difficulty: 'beginner', // Default - should be extracted from source
      duration: { min: 5, max: 13 }, // Default - should be calculated
      tags: [],
      prerequisites: [],
      learningOutcomes: []
    },
    
    // Age variants
    ageVariants: {}
  };
  
  // Migrate age expressions to ageVariants
  if (sourceData.age_expressions) {
    Object.keys(sourceData.age_expressions).forEach(oldAgeKey => {
      const newAgeKey = AGE_BUCKET_MAP[oldAgeKey];
      if (!newAgeKey) {
        console.warn(`⚠️  Unknown age key: ${oldAgeKey}, skipping`);
        return;
      }
      
      const sourceAgeData = sourceData.age_expressions[oldAgeKey];
      const kellyAge = KELLY_AGES[newAgeKey];
      const kellyPersona = KELLY_PERSONAS[newAgeKey];
      
      // Build voice profile
      const voiceProfile = {
        ...DEFAULT_VOICE_PROFILE,
        ...VOICE_PROFILE_ADJUSTMENTS[newAgeKey]
      };
      
      // Build language structure
      const language = {};
      ['en', 'es', 'fr'].forEach(lang => {
        const langKey = lang === 'en' ? 'en' : lang;
        if (sourceAgeData.concept_name_translations?.[langKey]) {
          language[lang] = {
            title: sourceAgeData.concept_name_translations[langKey],
            welcome: sourceAgeData.concept_name_translations[langKey] + '! Let\'s learn together!', // Placeholder
            mainContent: sourceAgeData.concept_name_translations[langKey] + ' This is amazing!', // Placeholder - needs actual content
            keyPoints: sourceAgeData.examples || [],
            interactionPrompts: [], // Needs to be extracted from interactions
            wisdomMoment: sourceAgeData.abstract_concepts_translations?.biological_systems?.[langKey] || 'Wonderful!',
            core_metaphor: sourceAgeData.core_metaphor_translations?.[langKey],
            abstract_concepts: sourceAgeData.abstract_concepts_translations ? 
              Object.keys(sourceAgeData.abstract_concepts_translations).reduce((acc, key) => {
                acc[key] = sourceAgeData.abstract_concepts_translations[key][langKey] || '';
                return acc;
              }, {}) : undefined
          };
        }
      });
      
      // Build vocabulary
      const vocabulary = {
        keyTerms: sourceAgeData.vocabulary || [],
        complexity: sourceAgeData.complexity_level?.includes('concrete') ? 'simple' :
                   sourceAgeData.complexity_level?.includes('systems') ? 'moderate' : 'complex',
        explanations: {}
      };
      
      // Build abstract concepts
      const abstract_concepts = sourceAgeData.abstract_concepts || {};
      const abstract_concepts_translations = sourceAgeData.abstract_concepts_translations || {};
      
      // Build age variant
      migrated.ageVariants[newAgeKey] = {
        title: sourceAgeData.concept_name_translations?.en || sourceAgeData.concept_name,
        description: sourceAgeData.concept_name_translations?.en || sourceAgeData.concept_name,
        video: `kelly_${migrated.id}_${newAgeKey}.mp4`, // Placeholder - needs to be generated
        script: sourceAgeData.concept_name_translations?.en || sourceAgeData.concept_name + '! Let\'s explore this together!',
        kellyAge: kellyAge,
        kellyPersona: kellyPersona,
        voiceProfile: voiceProfile,
        
        // Optional pedagogical fields
        core_metaphor: sourceAgeData.core_metaphor,
        core_metaphor_translations: sourceAgeData.core_metaphor_translations,
        complexity_level: sourceAgeData.complexity_level,
        attention_span: sourceAgeData.attention_span,
        cognitive_focus: sourceAgeData.cognitive_focus,
        examples: sourceAgeData.examples || [],
        
        // Language structure
        language: language,
        
        // Objectives (placeholder - needs actual objectives)
        objectives: sourceAgeData.examples?.slice(0, 3) || ['Learn about this topic', 'Understand key concepts'],
        
        // Vocabulary
        vocabulary: vocabulary,
        
        // Abstract concepts
        abstract_concepts: abstract_concepts,
        abstract_concepts_translations: abstract_concepts_translations,
        
        // Pacing
        pacing: DEFAULT_PACING[newAgeKey],
        
        // Teaching moments (placeholder - needs actual timestamps)
        teachingMoments: [
          {
            id: `tm1-${newAgeKey}`,
            timestamp: 15,
            type: 'explanation',
            content: sourceAgeData.concept_name_translations?.en || 'Key concept explanation'
          }
        ],
        
        // Expression cues (placeholder - needs actual timing)
        expressionCues: [
          {
            id: `ec1-${newAgeKey}`,
            momentRef: `tm1-${newAgeKey}`,
            type: 'micro-smile',
            offset: 0,
            duration: 2,
            intensity: 'medium',
            gazeTarget: 'camera'
          }
        ]
      };
      
      // Add tone if available
      if (sourceData.tone_delivery_dna) {
        // Map tone based on persona (simplified mapping)
        let toneKey = 'neutral';
        if (newAgeKey === '2-5' || newAgeKey === '6-12') toneKey = 'fun';
        if (newAgeKey === '61-102') toneKey = 'grandmother';
        
        const toneData = sourceData.tone_delivery_dna[toneKey];
        if (toneData) {
          migrated.ageVariants[newAgeKey].tone = {
            voice_character: toneData.voice_character,
            emotional_temperature: toneData.emotional_temperature,
            language_patterns: toneData.language_patterns,
            metaphor_style: toneData.metaphor_style,
            question_approach: toneData.question_approach,
            validation_style: toneData.validation_style
          };
          migrated.ageVariants[newAgeKey].tone_translations = toneData.language_patterns_translations;
        }
      }
    });
  }
  
  // Migrate interactions
  migrated.interactions = [];
  if (sourceData.core_lesson_structure) {
    Object.keys(sourceData.core_lesson_structure).forEach((questionKey, index) => {
      const questionData = sourceData.core_lesson_structure[questionKey];
      
      // Determine step based on question number
      const step = index === 0 ? 'welcome' : index === 1 ? 'teaching' : 'practice';
      
      const interaction = {
        step: step,
        question: questionData.concept_focus || 'Question about this topic', // Placeholder
        concept_focus: questionData.concept_focus,
        universal_principle: questionData.universal_principle,
        cognitive_target: questionData.cognitive_target,
        choices: [
          {
            text: questionData.choice_architecture?.option_a || 'Option A',
            nextStep: index < 2 ? 'teaching' : 'wisdom',
            response: questionData.teaching_moments?.option_a_response || 'Good thinking!',
            learningValue: 'moderate'
          },
          {
            text: questionData.choice_architecture?.option_b || 'Option B',
            nextStep: index < 2 ? 'teaching' : 'wisdom',
            response: questionData.teaching_moments?.option_b_response || 'Excellent!',
            learningValue: 'high'
          }
        ],
        ageAdaptations: {}
      };
      
      // Add age-specific scenarios if available
      if (sourceData.example_selector_data && sourceData.example_selector_data[questionKey + '_examples']) {
        const examples = sourceData.example_selector_data[questionKey + '_examples'];
        Object.keys(examples).forEach(oldAgeKey => {
          const newAgeKey = AGE_BUCKET_MAP[oldAgeKey];
          if (newAgeKey && examples[oldAgeKey]) {
            interaction.ageAdaptations[newAgeKey] = {
              scenario: examples[oldAgeKey].scenario,
              question: examples[oldAgeKey].scenario, // Use scenario as question placeholder
              choices: [
                {
                  text: examples[oldAgeKey].option_a,
                  nextStep: index < 2 ? 'teaching' : 'wisdom',
                  response: questionData.teaching_moments?.option_a_response || 'Good thinking!',
                  learningValue: 'moderate'
                },
                {
                  text: examples[oldAgeKey].option_b,
                  nextStep: index < 2 ? 'teaching' : 'wisdom',
                  response: questionData.teaching_moments?.option_b_response || 'Excellent!',
                  learningValue: 'high'
                }
              ]
            };
          }
        });
      }
      
      migrated.interactions.push(interaction);
    });
  }
  
  // Ensure all 6 age buckets are present - create 18-35 if missing
  if (!migrated.ageVariants['18-35'] && Object.keys(migrated.ageVariants).length > 0) {
    // Interpolate between 13-17 and 36-60
    const youngAdult = migrated.ageVariants['13-17'];
    const midlife = migrated.ageVariants['36-60'];
    
    if (youngAdult || midlife) {
      const source = midlife || youngAdult;
      migrated.ageVariants['18-35'] = {
        ...source,
        title: source.title || migrated.title,
        description: source.description || migrated.description,
        kellyAge: KELLY_AGES['18-35'],
        kellyPersona: KELLY_PERSONAS['18-35'],
        video: `kelly_${migrated.id}_18-35.mp4`,
        voiceProfile: {
          ...DEFAULT_VOICE_PROFILE,
          ...VOICE_PROFILE_ADJUSTMENTS['18-35']
        },
        pacing: DEFAULT_PACING['18-35']
      };
    }
  }
  
  // Add optional frameworks
  if (sourceData.daily_fortune_elements) {
    migrated.daily_fortune_elements = sourceData.daily_fortune_elements;
    migrated.daily_fortune_elements_translations = sourceData.daily_fortune_elements_translations;
  }
  
  if (sourceData.language_adaptation_framework) {
    migrated.language_adaptation_framework = sourceData.language_adaptation_framework;
  }
  
  if (sourceData.quality_validation_targets) {
    migrated.quality_validation_targets = sourceData.quality_validation_targets;
  }
  
  if (sourceData.example_selector_data) {
    migrated.example_selector_data = sourceData.example_selector_data;
  }
  
  // Write output
  fs.writeFileSync(outputPath, JSON.stringify(migrated, null, 2));
  console.log(`✅ Migration complete! Output: ${outputPath}\n`);
  console.log('⚠️  NOTE: This is a structural migration. You will need to:');
  console.log('   1. Add actual welcome/mainContent/wisdomMoment text');
  console.log('   2. Generate video files');
  console.log('   3. Add proper teaching moments with timestamps');
  console.log('   4. Add expression cues aligned to teaching moments');
  console.log('   5. Review and enhance all content\n');
  
  return migrated;
}

// CLI
if (require.main === module) {
  const sourcePath = process.argv[2];
  const outputPath = process.argv[3] || sourcePath.replace('.json', '_v2.json');
  
  if (!sourcePath) {
    console.error('Usage: node migrate-to-phasedna-v2.js <source.json> [output.json]');
    process.exit(1);
  }
  
  if (!fs.existsSync(sourcePath)) {
    console.error(`Error: File not found: ${sourcePath}`);
    process.exit(1);
  }
  
  migrateLesson(sourcePath, outputPath);
}

module.exports = { migrateLesson };

