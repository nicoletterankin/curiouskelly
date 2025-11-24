#!/usr/bin/env node

/**
 * Fix Migrated Lessons Script
 * Fixes common issues in migrated lessons:
 * - ID format (underscores to hyphens)
 * - Missing 18-35 age variant
 * - Language content completeness
 * - Missing cta/summary fields
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');

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

function fixLesson(lessonPath) {
  console.log(`\n🔧 Fixing: ${path.basename(lessonPath)}`);
  
  const data = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  let fixed = false;
  
  // 1. Fix ID format
  if (data.id && data.id.includes('_')) {
    const oldId = data.id;
    data.id = data.id.replace(/_/g, '-').replace(/[^a-z0-9-]/g, '').toLowerCase();
    console.log(`  ✅ Fixed ID: ${oldId} → ${data.id}`);
    fixed = true;
  }
  
  // 2. Fix description length
  if (data.description && data.description.length < 20) {
    const oldDesc = data.description;
    data.description = (data.learning_essence_translations?.en || data.learning_essence || data.description || 'Learning lesson').substring(0, 200);
    if (data.description.length < 20) {
      data.description = data.description + ' ' + 'Discover fascinating concepts and learn together!';
    }
    console.log(`  ✅ Fixed description length`);
    fixed = true;
  }
  
  // 3. Ensure all age variants have complete language structure
  if (data.ageVariants) {
    Object.keys(data.ageVariants).forEach(ageKey => {
      const variant = data.ageVariants[ageKey];
      
      // Ensure language structure exists for all languages
      if (!variant.language) {
        variant.language = {};
      }
      
      ['en', 'es', 'fr'].forEach(lang => {
        if (!variant.language[lang]) {
          variant.language[lang] = {};
        }
        
        const langData = variant.language[lang];
        const title = langData.title || variant.title || data.title || 'Learning';
        
        // Fix welcome
        if (!langData.welcome || langData.welcome.includes('Let\'s learn together') && lang !== 'en') {
          if (lang === 'en') {
            langData.welcome = title + '! Let\'s learn together!';
          } else if (lang === 'es') {
            langData.welcome = title + '! ¡Aprendamos juntos!';
          } else if (lang === 'fr') {
            langData.welcome = title + '! Apprenons ensemble!';
          }
          fixed = true;
        }
        
        // Fix mainContent
        if (!langData.mainContent || langData.mainContent.includes('This is amazing') && lang !== 'en') {
          const learningEssence = data.learning_essence_translations?.[lang] || data.learning_essence_translations?.en || data.learning_essence || '';
          if (learningEssence) {
            langData.mainContent = learningEssence;
          } else {
            langData.mainContent = title + '. ' + (lang === 'en' ? 'This is fascinating!' : lang === 'es' ? '¡Esto es fascinante!' : 'C\'est fascinant!');
          }
          fixed = true;
        }
        
        // Ensure keyPoints exists
        if (!langData.keyPoints || !Array.isArray(langData.keyPoints)) {
          langData.keyPoints = variant.examples || [];
          fixed = true;
        }
        
        // Ensure interactionPrompts exists and has at least one item
        if (!langData.interactionPrompts || !Array.isArray(langData.interactionPrompts) || langData.interactionPrompts.length === 0) {
          // Add default interaction prompts based on language
          if (lang === 'en') {
            langData.interactionPrompts = ['What do you think about this?', 'Can you share your thoughts?'];
          } else if (lang === 'es') {
            langData.interactionPrompts = ['¿Qué piensas sobre esto?', '¿Puedes compartir tus pensamientos?'];
          } else if (lang === 'fr') {
            langData.interactionPrompts = ['Qu\'en penses-tu?', 'Peux-tu partager tes pensées?'];
          }
          fixed = true;
        }
        
        // Ensure title exists in language object (required by schema)
        if (!langData.title) {
          langData.title = title;
          fixed = true;
        }
        
        // Ensure wisdomMoment exists
        if (!langData.wisdomMoment) {
          const wisdomKeys = Object.keys(variant.abstract_concepts_translations || {});
          if (wisdomKeys.length > 0) {
            langData.wisdomMoment = variant.abstract_concepts_translations[wisdomKeys[0]]?.[lang] || variant.abstract_concepts_translations[wisdomKeys[0]]?.en || '';
          }
          if (!langData.wisdomMoment) {
            langData.wisdomMoment = lang === 'en' ? 'Wonderful!' : lang === 'es' ? '¡Maravilloso!' : 'Merveilleux!';
          }
          fixed = true;
        }
        
        // Ensure cta exists
        if (!langData.cta) {
          langData.cta = lang === 'en' ? 'Keep exploring!' : lang === 'es' ? '¡Sigue explorando!' : 'Continuez à explorer!';
          fixed = true;
        }
        
        // Ensure summary exists
        if (!langData.summary) {
          langData.summary = lang === 'en' ? 'Great learning today!' : lang === 'es' ? '¡Gran aprendizaje hoy!' : 'Excellent apprentissage aujourd\'hui!';
          fixed = true;
        }
      });
    });
    
    // 4. Add missing 18-35 age variant
    if (!data.ageVariants['18-35'] && Object.keys(data.ageVariants).length > 0) {
      const youngAdult = data.ageVariants['13-17'];
      const midlife = data.ageVariants['36-60'];
      const source = midlife || youngAdult;
      
      if (source) {
        // Deep clone the source
        const newVariant = JSON.parse(JSON.stringify(source));
        
        // Update age-specific fields
        newVariant.kellyAge = KELLY_AGES['18-35'];
        newVariant.kellyPersona = KELLY_PERSONAS['18-35'];
        newVariant.video = `kelly_${data.id}_18-35.mp4`;
        newVariant.voiceProfile = {
          ...DEFAULT_VOICE_PROFILE,
          ...VOICE_PROFILE_ADJUSTMENTS['18-35']
        };
        newVariant.pacing = DEFAULT_PACING['18-35'];
        
        // Update language titles if needed
        ['en', 'es', 'fr'].forEach(lang => {
          if (newVariant.language && newVariant.language[lang]) {
            if (!newVariant.language[lang].title || newVariant.language[lang].title === source.language?.[lang]?.title) {
              newVariant.language[lang].title = data.title || newVariant.language[lang].title;
            }
          }
        });
        
        data.ageVariants['18-35'] = newVariant;
        console.log(`  ✅ Added missing 18-35 age variant`);
        fixed = true;
      }
    }
  }
  
  if (fixed) {
    // Write back
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
  
  console.log(`\n🔧 Fixing ${files.length} migrated lessons...\n`);
  
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

module.exports = { fixLesson };

