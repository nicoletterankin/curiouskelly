#!/usr/bin/env node

/**
 * Complete Molecular Biology Example Script
 * Adds missing age variants to molecular-biology-v2-example.json
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const FILE = path.join(LESSONS_DIR, 'molecular-biology-v2-example.json');

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

function completeAgeVariants() {
  console.log(`\n🔧 Completing age variants in molecular-biology-v2-example.json...\n`);
  
  const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));
  const source = data.ageVariants['6-12'] || data.ageVariants['2-5'];
  
  const missingAges = ['13-17', '18-35', '36-60', '61-102'].filter(age => !data.ageVariants[age]);
  
  missingAges.forEach(ageKey => {
    const newVariant = JSON.parse(JSON.stringify(source));
    
    // Update age-specific fields
    newVariant.title = data.title;
    newVariant.description = data.description;
    newVariant.kellyAge = KELLY_AGES[ageKey];
    newVariant.kellyPersona = KELLY_PERSONAS[ageKey];
    newVariant.video = `kelly_molecular-biology_${ageKey}.mp4`;
    newVariant.script = data.learning_essence_translations?.en || data.description;
    newVariant.voiceProfile = {
      ...DEFAULT_VOICE_PROFILE,
      ...VOICE_PROFILE_ADJUSTMENTS[ageKey]
    };
    newVariant.pacing = DEFAULT_PACING[ageKey];
    
    // Update teaching moments and expression cues IDs
    if (newVariant.teachingMoments) {
      newVariant.teachingMoments.forEach((tm, i) => {
        tm.id = `tm${i + 1}-${ageKey}`;
      });
    }
    
    if (newVariant.expressionCues) {
      newVariant.expressionCues.forEach((ec, i) => {
        ec.id = `ec${i + 1}-${ageKey}`;
        if (newVariant.teachingMoments && newVariant.teachingMoments[i]) {
          ec.momentRef = newVariant.teachingMoments[i].id;
        }
      });
    }
    
    // Ensure language structure exists
    if (!newVariant.language) {
      newVariant.language = {};
    }
    
    ['en', 'es', 'fr'].forEach(lang => {
      if (!newVariant.language[lang]) {
        newVariant.language[lang] = {
          title: data.title,
          welcome: lang === 'en' ? 'Welcome! Let\'s learn together!' : lang === 'es' ? '¡Bienvenido! ¡Aprendamos juntos!' : 'Bienvenue! Apprenons ensemble!',
          mainContent: data.learning_essence_translations?.[lang] || data.learning_essence_translations?.en || data.description,
          keyPoints: ['Key concept', 'Important idea'],
          interactionPrompts: lang === 'en' ? ['What do you think?', 'Can you share your thoughts?'] : lang === 'es' ? ['¿Qué piensas?', '¿Puedes compartir tus pensamientos?'] : ['Qu\'en penses-tu?', 'Peux-tu partager tes pensées?'],
          wisdomMoment: lang === 'en' ? 'Wonderful!' : lang === 'es' ? '¡Maravilloso!' : 'Merveilleux!',
          cta: lang === 'en' ? 'Keep exploring!' : lang === 'es' ? '¡Sigue explorando!' : 'Continuez à explorer!',
          summary: lang === 'en' ? 'Great learning today!' : lang === 'es' ? '¡Gran aprendizaje hoy!' : 'Excellent apprentissage aujourd\'hui!'
        };
      }
    });
    
    // Ensure all required fields exist
    if (!newVariant.objectives || !Array.isArray(newVariant.objectives)) {
      newVariant.objectives = ['Understand key concepts', 'Learn important ideas'];
    }
    
    if (!newVariant.vocabulary) {
      newVariant.vocabulary = {
        keyTerms: ['concept', 'idea'],
        complexity: 'moderate',
        explanations: {}
      };
    }
    
    if (!newVariant.teachingMoments || !Array.isArray(newVariant.teachingMoments) || newVariant.teachingMoments.length === 0) {
      newVariant.teachingMoments = [
        {
          id: `tm1-${ageKey}`,
          timestamp: 15,
          type: 'explanation',
          content: 'Key concept explanation'
        }
      ];
    }
    
    if (!newVariant.expressionCues || !Array.isArray(newVariant.expressionCues) || newVariant.expressionCues.length === 0) {
      newVariant.expressionCues = [
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
    }
    
    data.ageVariants[ageKey] = newVariant;
    console.log(`  ✅ Added age variant: ${ageKey}`);
  });
  
  fs.writeFileSync(FILE, JSON.stringify(data, null, 2));
  console.log(`\n✅ File updated\n`);
}

if (require.main === module) {
  completeAgeVariants();
}

module.exports = { completeAgeVariants };

