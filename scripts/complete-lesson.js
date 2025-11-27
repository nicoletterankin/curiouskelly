#!/usr/bin/env node
/**
 * One-Shot Lesson Completion System
 *
 * Takes an existing lesson DNA file and transforms it into a fully
 * validated, launch-ready lesson with proper phases and 2-choice interactions.
 *
 * Usage:
 *   node complete-lesson.js path/to/lesson-dna.json [--dry-run]
 *   node complete-lesson.js --all [--dry-run]
 *
 * The script:
 * 1. Reads existing lesson metadata, interactions, and language content
 * 2. Builds proper phases array for each age variant
 * 3. Generates Kelly's script content for each phase
 * 4. Creates 2-choice interactions with responses
 * 5. Validates against the canonical schema
 * 6. Saves the completed lesson
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// PHASE REQUIREMENTS
// ============================================================================

const REQUIRED_PHASES = ['welcome', 'teaching', 'practice', 'wisdom'];
const PHASES_REQUIRING_CHOICES = ['teaching', 'practice'];

const AGE_BUCKETS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];

// ============================================================================
// CONTENT TEMPLATES BY AGE
// ============================================================================

const AGE_VOICE = {
  '2-5': {
    greeting: "Hi friend!",
    transition: "And guess what?",
    closeWisdom: "You learned something amazing today!",
    responsePositive: "Yes! You got it!",
    responseRedirect: "That's a great guess! Here's something cool:",
    maxResponseLength: 60,
  },
  '6-12': {
    greeting: "Hey there!",
    transition: "Here's something really interesting:",
    closeWisdom: "Now you understand something most people don't!",
    responsePositive: "Exactly right!",
    responseRedirect: "Good thinking! Actually,",
    maxResponseLength: 80,
  },
  '13-17': {
    greeting: "Let's dive in.",
    transition: "Here's where it gets interesting:",
    closeWisdom: "This knowledge will serve you well.",
    responsePositive: "Correct.",
    responseRedirect: "Common misconception.",
    maxResponseLength: 100,
  },
  '18-35': {
    greeting: "Welcome.",
    transition: "The key insight here:",
    closeWisdom: "Apply this understanding in your daily life.",
    responsePositive: "Exactly.",
    responseRedirect: "A common assumption, but",
    maxResponseLength: 100,
  },
  '36-60': {
    greeting: "Good to see you.",
    transition: "Consider this:",
    closeWisdom: "This perspective enriches how we see the world.",
    responsePositive: "Precisely.",
    responseRedirect: "That's the conventional view. Actually,",
    maxResponseLength: 100,
  },
  '61-102': {
    greeting: "Come, let's learn together.",
    transition: "Here's the beautiful truth:",
    closeWisdom: "What wisdom this brings to our understanding.",
    responsePositive: "Indeed, that's the heart of it.",
    responseRedirect: "A thoughtful perspective. Consider also:",
    maxResponseLength: 100,
  },
};

// ============================================================================
// LESSON COMPLETER
// ============================================================================

class LessonCompleter {
  constructor() {
    this.stats = {
      phasesCreated: 0,
      choicesCreated: 0,
      variantsCompleted: 0,
    };
  }

  /**
   * Complete a lesson by building proper phases for each age variant
   */
  completeLesson(lesson, filePath = '') {
    const completedLesson = JSON.parse(JSON.stringify(lesson));

    // Generate id from filename if missing
    if (!completedLesson.id && filePath) {
      const baseName = path.basename(filePath, '.json')
        .replace(/_dna$/, '')
        .replace(/-dna$/, '')
        .replace(/_/g, '-');
      completedLesson.id = baseName;
    }
    if (!completedLesson.id) {
      completedLesson.id = 'lesson-' + Date.now();
    }

    // Generate title from id if missing
    if (!completedLesson.title) {
      completedLesson.title = completedLesson.id
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    }

    // Create calendar if missing
    if (!completedLesson.calendar) {
      completedLesson.calendar = {
        day: 1,
        date: 'January 1, 2025',
        month: 'January'
      };
    }

    // Fix calendar date if needed
    if (completedLesson.calendar && completedLesson.calendar.day) {
      const dayNum = completedLesson.calendar.day;
      const calendarDate = this.dayToCalendarDate(dayNum);
      completedLesson.calendar.date = calendarDate.date;
      completedLesson.calendar.month = calendarDate.month;
    }

    // Add category if missing
    if (!completedLesson.category && completedLesson.metadata?.category) {
      completedLesson.category = completedLesson.metadata.category;
    }
    if (!completedLesson.category) {
      completedLesson.category = 'general';
    }

    // Process each age variant
    const ageVariants = completedLesson.ageVariants || {};

    for (const bucket of AGE_BUCKETS) {
      if (!ageVariants[bucket]) {
        // Create minimal variant if missing
        ageVariants[bucket] = this.createVariantFromBase(completedLesson, bucket);
      }

      // Build phases for this variant
      ageVariants[bucket] = this.completeVariant(
        ageVariants[bucket],
        bucket,
        completedLesson,
        lesson.interactions || []
      );

      this.stats.variantsCompleted++;
    }

    completedLesson.ageVariants = ageVariants;

    return completedLesson;
  }

  /**
   * Create a variant from the base lesson if one doesn't exist
   */
  createVariantFromBase(lesson, bucket) {
    const voice = AGE_VOICE[bucket];
    const title = lesson.title || 'Untitled Lesson';

    return {
      title: title,
      description: lesson.description || title,
      language: {
        en: {
          title: title,
          welcome: `${voice.greeting} Today we're exploring ${title.toLowerCase()}.`,
          mainContent: lesson.learning_essence || lesson.description || title,
          wisdomMoment: voice.closeWisdom,
        }
      },
      phases: []
    };
  }

  /**
   * Complete a single age variant with proper phases
   */
  completeVariant(variant, bucket, lesson, interactions) {
    const voice = AGE_VOICE[bucket];
    const lang = variant.language?.en || {};

    // Build phases array
    const phases = [];

    // 1. WELCOME PHASE
    phases.push({
      id: 'welcome',
      type: 'welcome',
      content: this.buildWelcomeContent(variant, lang, voice, lesson),
      duration: 30,
    });

    // 2. TEACHING PHASE with 2 choices
    const teachingInteraction = interactions.find(i => i.step === 'teaching') || {};
    phases.push({
      id: 'teaching',
      type: 'teaching',
      content: this.buildTeachingContent(variant, lang, voice, lesson),
      duration: 60,
      choices: this.buildChoices(teachingInteraction, bucket, lesson, 'teaching'),
    });

    // 3. PRACTICE PHASE with 2 choices
    const practiceInteraction = interactions.find(i => i.step === 'practice') || {};
    phases.push({
      id: 'practice',
      type: 'practice',
      content: this.buildPracticeContent(variant, lang, voice, lesson),
      duration: 60,
      choices: this.buildChoices(practiceInteraction, bucket, lesson, 'practice'),
    });

    // 4. WISDOM PHASE
    phases.push({
      id: 'wisdom',
      type: 'wisdom',
      content: this.buildWisdomContent(variant, lang, voice, lesson),
      duration: 30,
    });

    variant.phases = phases;
    this.stats.phasesCreated += 4;

    // Ensure language content is complete
    if (!variant.language) {
      variant.language = { en: {} };
    }
    if (!variant.language.en) {
      variant.language.en = {};
    }

    // Fill in required language fields
    const enLang = variant.language.en;
    if (!enLang.title || enLang.title.length < 5) {
      enLang.title = variant.title || lesson.title;
    }
    if (!enLang.welcome || enLang.welcome.length < 20) {
      enLang.welcome = phases[0].content;
    }
    if (!enLang.mainContent || enLang.mainContent.length < 100) {
      enLang.mainContent = phases[1].content + ' ' + phases[2].content;
    }
    if (!enLang.wisdomMoment || enLang.wisdomMoment.length < 20) {
      enLang.wisdomMoment = phases[3].content;
    }

    return variant;
  }

  /**
   * Build welcome phase content
   */
  buildWelcomeContent(variant, lang, voice, lesson) {
    // Use existing welcome if it's substantial
    if (lang.welcome && lang.welcome.length > 30 && !lang.welcome.includes('Let\'s learn together!')) {
      return lang.welcome;
    }

    const title = variant.title || lesson.title || 'today\'s topic';
    return `${voice.greeting} Today we're discovering ${title.toLowerCase()}. Are you ready to learn something amazing?`;
  }

  /**
   * Build teaching phase content
   */
  buildTeachingContent(variant, lang, voice, lesson) {
    // Use mainContent if substantial
    if (lang.mainContent && lang.mainContent.length > 50) {
      // Extract first substantive portion
      const sentences = lang.mainContent.split(/[.!?]+/).filter(s => s.trim().length > 10);
      if (sentences.length > 0) {
        return sentences.slice(0, 2).join('. ').trim() + '.';
      }
    }

    // Build from abstract concepts if available
    const concepts = variant.abstract_concepts || lesson.abstract_concepts || {};
    const conceptValues = Object.values(concepts);
    if (conceptValues.length > 0) {
      const concept = typeof conceptValues[0] === 'string'
        ? conceptValues[0].replace(/_/g, ' ')
        : conceptValues[0]?.en || 'this fascinating concept';
      return `${voice.transition} ${concept}.`;
    }

    // Fallback
    const title = variant.title || lesson.title || 'this topic';
    return `${voice.transition} Let me explain how ${title.toLowerCase()} works.`;
  }

  /**
   * Build practice phase content
   */
  buildPracticeContent(variant, lang, voice, lesson) {
    // Use key points if available
    const keyPoints = lang.keyPoints || variant.examples || [];
    if (keyPoints.length > 0) {
      const point = typeof keyPoints[0] === 'string'
        ? keyPoints[0].replace(/_/g, ' ')
        : keyPoints[0];
      return `Now let's think about this: ${point}. How does this connect to what we just learned?`;
    }

    const title = variant.title || lesson.title || 'this';
    return `Let's put this into practice. Think about how ${title.toLowerCase()} shows up in your daily life.`;
  }

  /**
   * Build wisdom phase content
   */
  buildWisdomContent(variant, lang, voice, lesson) {
    // Use existing wisdom if substantial
    if (lang.wisdomMoment && lang.wisdomMoment.length > 30 && lang.wisdomMoment !== 'Wonderful!') {
      return lang.wisdomMoment;
    }

    // Use core principle if available
    const corePrinciple = lesson.core_principle_translations?.en || lesson.core_principle;
    if (corePrinciple && typeof corePrinciple === 'string') {
      return `${voice.closeWisdom} Remember: ${corePrinciple.replace(/_/g, ' ')}.`;
    }

    const title = variant.title || lesson.title || 'this';
    return `${voice.closeWisdom} You now understand ${title.toLowerCase()} in a deeper way. Carry this wisdom with you.`;
  }

  /**
   * Build 2-choice interaction from existing data or generate new
   */
  buildChoices(interaction, bucket, lesson, phaseType) {
    const voice = AGE_VOICE[bucket];
    const ageAdaptations = interaction.ageAdaptations || {};
    const bucketData = ageAdaptations[bucket] || {};

    // Try to use existing choices from age adaptations
    if (bucketData.choices && Array.isArray(bucketData.choices) && bucketData.choices.length >= 2) {
      const choices = bucketData.choices.slice(0, 2).map(c => ({
        text: this.humanizeText(c.text, voice.maxResponseLength),
        response: this.buildResponse(c, voice),
      }));
      this.stats.choicesCreated += 2;
      return choices;
    }

    // Try to use base interaction choices
    if (interaction.choices && Array.isArray(interaction.choices) && interaction.choices.length >= 2) {
      const choices = interaction.choices.slice(0, 2).map(c => ({
        text: this.humanizeText(c.text, voice.maxResponseLength),
        response: this.buildResponse(c, voice),
      }));
      this.stats.choicesCreated += 2;
      return choices;
    }

    // Generate fallback choices based on phase type
    const choices = this.generateFallbackChoices(phaseType, bucket, lesson, voice);
    this.stats.choicesCreated += 2;
    return choices;
  }

  /**
   * Convert underscore_text to human readable
   */
  humanizeText(text, maxLength) {
    if (!text) return 'I think I understand';

    let humanized = text.replace(/_/g, ' ');

    // Capitalize first letter
    humanized = humanized.charAt(0).toUpperCase() + humanized.slice(1);

    // Truncate if needed
    if (humanized.length > maxLength) {
      humanized = humanized.slice(0, maxLength - 3) + '...';
    }

    return humanized;
  }

  /**
   * Build response from choice data
   */
  buildResponse(choice, voice) {
    const isHighValue = choice.learningValue === 'high';
    const prefix = isHighValue ? voice.responsePositive : voice.responseRedirect;

    let response = choice.response || '';

    // Check if response is an instruction placeholder (starts with explain_, celebrate_, etc.)
    if (response.match(/^(explain|celebrate|describe|demonstrate|show|reveal)_/i)) {
      // Generate natural response based on learning value
      response = isHighValue
        ? 'That shows you understand the deeper principle at work here.'
        : 'there\'s a deeper connection to explore here.';
    } else {
      response = this.humanizeText(response, voice.maxResponseLength);
    }

    if (response.length < 10) {
      response = isHighValue
        ? 'That shows real understanding.'
        : 'Let me help you see it differently.';
    }

    return `${prefix} ${response}`;
  }

  /**
   * Generate fallback choices when none exist
   */
  generateFallbackChoices(phaseType, bucket, lesson, voice) {
    const title = lesson.title || 'this topic';

    if (phaseType === 'teaching') {
      return [
        {
          text: `I want to learn more about ${title.toLowerCase()}`,
          response: `${voice.responsePositive} Your curiosity will take you far. Let's explore this together.`,
        },
        {
          text: 'Tell me something I don\'t know',
          response: `${voice.responseRedirect} Here's something that might surprise you about ${title.toLowerCase()}.`,
        },
      ];
    } else {
      return [
        {
          text: 'I think I understand how this works',
          response: `${voice.responsePositive} You're grasping the key concepts well.`,
        },
        {
          text: 'I\'m not sure I fully get it',
          response: `${voice.responseRedirect} That's okay. Learning takes time. Let's think about it another way.`,
        },
      ];
    }
  }

  /**
   * Convert day number to calendar date
   */
  dayToCalendarDate(dayNumber) {
    const months = [
      { name: 'January', days: 31 },
      { name: 'February', days: 28 },
      { name: 'March', days: 31 },
      { name: 'April', days: 30 },
      { name: 'May', days: 31 },
      { name: 'June', days: 30 },
      { name: 'July', days: 31 },
      { name: 'August', days: 31 },
      { name: 'September', days: 30 },
      { name: 'October', days: 31 },
      { name: 'November', days: 30 },
      { name: 'December', days: 31 },
    ];

    let remaining = dayNumber;
    for (const month of months) {
      if (remaining <= month.days) {
        return {
          date: `${month.name} ${remaining}, 2025`,
          month: month.name,
        };
      }
      remaining -= month.days;
    }

    return { date: 'December 31, 2025', month: 'December' };
  }
}

// ============================================================================
// VALIDATOR (inline, matches schema)
// ============================================================================

class LessonValidator {
  validate(lesson, filePath) {
    const errors = [];
    const fileName = path.basename(filePath);

    // Required top-level fields
    if (!lesson.id) errors.push(`${fileName}: Missing id`);
    if (!lesson.title) errors.push(`${fileName}: Missing title`);
    if (!lesson.category) errors.push(`${fileName}: Missing category`);

    // Calendar
    if (!lesson.calendar) {
      errors.push(`${fileName}: Missing calendar`);
    } else {
      if (!lesson.calendar.day) errors.push(`${fileName}: Missing calendar.day`);
      if (!lesson.calendar.date) errors.push(`${fileName}: Missing calendar.date`);
      if (lesson.calendar.date && !/^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$/.test(lesson.calendar.date)) {
        errors.push(`${fileName}: Invalid calendar.date format: ${lesson.calendar.date}`);
      }
      if (!lesson.calendar.month) errors.push(`${fileName}: Missing calendar.month`);
    }

    // Age variants
    if (!lesson.ageVariants) {
      errors.push(`${fileName}: Missing ageVariants`);
    } else {
      if (!lesson.ageVariants['18-35']) {
        errors.push(`${fileName}: Missing required age variant 18-35`);
      }

      for (const [bucket, variant] of Object.entries(lesson.ageVariants)) {
        const ctx = `${fileName}[${bucket}]`;

        if (!variant.title) errors.push(`${ctx}: Missing title`);
        if (!variant.language?.en) errors.push(`${ctx}: Missing language.en`);

        // Check phases
        if (!variant.phases || !Array.isArray(variant.phases)) {
          errors.push(`${ctx}: Missing phases array`);
        } else {
          const phaseIds = variant.phases.map(p => p.id);
          for (const req of ['welcome', 'teaching', 'practice', 'wisdom']) {
            if (!phaseIds.includes(req)) {
              errors.push(`${ctx}: Missing required phase '${req}'`);
            }
          }

          // Check choices on teaching/practice
          for (const phase of variant.phases) {
            if (phase.id === 'teaching' || phase.id === 'practice') {
              if (!phase.choices || phase.choices.length < 2) {
                errors.push(`${ctx}.phases[${phase.id}]: Must have exactly 2 choices`);
              } else {
                for (let i = 0; i < phase.choices.length; i++) {
                  const choice = phase.choices[i];
                  if (!choice.text || choice.text.length < 5) {
                    errors.push(`${ctx}.phases[${phase.id}].choices[${i}]: Choice text too short`);
                  }
                  if (!choice.response || choice.response.length < 10) {
                    errors.push(`${ctx}.phases[${phase.id}].choices[${i}]: Response too short`);
                  }
                }
              }
            }

            if (!phase.content || phase.content.length < 10) {
              errors.push(`${ctx}.phases[${phase.id}]: Content too short`);
            }
          }
        }

        // Check language content
        if (variant.language?.en) {
          const en = variant.language.en;
          if (!en.title || en.title.length < 5) errors.push(`${ctx}.language.en: Missing or short title`);
          if (!en.welcome || en.welcome.length < 20) errors.push(`${ctx}.language.en: Missing or short welcome`);
          if (!en.mainContent || en.mainContent.length < 100) errors.push(`${ctx}.language.en: Missing or short mainContent`);
          if (!en.wisdomMoment || en.wisdomMoment.length < 20) errors.push(`${ctx}.language.en: Missing or short wisdomMoment`);
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      file: fileName,
    };
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const processAll = args.includes('--all');
  const specificFile = args.find(a => a.endsWith('.json'));

  console.log('='.repeat(60));
  console.log('ONE-SHOT LESSON COMPLETION SYSTEM');
  console.log('='.repeat(60));
  console.log(`Mode: ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  console.log('');

  let filesToProcess = [];

  if (specificFile) {
    filesToProcess = [path.resolve(specificFile)];
  } else if (processAll) {
    // Process both main lessons dir and archive
    const dirs = [
      path.join(__dirname, '..', 'lessons'),
      path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
    ];

    for (const dir of dirs) {
      if (fs.existsSync(dir)) {
        const files = fs.readdirSync(dir)
          .filter(f => f.endsWith('-dna.json') || f.endsWith('_dna.json'))
          .map(f => path.join(dir, f));
        filesToProcess.push(...files);
      }
    }

    // Remove duplicates based on filename
    const seen = new Set();
    filesToProcess = filesToProcess.filter(f => {
      const name = path.basename(f);
      if (seen.has(name)) return false;
      seen.add(name);
      return true;
    });
  } else {
    console.log('Usage:');
    console.log('  node complete-lesson.js path/to/lesson-dna.json [--dry-run]');
    console.log('  node complete-lesson.js --all [--dry-run]');
    process.exit(0);
  }

  const completer = new LessonCompleter();
  const validator = new LessonValidator();

  let successCount = 0;
  let failCount = 0;

  for (const file of filesToProcess) {
    const fileName = path.basename(file);
    console.log(`\nProcessing: ${fileName}`);

    try {
      const content = fs.readFileSync(file, 'utf8');
      const lesson = JSON.parse(content);

      // Complete the lesson
      const completed = completer.completeLesson(lesson, file);

      // Validate
      const result = validator.validate(completed, file);

      if (result.valid) {
        console.log(`  ✅ Valid`);
        if (!dryRun) {
          fs.writeFileSync(file, JSON.stringify(completed, null, 2) + '\n', 'utf8');
          console.log(`  💾 Saved`);
        }
        successCount++;
      } else {
        console.log(`  ❌ ${result.errors.length} validation errors:`);
        result.errors.slice(0, 5).forEach(e => console.log(`     - ${e}`));
        if (result.errors.length > 5) {
          console.log(`     ... and ${result.errors.length - 5} more`);
        }
        failCount++;
      }
    } catch (error) {
      console.log(`  ❌ Error: ${error.message}`);
      failCount++;
    }
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('COMPLETION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Files processed: ${filesToProcess.length}`);
  console.log(`Successful: ${successCount}`);
  console.log(`Failed: ${failCount}`);
  console.log(`Phases created: ${completer.stats.phasesCreated}`);
  console.log(`Choices created: ${completer.stats.choicesCreated}`);
  console.log(`Variants completed: ${completer.stats.variantsCompleted}`);

  if (dryRun) {
    console.log('\nRun without --dry-run to apply changes.');
  }

  process.exit(failCount > 0 ? 1 : 0);
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(2);
});
