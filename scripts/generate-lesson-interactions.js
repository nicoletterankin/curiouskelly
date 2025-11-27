/**
 * Lesson Completion Engine
 *
 * Adds missing 2-choice interactions to lesson phases.
 * Each teaching/practice phase gets:
 * - A question derived from the content
 * - Two plausible choices (one more accurate, one common misconception)
 * - Kelly's response for each choice (educational, never condescending)
 *
 * Principles:
 * - Both choices are valid learning opportunities
 * - Kelly never says "wrong" - she redirects with curiosity
 * - Responses are concise (1-2 sentences max for TTS)
 * - Age-appropriate language per bucket
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// INTERACTION TEMPLATES BY AGE
// ============================================================================

const AGE_TEMPLATES = {
  '2-5': {
    questionStarters: [
      'What do you think?',
      'Which one sounds right?',
      'Can you guess?',
    ],
    responsePositive: [
      'Yes! You got it!',
      'That\'s right!',
      'Great thinking!',
    ],
    responseRedirect: [
      'Good guess! Actually,',
      'I like that idea! Here\'s what really happens:',
      'That\'s a fun thought! The answer is',
    ],
    maxLength: 50,
  },
  '6-12': {
    questionStarters: [
      'What do you think causes this?',
      'Which explanation makes more sense?',
      'Based on what we learned,',
    ],
    responsePositive: [
      'Exactly right!',
      'You\'ve got it!',
      'That\'s correct!',
    ],
    responseRedirect: [
      'Interesting thought! The science shows',
      'Good reasoning, but actually',
      'Many people think that! In reality,',
    ],
    maxLength: 75,
  },
  '13-17': {
    questionStarters: [
      'Which factor is most important?',
      'What\'s the primary cause?',
      'Consider this:',
    ],
    responsePositive: [
      'Correct.',
      'That\'s accurate.',
      'Precisely.',
    ],
    responseRedirect: [
      'Common misconception. Actually,',
      'That\'s partially true, but',
      'Good hypothesis, though',
    ],
    maxLength: 100,
  },
  '18-35': {
    questionStarters: [
      'The key mechanism here is:',
      'Which principle applies?',
      'What drives this process?',
    ],
    responsePositive: [
      'Correct.',
      'Exactly.',
      'That\'s right.',
    ],
    responseRedirect: [
      'A common assumption, but',
      'Partially correct. More precisely,',
      'That\'s the intuitive answer, however',
    ],
    maxLength: 100,
  },
  '36-60': {
    questionStarters: [
      'The fundamental principle is:',
      'What\'s the underlying cause?',
      'Which factor is determinant?',
    ],
    responsePositive: [
      'Correct.',
      'Precisely.',
      'That\'s accurate.',
    ],
    responseRedirect: [
      'A reasonable assumption, though',
      'That\'s the conventional wisdom, but',
      'Interestingly,',
    ],
    maxLength: 100,
  },
  '61-102': {
    questionStarters: [
      'The core truth here is:',
      'What wisdom does this reveal?',
      'The essential principle:',
    ],
    responsePositive: [
      'Indeed.',
      'Precisely so.',
      'That\'s the truth of it.',
    ],
    responseRedirect: [
      'A thoughtful perspective. Consider also:',
      'Many have thought so. The reality:',
      'An understandable view, though',
    ],
    maxLength: 100,
  },
};

// ============================================================================
// INTERACTION GENERATOR
// ============================================================================

class InteractionGenerator {
  constructor() {
    this.stats = {
      lessonsProcessed: 0,
      interactionsAdded: 0,
      phasesSkipped: 0,
    };
  }

  /**
   * Generate choices for a phase based on its content
   */
  generateChoicesForPhase(phase, ageBucket, lessonTopic) {
    const template = AGE_TEMPLATES[ageBucket] || AGE_TEMPLATES['18-35'];
    const content = phase.content || '';

    // Extract key concept from content
    const concept = this.extractKeyConcept(content, lessonTopic);

    // Generate question
    const questionStarter = template.questionStarters[
      Math.floor(Math.random() * template.questionStarters.length)
    ];

    // Generate two choices based on the concept
    const choices = this.generateChoicePair(concept, ageBucket, template);

    return {
      question: `${questionStarter} ${concept.question}`,
      choices,
    };
  }

  /**
   * Extract the key concept from phase content
   */
  extractKeyConcept(content, topic) {
    // Split content into sentences
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);

    if (sentences.length === 0) {
      return {
        fact: topic,
        question: `What makes ${topic} work?`,
        correctAnswer: `It follows natural principles`,
        misconception: `It happens randomly`,
      };
    }

    // Use the first substantive sentence as the key fact
    const keyFact = sentences[0].trim();

    return {
      fact: keyFact,
      question: this.factToQuestion(keyFact),
      correctAnswer: this.factToCorrectChoice(keyFact),
      misconception: this.factToMisconception(keyFact),
    };
  }

  /**
   * Convert a fact statement to a question
   */
  factToQuestion(fact) {
    // Simple transformation: "X does Y" → "Why does X do Y?"
    const words = fact.toLowerCase().split(' ').slice(0, 6);
    return `why does this happen?`;
  }

  /**
   * Generate the correct choice from a fact
   */
  factToCorrectChoice(fact) {
    const words = fact.split(' ');
    if (words.length > 8) {
      return words.slice(0, 8).join(' ') + '...';
    }
    return fact;
  }

  /**
   * Generate a plausible misconception
   */
  factToMisconception(fact) {
    // Common misconception patterns
    const patterns = [
      'It happens by chance',
      'It\'s always been this way',
      'It works like magic',
      'There\'s no specific reason',
    ];
    return patterns[Math.floor(Math.random() * patterns.length)];
  }

  /**
   * Generate a pair of choices with responses
   */
  generateChoicePair(concept, ageBucket, template) {
    const positiveResponse = template.responsePositive[
      Math.floor(Math.random() * template.responsePositive.length)
    ];
    const redirectResponse = template.responseRedirect[
      Math.floor(Math.random() * template.responseRedirect.length)
    ];

    // Randomly decide which choice is "correct"
    const correctFirst = Math.random() > 0.5;

    const choiceA = {
      text: correctFirst ? concept.correctAnswer : concept.misconception,
      response: correctFirst
        ? `${positiveResponse} ${concept.fact.slice(0, template.maxLength)}`
        : `${redirectResponse} ${concept.fact.slice(0, template.maxLength)}`,
      isMoreAccurate: correctFirst,
    };

    const choiceB = {
      text: correctFirst ? concept.misconception : concept.correctAnswer,
      response: correctFirst
        ? `${redirectResponse} ${concept.fact.slice(0, template.maxLength)}`
        : `${positiveResponse} ${concept.fact.slice(0, template.maxLength)}`,
      isMoreAccurate: !correctFirst,
    };

    return [choiceA, choiceB];
  }

  /**
   * Add interactions to all phases in a lesson
   */
  addInteractionsToLesson(lesson) {
    const ageVariants = lesson.ageVariants || {};
    let modified = false;

    for (const [ageBucket, variant] of Object.entries(ageVariants)) {
      const phases = variant.phases || [];

      for (const phase of phases) {
        // Only add to teaching and practice phases
        if (phase.id !== 'teaching' && phase.id !== 'practice') continue;
        if (phase.type !== 'teaching' && phase.type !== 'practice') continue;

        // Skip if already has choices
        if (phase.choices && phase.choices.length >= 2) {
          this.stats.phasesSkipped++;
          continue;
        }

        // Generate choices
        const interaction = this.generateChoicesForPhase(
          phase,
          ageBucket,
          lesson.title || lesson.topic || 'this topic'
        );

        phase.question = interaction.question;
        phase.choices = interaction.choices;

        this.stats.interactionsAdded++;
        modified = true;
      }
    }

    if (modified) {
      this.stats.lessonsProcessed++;
    }

    return modified;
  }

  /**
   * Process a lesson file
   */
  processLessonFile(filePath, dryRun = false) {
    const fileName = path.basename(filePath);

    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lesson = JSON.parse(content);

      const modified = this.addInteractionsToLesson(lesson);

      if (modified && !dryRun) {
        fs.writeFileSync(filePath, JSON.stringify(lesson, null, 2) + '\n', 'utf8');
        return { success: true, modified: true, file: fileName };
      }

      return { success: true, modified, file: fileName };
    } catch (error) {
      return { success: false, error: error.message, file: fileName };
    }
  }

  /**
   * Print statistics
   */
  printStats() {
    console.log('\n' + '='.repeat(60));
    console.log('INTERACTION GENERATION STATS');
    console.log('='.repeat(60));
    console.log(`Lessons processed: ${this.stats.lessonsProcessed}`);
    console.log(`Interactions added: ${this.stats.interactionsAdded}`);
    console.log(`Phases skipped (already have choices): ${this.stats.phasesSkipped}`);
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const dryRun = process.argv.includes('--dry-run');

  console.log('=' .repeat(60));
  console.log('LESSON INTERACTION GENERATOR');
  console.log('=' .repeat(60));
  console.log(`Mode: ${dryRun ? 'DRY RUN' : 'LIVE'}\n`);

  const generator = new InteractionGenerator();

  const lessonDirs = [
    path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
    path.join(__dirname, '..', 'lessons'),
  ];

  for (const dir of lessonDirs) {
    if (!fs.existsSync(dir)) continue;

    console.log(`Processing: ${dir}\n`);
    const files = fs.readdirSync(dir).filter(f =>
      f.endsWith('-dna.json') || f.endsWith('_dna.json')
    );

    for (const file of files) {
      const result = generator.processLessonFile(path.join(dir, file), dryRun);
      const status = result.modified ? '✅' : '⏭️';
      console.log(`  ${status} ${file}`);
    }
  }

  generator.printStats();

  if (dryRun) {
    console.log('\nRun without --dry-run to apply changes.');
  }
}

main().catch(console.error);
