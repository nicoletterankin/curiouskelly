/**
 * Lesson Content Audit Script
 * 
 * Connects to Supabase and audits all 365 lessons for content completeness.
 * Checks for missing atoms, incomplete variants, and content gaps.
 * 
 * Run from project root: node scripts/audit_lessons.js
 */

import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

// Create require to load from daily-lesson-marketing node_modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const { createClient } = require('@supabase/supabase-js');

// Supabase configuration (hardcoded for now)
const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Expected phases for each lesson
const EXPECTED_PHASES = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];
const EXPECTED_ARCHETYPES = ['Sage', 'Jester', 'Ruler'];
const EXPECTED_AGE_GROUPS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61+'];
const EXPECTED_LANGUAGES = ['en', 'es', 'fr'];

// Audit results
const results = {
  totalLessons: 0,
  lessonsWithContent: 0,
  lessonsWithoutContent: 0,
  missingAtoms: [],
  incompleteVariants: [],
  missingChoices: [],
  summary: {}
};

async function auditLessons() {
  console.log('🔍 CURIOUS KELLY LESSON AUDIT');
  console.log('=' .repeat(60));
  console.log(`Supabase: ${SUPABASE_URL}`);
  console.log(`Expected: 365 lessons × 3 archetypes × 5 phases = 5,475 atoms`);
  console.log('=' .repeat(60));
  console.log('');

  try {
    // Step 1: Get all core lessons
    console.log('📚 Step 1: Fetching core lessons...');
    const { data: coreLessons, error: lessonError } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic, universal_truth')
      .order('day_number');

    if (lessonError) {
      console.error('❌ Error fetching core lessons:', lessonError);
      return;
    }

    results.totalLessons = coreLessons.length;
    console.log(`✅ Found ${coreLessons.length} core lessons`);
    console.log('');

    // Step 2: Audit each lesson
    console.log('🔎 Step 2: Auditing lesson atoms...');
    console.log('');

    for (const lesson of coreLessons) {
      const dayNum = lesson.day_number;
      const topic = lesson.topic;

      // Get atoms for this lesson
      const { data: atoms, error: atomError } = await supabase
        .from('lesson_atoms')
        .select('id, phase, archetype, content')
        .eq('core_lesson_id', lesson.id);

      if (atomError) {
        console.error(`❌ Day ${dayNum} (${topic}): Error fetching atoms:`, atomError);
        continue;
      }

      // Check if lesson has any content
      if (!atoms || atoms.length === 0) {
        results.lessonsWithoutContent++;
        results.missingAtoms.push({
          day: dayNum,
          topic: topic,
          issue: 'No atoms found'
        });
        console.log(`❌ Day ${dayNum}: ${topic} - NO ATOMS`);
        continue;
      }

      results.lessonsWithContent++;

      // Check for missing archetypes
      const archetypesPresent = [...new Set(atoms.map(a => a.archetype))];
      const missingArchetypes = EXPECTED_ARCHETYPES.filter(a => !archetypesPresent.includes(a));

      if (missingArchetypes.length > 0) {
        results.missingAtoms.push({
          day: dayNum,
          topic: topic,
          issue: `Missing archetypes: ${missingArchetypes.join(', ')}`
        });
      }

      // Check each archetype for missing phases
      for (const archetype of EXPECTED_ARCHETYPES) {
        const archetypeAtoms = atoms.filter(a => a.archetype === archetype);
        const phasesPresent = archetypeAtoms.map(a => a.phase);
        const missingPhases = EXPECTED_PHASES.filter(p => !phasesPresent.includes(p));

        if (missingPhases.length > 0) {
          results.missingAtoms.push({
            day: dayNum,
            topic: topic,
            archetype: archetype,
            issue: `Missing phases: ${missingPhases.join(', ')}`
          });
        }

        // Check for interactive choices in question phases
        const questionPhases = archetypeAtoms.filter(a => ['Fact1', 'Fact2', 'Fact3'].includes(a.phase));
        for (const atom of questionPhases) {
          const content = atom.content;
          
          // Check if choices exist
          if (!content.choices) {
            results.missingChoices.push({
              day: dayNum,
              topic: topic,
              archetype: archetype,
              phase: atom.phase,
              issue: 'No choices field'
            });
            continue;
          }

          // Check for all age groups
          const ageGroupsPresent = Object.keys(content.choices);
          const missingAgeGroups = EXPECTED_AGE_GROUPS.filter(ag => !ageGroupsPresent.includes(ag));

          if (missingAgeGroups.length > 0) {
            results.incompleteVariants.push({
              day: dayNum,
              topic: topic,
              archetype: archetype,
              phase: atom.phase,
              issue: `Missing age groups: ${missingAgeGroups.join(', ')}`
            });
          }

          // Check for all languages in each age group
          for (const ageGroup of ageGroupsPresent) {
            const languagesPresent = Object.keys(content.choices[ageGroup] || {});
            const missingLanguages = EXPECTED_LANGUAGES.filter(lang => !languagesPresent.includes(lang));

            if (missingLanguages.length > 0) {
              results.incompleteVariants.push({
                day: dayNum,
                topic: topic,
                archetype: archetype,
                phase: atom.phase,
                ageGroup: ageGroup,
                issue: `Missing languages: ${missingLanguages.join(', ')}`
              });
            }

            // Check for at least 2 choices per language
            for (const lang of languagesPresent) {
              const choices = content.choices[ageGroup][lang];
              if (!Array.isArray(choices) || choices.length < 2) {
                results.incompleteVariants.push({
                  day: dayNum,
                  topic: topic,
                  archetype: archetype,
                  phase: atom.phase,
                  ageGroup: ageGroup,
                  language: lang,
                  issue: `Only ${choices?.length || 0} choices (need 2+)`
                });
              }
            }
          }
        }
      }

      // Progress indicator
      if (dayNum % 50 === 0) {
        console.log(`   Progress: Day ${dayNum}/365...`);
      }
    }

    console.log('');
    console.log('=' .repeat(60));
    console.log('📊 AUDIT RESULTS');
    console.log('=' .repeat(60));
    console.log('');

    // Summary
    console.log('📈 SUMMARY:');
    console.log(`   Total lessons: ${results.totalLessons}`);
    console.log(`   Lessons with content: ${results.lessonsWithContent} (${Math.round(results.lessonsWithContent / results.totalLessons * 100)}%)`);
    console.log(`   Lessons without content: ${results.lessonsWithoutContent} (${Math.round(results.lessonsWithoutContent / results.totalLessons * 100)}%)`);
    console.log('');

    // Missing atoms
    if (results.missingAtoms.length > 0) {
      console.log(`❌ MISSING ATOMS: ${results.missingAtoms.length} issues`);
      console.log('');
      console.log('   First 10 issues:');
      results.missingAtoms.slice(0, 10).forEach(issue => {
        console.log(`   - Day ${issue.day} (${issue.topic}): ${issue.issue}`);
      });
      if (results.missingAtoms.length > 10) {
        console.log(`   ... and ${results.missingAtoms.length - 10} more`);
      }
      console.log('');
    } else {
      console.log('✅ All lessons have atoms for all archetypes and phases');
      console.log('');
    }

    // Missing choices
    if (results.missingChoices.length > 0) {
      console.log(`❌ MISSING INTERACTIVE CHOICES: ${results.missingChoices.length} issues`);
      console.log('');
      console.log('   First 10 issues:');
      results.missingChoices.slice(0, 10).forEach(issue => {
        console.log(`   - Day ${issue.day} (${issue.topic}) ${issue.archetype}/${issue.phase}: ${issue.issue}`);
      });
      if (results.missingChoices.length > 10) {
        console.log(`   ... and ${results.missingChoices.length - 10} more`);
      }
      console.log('');
    } else {
      console.log('✅ All question phases have interactive choices');
      console.log('');
    }

    // Incomplete variants
    if (results.incompleteVariants.length > 0) {
      console.log(`⚠️  INCOMPLETE VARIANTS: ${results.incompleteVariants.length} issues`);
      console.log('');
      console.log('   First 10 issues:');
      results.incompleteVariants.slice(0, 10).forEach(issue => {
        const location = `Day ${issue.day} (${issue.topic}) ${issue.archetype}/${issue.phase}`;
        const detail = issue.ageGroup ? ` [${issue.ageGroup}]` : '';
        console.log(`   - ${location}${detail}: ${issue.issue}`);
      });
      if (results.incompleteVariants.length > 10) {
        console.log(`   ... and ${results.incompleteVariants.length - 10} more`);
      }
      console.log('');
    } else {
      console.log('✅ All choices have complete variants (age groups + languages)');
      console.log('');
    }

    // Priority fixes
    console.log('=' .repeat(60));
    console.log('🔧 PRIORITY FIXES');
    console.log('=' .repeat(60));
    console.log('');

    if (results.lessonsWithoutContent > 0) {
      console.log('🚨 CRITICAL: Generate missing lessons');
      console.log(`   ${results.lessonsWithoutContent} lessons have NO content at all`);
      console.log('   Run: python scripts/generate_curious_kelly_content.py --days <missing_days>');
      console.log('');
    }

    if (results.missingChoices.length > 0) {
      console.log('🚨 CRITICAL: Add interactive choices');
      console.log(`   ${results.missingChoices.length} question phases lack choices`);
      console.log('   Run: python scripts/generate_all_choices.py');
      console.log('');
    }

    if (results.incompleteVariants.length > 0) {
      console.log('⚠️  IMPORTANT: Complete variant coverage');
      console.log(`   ${results.incompleteVariants.length} choices missing age/language variants`);
      console.log('   Run: python scripts/fix_incomplete_variants.py');
      console.log('');
    }

    // Days 1-30 check
    console.log('=' .repeat(60));
    console.log('🎯 LAUNCH READINESS (Days 1-30)');
    console.log('=' .repeat(60));
    console.log('');

    const firstThirtyDays = coreLessons.filter(l => l.day_number >= 1 && l.day_number <= 30);
    const firstThirtyWithContent = firstThirtyDays.filter(l => {
      return !results.missingAtoms.some(issue => issue.day === l.day_number);
    });

    console.log(`Days 1-30: ${firstThirtyWithContent.length}/30 have complete content`);
    
    if (firstThirtyWithContent.length < 30) {
      const missingDays = firstThirtyDays
        .filter(l => !firstThirtyWithContent.includes(l))
        .map(l => l.day_number);
      console.log(`❌ Missing content for days: ${missingDays.join(', ')}`);
    } else {
      console.log('✅ First 30 days ready for launch!');
    }

    console.log('');
    console.log('=' .repeat(60));
    console.log('✅ AUDIT COMPLETE');
    console.log('=' .repeat(60));

  } catch (error) {
    console.error('❌ Audit failed:', error);
  }
}

// Run audit
auditLessons().catch(console.error);

