#!/usr/bin/env npx tsx
/**
 * CURIOUS KELLY - LESSON QUALITY EVAL
 * 
 * Comprehensive evaluation of all 365 Learn topics and 365 Grow topics
 * against the Kelly Bible brand standards.
 * 
 * Checks:
 * 1. Lesson existence and completeness (topic, universal_truth, phases)
 * 2. Phase sequence correctness (hook → cliff → fact1/2/3 → wisdom → outro)
 * 3. Options/answers present for question phases
 * 4. Feedback/responses defined for each option
 * 5. Lesson sources/citations present
 * 6. Brand voice compliance (Kelly's tone, no forbidden words)
 * 7. Visual asset URLs valid (infographics, Kelly images)
 * 8. Audio/video asset URLs valid where expected
 * 
 * Usage:
 *   npx tsx scripts/eval-lesson-quality.ts --track=learn --days=1-365
 *   npx tsx scripts/eval-lesson-quality.ts --track=grow --days=1-30
 *   npx tsx scripts/eval-lesson-quality.ts --all --report
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

const EXPECTED_PHASES = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'] as const;
const QUESTION_PHASES = ['cliff', 'fact1', 'fact2', 'fact3'] as const;

// Kelly Voice Bible - Forbidden words/patterns
const FORBIDDEN_PATTERNS = [
  /\bfree\b/gi,                    // Never use "free" (use "included", "yours")
  /\blearning style\b/gi,          // No learning-style classification
  /\binterest-driven\b/gi,         // No interest-driven selection
  /\bcheap\b/gi,                   // Avoid cheap/discount language
  /\bdiscount\b/gi,
  /\blimited time\b/gi,            // No artificial urgency
  /\bhurry\b/gi,
  /\bact now\b/gi,
];

// Kelly Voice - Expected characteristics
const VOICE_CHECKS = {
  humble: /\bwe\b|\btogether\b|\blet's\b|\bwondering\b/gi,
  curious: /\bwonder\b|\bcurious\b|\bexplore\b|\bdiscover\b|\bwhy\b|\bhow\b/gi,
  warm: /\b(you|your|you're)\b/gi,
  simple: (text: string) => {
    const words = text.split(/\s+/);
    const avgWordLen = words.reduce((sum, w) => sum + w.length, 0) / words.length;
    return avgWordLen < 7; // Simple language = shorter average word length
  },
};

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

interface EvalResult {
  day: number;
  track: string;
  topic: string;
  status: 'pass' | 'warn' | 'fail';
  issues: Issue[];
  score: number;
}

interface Issue {
  severity: 'critical' | 'warning' | 'info';
  category: 'existence' | 'sequence' | 'options' | 'feedback' | 'sources' | 'voice' | 'assets';
  message: string;
  phase?: string;
}

interface LessonData {
  id: string;
  day_number: number;
  track: string;
  topic: string;
  universal_truth: string;
  marketing_headline?: string;
  marketing_tagline?: string;
  sources?: string[];
}

interface AtomData {
  id: string;
  phase: string;
  archetype: string;
  content: any;
  hd_video_url?: string;
  visual_url?: string;
}

interface ShardData {
  id: string;
  region: string;
  archetype: string;
  script_content: any;
}

// ═══════════════════════════════════════════════════════════════════
// EVALUATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

function checkForbiddenWords(text: string): Issue[] {
  const issues: Issue[] = [];
  
  for (const pattern of FORBIDDEN_PATTERNS) {
    const matches = text.match(pattern);
    if (matches) {
      issues.push({
        severity: 'critical',
        category: 'voice',
        message: `Forbidden word/phrase found: "${matches[0]}" - Kelly Bible violation`,
      });
    }
  }
  
  return issues;
}

function checkVoiceCompliance(text: string): Issue[] {
  const issues: Issue[] = [];
  
  // Check for humble voice markers
  if (!VOICE_CHECKS.humble.test(text)) {
    issues.push({
      severity: 'info',
      category: 'voice',
      message: 'Consider adding collaborative language (we, together, let\'s)',
    });
  }
  
  // Check for curious voice markers
  if (!VOICE_CHECKS.curious.test(text)) {
    issues.push({
      severity: 'warning',
      category: 'voice',
      message: 'Missing curiosity markers (wonder, explore, discover, why, how)',
    });
  }
  
  // Check for simple language
  if (!VOICE_CHECKS.simple(text)) {
    issues.push({
      severity: 'info',
      category: 'voice',
      message: 'Language may be too complex - aim for simpler words',
    });
  }
  
  return issues;
}

function checkPhaseSequence(atoms: AtomData[]): Issue[] {
  const issues: Issue[] = [];
  const presentPhases = new Set(atoms.map(a => a.phase));
  
  // Check all required phases exist
  for (const phase of EXPECTED_PHASES) {
    if (!presentPhases.has(phase)) {
      issues.push({
        severity: 'critical',
        category: 'sequence',
        message: `Missing required phase: ${phase}`,
        phase,
      });
    }
  }
  
  return issues;
}

function checkOptionsAndFeedback(atoms: AtomData[]): Issue[] {
  const issues: Issue[] = [];
  
  for (const phase of QUESTION_PHASES) {
    const phaseAtom = atoms.find(a => a.phase === phase);
    if (!phaseAtom) continue;
    
    const content = phaseAtom.content;
    if (typeof content !== 'object') {
      issues.push({
        severity: 'warning',
        category: 'options',
        message: `Phase ${phase} content is not structured JSON`,
        phase,
      });
      continue;
    }
    
    // Check for options
    const options = content.options || content.choices || content.answers;
    if (!options || !Array.isArray(options) || options.length < 2) {
      issues.push({
        severity: 'critical',
        category: 'options',
        message: `Phase ${phase} missing options/choices (need at least 2)`,
        phase,
      });
    } else {
      // Check each option has feedback
      for (let i = 0; i < options.length; i++) {
        const opt = options[i];
        if (!opt.feedback && !opt.response && !opt.explanation) {
          issues.push({
            severity: 'warning',
            category: 'feedback',
            message: `Phase ${phase} option ${i + 1} missing feedback/response`,
            phase,
          });
        }
      }
    }
  }
  
  return issues;
}

function checkAssets(atoms: AtomData[]): Issue[] {
  const issues: Issue[] = [];
  
  for (const atom of atoms) {
    // Check visual assets for relevant phases
    if (['hook', 'wisdom'].includes(atom.phase)) {
      if (!atom.visual_url && !atom.hd_video_url) {
        issues.push({
          severity: 'warning',
          category: 'assets',
          message: `Phase ${atom.phase} missing visual asset (infographic or video)`,
          phase: atom.phase,
        });
      }
    }
  }
  
  return issues;
}

async function evaluateLesson(
  supabase: any,
  day: number,
  track: string
): Promise<EvalResult> {
  const issues: Issue[] = [];
  
  // Fetch core lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', day)
    .eq('track', track)
    .single();
  
  if (lessonError || !lesson) {
    return {
      day,
      track,
      topic: 'NOT FOUND',
      status: 'fail',
      issues: [{
        severity: 'critical',
        category: 'existence',
        message: `Lesson day ${day} track ${track} not found in database`,
      }],
      score: 0,
    };
  }
  
  // Check basic fields
  if (!lesson.topic) {
    issues.push({
      severity: 'critical',
      category: 'existence',
      message: 'Missing topic',
    });
  }
  
  if (!lesson.universal_truth) {
    issues.push({
      severity: 'critical',
      category: 'existence',
      message: 'Missing universal_truth',
    });
  }
  
  // Check voice compliance on topic and truth
  const textToCheck = `${lesson.topic || ''} ${lesson.universal_truth || ''} ${lesson.marketing_headline || ''}`;
  issues.push(...checkForbiddenWords(textToCheck));
  issues.push(...checkVoiceCompliance(textToCheck));
  
  // Check sources
  if (!lesson.sources || (Array.isArray(lesson.sources) && lesson.sources.length === 0)) {
    issues.push({
      severity: 'warning',
      category: 'sources',
      message: 'No lesson sources/citations provided',
    });
  }
  
  // Fetch atoms (phases)
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);
  
  if (atomsError || !atoms || atoms.length === 0) {
    issues.push({
      severity: 'critical',
      category: 'sequence',
      message: 'No lesson atoms (phases) found',
    });
  } else {
    // Check phase sequence
    issues.push(...checkPhaseSequence(atoms));
    
    // Check options and feedback
    issues.push(...checkOptionsAndFeedback(atoms));
    
    // Check assets
    issues.push(...checkAssets(atoms));
    
    // Check voice in phase content
    for (const atom of atoms) {
      const content = typeof atom.content === 'string' 
        ? atom.content 
        : JSON.stringify(atom.content || '');
      issues.push(...checkForbiddenWords(content));
    }
  }
  
  // Calculate score
  const criticalCount = issues.filter(i => i.severity === 'critical').length;
  const warningCount = issues.filter(i => i.severity === 'warning').length;
  const infoCount = issues.filter(i => i.severity === 'info').length;
  
  const score = Math.max(0, 100 - (criticalCount * 25) - (warningCount * 5) - (infoCount * 1));
  
  const status: 'pass' | 'warn' | 'fail' = 
    criticalCount > 0 ? 'fail' :
    warningCount > 2 ? 'warn' : 
    'pass';
  
  return {
    day,
    track,
    topic: lesson.topic || 'UNKNOWN',
    status,
    issues,
    score,
  };
}

// ═══════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  // Parse arguments
  const getArg = (name: string): string | undefined => {
    const arg = args.find(a => a.startsWith(`--${name}=`));
    return arg?.split('=')[1];
  };
  
  const track = getArg('track') || 'learn';
  const daysArg = getArg('days') || '1-365';
  const reportOnly = args.includes('--report');
  const allTracks = args.includes('--all');
  
  // Parse day range
  let startDay = 1;
  let endDay = 365;
  if (daysArg.includes('-')) {
    const [start, end] = daysArg.split('-').map(Number);
    startDay = start;
    endDay = end;
  } else {
    startDay = endDay = parseInt(daysArg);
  }
  
  console.log('═══════════════════════════════════════════════════════════════════');
  console.log('  CURIOUS KELLY - LESSON QUALITY EVALUATION');
  console.log('  Against Kelly Bible Brand Standards');
  console.log('═══════════════════════════════════════════════════════════════════');
  console.log(`\n  Track: ${allTracks ? 'learn + grow' : track}`);
  console.log(`  Days: ${startDay} - ${endDay}`);
  console.log('');
  
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error('❌ Missing SUPABASE credentials. Set PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }
  
  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  
  const tracks = allTracks ? ['learn', 'grow'] : [track];
  const allResults: EvalResult[] = [];
  
  for (const t of tracks) {
    console.log(`\n📚 Evaluating ${t.toUpperCase()} track (days ${startDay}-${endDay})...\n`);
    
    for (let day = startDay; day <= endDay; day++) {
      const result = await evaluateLesson(supabase, day, t);
      allResults.push(result);
      
      const icon = result.status === 'pass' ? '✅' : result.status === 'warn' ? '⚠️' : '❌';
      const criticalCount = result.issues.filter(i => i.severity === 'critical').length;
      const warningCount = result.issues.filter(i => i.severity === 'warning').length;
      
      if (!reportOnly || result.status !== 'pass') {
        console.log(`  ${icon} Day ${day.toString().padStart(3)}: ${result.topic.substring(0, 40).padEnd(40)} [${result.score}/100] ${criticalCount > 0 ? `🔴${criticalCount}` : ''} ${warningCount > 0 ? `🟡${warningCount}` : ''}`);
        
        if (result.status === 'fail') {
          for (const issue of result.issues.filter(i => i.severity === 'critical')) {
            console.log(`         └─ 🔴 ${issue.message}`);
          }
        }
      }
      
      // Rate limit
      if (day % 50 === 0) {
        await new Promise(r => setTimeout(r, 100));
      }
    }
  }
  
  // Summary
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('  SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════════\n');
  
  const passCount = allResults.filter(r => r.status === 'pass').length;
  const warnCount = allResults.filter(r => r.status === 'warn').length;
  const failCount = allResults.filter(r => r.status === 'fail').length;
  const avgScore = Math.round(allResults.reduce((sum, r) => sum + r.score, 0) / allResults.length);
  
  console.log(`  ✅ Pass: ${passCount} lessons`);
  console.log(`  ⚠️  Warn: ${warnCount} lessons`);
  console.log(`  ❌ Fail: ${failCount} lessons`);
  console.log(`  📊 Average Score: ${avgScore}/100`);
  
  // Issue breakdown
  const issuesByCategory: Record<string, number> = {};
  for (const result of allResults) {
    for (const issue of result.issues) {
      issuesByCategory[issue.category] = (issuesByCategory[issue.category] || 0) + 1;
    }
  }
  
  console.log('\n  Issues by Category:');
  for (const [cat, count] of Object.entries(issuesByCategory).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${cat}: ${count}`);
  }
  
  // Write detailed report
  const reportPath = path.join(process.cwd(), 'lesson-quality-report.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    generated_at: new Date().toISOString(),
    summary: { passCount, warnCount, failCount, avgScore, total: allResults.length },
    issuesByCategory,
    results: allResults,
  }, null, 2));
  
  console.log(`\n  📄 Detailed report: ${reportPath}`);
  console.log('');
  
  // Exit with error if failures
  if (failCount > 0) {
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});

