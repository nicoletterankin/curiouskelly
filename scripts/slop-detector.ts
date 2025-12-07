/**
 * AI Slop Detection Pipeline for Curious Kelly
 * 
 * Detects quality issues in lesson content:
 * - Topic/headline mismatches
 * - Duplicate content
 * - Generic AI patterns
 * - Template testimonials
 * - Missing visuals
 * 
 * Run: npx ts-node scripts/slop-detector.ts
 */

import { createClient } from '@supabase/supabase-js';

// Configuration
const SUPABASE_URL = process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || '';

if (!SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing SUPABASE_SERVICE_ROLE_KEY environment variable');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// Types
interface SlopIssue {
  content_type: string;
  content_id: string;
  day_number: number;
  issue_type: string;
  severity: string;
  field_name: string;
  field_value: string;
  expected_pattern?: string;
  actual_pattern?: string;
  detection_method: string;
  confidence_score: number;
}

interface CoreLesson {
  id: string;
  day_number: number;
  topic: string;
  marketing_headline: string | null;
  marketing_tagline: string | null;
  universal_truth: string | null;
  sample_testimonial: string | null;
  hero_image_url: string | null;
  thumbnail_url: string | null;
  recommended_videos: any[] | null;
  recommended_books: any[] | null;
}

interface LessonAtom {
  id: string;
  archetype: string;
  phase: string;
  content: {
    script?: string;
    options?: string[];
    responses?: Record<string, string>;
  };
  core_lessons?: {
    day_number: number;
  };
}

// ============================================
// STOP WORDS FOR KEYWORD EXTRACTION
// ============================================
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
  'of', 'with', 'by', 'from', 'how', 'what', 'why', 'where', 'when', 
  'your', 'you', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'this', 'that', 'these', 'those', 'we', 'they', 'their', 'our', 'its',
  'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'can', 'all', 'each', 'every', 'both'
]);

// ============================================
// AI PUN PATTERNS TO DETECT
// ============================================
const PUN_PATTERNS = [
  /\b\w+\s+It\s+to\s+\w+/i,           // "Leaf It to Nature"
  /\bUn\w+\s+the\s+Secrets?\b/i,      // "Unlock the Secrets"
  /\bLight\s+Up\s+Your\b/i,           // "Light Up Your World"
  /\bHarness\s+the\s+Power\b/i,       // "Harness the Power"
  /\bMaster\s+the\s+Art\b/i,          // "Master the Art"
  /\bDiscover\s+the\s+Magic\b/i,      // "Discover the Magic"
  /\bJourney\s+(Into|Through)\b/i,    // "Journey Into/Through"
  /\bDive\s+(Into|Deep|In)\b/i,       // "Dive Into"
  /\bUnleash\s+(the|Your)\b/i,        // "Unleash the Power"
  /\bUnlock\s+(the|Your)\b/i,         // "Unlock the Secrets"
  /\bUncover\s+the\b/i,               // "Uncover the Secrets"
  /\bUnearth\s+the\b/i,               // "Unearth the Past"
  /\bEmbrace\s+the\b/i,               // "Embrace the Change"
  /\bExplore\s+the\s+(World|Wonders|Magic)\b/i,
  /!\s*$/,                             // Ends with exclamation mark
  /^(The\s+)?(Ultimate|Complete|Essential)\s+Guide/i,
  /\bPower\s+of\s+\w+!/i,             // "Power of X!"
  /\bSecrets?\s+of\b/i,               // "Secrets of"
  /\bMagic\s+of\b/i,                  // "Magic of"
];

// ============================================
// TESTIMONIAL TEMPLATE PATTERNS
// ============================================
const TESTIMONIAL_PATTERNS = [
  /My (child|kid|son|daughter|kids)\s+(loved|absolutely loved|was completely captivated)/i,
  /Highly recommend!/i,
  /made\s+\w+\s+so\s+(fun|engaging|easy|understandable)/i,
  /truly\s+(engaging|educational|wonderful|amazing)/i,
  /fantastic\s+(learning\s+)?experience/i,
  /"-\s*(Parent|Teacher|Educator)\s*Review"?$/i,
  /A\s+truly\s+\w+\s+(and\s+\w+\s+)?experience/i,
  /helped\s+(them|my\s+\w+)\s+understand/i,
];

// ============================================
// HELPER FUNCTIONS
// ============================================

function extractKeywords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 3 && !STOP_WORDS.has(word));
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();
}

function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
}

function calculateKeywordOverlap(text1: string, text2: string): number {
  const keywords1 = extractKeywords(text1);
  const keywords2 = extractKeywords(text2);
  
  if (keywords1.length === 0) return 0;
  
  const overlap = keywords1.filter(k => 
    keywords2.some(h => h.includes(k) || k.includes(h))
  );
  
  return overlap.length / keywords1.length;
}

// ============================================
// DETECTION FUNCTIONS
// ============================================

/**
 * CRITICAL: Topic-Headline Mismatch Detection
 * Uses keyword extraction to compare topic vs headline
 */
async function detectTopicHeadlineMismatch(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, marketing_headline, universal_truth')
    .not('marketing_headline', 'is', null);

  if (error) {
    console.error('Error fetching lessons:', error);
    return issues;
  }

  for (const lesson of lessons || []) {
    const topicKeywords = extractKeywords(lesson.topic);
    const headlineKeywords = extractKeywords(lesson.marketing_headline || '');
    
    // Calculate keyword overlap
    const overlapRatio = calculateKeywordOverlap(lesson.topic, lesson.marketing_headline || '');
    
    // If less than 20% keyword overlap, flag it
    if (overlapRatio < 0.2 && topicKeywords.length >= 2) {
      issues.push({
        content_type: 'core_lesson',
        content_id: lesson.id,
        day_number: lesson.day_number,
        issue_type: 'topic_headline_mismatch',
        severity: 'critical',
        field_name: 'marketing_headline',
        field_value: lesson.marketing_headline || '',
        expected_pattern: `Topic keywords: ${topicKeywords.join(', ')}`,
        actual_pattern: `Headline keywords: ${headlineKeywords.join(', ')}`,
        detection_method: 'semantic',
        confidence_score: 1 - overlapRatio
      });
    }
    
    // Also check universal_truth vs topic
    if (lesson.universal_truth) {
      const truthOverlap = calculateKeywordOverlap(lesson.topic, lesson.universal_truth);
      if (truthOverlap < 0.15 && topicKeywords.length >= 2) {
        issues.push({
          content_type: 'core_lesson',
          content_id: lesson.id,
          day_number: lesson.day_number,
          issue_type: 'topic_truth_mismatch',
          severity: 'critical',
          field_name: 'universal_truth',
          field_value: lesson.universal_truth,
          expected_pattern: `Topic keywords: ${topicKeywords.join(', ')}`,
          actual_pattern: `Truth keywords: ${extractKeywords(lesson.universal_truth).join(', ')}`,
          detection_method: 'semantic',
          confidence_score: 1 - truthOverlap
        });
      }
    }
  }
  
  return issues;
}

/**
 * CRITICAL: Duplicate Content Detection
 * Hashes content blocks and finds duplicates
 */
async function detectDuplicateContent(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  const contentHashes = new Map<string, { id: string; day: number; archetype: string }[]>();
  
  const { data: atoms, error } = await supabase
    .from('lesson_atoms')
    .select(`
      id, 
      archetype, 
      phase, 
      content,
      core_lesson_id
    `);

  if (error) {
    console.error('Error fetching atoms:', error);
    return issues;
  }

  // Also fetch day numbers for each atom
  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('id, day_number');
  
  const lessonDays = new Map(lessons?.map(l => [l.id, l.day_number]) || []);

  for (const atom of atoms || []) {
    const script = (atom.content as any)?.script || '';
    if (script.length < 50) continue; // Skip very short content
    
    const hash = simpleHash(normalizeText(script));
    const dayNumber = lessonDays.get(atom.core_lesson_id) || 0;
    
    if (!contentHashes.has(hash)) {
      contentHashes.set(hash, []);
    }
    contentHashes.get(hash)!.push({
      id: atom.id,
      day: dayNumber,
      archetype: atom.archetype
    });
  }
  
  // Find duplicates (same content in different days)
  for (const [hash, occurrences] of contentHashes) {
    const uniqueDays = [...new Set(occurrences.map(o => o.day))];
    if (uniqueDays.length > 1) {
      // Only flag if same content appears in different days
      for (const occ of occurrences) {
        issues.push({
          content_type: 'lesson_atom',
          content_id: occ.id,
          day_number: occ.day,
          issue_type: 'duplicate_content',
          severity: 'critical',
          field_name: 'content.script',
          field_value: `Hash: ${hash}`,
          expected_pattern: 'Unique content per day',
          actual_pattern: `Found in days: ${uniqueDays.join(', ')}`,
          detection_method: 'hash',
          confidence_score: 1.0
        });
      }
    }
  }
  
  return issues;
}

/**
 * WARNING: Generic Pun Headline Detection
 * Matches common AI-generated pun patterns
 */
async function detectGenericPunHeadlines(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, marketing_headline')
    .not('marketing_headline', 'is', null);

  if (error) {
    console.error('Error fetching lessons:', error);
    return issues;
  }

  for (const lesson of lessons || []) {
    const matchedPatterns = PUN_PATTERNS.filter(p => p.test(lesson.marketing_headline || ''));
    
    // Only flag if multiple patterns match (more confident it's AI slop)
    if (matchedPatterns.length >= 2) {
      issues.push({
        content_type: 'core_lesson',
        content_id: lesson.id,
        day_number: lesson.day_number,
        issue_type: 'generic_pun_headline',
        severity: 'warning',
        field_name: 'marketing_headline',
        field_value: lesson.marketing_headline || '',
        expected_pattern: 'Original, non-template headline',
        actual_pattern: `Matched ${matchedPatterns.length} AI patterns`,
        detection_method: 'regex',
        confidence_score: Math.min(matchedPatterns.length * 0.25, 1.0)
      });
    }
  }
  
  return issues;
}

/**
 * WARNING: Template Testimonial Detection
 * Identifies AI-generated testimonial patterns
 */
async function detectTemplateTestimonials(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, sample_testimonial')
    .not('sample_testimonial', 'is', null);

  if (error) {
    console.error('Error fetching lessons:', error);
    return issues;
  }

  for (const lesson of lessons || []) {
    const matchedPatterns = TESTIMONIAL_PATTERNS.filter(p => 
      p.test(lesson.sample_testimonial || '')
    );
    
    if (matchedPatterns.length >= 2) {
      issues.push({
        content_type: 'core_lesson',
        content_id: lesson.id,
        day_number: lesson.day_number,
        issue_type: 'repetitive_testimonial',
        severity: 'warning',
        field_name: 'sample_testimonial',
        field_value: (lesson.sample_testimonial || '').substring(0, 100) + '...',
        expected_pattern: 'Unique, specific testimonial',
        actual_pattern: `Matched ${matchedPatterns.length} template patterns`,
        detection_method: 'regex',
        confidence_score: Math.min(matchedPatterns.length * 0.25, 1.0)
      });
    }
  }
  
  return issues;
}

/**
 * INFO: Missing Visual URLs
 */
async function detectMissingVisuals(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, hero_image_url, thumbnail_url');

  if (error) {
    console.error('Error fetching lessons:', error);
    return issues;
  }

  for (const lesson of lessons || []) {
    if (!lesson.hero_image_url) {
      issues.push({
        content_type: 'core_lesson',
        content_id: lesson.id,
        day_number: lesson.day_number,
        issue_type: 'missing_visual_url',
        severity: 'info',
        field_name: 'hero_image_url',
        field_value: 'NULL',
        detection_method: 'null_check',
        confidence_score: 1.0
      });
    }
  }
  
  return issues;
}

/**
 * WARNING: Cross-lesson Repetition Detection
 * Finds phrases that appear in multiple lessons
 */
async function detectCrossLessonRepetition(): Promise<SlopIssue[]> {
  const issues: SlopIssue[] = [];
  const phraseOccurrences = new Map<string, number[]>();
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, marketing_headline, marketing_tagline, marketing_pitch');

  if (error) {
    console.error('Error fetching lessons:', error);
    return issues;
  }

  // Extract 3-word phrases from marketing copy
  for (const lesson of lessons || []) {
    const texts = [
      lesson.marketing_headline,
      lesson.marketing_tagline,
      lesson.marketing_pitch
    ].filter(Boolean).join(' ');
    
    const words = normalizeText(texts).split(' ').filter(w => w.length > 2);
    
    for (let i = 0; i < words.length - 2; i++) {
      const phrase = `${words[i]} ${words[i+1]} ${words[i+2]}`;
      if (!phraseOccurrences.has(phrase)) {
        phraseOccurrences.set(phrase, []);
      }
      if (!phraseOccurrences.get(phrase)!.includes(lesson.day_number)) {
        phraseOccurrences.get(phrase)!.push(lesson.day_number);
      }
    }
  }
  
  // Find phrases that appear in 5+ different lessons
  for (const [phrase, days] of phraseOccurrences) {
    if (days.length >= 5 && !STOP_WORDS.has(phrase.split(' ')[0])) {
      // Just log one issue for the repeated phrase
      issues.push({
        content_type: 'core_lesson',
        content_id: 'multiple',
        day_number: days[0],
        issue_type: 'cross_lesson_repetition',
        severity: 'info',
        field_name: 'marketing_copy',
        field_value: phrase,
        expected_pattern: 'Unique phrasing',
        actual_pattern: `Found in ${days.length} lessons: Days ${days.slice(0, 5).join(', ')}${days.length > 5 ? '...' : ''}`,
        detection_method: 'ngram',
        confidence_score: Math.min(days.length / 10, 1.0)
      });
    }
  }
  
  return issues;
}

// ============================================
// MAIN PIPELINE
// ============================================

async function runFullAudit(reportOnly: boolean = false) {
  console.log('🔍 AI SLOP DETECTION PIPELINE');
  console.log('=' .repeat(60));
  console.log(`Supabase: ${SUPABASE_URL}`);
  console.log(`Mode: ${reportOnly ? 'Report Only' : 'Full Audit + Save'}`);
  console.log('=' .repeat(60));
  console.log('');

  const detectors = [
    { name: 'Topic-Headline Mismatch', fn: detectTopicHeadlineMismatch, emoji: '🎯' },
    { name: 'Duplicate Content', fn: detectDuplicateContent, emoji: '📋' },
    { name: 'Generic Pun Headlines', fn: detectGenericPunHeadlines, emoji: '🎪' },
    { name: 'Template Testimonials', fn: detectTemplateTestimonials, emoji: '💬' },
    { name: 'Cross-Lesson Repetition', fn: detectCrossLessonRepetition, emoji: '🔄' },
    { name: 'Missing Visuals', fn: detectMissingVisuals, emoji: '🖼️' },
  ];
  
  const allIssues: SlopIssue[] = [];
  
  for (const detector of detectors) {
    console.log(`${detector.emoji} Running: ${detector.name}...`);
    try {
      const issues = await detector.fn();
      console.log(`   ✓ Found ${issues.length} issues\n`);
      allIssues.push(...issues);
    } catch (error) {
      console.error(`   ✗ Error: ${error}\n`);
    }
  }
  
  if (!reportOnly && allIssues.length > 0) {
    console.log('💾 Saving issues to database...');
    
    // Clear old unresolved issues from this run
    const { error: deleteError } = await supabase
      .from('content_validation_results')
      .delete()
      .is('resolved_at', null);
    
    if (deleteError) {
      console.error('Error clearing old issues:', deleteError);
    }
    
    // Insert new issues in batches
    const batchSize = 100;
    for (let i = 0; i < allIssues.length; i += batchSize) {
      const batch = allIssues.slice(i, i + batchSize);
      const { error } = await supabase
        .from('content_validation_results')
        .insert(batch);
      
      if (error) {
        console.error(`Error inserting batch ${i / batchSize + 1}:`, error);
      }
    }
    console.log(`   ✓ Saved ${allIssues.length} issues\n`);
  }
  
  // Generate summary
  const summary = {
    total: allIssues.length,
    critical: allIssues.filter(i => i.severity === 'critical').length,
    warning: allIssues.filter(i => i.severity === 'warning').length,
    info: allIssues.filter(i => i.severity === 'info').length,
    byType: {} as Record<string, number>
  };
  
  for (const issue of allIssues) {
    summary.byType[issue.issue_type] = (summary.byType[issue.issue_type] || 0) + 1;
  }
  
  console.log('');
  console.log('=' .repeat(60));
  console.log('📊 SLOP DETECTION SUMMARY');
  console.log('=' .repeat(60));
  console.log('');
  console.log(`🔴 Critical: ${summary.critical}`);
  console.log(`🟡 Warning:  ${summary.warning}`);
  console.log(`🟢 Info:     ${summary.info}`);
  console.log(`📝 Total:    ${summary.total}`);
  console.log('');
  console.log('By Issue Type:');
  for (const [type, count] of Object.entries(summary.byType).sort((a, b) => b[1] - a[1])) {
    const emoji = type.includes('mismatch') ? '🚨' : 
                  type.includes('duplicate') ? '📋' :
                  type.includes('pun') ? '🎪' :
                  type.includes('testimonial') ? '💬' :
                  type.includes('visual') ? '🖼️' : '📌';
    console.log(`  ${emoji} ${type}: ${count}`);
  }
  
  // Show critical issues details
  if (summary.critical > 0) {
    console.log('');
    console.log('=' .repeat(60));
    console.log('🚨 CRITICAL ISSUES (First 10)');
    console.log('=' .repeat(60));
    
    const criticalIssues = allIssues.filter(i => i.severity === 'critical').slice(0, 10);
    for (const issue of criticalIssues) {
      console.log(`\nDay ${issue.day_number}: ${issue.issue_type}`);
      console.log(`  Field: ${issue.field_name}`);
      console.log(`  Value: ${issue.field_value?.substring(0, 60)}...`);
      if (issue.expected_pattern) {
        console.log(`  Expected: ${issue.expected_pattern}`);
      }
      if (issue.actual_pattern) {
        console.log(`  Actual: ${issue.actual_pattern}`);
      }
    }
  }
  
  console.log('');
  console.log('=' .repeat(60));
  console.log('✅ AUDIT COMPLETE');
  console.log('=' .repeat(60));
  
  return summary;
}

// CLI handling
const args = process.argv.slice(2);
const reportOnly = args.includes('--report-only');

runFullAudit(reportOnly).catch(console.error);

export { runFullAudit, detectTopicHeadlineMismatch, detectDuplicateContent };


