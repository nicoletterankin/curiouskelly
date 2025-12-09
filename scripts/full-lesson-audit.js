/**
 * PICKY NICKY'S FULL LESSON AUDIT
 * 
 * Zero Trust. Full Transparency. Every lesson checked.
 * Every finding recorded in the Commons for all to see.
 * 
 * "Trust, but verify" - NO. "Verify, then verify again."
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_KEY) {
    console.error('❌ Missing Supabase key');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Slop patterns Picky Nicky HATES
const SLOP_PATTERNS = [
    { pattern: /\bunlock\b/i, name: 'unlock', severity: 'high' },
    { pattern: /\bdiscover\b/i, name: 'discover', severity: 'medium' },
    { pattern: /\bjourney\b/i, name: 'journey', severity: 'high' },
    { pattern: /\bdive into\b/i, name: 'dive_into', severity: 'high' },
    { pattern: /\bembark\b/i, name: 'embark', severity: 'high' },
    { pattern: /\bdelve\b/i, name: 'delve', severity: 'high' },
    { pattern: /\bexplore the world of\b/i, name: 'explore_world_of', severity: 'high' },
    { pattern: /\bsecrets of\b/i, name: 'secrets_of', severity: 'medium' },
    { pattern: /\bpower of\b/i, name: 'power_of', severity: 'low' },
    { pattern: /\bmagic of\b/i, name: 'magic_of', severity: 'medium' },
    { pattern: /\btap into\b/i, name: 'tap_into', severity: 'high' },
    { pattern: /\bgame.?changer\b/i, name: 'game_changer', severity: 'high' },
    { pattern: /\bmaster the art\b/i, name: 'master_the_art', severity: 'high' },
];

// Check if topic words appear in text
function hasTopicRelevance(topic, text) {
    if (!topic || !text) return { relevant: false, matchedWords: [], score: 0 };
    
    const topicWords = topic.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length > 3)
        .filter(w => !['what', 'where', 'when', 'which', 'that', 'this', 'from', 'your', 'with', 'have', 'does', 'made'].includes(w));
    
    const textLower = text.toLowerCase();
    const matchedWords = topicWords.filter(word => textLower.includes(word));
    const score = topicWords.length > 0 ? matchedWords.length / topicWords.length : 0;
    
    return {
        relevant: score > 0.2 || matchedWords.length >= 1,
        matchedWords,
        score,
        topicWords
    };
}

// Check for slop
function detectSlop(text) {
    if (!text) return [];
    const found = [];
    for (const slop of SLOP_PATTERNS) {
        if (slop.pattern.test(text)) {
            found.push({ name: slop.name, severity: slop.severity });
        }
    }
    return found;
}

// Main audit function
async function runFullAudit() {
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   🔬 PICKY NICKY\'S COMPREHENSIVE LESSON AUDIT');
    console.log('   Zero Trust • Full Transparency • Every Lesson Checked');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('\n');

    // Clear previous audits (fresh start)
    console.log('🧹 Clearing previous audit records...');
    await supabase.from('lesson_audits').delete().neq('id', '00000000-0000-0000-0000-000000000000');

    // Fetch all lessons
    console.log('📚 Fetching all 365 lessons...\n');
    const { data: lessons, error } = await supabase
        .from('core_lessons')
        .select('*')
        .order('day_number');

    if (error) {
        console.error('❌ Failed to fetch lessons:', error);
        return;
    }

    const audits = [];
    const stats = {
        total: lessons.length,
        headlineMismatches: 0,
        truthMismatches: 0,
        slopFound: 0,
        missingBooks: 0,
        missingVideos: 0,
        missingFunFacts: 0,
        incompleteFields: 0,
        passed: 0
    };

    console.log('🔍 Auditing each lesson...\n');

    for (const lesson of lessons) {
        const dayAudits = [];
        let dayPassed = true;

        // 1. HEADLINE-TOPIC MATCH CHECK
        const headlineCheck = hasTopicRelevance(lesson.topic, lesson.marketing_headline);
        if (!headlineCheck.relevant) {
            dayPassed = false;
            stats.headlineMismatches++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'headline_topic_match',
                status: 'fail',
                field_name: 'marketing_headline',
                original_value: lesson.marketing_headline,
                expected_pattern: `Should contain topic keywords: ${headlineCheck.topicWords?.join(', ')}`,
                actual_issue: `No topic relevance found. Score: ${(headlineCheck.score * 100).toFixed(0)}%`,
                confidence_score: 0.8,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 2. UNIVERSAL TRUTH-TOPIC MATCH CHECK
        const truthCheck = hasTopicRelevance(lesson.topic, lesson.universal_truth);
        if (!truthCheck.relevant) {
            dayPassed = false;
            stats.truthMismatches++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'universal_truth_match',
                status: 'fail',
                field_name: 'universal_truth',
                original_value: lesson.universal_truth,
                expected_pattern: `Should relate to: ${lesson.topic}`,
                actual_issue: `Universal truth doesn't match topic. Score: ${(truthCheck.score * 100).toFixed(0)}%`,
                confidence_score: 0.8,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 3. SLOP DETECTION IN HEADLINE
        const headlineSlop = detectSlop(lesson.marketing_headline);
        if (headlineSlop.length > 0) {
            dayPassed = false;
            stats.slopFound++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'slop_detection',
                status: 'warning',
                field_name: 'marketing_headline',
                original_value: lesson.marketing_headline,
                expected_pattern: 'No AI slop patterns',
                actual_issue: `Slop detected: ${headlineSlop.map(s => s.name).join(', ')}`,
                confidence_score: 0.9,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 4. BOOK VERIFICATION
        const hasBooks = lesson.recommended_books && 
                        Array.isArray(lesson.recommended_books) && 
                        lesson.recommended_books.length > 0;
        if (!hasBooks) {
            stats.missingBooks++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'book_verification',
                status: 'warning',
                field_name: 'recommended_books',
                original_value: null,
                expected_pattern: 'At least 1 real book',
                actual_issue: 'No books recommended',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 5. VIDEO VERIFICATION
        const hasVideos = lesson.recommended_videos && 
                         Array.isArray(lesson.recommended_videos) && 
                         lesson.recommended_videos.length > 0;
        if (!hasVideos) {
            stats.missingVideos++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'video_verification',
                status: 'warning',
                field_name: 'recommended_videos',
                original_value: null,
                expected_pattern: 'At least 1 real video',
                actual_issue: 'No videos recommended',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 6. FUN FACTS CHECK
        const hasFunFacts = lesson.fun_facts && 
                          Array.isArray(lesson.fun_facts) && 
                          lesson.fun_facts.length > 0 &&
                          lesson.fun_facts[0]?.fact;
        if (!hasFunFacts) {
            stats.missingFunFacts++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'fun_fact_quality',
                status: 'warning',
                field_name: 'fun_facts',
                original_value: JSON.stringify(lesson.fun_facts),
                expected_pattern: 'At least 1 fun fact with content',
                actual_issue: 'Fun facts missing or empty',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v2'
            });
        }

        // 7. CONTENT COMPLETENESS
        const requiredFields = ['topic', 'universal_truth', 'marketing_headline', 'extended_explanation'];
        const missingFields = requiredFields.filter(f => !lesson[f] || lesson[f].trim() === '');
        if (missingFields.length > 0) {
            dayPassed = false;
            stats.incompleteFields++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'content_completeness',
                status: 'fail',
                field_name: missingFields.join(', '),
                original_value: null,
                expected_pattern: 'All required fields populated',
                actual_issue: `Missing: ${missingFields.join(', ')}`,
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v2'
            });
        }

        // Record full audit status
        if (dayPassed) {
            stats.passed++;
            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'full_audit',
                status: 'pass',
                field_name: null,
                original_value: null,
                expected_pattern: null,
                actual_issue: null,
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v2'
            });
        }

        audits.push(...dayAudits);

        // Progress indicator
        if (lesson.day_number % 50 === 0 || lesson.day_number === 365) {
            console.log(`   Audited Day ${lesson.day_number}/365...`);
        }
    }

    // Insert all audits
    console.log('\n📝 Recording audit results in the Commons...');
    
    // Insert in batches
    const batchSize = 100;
    for (let i = 0; i < audits.length; i += batchSize) {
        const batch = audits.slice(i, i + batchSize);
        const { error: insertError } = await supabase
            .from('lesson_audits')
            .upsert(batch, { 
                onConflict: 'day_number,audit_type,field_name',
                ignoreDuplicates: false 
            });
        
        if (insertError) {
            console.error(`   ❌ Batch insert error:`, insertError.message);
        }
    }

    // Print summary
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   📊 AUDIT COMPLETE - FULL TRANSPARENCY REPORT');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('\n');
    console.log(`   Total Lessons Audited:     ${stats.total}`);
    console.log(`   ✅ Passed All Checks:      ${stats.passed} (${(stats.passed/stats.total*100).toFixed(1)}%)`);
    console.log(`   ❌ Headline Mismatches:    ${stats.headlineMismatches}`);
    console.log(`   ❌ Truth Mismatches:       ${stats.truthMismatches}`);
    console.log(`   ⚠️  Slop Detected:          ${stats.slopFound}`);
    console.log(`   📚 Missing Books:          ${stats.missingBooks}`);
    console.log(`   🎬 Missing Videos:         ${stats.missingVideos}`);
    console.log(`   💡 Missing Fun Facts:      ${stats.missingFunFacts}`);
    console.log(`   📝 Incomplete Fields:      ${stats.incompleteFields}`);
    console.log('\n');
    console.log(`   Total Audit Records:       ${audits.length}`);
    console.log('\n');
    console.log('   All results stored in lesson_audits table.');
    console.log('   View them at: /public/lesson-commons.html');
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');

    return stats;
}

// Run the audit
runFullAudit().catch(console.error);


