/**
 * PICKY NICKY'S SMART LESSON AUDIT v3
 * 
 * Zero Trust + Intelligence = Smart Audit
 * 
 * Three levels:
 * 1. KEYWORD MATCH - exact words from topic appear in content
 * 2. SEMANTIC MATCH - related concepts even without exact words  
 * 3. HUMAN REVIEW - flagged for manual inspection
 * 
 * Everything recorded. Nothing hidden. Full transparency.
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_KEY) {
    console.error('❌ Missing Supabase key');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Semantic relationships - words/concepts that are related
const SEMANTIC_MAP = {
    'listen': ['hear', 'attention', 'understand', 'respond', 'speaker', 'conversation'],
    'listening': ['hear', 'attention', 'understand', 'respond', 'speaker', 'conversation'],
    'patience': ['wait', 'delay', 'gratification', 'marshmallow', 'time', 'persever'],
    'pays': ['reward', 'outcome', 'success', 'benefit', 'worth'],
    'gratitude': ['grateful', 'thankful', 'appreciate', 'blessing'],
    'changes': ['transform', 'brain', 'effect', 'impact', 'different'],
    'breathing': ['breath', 'air', 'oxygen', 'lungs', 'inhale', 'exhale'],
    'matters': ['important', 'essential', 'vital', 'critical', 'key'],
    'rest': ['sleep', 'recover', 'repair', 'relax', 'recharge'],
    'happens': ['occur', 'process', 'during', 'when'],
    'patterns': ['pattern', 'recognition', 'repeat', 'sequence', 'predict'],
    'everywhere': ['all', 'around', 'world', 'ubiquitous', 'find'],
    'memories': ['remember', 'recall', 'memory', 'forget', 'past'],
    'sun': ['solar', 'star', 'energy', 'light', 'heat'],
    'powers': ['energy', 'fuel', 'drive', 'sustain'],
    'earth': ['planet', 'world', 'ground', 'land'],
    'fossils': ['fossil', 'preserved', 'ancient', 'old', 'stone', 'rock'],
    'stories': ['tale', 'history', 'past', 'record'],
    'trapped': ['preserved', 'encased', 'held', 'contained'],
    'stone': ['rock', 'mineral', 'fossil'],
    'earthquakes': ['earthquake', 'seismic', 'shake', 'tremor', 'quake'],
    'ground': ['earth', 'land', 'surface', 'floor'],
    'shakes': ['shake', 'tremble', 'vibrate', 'move'],
    'lakes': ['lake', 'water', 'pond', 'body'],
    'come': ['form', 'start', 'begin', 'origin'],
    'forests': ['forest', 'trees', 'woods', 'jungle'],
    'secret': ['hidden', 'underground', 'invisible', 'unseen'],
    'life': ['living', 'alive', 'organism', 'species'],
    'jungles': ['jungle', 'rainforest', 'tropical', 'dense'],
    'alive': ['living', 'life', 'teeming', 'species'],
    'cities': ['city', 'metropolis', 'urban', 'habitat'],
    'under': ['underwater', 'beneath', 'below', 'ocean'],
    'islands': ['island', 'hawaii', 'land', 'emerge'],
    'born': ['create', 'form', 'build', 'made', 'origin'],
    'volcanoes': ['volcano', 'volcanic', 'lava', 'eruption'],
    'ocean': ['sea', 'water', 'marine', 'underwater'],
    'coral': ['reef', 'marine', 'ocean', 'underwater'],
    'reefs': ['coral', 'marine', 'ecosystem'],
    'trees': ['tree', 'forest', 'plant', 'wood'],
    'fungi': ['fungus', 'mushroom', 'underground', 'mycelium'],
};

// Check keyword relevance
function keywordMatch(topic, text) {
    if (!topic || !text) return { score: 0, matched: [], keywords: [] };
    
    const keywords = topic.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length > 3)
        .filter(w => !['what', 'where', 'when', 'which', 'that', 'this', 'from', 'your', 'with', 'have', 'does', 'made', 'really', 'why'].includes(w));
    
    const textLower = text.toLowerCase();
    const matched = keywords.filter(word => textLower.includes(word));
    
    return {
        score: keywords.length > 0 ? matched.length / keywords.length : 0,
        matched,
        keywords
    };
}

// Check semantic relevance
function semanticMatch(topic, text) {
    if (!topic || !text) return { score: 0, matched: [], expanded: [] };
    
    const keywords = topic.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length > 3);
    
    const textLower = text.toLowerCase();
    const expanded = [];
    const matched = [];
    
    for (const keyword of keywords) {
        const semanticAlternatives = SEMANTIC_MAP[keyword] || [];
        expanded.push({ keyword, alternatives: semanticAlternatives });
        
        // Check if keyword or any alternative is in text
        if (textLower.includes(keyword)) {
            matched.push(keyword);
        } else {
            for (const alt of semanticAlternatives) {
                if (textLower.includes(alt)) {
                    matched.push(`${keyword}→${alt}`);
                    break;
                }
            }
        }
    }
    
    return {
        score: keywords.length > 0 ? matched.length / keywords.length : 0,
        matched,
        expanded
    };
}

// Determine issue severity
function classifyIssue(keywordResult, semanticResult) {
    if (keywordResult.score >= 0.3) {
        return { severity: 'none', reason: 'Keyword match sufficient' };
    }
    if (semanticResult.score >= 0.5) {
        return { severity: 'low', reason: 'Semantically related, likely fine' };
    }
    if (semanticResult.score >= 0.25) {
        return { severity: 'medium', reason: 'Partial semantic match, needs review' };
    }
    return { severity: 'high', reason: 'No clear connection found' };
}

// Main audit function
async function runSmartAudit() {
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   🧠 PICKY NICKY\'S SMART LESSON AUDIT v3');
    console.log('   Zero Trust • Semantic Intelligence • Full Transparency');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('\n');

    // Clear previous audits
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
        passed: 0,
        headlineMismatch: { high: 0, medium: 0, low: 0 },
        truthMismatch: { high: 0, medium: 0, low: 0 },
        missingBooks: 0,
        missingVideos: 0,
        missingFunFacts: 0
    };

    console.log('🔍 Auditing each lesson with smart analysis...\n');

    for (const lesson of lessons) {
        const dayAudits = [];
        let hasHighSeverity = false;

        // 1. HEADLINE-TOPIC ANALYSIS
        const headlineKw = keywordMatch(lesson.topic, lesson.marketing_headline);
        const headlineSem = semanticMatch(lesson.topic, lesson.marketing_headline);
        const headlineClass = classifyIssue(headlineKw, headlineSem);

        if (headlineClass.severity !== 'none') {
            if (headlineClass.severity === 'high') {
                stats.headlineMismatch.high++;
                hasHighSeverity = true;
            } else if (headlineClass.severity === 'medium') {
                stats.headlineMismatch.medium++;
            } else {
                stats.headlineMismatch.low++;
            }

            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'headline_topic_match',
                status: headlineClass.severity === 'high' ? 'fail' : headlineClass.severity === 'medium' ? 'needs_review' : 'warning',
                field_name: 'marketing_headline',
                original_value: lesson.marketing_headline,
                expected_pattern: `Topic: "${lesson.topic}" | Keywords: ${headlineKw.keywords.join(', ')}`,
                actual_issue: `${headlineClass.reason} | Keyword: ${(headlineKw.score * 100).toFixed(0)}% | Semantic: ${(headlineSem.score * 100).toFixed(0)}% | Matched: ${headlineSem.matched.join(', ') || 'none'}`,
                confidence_score: headlineSem.score,
                audited_by: 'picky_nicky_v3'
            });
        }

        // 2. UNIVERSAL TRUTH-TOPIC ANALYSIS
        const truthKw = keywordMatch(lesson.topic, lesson.universal_truth);
        const truthSem = semanticMatch(lesson.topic, lesson.universal_truth);
        const truthClass = classifyIssue(truthKw, truthSem);

        if (truthClass.severity !== 'none') {
            if (truthClass.severity === 'high') {
                stats.truthMismatch.high++;
                hasHighSeverity = true;
            } else if (truthClass.severity === 'medium') {
                stats.truthMismatch.medium++;
            } else {
                stats.truthMismatch.low++;
            }

            dayAudits.push({
                day_number: lesson.day_number,
                audit_type: 'universal_truth_match',
                status: truthClass.severity === 'high' ? 'fail' : truthClass.severity === 'medium' ? 'needs_review' : 'warning',
                field_name: 'universal_truth',
                original_value: lesson.universal_truth,
                expected_pattern: `Topic: "${lesson.topic}" | Keywords: ${truthKw.keywords.join(', ')}`,
                actual_issue: `${truthClass.reason} | Keyword: ${(truthKw.score * 100).toFixed(0)}% | Semantic: ${(truthSem.score * 100).toFixed(0)}%`,
                confidence_score: truthSem.score,
                audited_by: 'picky_nicky_v3'
            });
        }

        // 3. BOOK VERIFICATION
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
                expected_pattern: 'At least 1 book recommended',
                actual_issue: 'No books recommended',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v3'
            });
        }

        // 4. VIDEO VERIFICATION
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
                expected_pattern: 'At least 1 video recommended',
                actual_issue: 'No videos recommended',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v3'
            });
        }

        // 5. FUN FACTS CHECK
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
                original_value: null,
                expected_pattern: 'At least 1 fun fact with content',
                actual_issue: 'Fun facts missing or empty',
                confidence_score: 1.0,
                audited_by: 'picky_nicky_v3'
            });
        }

        // Record full audit status
        if (dayAudits.length === 0) {
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
                audited_by: 'picky_nicky_v3'
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
    console.log('   📊 SMART AUDIT COMPLETE - FULL TRANSPARENCY REPORT');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('\n');
    console.log(`   Total Lessons Audited:       ${stats.total}`);
    console.log(`   ✅ Perfect Score:            ${stats.passed} (${(stats.passed/stats.total*100).toFixed(1)}%)`);
    console.log('\n   HEADLINE-TOPIC ALIGNMENT:');
    console.log(`   🔴 High Severity (FAIL):     ${stats.headlineMismatch.high}`);
    console.log(`   🟡 Medium (NEEDS REVIEW):    ${stats.headlineMismatch.medium}`);
    console.log(`   ⚪ Low (semantically OK):    ${stats.headlineMismatch.low}`);
    console.log('\n   UNIVERSAL TRUTH ALIGNMENT:');
    console.log(`   🔴 High Severity (FAIL):     ${stats.truthMismatch.high}`);
    console.log(`   🟡 Medium (NEEDS REVIEW):    ${stats.truthMismatch.medium}`);
    console.log(`   ⚪ Low (semantically OK):    ${stats.truthMismatch.low}`);
    console.log('\n   RESOURCE GAPS:');
    console.log(`   📚 Missing Books:            ${stats.missingBooks}`);
    console.log(`   🎬 Missing Videos:           ${stats.missingVideos}`);
    console.log(`   💡 Missing Fun Facts:        ${stats.missingFunFacts}`);
    console.log('\n');
    console.log(`   Total Audit Records:         ${audits.length}`);
    console.log('\n');
    console.log('   🎯 PRIORITIZED ACTION ITEMS:');
    console.log(`   1. Fix ${stats.headlineMismatch.high} headlines with HIGH severity`);
    console.log(`   2. Fix ${stats.truthMismatch.high} universal truths with HIGH severity`);
    console.log(`   3. Review ${stats.headlineMismatch.medium + stats.truthMismatch.medium} items flagged for human review`);
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');

    return stats;
}

// Run the audit
runSmartAudit().catch(console.error);

