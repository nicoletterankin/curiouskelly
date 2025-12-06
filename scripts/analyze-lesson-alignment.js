/**
 * LESSON ALIGNMENT ANALYZER
 * 
 * Finds where content got shuffled to wrong topics
 * by detecting headline/truth pairs that match each other
 * but don't match their assigned topic.
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function analyzeAlignment() {
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('   🔬 LESSON ALIGNMENT ANALYZER');
    console.log('   Finding misaligned content pairs');
    console.log('═══════════════════════════════════════════════════════════════\n');

    const { data: lessons, error } = await supabase
        .from('core_lessons')
        .select('day_number, topic, marketing_headline, universal_truth')
        .order('day_number');

    if (error) {
        console.error('Error:', error);
        return;
    }

    // Analyze each lesson for alignment
    const misaligned = [];
    const probablyOk = [];
    const needsReview = [];

    for (const lesson of lessons) {
        const topicWords = extractKeywords(lesson.topic);
        const headlineWords = extractKeywords(lesson.marketing_headline);
        const truthWords = extractKeywords(lesson.universal_truth);

        // Check if headline and truth match each other
        const headlineTruthOverlap = calculateOverlap(headlineWords, truthWords);
        
        // Check if topic matches headline
        const topicHeadlineOverlap = calculateOverlap(topicWords, headlineWords);
        
        // Check if topic matches truth
        const topicTruthOverlap = calculateOverlap(topicWords, truthWords);

        // Classification logic:
        // If headline matches truth well (>20%) but topic doesn't match either (<10%)
        // = Content pair is misaligned with topic
        if (headlineTruthOverlap >= 0.15 && topicHeadlineOverlap < 0.1 && topicTruthOverlap < 0.1) {
            misaligned.push({
                day: lesson.day_number,
                topic: lesson.topic,
                headline: lesson.marketing_headline,
                truth: lesson.universal_truth,
                headlineTruthOverlap,
                topicHeadlineOverlap,
                topicTruthOverlap,
                diagnosis: 'MISALIGNED: Headline+Truth match each other but not topic'
            });
        } else if (topicHeadlineOverlap >= 0.15 || topicTruthOverlap >= 0.15) {
            probablyOk.push({
                day: lesson.day_number,
                topic: lesson.topic,
                topicHeadlineOverlap,
                topicTruthOverlap
            });
        } else {
            needsReview.push({
                day: lesson.day_number,
                topic: lesson.topic,
                headline: lesson.marketing_headline,
                truth: lesson.universal_truth,
                headlineTruthOverlap,
                topicHeadlineOverlap,
                topicTruthOverlap,
                diagnosis: 'UNCLEAR: Needs human review'
            });
        }
    }

    console.log(`📊 ALIGNMENT ANALYSIS RESULTS:\n`);
    console.log(`   ✅ Probably OK:     ${probablyOk.length} lessons`);
    console.log(`   🔴 MISALIGNED:      ${misaligned.length} lessons`);
    console.log(`   🟡 Needs Review:    ${needsReview.length} lessons`);
    console.log('\n');

    // Show misaligned lessons with details
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   🔴 CLEARLY MISALIGNED LESSONS (Content doesn\'t match topic)');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Group by ranges to see patterns
    const ranges = {};
    misaligned.forEach(m => {
        const range = Math.floor(m.day / 10) * 10;
        if (!ranges[range]) ranges[range] = [];
        ranges[range].push(m);
    });

    for (const [range, items] of Object.entries(ranges).sort((a, b) => a[0] - b[0])) {
        console.log(`\n   Days ${range}-${parseInt(range) + 9}: ${items.length} misaligned`);
        items.slice(0, 3).forEach(m => {
            console.log(`      Day ${m.day}: "${m.topic.substring(0, 30)}..."`);
            console.log(`         Headline about: "${m.headline.substring(0, 50)}..."`);
        });
        if (items.length > 3) {
            console.log(`         ... and ${items.length - 3} more`);
        }
    }

    // Find potential swaps - where content from day X would fit day Y
    console.log('\n\n═══════════════════════════════════════════════════════════════');
    console.log('   🔄 POTENTIAL CONTENT LOCATIONS');
    console.log('   Where the misaligned content might belong');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Look for topics that match misaligned headlines/truths
    const contentSearches = [
        { search: 'egypt', content: 'Ancient Egypt' },
        { search: 'greek', content: 'Ancient Greeks' },
        { search: 'rome', content: 'Roman Empire' },
        { search: 'medieval', content: 'Middle Ages' },
        { search: 'renaissance', content: 'Renaissance' },
        { search: 'scientific', content: 'Scientific Revolution' },
        { search: 'solar system', content: 'Solar System' },
        { search: 'venus', content: 'Venus planet' },
        { search: 'mars', content: 'Mars planet' },
        { search: 'jupiter', content: 'Jupiter planet' },
    ];

    for (const { search, content } of contentSearches) {
        // Find days where topic mentions this
        const topicMatches = lessons.filter(l => 
            l.topic.toLowerCase().includes(search)
        );
        // Find days where headline mentions this
        const headlineMatches = lessons.filter(l => 
            l.marketing_headline.toLowerCase().includes(search)
        );

        if (topicMatches.length > 0 || headlineMatches.length > 0) {
            console.log(`   "${content}":`);
            if (topicMatches.length > 0) {
                console.log(`      Topic appears in: Day ${topicMatches.map(l => l.day_number).join(', ')}`);
            }
            if (headlineMatches.length > 0) {
                console.log(`      Headline appears in: Day ${headlineMatches.map(l => l.day_number).join(', ')}`);
            }
            console.log('');
        }
    }

    // Export findings
    const findings = {
        summary: {
            ok: probablyOk.length,
            misaligned: misaligned.length,
            needsReview: needsReview.length
        },
        misaligned,
        needsReview
    };

    // Insert findings into audit trail
    const auditRecords = misaligned.map(m => ({
        day_number: m.day,
        audit_type: 'content_completeness',
        status: 'fail',
        field_name: 'alignment',
        original_value: `Topic: ${m.topic} | Headline: ${m.headline}`,
        expected_pattern: 'Headline and Universal Truth should match Topic',
        actual_issue: m.diagnosis,
        confidence_score: Math.max(m.headlineTruthOverlap, 0),
        audited_by: 'alignment_analyzer_v1'
    }));

    console.log('\n📝 Recording findings in audit trail...');
    if (auditRecords.length > 0) {
        await supabase.from('lesson_audits').upsert(auditRecords, {
            onConflict: 'day_number,audit_type,field_name',
            ignoreDuplicates: false
        });
    }

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('   Analysis complete. Check the Lesson Commons for details.');
    console.log('═══════════════════════════════════════════════════════════════\n');

    return findings;
}

function extractKeywords(text) {
    if (!text) return new Set();
    const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'and', 'or', 'but', 'if', 'because', 'than', 'that', 'which', 'who', 'whom', 'what', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'even', 'not', 'your',
        'you', 'it', 'its', 'they', 'them', 'their', 'this', 'these', 'those', 'about']);
    
    return new Set(
        text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .split(/\s+/)
            .filter(w => w.length > 3 && !stopWords.has(w))
    );
}

function calculateOverlap(set1, set2) {
    if (set1.size === 0 || set2.size === 0) return 0;
    let overlap = 0;
    for (const word of set1) {
        if (set2.has(word)) overlap++;
    }
    const minSize = Math.min(set1.size, set2.size);
    return overlap / minSize;
}

analyzeAlignment().catch(console.error);

