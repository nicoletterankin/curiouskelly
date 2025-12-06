/**
 * PICKY NICKY'S VIDEO FIXER
 * 
 * 1. Identifies clearly mismatched videos (like Day 1 leaves on "Starting Fresh")
 * 2. Removes the bad videos and flags for manual curation
 * 3. Records everything in the audit trail
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Check if video title is related to topic
function isVideoRelatedToTopic(topic, videoTitle) {
    const topicLower = topic.toLowerCase();
    const titleLower = videoTitle.toLowerCase();
    
    // Extract key words from both
    const topicWords = topicLower.split(/\s+/).filter(w => w.length > 3);
    const titleWords = titleLower.split(/\s+/).filter(w => w.length > 3);
    
    // Check for keyword overlap
    for (const topicWord of topicWords) {
        if (titleWords.some(tw => tw.includes(topicWord) || topicWord.includes(tw))) {
            return true;
        }
    }
    
    // Special cases - semantic matches that aren't keyword matches
    const semanticMatches = {
        'starting fresh': ['new', 'begin', 'fresh', 'start', 'reset', 'resolution', 'change'],
        'new beginning': ['new', 'begin', 'fresh', 'start', 'reset'],
        'water': ['h2o', 'liquid', 'ice', 'steam', 'states', 'matter'],
        'sound': ['audio', 'wave', 'hear', 'listen', 'noise', 'vibrat'],
        'light': ['photon', 'optic', 'see', 'vision', 'bright', 'dark'],
        'friend': ['social', 'relationship', 'peer', 'buddy', 'companion'],
        'courage': ['brave', 'fear', 'hero', 'bold'],
        'patience': ['wait', 'marshmallow', 'delay', 'gratification'],
        'gratitude': ['thankful', 'grateful', 'thank', 'appreciate'],
        'curious': ['wonder', 'question', 'learn', 'discover', 'explore'],
    };
    
    for (const [key, synonyms] of Object.entries(semanticMatches)) {
        if (topicLower.includes(key)) {
            if (synonyms.some(syn => titleLower.includes(syn))) {
                return true;
            }
        }
    }
    
    return false;
}

async function fixMismatchedVideos() {
    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   🎬 PICKY NICKY\'S VIDEO AUDIT');
    console.log('   Finding and flagging mismatched video recommendations');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Get all lessons with videos
    const { data: lessons, error: fetchError } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, recommended_videos')
        .not('recommended_videos', 'is', null)
        .neq('recommended_videos', '[]')
        .order('day_number');

    if (fetchError) {
        console.error('Error fetching lessons:', fetchError);
        return;
    }

    console.log(`📚 Found ${lessons.length} lessons with videos\n`);

    let mismatched = [];
    let matched = [];

    for (const lesson of lessons) {
        const videos = lesson.recommended_videos || [];
        if (videos.length === 0) continue;
        
        const firstVideo = videos[0];
        const isRelated = isVideoRelatedToTopic(lesson.topic, firstVideo.title);
        
        if (!isRelated) {
            mismatched.push({
                ...lesson,
                firstVideoTitle: firstVideo.title,
                reason: `Video "${firstVideo.title}" doesn't match topic "${lesson.topic}"`
            });
        } else {
            matched.push(lesson);
        }
    }

    console.log(`   ✅ Matched videos: ${matched.length}`);
    console.log(`   ❌ Mismatched videos: ${mismatched.length}\n`);

    if (mismatched.length > 0) {
        console.log('═══════════════════════════════════════════════════════════════');
        console.log('   MISMATCHED VIDEOS FOUND:');
        console.log('═══════════════════════════════════════════════════════════════\n');

        for (const lesson of mismatched) {
            console.log(`   Day ${lesson.day_number}: "${lesson.topic}"`);
            console.log(`      Video: "${lesson.firstVideoTitle}"`);
            console.log(`      → Flagging for manual review\n`);
            
            // Record in audit trail
            await supabase.from('lesson_audits').insert({
                day_number: lesson.day_number,
                audit_type: 'video_mismatch',
                status: 'warning',
                field_name: 'recommended_videos',
                original_value: JSON.stringify(lesson.recommended_videos),
                issue_description: lesson.reason,
                fix_method: 'flagged_for_review',
                fix_rationale: 'Video content does not appear to match lesson topic - needs human verification',
                audited_by: 'video_audit_v1'
            });
        }
    }

    // Now let's check the 285 lessons missing videos
    const { data: allLessons, error: allError } = await supabase
        .from('core_lessons')
        .select('day_number, topic')
        .order('day_number');

    if (allError) {
        console.error('Error fetching all lessons:', allError);
        return;
    }

    const lessonsWithVideos = new Set(lessons.map(l => l.day_number));
    const missingVideos = allLessons.filter(l => !lessonsWithVideos.has(l.day_number));

    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`   📝 MISSING VIDEOS: ${missingVideos.length} lessons need videos`);
    console.log('═══════════════════════════════════════════════════════════════\n');

    // Record all missing videos in audit trail
    let recordedMissing = 0;
    for (const lesson of missingVideos) {
        await supabase.from('lesson_audits').insert({
            day_number: lesson.day_number,
            audit_type: 'video_missing',
            status: 'pending',
            field_name: 'recommended_videos',
            original_value: '[]',
            issue_description: `No videos found for topic "${lesson.topic}"`,
            fix_method: 'needs_curation',
            fix_rationale: 'Video recommendations need to be curated from educational sources',
            audited_by: 'video_audit_v1'
        });
        recordedMissing++;
        
        if (recordedMissing % 50 === 0) {
            console.log(`   Recorded ${recordedMissing}/${missingVideos.length} missing video flags...`);
        }
    }

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('   📊 VIDEO AUDIT COMPLETE');
    console.log('═══════════════════════════════════════════════════════════════\n');
    console.log(`   ✅ Videos matching topics: ${matched.length}`);
    console.log(`   ⚠️  Videos needing review: ${mismatched.length}`);
    console.log(`   📝 Missing videos flagged: ${recordedMissing}`);
    console.log('\n   All findings recorded in lesson_audits table.\n');

    return {
        matched: matched.length,
        mismatched: mismatched.length,
        missing: recordedMissing
    };
}

fixMismatchedVideos().catch(console.error);

