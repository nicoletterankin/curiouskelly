/**
 * Fix Hallucinated Resources
 * 
 * This script:
 * 1. Analyzes all books/videos in the database
 * 2. Identifies duplicates (same ISBN/URL used for different titles = hallucination)
 * 3. Removes hallucinated entries, keeping only verified unique ones
 * 4. Updates the database with clean data
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

if (!SUPABASE_SERVICE_KEY) {
    console.error('❌ Missing SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY');
    console.log('Set it with: $env:SUPABASE_SERVICE_ROLE_KEY = "your-key"');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

async function analyzeAndFix() {
    console.log('\n🔬 HALLUCINATION FIX SCRIPT');
    console.log('============================\n');

    // Fetch all lessons
    const { data: lessons, error } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, recommended_books, recommended_videos')
        .order('day_number');

    if (error) {
        console.error('❌ Failed to fetch lessons:', error);
        return;
    }

    console.log(`📚 Loaded ${lessons.length} lessons\n`);

    // STEP 1: Analyze books
    console.log('📖 ANALYZING BOOKS...');
    const isbnUsage = new Map(); // isbn -> [{title, dayNumber, lessonId}]
    
    lessons.forEach(lesson => {
        if (lesson.recommended_books && Array.isArray(lesson.recommended_books)) {
            lesson.recommended_books.forEach(book => {
                if (book.isbn) {
                    if (!isbnUsage.has(book.isbn)) {
                        isbnUsage.set(book.isbn, []);
                    }
                    isbnUsage.get(book.isbn).push({
                        title: book.title,
                        author: book.author,
                        dayNumber: lesson.day_number,
                        lessonId: lesson.id,
                        topic: lesson.topic
                    });
                }
            });
        }
    });

    // Find duplicate ISBNs (same ISBN, different titles)
    const duplicateIsbns = new Set();
    isbnUsage.forEach((uses, isbn) => {
        const uniqueTitles = new Set(uses.map(u => u.title));
        if (uniqueTitles.size > 1) {
            duplicateIsbns.add(isbn);
        }
    });

    console.log(`   Total unique ISBNs: ${isbnUsage.size}`);
    console.log(`   ISBNs with multiple titles (hallucinated): ${duplicateIsbns.size}`);

    // STEP 2: Analyze videos
    console.log('\n🎬 ANALYZING VIDEOS...');
    const urlUsage = new Map(); // url -> [{title, dayNumber, lessonId}]
    
    lessons.forEach(lesson => {
        if (lesson.recommended_videos && Array.isArray(lesson.recommended_videos)) {
            lesson.recommended_videos.forEach(video => {
                if (video.url) {
                    if (!urlUsage.has(video.url)) {
                        urlUsage.set(video.url, []);
                    }
                    urlUsage.get(video.url).push({
                        title: video.title,
                        dayNumber: lesson.day_number,
                        lessonId: lesson.id,
                        topic: lesson.topic
                    });
                }
            });
        }
    });

    // Find duplicate URLs (same URL, different titles)
    const duplicateUrls = new Set();
    urlUsage.forEach((uses, url) => {
        const uniqueTitles = new Set(uses.map(u => u.title));
        if (uniqueTitles.size > 1) {
            duplicateUrls.add(url);
        }
    });

    console.log(`   Total unique URLs: ${urlUsage.size}`);
    console.log(`   URLs with multiple titles (hallucinated): ${duplicateUrls.size}`);

    // STEP 3: Clean up the data
    console.log('\n🧹 CLEANING DATA...');
    
    let booksRemoved = 0;
    let videosRemoved = 0;
    let lessonsToUpdate = [];

    lessons.forEach(lesson => {
        let modified = false;
        let cleanBooks = [];
        let cleanVideos = [];

        // Clean books - keep only those with unique ISBNs
        if (lesson.recommended_books && Array.isArray(lesson.recommended_books)) {
            lesson.recommended_books.forEach(book => {
                if (book.isbn && !duplicateIsbns.has(book.isbn)) {
                    cleanBooks.push(book);
                } else if (book.isbn) {
                    booksRemoved++;
                    modified = true;
                }
            });
        }

        // Clean videos - keep only those with unique URLs
        if (lesson.recommended_videos && Array.isArray(lesson.recommended_videos)) {
            lesson.recommended_videos.forEach(video => {
                if (video.url && !duplicateUrls.has(video.url)) {
                    cleanVideos.push(video);
                } else if (video.url) {
                    videosRemoved++;
                    modified = true;
                }
            });
        }

        // If books/videos were different before, we need to update
        const originalBookCount = lesson.recommended_books?.length || 0;
        const originalVideoCount = lesson.recommended_videos?.length || 0;
        
        if (cleanBooks.length !== originalBookCount || cleanVideos.length !== originalVideoCount) {
            lessonsToUpdate.push({
                id: lesson.id,
                day_number: lesson.day_number,
                recommended_books: cleanBooks.length > 0 ? cleanBooks : null,
                recommended_videos: cleanVideos.length > 0 ? cleanVideos : null
            });
        }
    });

    console.log(`   Books to remove: ${booksRemoved}`);
    console.log(`   Videos to remove: ${videosRemoved}`);
    console.log(`   Lessons to update: ${lessonsToUpdate.length}`);

    // STEP 4: Preview changes
    console.log('\n📋 PREVIEW (first 5 affected lessons):');
    lessonsToUpdate.slice(0, 5).forEach(lesson => {
        console.log(`   Day ${lesson.day_number}: ${lesson.recommended_books?.length || 0} books, ${lesson.recommended_videos?.length || 0} videos remaining`);
    });

    // STEP 5: Apply changes
    if (lessonsToUpdate.length === 0) {
        console.log('\n✅ No changes needed - data is already clean!');
        return;
    }

    console.log('\n⚡ APPLYING CHANGES...');
    
    let successCount = 0;
    let errorCount = 0;

    // Process in batches of 50
    const batchSize = 50;
    for (let i = 0; i < lessonsToUpdate.length; i += batchSize) {
        const batch = lessonsToUpdate.slice(i, i + batchSize);
        
        for (const lesson of batch) {
            const { error: updateError } = await supabase
                .from('core_lessons')
                .update({
                    recommended_books: lesson.recommended_books,
                    recommended_videos: lesson.recommended_videos,
                    updated_at: new Date().toISOString()
                })
                .eq('id', lesson.id);

            if (updateError) {
                console.error(`   ❌ Day ${lesson.day_number}: ${updateError.message}`);
                errorCount++;
            } else {
                successCount++;
            }
        }

        // Progress indicator
        const progress = Math.min(i + batchSize, lessonsToUpdate.length);
        process.stdout.write(`\r   Updated ${progress}/${lessonsToUpdate.length} lessons...`);
    }

    console.log('\n');
    console.log('============================');
    console.log('🎉 CLEANUP COMPLETE!');
    console.log('============================');
    console.log(`   ✅ Successfully updated: ${successCount} lessons`);
    console.log(`   ❌ Errors: ${errorCount}`);
    console.log(`   📚 Books removed: ${booksRemoved}`);
    console.log(`   🎬 Videos removed: ${videosRemoved}`);

    // STEP 6: Verify
    console.log('\n🔍 VERIFYING RESULTS...');
    await verifyCleanup();
}

async function verifyCleanup() {
    const { data: lessons } = await supabase
        .from('core_lessons')
        .select('day_number, recommended_books, recommended_videos');

    // Re-analyze
    const isbnUsage = new Map();
    const urlUsage = new Map();
    let totalBooks = 0;
    let totalVideos = 0;
    let lessonsWithBooks = 0;
    let lessonsWithVideos = 0;

    lessons.forEach(lesson => {
        if (lesson.recommended_books && lesson.recommended_books.length > 0) {
            lessonsWithBooks++;
            lesson.recommended_books.forEach(book => {
                totalBooks++;
                if (book.isbn) {
                    if (!isbnUsage.has(book.isbn)) isbnUsage.set(book.isbn, new Set());
                    isbnUsage.get(book.isbn).add(book.title);
                }
            });
        }
        if (lesson.recommended_videos && lesson.recommended_videos.length > 0) {
            lessonsWithVideos++;
            lesson.recommended_videos.forEach(video => {
                totalVideos++;
                if (video.url) {
                    if (!urlUsage.has(video.url)) urlUsage.set(video.url, new Set());
                    urlUsage.get(video.url).add(video.title);
                }
            });
        }
    });

    // Count remaining duplicates
    let duplicateIsbns = 0;
    let duplicateUrls = 0;
    isbnUsage.forEach((titles) => { if (titles.size > 1) duplicateIsbns++; });
    urlUsage.forEach((titles) => { if (titles.size > 1) duplicateUrls++; });

    console.log('\n📊 POST-CLEANUP STATS:');
    console.log(`   Total books remaining: ${totalBooks}`);
    console.log(`   Lessons with books: ${lessonsWithBooks}/365`);
    console.log(`   Duplicate ISBNs remaining: ${duplicateIsbns}`);
    console.log(`   Total videos remaining: ${totalVideos}`);
    console.log(`   Lessons with videos: ${lessonsWithVideos}/365`);
    console.log(`   Duplicate URLs remaining: ${duplicateUrls}`);

    if (duplicateIsbns === 0 && duplicateUrls === 0) {
        console.log('\n✅ ALL HALLUCINATIONS REMOVED! Data is now clean.');
    } else {
        console.log('\n⚠️ Some duplicates still remain. May need another pass.');
    }
}

// Run the fix
analyzeAndFix().catch(console.error);



