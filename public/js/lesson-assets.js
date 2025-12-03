/**
 * ═══════════════════════════════════════════════════════════════════════════
 * LESSON ASSETS SYSTEM
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * The core engine for lesson asset management. This system ensures:
 * 1. Assets are generated ONCE and cached forever
 * 2. Students never regenerate content that already exists
 * 3. Every phase has defined asset slots with fallback chains
 * 4. Variants (age, language, archetype, tone) are tracked and reused
 * 
 * Architecture:
 * - Local files in /kelly/ serve as the primary asset store
 * - Supabase tracks metadata and variant cache status
 * - lesson_assets table stores all generated content
 * - lesson_variant_cache tracks what's been generated per variant
 * 
 * Created: December 3, 2025
 * ═══════════════════════════════════════════════════════════════════════════
 */

class LessonAssetManager {
    constructor(supabaseClient) {
        this.supabase = supabaseClient;
        this.cache = new Map();
        this.preloadQueue = [];
        this.thumbnailSlugMap = new Map(); // day_number -> slug
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ASSET TYPES & PHASES
    // ═══════════════════════════════════════════════════════════════════════

    static PHASES = {
        // Lesson flow phases
        HERO: 'hero',           // Thumbnail/card image
        INTRO: 'intro',         // Kelly welcomes learner
        Q1: 'q1',               // Question 1 pose
        Q2: 'q2',               // Question 2 pose
        Q3: 'q3',               // Question 3 pose
        HOOK: 'hook',           // The surprising reveal
        WISDOM: 'wisdom',       // Final insight
        
        // Reaction states
        CORRECT: 'correct',     // Learner chose correct
        INCORRECT: 'incorrect', // Learner chose different
        THINKING: 'thinking',   // Kelly pondering
        EXCITED: 'excited',     // Discovery moment
        ENCOURAGING: 'encouraging', // Support after wrong answer
    };

    static ASSET_TYPES = {
        THUMBNAIL: 'thumbnail',
        PHASE_IMAGE: 'phase_image',
        AUDIO: 'audio',
        VIDEO: 'video',
        ANIMATION: 'animation'
    };

    static AGE_BUCKETS = {
        TODDLER: 'toddler',     // 2-4
        CHILD: 'child',         // 5-9
        TWEEN: 'tween',         // 10-12
        TEEN: 'teen',           // 13-17
        YOUNG_ADULT: 'young_adult', // 18-25
        ADULT: 'adult',         // 26-59
        SENIOR: 'senior'        // 60+
    };

    // ═══════════════════════════════════════════════════════════════════════
    // THUMBNAIL METHODS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get thumbnail URL for a lesson
     * Uses the canonical thumbnail_slug from the database
     */
    getThumbnailUrl(lesson) {
        const dayStr = String(lesson.day_number).padStart(3, '0');
        
        // Use cached slug if available
        if (lesson.thumbnail_slug) {
            return `/kelly/thumbnails/raw/lesson-${dayStr}-${lesson.thumbnail_slug}.png`;
        }
        
        // Check local cache
        if (this.thumbnailSlugMap.has(lesson.day_number)) {
            const slug = this.thumbnailSlugMap.get(lesson.day_number);
            return `/kelly/thumbnails/raw/lesson-${dayStr}-${slug}.png`;
        }
        
        // Fallback to per-lesson folder
        return `/kelly/lessons/${dayStr}/lesson-${lesson.day_number}-hero.png`;
    }

    /**
     * Preload thumbnail slugs from database
     */
    async loadThumbnailSlugs() {
        try {
            const { data, error } = await this.supabase
                .from('core_lessons')
                .select('day_number, thumbnail_slug')
                .not('thumbnail_slug', 'is', null);
            
            if (error) throw error;
            
            data?.forEach(lesson => {
                this.thumbnailSlugMap.set(lesson.day_number, lesson.thumbnail_slug);
            });
            
            console.log(`📸 Loaded ${this.thumbnailSlugMap.size} thumbnail slugs`);
            return this.thumbnailSlugMap;
        } catch (err) {
            console.error('Error loading thumbnail slugs:', err);
            return this.thumbnailSlugMap;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE IMAGE METHODS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get image URL for a specific lesson phase
     * Follows fallback chain: lesson-specific → base pose → default
     */
    getPhaseImageUrl(dayNumber, phase, options = {}) {
        const dayStr = String(dayNumber).padStart(3, '0');
        const { archetype, age, language } = options;
        
        // 1. Try lesson-specific phase image
        const lessonPath = `/kelly/lessons/${dayStr}/lesson-${dayNumber}-${phase}.png`;
        
        // 2. Fallback to base poses
        const basePoseMap = {
            [LessonAssetManager.PHASES.Q1]: 'kelly_thinking.png',
            [LessonAssetManager.PHASES.Q2]: 'kelly_curious.png',
            [LessonAssetManager.PHASES.Q3]: 'kelly_thinking.png',
            [LessonAssetManager.PHASES.INTRO]: 'kelly_welcome.png',
            [LessonAssetManager.PHASES.HOOK]: 'kelly_excited.png',
            [LessonAssetManager.PHASES.WISDOM]: 'kelly_clasp.png',
            [LessonAssetManager.PHASES.CORRECT]: 'kelly_excited.png',
            [LessonAssetManager.PHASES.INCORRECT]: 'kelly_encouraging.png',
            [LessonAssetManager.PHASES.THINKING]: 'kelly_thinking.png',
        };
        
        const basePose = basePoseMap[phase] || 'kelly_idle.png';
        const basePath = `/kelly/poses/${basePose}`;
        
        return {
            primary: lessonPath,
            fallback: basePath,
            phase
        };
    }

    /**
     * Get all images needed for a lesson (for preloading)
     */
    async getFullLessonAssetManifest(dayNumber) {
        const dayStr = String(dayNumber).padStart(3, '0');
        const phases = Object.values(LessonAssetManager.PHASES);
        
        const manifest = {
            dayNumber,
            thumbnail: this.getThumbnailUrl({ day_number: dayNumber, thumbnail_slug: this.thumbnailSlugMap.get(dayNumber) }),
            phases: {},
            reactions: {},
            audio: {}
        };
        
        // Phase images
        phases.forEach(phase => {
            manifest.phases[phase] = this.getPhaseImageUrl(dayNumber, phase);
        });
        
        // Reactions (shared across lessons)
        manifest.reactions = {
            choice_left: '/kelly/choices/choice_left.png',
            choice_right: '/kelly/choices/choice_right.png',
            pointing_left: '/kelly/poses/kelly_choice_left.png',
            pointing_right: '/kelly/poses/kelly_choice_right.png',
        };
        
        return manifest;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VARIANT CACHE METHODS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Check if a variant has been generated
     * If not, queue it for generation
     */
    async checkVariantCache(dayNumber, phase, options = {}) {
        const { language = 'en', age = 25, archetype = null, tone = 'curious' } = options;
        const ageBucket = this.ageToAgeBucket(age);
        
        try {
            const { data, error } = await this.supabase
                .rpc('get_or_create_variant_cache', {
                    p_lesson_day: dayNumber,
                    p_phase: phase,
                    p_language: language,
                    p_age_bucket: ageBucket,
                    p_archetype: archetype,
                    p_tone: tone
                });
            
            if (error) throw error;
            
            return {
                cacheId: data?.[0]?.cache_id,
                isComplete: data?.[0]?.is_complete || false,
                assetsReady: data?.[0]?.assets_ready || {},
                needsGeneration: data?.[0]?.needs_generation || true
            };
        } catch (err) {
            console.error('Error checking variant cache:', err);
            // Return safe defaults - assume cache miss
            return {
                cacheId: null,
                isComplete: false,
                assetsReady: {},
                needsGeneration: true
            };
        }
    }

    /**
     * Convert numeric age to age bucket
     */
    ageToAgeBucket(age) {
        if (age <= 4) return LessonAssetManager.AGE_BUCKETS.TODDLER;
        if (age <= 9) return LessonAssetManager.AGE_BUCKETS.CHILD;
        if (age <= 12) return LessonAssetManager.AGE_BUCKETS.TWEEN;
        if (age <= 17) return LessonAssetManager.AGE_BUCKETS.TEEN;
        if (age <= 25) return LessonAssetManager.AGE_BUCKETS.YOUNG_ADULT;
        if (age <= 59) return LessonAssetManager.AGE_BUCKETS.ADULT;
        return LessonAssetManager.AGE_BUCKETS.SENIOR;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRELOADING
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Preload all images for a lesson
     */
    async preloadLessonAssets(dayNumber, options = {}) {
        const manifest = await this.getFullLessonAssetManifest(dayNumber);
        const urls = [];
        
        // Collect all URLs
        urls.push(manifest.thumbnail);
        Object.values(manifest.phases).forEach(phase => {
            urls.push(phase.primary);
            urls.push(phase.fallback);
        });
        Object.values(manifest.reactions).forEach(url => urls.push(url));
        
        // Preload all
        const results = await Promise.allSettled(
            urls.map(url => this.preloadImage(url))
        );
        
        const successful = results.filter(r => r.status === 'fulfilled').length;
        console.log(`📦 Preloaded ${successful}/${urls.length} assets for day ${dayNumber}`);
        
        return manifest;
    }

    /**
     * Preload a single image
     */
    preloadImage(url) {
        return new Promise((resolve, reject) => {
            if (this.cache.has(url)) {
                resolve(this.cache.get(url));
                return;
            }
            
            const img = new Image();
            img.onload = () => {
                this.cache.set(url, true);
                resolve(url);
            };
            img.onerror = () => {
                this.cache.set(url, false);
                reject(new Error(`Failed to load: ${url}`));
            };
            img.src = url;
        });
    }

    /**
     * Check if an image exists (was successfully preloaded)
     */
    hasImage(url) {
        return this.cache.get(url) === true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROMPT LIBRARY FOR IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_PROMPT_LIBRARY = {
    // Base character description (immutable)
    character: `A warm, intelligent woman in her late 20s named Kelly. Oval face with soft features, warm brown expressive eyes, natural well-groomed eyebrows, straight proportional nose, natural pink lips in a genuine warm smile. Medium to light brown hair with subtle caramel highlights, long soft waves past shoulders. Warm olive Mediterranean complexion. Wearing a comfortable light blue crewneck sweater. Seated in a vintage Hollywood director's chair with wood frame and black canvas, in a bright clean studio with white/light gray background, soft natural light from camera-right.`,
    
    // Phase-specific prompts
    phases: {
        hero: (topic) => `Kelly introducing today's lesson about "${topic}". She has a curious, inviting expression - ready to explore this fascinating topic together.`,
        
        intro: (topic) => `Kelly warmly welcomes the learner. Her face shows genuine excitement - eyes bright, warm smile. She gestures openly as if saying "I'm so glad you're here for today's lesson about ${topic}!"`,
        
        q1: (topic) => `Kelly presents the first question about "${topic}". Her expression is curious and encouraging - head tilted slightly, one eyebrow raised, gentle smile that says "I believe in you."`,
        
        q2: (topic) => `Kelly asks the second question. She leans forward slightly, genuinely interested in what the learner will choose. Her posture is open, hands perhaps gesturing toward two options.`,
        
        q3: (topic) => `Kelly presents the final question. Her expression is thoughtful yet supportive - she knows this might be the trickiest question but trusts the learner.`,
        
        hook: (topic) => `Kelly reveals the surprising insight about "${topic}". Her eyes are wide with excitement, mouth slightly open in delighted surprise - the "aha!" moment of discovery.`,
        
        wisdom: (topic) => `Kelly shares the final wisdom about "${topic}". Her expression is warm and contemplative - the look of a teacher who has just imparted something meaningful she hopes will stay with the learner forever.`,
        
        correct: () => `Kelly's face lights up with genuine pride and joy. Not exaggerated, but the authentic happiness of a teacher watching a student succeed. She might give a subtle thumbs up.`,
        
        incorrect: () => `Kelly responds warmly to a different answer. Her expression is understanding and encouraging - no judgment, just gentle support. The kind of look that says "That's an interesting perspective, let's explore it together."`,
    },
    
    // Quality modifiers
    quality: `Professional photography, 8k resolution, sharp focus, natural skin texture, authentic expression, studio lighting, clean composition.`,
    
    // Negative prompt
    negative: `cartoon, anime, illustration, painting, drawing, sketch, 3D render, CGI, plastic, doll-like, uncanny valley, harsh lighting, dark shadows, busy background, text, watermarks, logos, different clothing, different hair color, masculine features, uncomfortable expression, forced smile`
};

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LessonAssetManager, KELLY_PROMPT_LIBRARY };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.LessonAssetManager = LessonAssetManager;
    window.KELLY_PROMPT_LIBRARY = KELLY_PROMPT_LIBRARY;
}


