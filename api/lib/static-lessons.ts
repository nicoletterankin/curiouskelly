/**
 * Static Lessons Loader
 * 
 * Reads lesson data from pre-generated static JS files in /public/data/
 * This eliminates Supabase dependency for lesson content delivery.
 * 
 * Benefits:
 * - Zero database queries for lesson content
 * - Works during Supabase outages
 * - Faster cold starts (no DB connection)
 * - Lower costs (no DB egress)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as vm from 'vm';

interface LessonMeta {
  created_at: string;
  day_number: number;
  version: string;
  is_launch_day?: boolean;
}

interface LessonCore {
  day_number: number;
  topic: string;
  headline: string;
  universal_truth: string;
  emoji?: string;
  category?: string;
  thumbnail_url?: string;
  kelly_images?: Record<string, string>;
}

interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;
  phase: string;
  content: {
    script: string;
    options?: Array<{
      text: string;
      letter: string;
      quality: string;
      response: string;
    }>;
    kellyPose?: string;
    kellyEmotion?: string;
    visual_cue?: string;
  };
  created_at?: string;
  visual_url?: string;
}

export interface StaticLessonPack {
  meta: LessonMeta;
  lesson: LessonCore;
  atoms: LessonAtom[];
}

// Cache loaded lessons in memory (within single request lifecycle for serverless)
const lessonCache = new Map<number, StaticLessonPack>();

/**
 * Extract text from i18n object (supports { en: "...", es: "..." } format)
 */
function extractI18n(value: any, lang = 'en'): string {
  if (!value) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'object') return value[lang] || value.en || '';
  return '';
}

/**
 * Convert phases format (v5.0 with i18n) to atoms format for compatibility
 */
function convertPhasesToAtoms(phases: Record<string, any>, dayNumber: number, lang = 'en'): LessonAtom[] {
  const atoms: LessonAtom[] = [];
  
  for (const [phaseName, phase] of Object.entries(phases)) {
    if (!phase || typeof phase !== 'object') continue;
    
    // Handle i18n script format: { script: { en: "...", es: "..." } }
    const script = extractI18n(phase.talk?.script || phase.script, lang);
    const prompt = extractI18n(phase.prompt, lang);
    
    atoms.push({
      id: `static-${dayNumber}-${phaseName}`,
      core_lesson_id: `static-${dayNumber}`,
      archetype: 'The Explorer',
      phase: phaseName,
      content: {
        script,
        options: phase.options?.map((opt: any) => ({
          text: extractI18n(opt.text, lang),
          letter: opt.letter,
          quality: opt.quality || 'good',
          response: extractI18n(opt.response || phase.responses?.[opt.letter]?.script, lang)
        })),
        kellyPose: phase.talk?.kellyPose,
        kellyEmotion: phase.talk?.kellyEmotion,
        prompt,
      }
    });
  }
  
  return atoms;
}

/**
 * Get the path to a static lesson JSON file
 */
function getLessonJsonPath(dayNumber: number): string {
  // Try multiple naming patterns
  return path.join(process.cwd(), 'public', 'lessons', `day-${dayNumber}.json`);
}

/**
 * Get the path to a static lesson file (legacy JS format)
 */
function getLessonFilePath(dayNumber: number): string {
  const paddedDay = dayNumber.toString().padStart(3, '0');
  return path.join(process.cwd(), 'public', 'data', `day-${paddedDay}-complete.js`);
}

/**
 * Parse a static lesson JS file
 * These files set window.CURIOUS_KELLY.DAY_XXX = { ... }
 */
function parseLessonFile(content: string, dayNumber: number): StaticLessonPack | null {
  try {
    // Create a sandbox to safely extract the data
    const sandbox = {
      window: {
        CURIOUS_KELLY: {
          LOCAL_PACKS: {}
        }
      }
    };
    
    // Execute the JS to populate the sandbox
    vm.runInNewContext(content, sandbox);
    
    // Try to find the lesson data (check both padded and unpadded keys)
    const paddedKey = `DAY_${dayNumber.toString().padStart(3, '0')}`;
    const unPaddedKey = `DAY_${dayNumber}`;
    const lessonData = (sandbox.window.CURIOUS_KELLY as any)[paddedKey] 
                    || (sandbox.window.CURIOUS_KELLY as any)[unPaddedKey];
    
    if (lessonData && lessonData.lesson) {
      // Convert phases format to atoms format if needed
      if (lessonData.phases && !lessonData.atoms) {
        lessonData.atoms = convertPhasesToAtoms(lessonData.phases, dayNumber);
      }
      return lessonData as StaticLessonPack;
    }
    
    return null;
  } catch (e) {
    console.warn(`[static-lessons] Failed to parse day ${dayNumber}:`, e);
    return null;
  }
}

/**
 * Parse JSON lesson file (v5.0 i18n format from public/lessons/)
 */
function parseJsonLessonFile(content: string, dayNumber: number): StaticLessonPack | null {
  try {
    const data = JSON.parse(content);
    
    // Build the lesson core
    const lesson: LessonCore = {
      day_number: dayNumber,
      topic: extractI18n(data.meta?.topic || data.topic, 'en'),
      headline: extractI18n(data.headline, 'en'),
      universal_truth: extractI18n(data.universal_truth, 'en'),
      emoji: data.meta?.emoji || data.emoji,
      category: data.meta?.category,
      thumbnail_url: `/generated-visuals/day-${dayNumber.toString().padStart(3, '0')}/thumbnail.png`,
    };
    
    // Convert phases to atoms
    const atoms = data.phases ? convertPhasesToAtoms(data.phases, dayNumber, 'en') : [];
    
    return {
      meta: {
        created_at: new Date().toISOString(),
        day_number: dayNumber,
        version: data.meta?.version || 'v5.0',
      },
      lesson,
      atoms,
    };
  } catch (e) {
    console.warn(`[static-lessons] Failed to parse JSON for day ${dayNumber}:`, e);
    return null;
  }
}

/**
 * Load a lesson from static files
 * Returns null if not found (caller should fall back to Supabase or emergency data)
 */
export function loadStaticLesson(dayNumber: number): StaticLessonPack | null {
  // Check cache first
  if (lessonCache.has(dayNumber)) {
    return lessonCache.get(dayNumber)!;
  }
  
  // PRIORITY 1: Try JSON files in public/lessons/ (the actual content location)
  const jsonPath = getLessonJsonPath(dayNumber);
  if (fs.existsSync(jsonPath)) {
    try {
      const content = fs.readFileSync(jsonPath, 'utf-8');
      const parsed = parseJsonLessonFile(content, dayNumber);
      if (parsed && parsed.atoms.length > 0) {
        lessonCache.set(dayNumber, parsed);
        return parsed;
      }
    } catch (e) {
      console.warn(`[static-lessons] Failed to read JSON ${jsonPath}:`, e);
    }
  }
  
  // PRIORITY 2: Try legacy JS files in public/data/
  const filePath = getLessonFilePath(dayNumber);
  if (fs.existsSync(filePath)) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const parsed = parseLessonFile(content, dayNumber);
      if (parsed) {
        lessonCache.set(dayNumber, parsed);
        return parsed;
      }
    } catch (e) {
      console.warn(`[static-lessons] Failed to read ${filePath}:`, e);
    }
  }
  
  // Try non-padded JS version
  const altPath = path.join(process.cwd(), 'public', 'data', `day-${dayNumber}-complete.js`);
  if (fs.existsSync(altPath)) {
    try {
      const content = fs.readFileSync(altPath, 'utf-8');
      const parsed = parseLessonFile(content, dayNumber);
      if (parsed) {
        lessonCache.set(dayNumber, parsed);
      }
      return parsed;
    } catch (e) {
      console.warn(`[static-lessons] Failed to read ${altPath}:`, e);
    }
  }
  
  return null;
}

/**
 * Check if a static lesson exists without loading it
 */
export function staticLessonExists(dayNumber: number): boolean {
  const paddedPath = getLessonFilePath(dayNumber);
  const plainPath = path.join(process.cwd(), 'public', 'data', `day-${dayNumber}-complete.js`);
  return fs.existsSync(paddedPath) || fs.existsSync(plainPath);
}

/**
 * Convert static lesson format to API response format
 * (matches what the existing API returns from Supabase)
 */
export function formatLessonForApi(pack: StaticLessonPack, archetype?: string, ageBucket?: string) {
  return {
    source: 'static-file',
    lesson: {
      id: `static-${pack.meta.day_number}`,
      day_number: pack.meta.day_number,
      topic: pack.lesson.topic,
      universal_truth: pack.lesson.universal_truth,
      marketing_headline: pack.lesson.headline,
      marketing_tagline: pack.lesson.category || 'Learn something new',
      thumbnail_url: pack.lesson.thumbnail_url,
      kelly_images: pack.lesson.kelly_images,
      emoji: pack.lesson.emoji,
    },
    atoms: pack.atoms.filter(a => 
      !archetype || a.archetype === archetype || a.archetype === 'The Explorer'
    ),
    shards: [], // Static files don't have shards (age variants are in atoms)
    dayNumber: pack.meta.day_number,
    archetype: archetype || 'The Explorer',
    ageBucket: ageBucket || 'adult',
  };
}
