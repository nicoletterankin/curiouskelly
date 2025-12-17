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
 * Convert phases format (v5.0) to atoms format for compatibility
 */
function convertPhasesToAtoms(phases: Record<string, any>, dayNumber: number): LessonAtom[] {
  const atoms: LessonAtom[] = [];
  
  for (const [phaseName, phase] of Object.entries(phases)) {
    if (!phase || typeof phase !== 'object') continue;
    
    atoms.push({
      id: `static-${dayNumber}-${phaseName}`,
      core_lesson_id: `static-${dayNumber}`,
      archetype: 'The Explorer',
      phase: phaseName,
      content: {
        script: phase.talk?.script || phase.script || '',
        options: phase.options?.map((opt: any) => ({
          text: opt.text,
          letter: opt.letter,
          quality: opt.quality || 'good',
          response: phase.responses?.[opt.letter]?.script || ''
        })),
        kellyPose: phase.talk?.kellyPose,
        kellyEmotion: phase.talk?.kellyEmotion,
      }
    });
  }
  
  return atoms;
}

/**
 * Get the path to a static lesson file
 */
function getLessonFilePath(dayNumber: number): string {
  const paddedDay = dayNumber.toString().padStart(3, '0');
  // In Vercel, public/ is at the root
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
 * Load a lesson from static files
 * Returns null if not found (caller should fall back to Supabase or emergency data)
 */
export function loadStaticLesson(dayNumber: number): StaticLessonPack | null {
  // Check cache first
  if (lessonCache.has(dayNumber)) {
    return lessonCache.get(dayNumber)!;
  }
  
  const filePath = getLessonFilePath(dayNumber);
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    // Try non-padded version (some files might be day-17 vs day-017)
    const altPath = path.join(process.cwd(), 'public', 'data', `day-${dayNumber}-complete.js`);
    if (!fs.existsSync(altPath)) {
      return null;
    }
    // Use alternative path
    try {
      const content = fs.readFileSync(altPath, 'utf-8');
      const parsed = parseLessonFile(content, dayNumber);
      if (parsed) {
        lessonCache.set(dayNumber, parsed);
      }
      return parsed;
    } catch (e) {
      console.warn(`[static-lessons] Failed to read ${altPath}:`, e);
      return null;
    }
  }
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const parsed = parseLessonFile(content, dayNumber);
    if (parsed) {
      lessonCache.set(dayNumber, parsed);
    }
    return parsed;
  } catch (e) {
    console.warn(`[static-lessons] Failed to read ${filePath}:`, e);
    return null;
  }
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
