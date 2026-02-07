/**
 * Sprint F: Lesson Loader
 * Loads a complete lesson for the player, with fallbacks
 * Input: day_number, age_group, language
 * Output: Complete lesson object with all 7 phases
 */

interface LessonPhaseData {
  phase: number;
  name: string;
  options: Array<{
    option: number;
    script: string;
    word_count: number;
    duration_seconds: number;
  }>;
  audio_url: string | null;
  video_url: string | null;
}

export interface LessonPayload {
  day_number: number;
  title: string;
  subject: string;
  learning_objective: string;
  category: string;
  difficulty: string;
  phases: LessonPhaseData[];
  metadata: {
    age_group: string;
    language: string;
    loaded_at: string;
    source: 'cache' | 'database';
    completeness: {
      total_phases: number;
      with_scripts: number;
      with_audio: number;
      with_video: number;
    };
  };
}

const PHASE_NAMES: Record<number, string> = {
  1: 'hook',
  2: 'teach',
  3: 'example',
  4: 'practice',
  5: 'reflect',
  6: 'apply',
  7: 'close',
};

const DEFAULT_SCRIPTS: Record<string, string> = {
  hook: "Welcome back to today's lesson! I have something really exciting to share with you.",
  teach: "Let me tell you about something fascinating that will change how you see the world.",
  example: "Here's a real-world example that brings this concept to life.",
  practice: "Now it's your turn! Try this thought experiment.",
  reflect: "Take a moment to think about what you've learned today.",
  apply: "Here's how you can use this knowledge in your daily life.",
  close: "Remember, every day is an opportunity to learn something new. See you tomorrow!",
};

/**
 * Ensures a lesson has all 7 phases with valid data
 * Fills gaps with defaults if necessary
 */
function ensureMvpLessonShape(lesson: any, ageGroup: string, language: string): LessonPayload {
  const phases: LessonPhaseData[] = [];
  
  for (let p = 1; p <= 7; p++) {
    const phaseName = PHASE_NAMES[p];
    const existingPhases = (lesson.phases || []).filter((ph: any) => ph.phase === p);
    
    const options = [];
    
    // Group by option number
    const opt1 = existingPhases.find((ph: any) => ph.option === 1);
    const opt2 = existingPhases.find((ph: any) => ph.option === 2);
    
    options.push({
      option: 1,
      script: opt1?.script || DEFAULT_SCRIPTS[phaseName] || DEFAULT_SCRIPTS.hook,
      word_count: opt1?.word_count || 0,
      duration_seconds: opt1?.duration_seconds || 15,
    });
    
    if (opt2?.script) {
      options.push({
        option: 2,
        script: opt2.script,
        word_count: opt2.word_count || 0,
        duration_seconds: opt2.duration_seconds || 15,
      });
    }
    
    phases.push({
      phase: p,
      name: phaseName,
      options,
      audio_url: opt1?.audio_url || null,
      video_url: opt1?.video_url || null,
    });
  }
  
  const withScripts = phases.filter(p => p.options[0].script !== DEFAULT_SCRIPTS[p.name]).length;
  const withAudio = phases.filter(p => p.audio_url).length;
  const withVideo = phases.filter(p => p.video_url).length;
  
  return {
    day_number: lesson.day_number,
    title: lesson.title || `Lesson ${lesson.day_number}`,
    subject: lesson.subject || 'general',
    learning_objective: lesson.learning_objective || '',
    category: lesson.category || 'general',
    difficulty: lesson.difficulty || 'beginner',
    phases,
    metadata: {
      age_group: ageGroup,
      language,
      loaded_at: new Date().toISOString(),
      source: 'database',
      completeness: {
        total_phases: 7,
        with_scripts: withScripts,
        with_audio: withAudio,
        with_video: withVideo,
      },
    },
  };
}

/**
 * Load a lesson by day number
 * Queries the database and fills gaps with defaults
 */
export async function loadLesson(
  dayNumber: number,
  ageGroup: string = 'adult',
  language: string = 'en'
): Promise<LessonPayload> {
  // Dynamic import to avoid module issues
  const { getLessonByDay } = await import('../db/lessons');
  
  const lesson = await getLessonByDay(dayNumber, ageGroup, language);
  
  if (!lesson) {
    // Return a minimal lesson with defaults
    return ensureMvpLessonShape(
      { day_number: dayNumber, title: `Lesson ${dayNumber}`, phases: [] },
      ageGroup,
      language
    );
  }
  
  return ensureMvpLessonShape(lesson, ageGroup, language);
}

/**
 * Get today's lesson based on day of year
 */
export async function loadTodayLesson(
  ageGroup: string = 'adult',
  language: string = 'en'
): Promise<LessonPayload> {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  const dayOfYear = Math.floor(diff / oneDay);
  const lessonDay = ((dayOfYear - 1) % 365) + 1; // 1-365
  
  return loadLesson(lessonDay, ageGroup, language);
}

export { ensureMvpLessonShape };
