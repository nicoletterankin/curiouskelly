/**
 * Database access layer for lessons
 * Queries core_lessons_v2, lesson_atoms, lesson_scripts
 */
import { getPool } from './connection';

export interface LessonPhase {
  phase: number;
  name: string;
  script: string;
  option: number;
  word_count: number;
  duration_seconds: number;
  audio_url: string | null;
  video_url: string | null;
}

export interface FullLesson {
  day_number: number;
  title: string;
  subject: string;
  learning_objective: string;
  category: string;
  difficulty: string;
  phases: LessonPhase[];
  seed_data: Record<string, any>;
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

export async function getLessonByDay(
  dayNumber: number,
  ageGroup: string = 'adult',
  language: string = 'en'
): Promise<FullLesson | null> {
  const pool = getPool();

  // Get core lesson
  const lessonRes = await pool.query(
    'SELECT * FROM core_lessons_v2 WHERE day_number = $1',
    [dayNumber]
  );
  if (lessonRes.rows.length === 0) return null;
  const lesson = lessonRes.rows[0];

  // Get all phases with scripts
  const phasesRes = await pool.query(
    `SELECT la.phase, la.variant, la.audio_url, la.video_url,
            ls.option_number, ls.content, ls.word_count, ls.duration_seconds
     FROM lesson_atoms la
     LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
     WHERE la.lesson_id = $1 AND la.age_group = $2 AND la.language = $3
     ORDER BY la.phase, ls.option_number`,
    [lesson.id, ageGroup, language]
  );

  const phases: LessonPhase[] = phasesRes.rows.map((r: any) => ({
    phase: r.phase,
    name: PHASE_NAMES[r.phase] || `phase_${r.phase}`,
    script: r.content || '',
    option: r.option_number || 1,
    word_count: r.word_count || 0,
    duration_seconds: r.duration_seconds || 0,
    audio_url: r.audio_url || null,
    video_url: r.video_url || null,
  }));

  return {
    day_number: lesson.day_number,
    title: lesson.title,
    subject: lesson.subject,
    learning_objective: lesson.learning_objective,
    category: lesson.category,
    difficulty: lesson.difficulty,
    phases,
    seed_data: lesson.seed_data || {},
  };
}

export async function getLessonCalendar(): Promise<
  Array<{
    day_number: number;
    title: string;
    subject: string;
    has_scripts: boolean;
    has_audio: boolean;
    has_video: boolean;
  }>
> {
  const pool = getPool();

  const res = await pool.query(`
    SELECT cl.day_number, cl.title, cl.subject,
           COUNT(DISTINCT ls.id) > 0 as has_scripts,
           COUNT(DISTINCT CASE WHEN la.audio_url IS NOT NULL THEN la.id END) > 0 as has_audio,
           COUNT(DISTINCT CASE WHEN la.video_url IS NOT NULL THEN la.id END) > 0 as has_video
    FROM core_lessons_v2 cl
    LEFT JOIN lesson_atoms la ON la.lesson_id = cl.id
    LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
    GROUP BY cl.id, cl.day_number, cl.title, cl.subject
    ORDER BY cl.day_number
  `);

  return res.rows;
}

export async function getScriptCount(): Promise<{ total: number; with_scripts: number; with_audio: number }> {
  const pool = getPool();
  const res = await pool.query(`
    SELECT 
      (SELECT COUNT(*) FROM core_lessons_v2) as total,
      (SELECT COUNT(DISTINCT cl.id) FROM core_lessons_v2 cl 
       JOIN lesson_atoms la ON la.lesson_id = cl.id
       JOIN lesson_scripts ls ON ls.atom_id = la.id) as with_scripts,
      (SELECT COUNT(DISTINCT cl.id) FROM core_lessons_v2 cl
       JOIN lesson_atoms la ON la.lesson_id = cl.id
       WHERE la.audio_url IS NOT NULL) as with_audio
  `);
  return res.rows[0];
}
