/**
 * Kelly OS — Data Layer
 * Shared data utilities for curriculum, state management, and Supabase integration
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════
const KELLY_CONFIG = {
  SUPABASE_URL: window.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co',
  SUPABASE_ANON_KEY: window.SUPABASE_ANON_KEY ||
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4NjI0NzgsImV4cCI6MjA0OTQzODQ3OH0.qfTs_t0tLmVHFNlKlOqXxvbmEgUEZpHdnVAFbQdJv1c',

  // Stripe
  STRIPE_MONTHLY_PRICE: 'price_monthly_999',
  STRIPE_ANNUAL_PRICE: 'price_annual_7999',

  // Limits
  FREE_LESSONS_PER_DAY: 1,
  GUEST_MAX_LESSONS: 3,

  // Variants
  AGE_GROUPS: ['2-5', '6-12', '13-17', '18-35', '36-60', '61+'],
  LANGUAGES: ['en', 'es', 'fr'],
  TONES: ['curious', 'playful', 'serious'],
  DIFFICULTIES: [2, 3],
  MODES: ['2D', '3D']
};

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL STORAGE KEYS
// ═══════════════════════════════════════════════════════════════════════════════
const STORAGE_KEYS = {
  // User preferences
  AGE: 'kelly_age',
  LANGUAGE: 'kelly_language',
  TONE: 'kelly_tone',
  DIFFICULTY: 'kelly_difficulty',
  MODE: 'kelly_mode',

  // Progress
  STREAK: 'kelly_streak',
  TOTAL_LESSONS: 'kelly_total_lessons',
  COMPLETED_DAYS: 'kelly_completed_days',
  LAST_LESSON_DAY: 'kelly_last_lesson_day',
  LAST_LESSON_DATE: 'kelly_last_lesson_date',

  // Birthday
  BIRTHDAY_MONTH: 'kelly_birthday_month',
  BIRTHDAY_DAY: 'kelly_birthday_day',

  // Session
  CURRENT_LESSON: 'kelly_current_lesson',
  CURRENT_PHASE: 'kelly_current_phase',
  CHOICES_MADE: 'kelly_choices_made',

  // Cache
  CURRICULUM_CACHE: 'kelly_curriculum_cache',
  CURRICULUM_VERSION: 'kelly_curriculum_version'
};

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

let curriculumData = null;

/**
 * Load the 365-day curriculum from JSON file
 * Caches in localStorage for faster subsequent loads
 */
async function loadCurriculum() {
  // Check cache first
  const cachedVersion = localStorage.getItem(STORAGE_KEYS.CURRICULUM_VERSION);
  const cachedData = localStorage.getItem(STORAGE_KEYS.CURRICULUM_CACHE);

  if (cachedVersion === '1.0.0' && cachedData) {
    try {
      curriculumData = JSON.parse(cachedData);
      return curriculumData;
    } catch (e) {
      console.warn('Failed to parse cached curriculum, reloading...');
    }
  }

  // Load from file
  try {
    const response = await fetch('/data/365_day_calendar.json');
    const data = await response.json();
    curriculumData = data;

    // Cache it
    try {
      localStorage.setItem(STORAGE_KEYS.CURRICULUM_VERSION, data.version || '1.0.0');
      localStorage.setItem(STORAGE_KEYS.CURRICULUM_CACHE, JSON.stringify(data));
    } catch (e) {
      console.warn('Failed to cache curriculum (storage full?)');
    }

    return data;
  } catch (error) {
    console.error('Failed to load curriculum:', error);
    return null;
  }
}

/**
 * Get lesson for a specific day number (1-365)
 */
function getLesson(dayNumber) {
  if (!curriculumData || !curriculumData.lessons) return null;
  return curriculumData.lessons.find((l) => l.day === dayNumber) || null;
}

/**
 * Get today's lesson
 */
function getTodayLesson() {
  const dayNum = getDayOfYear();
  return getLesson(dayNum);
}

/**
 * Get lesson by date
 */
function getLessonByDate(month, day) {
  const date = new Date(new Date().getFullYear(), month - 1, day);
  const dayNum = getDayOfYear(date);
  return getLesson(dayNum);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATE UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Get day of year (1-365/366)
 */
function getDayOfYear(date = new Date()) {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date - start;
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

/**
 * Get date from day of year
 */
function getDateFromDayOfYear(dayNum, year = new Date().getFullYear()) {
  const date = new Date(year, 0);
  date.setDate(dayNum);
  return date;
}

/**
 * Format date as "Friday, November 28, 2025"
 */
function formatDateFull(date) {
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const months = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
  ];
  return `${days[date.getDay()]}, ${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}

/**
 * Format date as "November 28"
 */
function formatDateShort(date) {
  const months = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
  ];
  return `${months[date.getMonth()]} ${date.getDate()}`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// USER STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Get current user preferences
 */
function getUserPreferences() {
  return {
    age: localStorage.getItem(STORAGE_KEYS.AGE) || '18-35',
    language: localStorage.getItem(STORAGE_KEYS.LANGUAGE) || 'en',
    tone: localStorage.getItem(STORAGE_KEYS.TONE) || 'curious',
    difficulty: parseInt(localStorage.getItem(STORAGE_KEYS.DIFFICULTY) || '2'),
    mode: localStorage.getItem(STORAGE_KEYS.MODE) || '2D'
  };
}

/**
 * Save user preference
 */
function saveUserPreference(key, value) {
  const storageKey = STORAGE_KEYS[key.toUpperCase()];
  if (storageKey) {
    localStorage.setItem(storageKey, value.toString());
  }
}

/**
 * Get user progress stats
 */
function getUserProgress() {
  return {
    streak: parseInt(localStorage.getItem(STORAGE_KEYS.STREAK) || '0'),
    totalLessons: parseInt(localStorage.getItem(STORAGE_KEYS.TOTAL_LESSONS) || '0'),
    completedDays: JSON.parse(localStorage.getItem(STORAGE_KEYS.COMPLETED_DAYS) || '[]'),
    lastLessonDay: parseInt(localStorage.getItem(STORAGE_KEYS.LAST_LESSON_DAY) || '0'),
    lastLessonDate: localStorage.getItem(STORAGE_KEYS.LAST_LESSON_DATE) || null
  };
}

/**
 * Record lesson completion
 */
function recordLessonCompletion(dayNumber) {
  const now = new Date();
  const todayStr = now.toISOString().split('T')[0];
  const progress = getUserProgress();

  // Add to completed days if not already there
  if (!progress.completedDays.includes(dayNumber)) {
    progress.completedDays.push(dayNumber);
    localStorage.setItem(STORAGE_KEYS.COMPLETED_DAYS, JSON.stringify(progress.completedDays));
  }

  // Update total lessons
  const totalLessons = progress.totalLessons + 1;
  localStorage.setItem(STORAGE_KEYS.TOTAL_LESSONS, totalLessons.toString());

  // Update streak
  let streak = progress.streak;
  const lastDate = progress.lastLessonDate;

  if (lastDate) {
    const lastDateObj = new Date(lastDate);
    const daysSinceLast = Math.floor((now - lastDateObj) / (1000 * 60 * 60 * 24));

    if (daysSinceLast === 1) {
      // Consecutive day - increase streak
      streak += 1;
    } else if (daysSinceLast === 0) {
      // Same day - keep streak
    } else {
      // Streak broken - reset to 1
      streak = 1;
    }
  } else {
    // First lesson ever
    streak = 1;
  }

  localStorage.setItem(STORAGE_KEYS.STREAK, streak.toString());
  localStorage.setItem(STORAGE_KEYS.LAST_LESSON_DAY, dayNumber.toString());
  localStorage.setItem(STORAGE_KEYS.LAST_LESSON_DATE, todayStr);

  return { streak, totalLessons, completedDays: progress.completedDays };
}

/**
 * Get user's birthday lesson
 */
function getBirthdayLesson() {
  const month = parseInt(localStorage.getItem(STORAGE_KEYS.BIRTHDAY_MONTH) || '0');
  const day = parseInt(localStorage.getItem(STORAGE_KEYS.BIRTHDAY_DAY) || '0');

  if (!month || !day) return null;

  const birthdayDate = new Date(new Date().getFullYear(), month - 1, day);
  const dayNum = getDayOfYear(birthdayDate);
  const lesson = getLesson(dayNum);

  return {
    month,
    day,
    dayNumber: dayNum,
    lesson,
    formatted: formatDateShort(birthdayDate)
  };
}

/**
 * Set user's birthday
 */
function setUserBirthday(month, day) {
  localStorage.setItem(STORAGE_KEYS.BIRTHDAY_MONTH, month.toString());
  localStorage.setItem(STORAGE_KEYS.BIRTHDAY_DAY, day.toString());
}

// ═══════════════════════════════════════════════════════════════════════════════
// KELLY EXPRESSION MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

const KELLY_IMAGES = {
  curious: '/assets/kelly/production/avatars/curious/kelly-curious-512.webp',
  explaining: '/assets/kelly/production/avatars/explaining/kelly-explaining-512.webp',
  listening: '/assets/kelly/production/avatars/listening/kelly-listening-512.webp',
  wisdom: '/assets/kelly/production/avatars/wisdom/kelly-wisdom-512.webp',
  celebrating: '/assets/kelly/production/avatars/celebrating/kelly-celebrating-512.webp'
};

/**
 * Get Kelly image URL for expression
 */
function getKellyImage(expression) {
  return KELLY_IMAGES[expression] || KELLY_IMAGES.explaining;
}

/**
 * Preload all Kelly images
 */
function preloadKellyImages() {
  Object.values(KELLY_IMAGES).forEach((src) => {
    const img = new Image();
    img.src = src;
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Phase names and types
 */
const PHASE_CONFIG = [
  { order: 1, type: 'welcome', name: 'Welcome', expression: 'curious' },
  { order: 2, type: 'question', name: 'Q1', expression: 'explaining' },
  { order: 3, type: 'question', name: 'Q2', expression: 'explaining' },
  { order: 4, type: 'question', name: 'Q3', expression: 'listening' },
  { order: 5, type: 'wisdom', name: 'Wisdom', expression: 'wisdom' }
];

/**
 * Generate sample phase content for a lesson
 * In production, this would come from Supabase with full variant support
 */
function generatePhaseContent(lesson, preferences) {
  if (!lesson) return null;

  const { age, language, tone, difficulty } = preferences;

  return [
    {
      order: 1,
      type: 'welcome',
      name: 'Welcome',
      expression: 'curious',
      text: `Welcome! Today we're exploring ${lesson.title}. ${lesson.learning_essence || lesson.learning_objective}`,
      hint: null,
      choices: null
    },
    {
      order: 2,
      type: 'question',
      name: 'Q1',
      expression: 'explaining',
      text: `Let's start with a question about ${lesson.title}. What do you think is the most important aspect?`,
      hint: 'Think about what connects to your daily life...',
      choices: [
        { letter: 'A', text: 'Understanding the basic concepts' },
        { letter: 'B', text: 'Seeing how it connects to everything else' },
        { letter: 'C', text: 'Both understanding AND seeing connections' }
      ].slice(0, difficulty)
    },
    {
      order: 3,
      type: 'question',
      name: 'Q2',
      expression: 'explaining',
      text: `${lesson.core_principle || 'This concept has many applications.'}. How might this apply to you?`,
      hint: 'Think about your own experiences...',
      choices: [
        { letter: 'A', text: 'In my personal life and relationships' },
        { letter: 'B', text: 'In my community and the wider world' },
        { letter: 'C', text: 'Both personally AND in my community' }
      ].slice(0, difficulty)
    },
    {
      order: 4,
      type: 'question',
      name: 'Q3',
      expression: 'listening',
      text: `What action could you take based on what you've learned today?`,
      hint: 'Small steps count too...',
      choices: [
        { letter: 'A', text: 'Learn more about this topic' },
        { letter: 'B', text: 'Share what I learned with someone' },
        { letter: 'C', text: 'Both learn more AND share with others' }
      ].slice(0, difficulty)
    },
    {
      order: 5,
      type: 'wisdom',
      name: 'Wisdom',
      expression: 'wisdom',
      text: `Here's something to carry with you: ${lesson.universal_concept ? lesson.universal_concept.replace(/_/g, ' ') : 'Every small step of learning adds up to great wisdom over time.'}. What one insight will you remember from today?`,
      hint: null,
      choices: null
    }
  ];
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS (for use as module)
// ═══════════════════════════════════════════════════════════════════════════════

// Make available globally
window.KellyData = {
  // Config
  CONFIG: KELLY_CONFIG,
  STORAGE_KEYS,

  // Curriculum
  loadCurriculum,
  getLesson,
  getTodayLesson,
  getLessonByDate,

  // Dates
  getDayOfYear,
  getDateFromDayOfYear,
  formatDateFull,
  formatDateShort,

  // User state
  getUserPreferences,
  saveUserPreference,
  getUserProgress,
  recordLessonCompletion,
  getBirthdayLesson,
  setUserBirthday,

  // Kelly
  KELLY_IMAGES,
  getKellyImage,
  preloadKellyImages,

  // Phases
  PHASE_CONFIG,
  generatePhaseContent
};

// Auto-preload Kelly images
document.addEventListener('DOMContentLoaded', preloadKellyImages);

console.log('📚 Kelly Data Layer loaded');

