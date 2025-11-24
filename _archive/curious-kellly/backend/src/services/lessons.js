/**
 * Lesson Service
 * Manages daily topics and lesson content
 */

const fs = require('fs').promises;
const path = require('path');
const {
  buildLocalizationBundle,
  DEFAULT_FALLBACK_LANGUAGE
} = require('../utils/localizationMapper');

class LessonService {
  constructor() {
    this.lessonsDir = path.join(__dirname, '../../config/lessons');
    this.schemaPath = path.join(__dirname, '../../config/lesson-dna-schema.json');
    this.lessonsCache = new Map();
    this.localizationCache = new Map();
  }

  /**
   * Get age bucket from specific age
   */
  getAgeBucket(age) {
    if (age >= 2 && age <= 5) return '2-5';
    if (age >= 6 && age <= 12) return '6-12';
    if (age >= 13 && age <= 17) return '13-17';
    if (age >= 18 && age <= 35) return '18-35';
    if (age >= 36 && age <= 60) return '36-60';
    if (age >= 61 && age <= 102) return '61-102';
    throw new Error(`Invalid age: ${age}. Must be between 2 and 102.`);
  }

  /**
   * Get Kelly's age for a given learner age
   */
  getKellyAge(learnerAge) {
    if (learnerAge <= 5) return 3;
    if (learnerAge <= 12) return 9;
    if (learnerAge <= 17) return 15;
    if (learnerAge <= 35) return 27;
    if (learnerAge <= 60) return 48;
    return 82;
  }

  /**
   * Get Kelly's persona for a given age
   */
  getKellyPersona(learnerAge) {
    if (learnerAge <= 5) return 'playful-toddler';
    if (learnerAge <= 12) return 'curious-kid';
    if (learnerAge <= 17) return 'enthusiastic-teen';
    if (learnerAge <= 35) return 'knowledgeable-adult';
    if (learnerAge <= 60) return 'wise-mentor';
    return 'reflective-elder';
  }

  /**
   * Load a lesson by ID
   */
  async loadLesson(lessonId) {
    try {
      // Check cache first
      if (this.lessonsCache.has(lessonId)) {
        return this.lessonsCache.get(lessonId);
      }

      // Load from file
      const lessonPath = path.join(this.lessonsDir, `${lessonId}.json`);
      const content = await fs.readFile(lessonPath, 'utf-8');
      const lesson = JSON.parse(content);

      // Build localization bundle once per lesson to avoid repeated traversal
      const localizationBundle = buildLocalizationBundle(lesson);
      this.localizationCache.set(lessonId, localizationBundle);

      // Cache it
      this.lessonsCache.set(lessonId, lesson);

      return lesson;
    } catch (error) {
      if (error.code === 'ENOENT') {
        throw new Error(`Lesson not found: ${lessonId}`);
      }
      throw error;
    }
  }

  /**
   * Get all available lessons
   */
  async getAllLessons() {
    try {
      const files = await fs.readdir(this.lessonsDir);
      const lessonFiles = files.filter(f => f.endsWith('.json'));
      
      const lessons = await Promise.all(
        lessonFiles.map(async (file) => {
          const lessonId = file.replace('.json', '');
          const lesson = await this.loadLesson(lessonId);
          return {
            id: lesson.id,
            title: lesson.title,
            description: lesson.description,
            category: lesson.metadata.category,
            duration: lesson.metadata.duration
          };
        })
      );

      return lessons;
    } catch (error) {
      throw new Error(`Failed to load lessons: ${error.message}`);
    }
  }

  /**
   * Get today's lesson
   * Based on day of year (1-365)
   */
  async getTodaysLesson() {
    const dayOfYear = this.getDayOfYear();
    
    // For now, we only have one lesson (leaves)
    // In production, this would rotate through 365 lessons
    // Lesson index = dayOfYear % totalLessons
    
    const lessons = await this.getAllLessons();
    if (lessons.length === 0) {
      throw new Error('No lessons available');
    }

    const lessonIndex = (dayOfYear - 1) % lessons.length;
    const todaysLessonInfo = lessons[lessonIndex];
    
    return await this.loadLesson(todaysLessonInfo.id);
  }

  /**
   * Get lesson for specific age
   */
  async getLessonForAge(lessonId, age, locale = DEFAULT_FALLBACK_LANGUAGE) {
    const lesson = await this.loadLesson(lessonId);
    const ageBucket = this.getAgeBucket(age);
    const ageVariant = lesson.ageVariants[ageBucket];

    if (!ageVariant) {
      throw new Error(`No content for age ${age} in lesson ${lessonId}`);
    }

    const localizationBundle = await this.getLocalizationBundle(lessonId);
    const availableLocales = localizationBundle.supportedLanguages || [];
    const normalizedLocale = availableLocales.includes(locale)
      ? locale
      : DEFAULT_FALLBACK_LANGUAGE;
    const localizedLesson =
      localizationBundle.locales[normalizedLocale] || lesson;
    const localizedAgeVariant = localizedLesson.ageVariants
      ? localizedLesson.ageVariants[ageBucket]
      : null;

    const kellyAge = this.getKellyAge(age);
    const kellyPersona = this.getKellyPersona(age);

    const englishContent = {
      ...ageVariant,
      kellyAge,
      kellyPersona,
      learnerAge: age,
      ageBucket
    };

    const localizedContent = localizedAgeVariant
      ? {
          ...localizedAgeVariant,
          kellyAge,
          kellyPersona,
          learnerAge: age,
          ageBucket
        }
      : englishContent;

    return {
      id: lesson.id,
      title: lesson.title,
      description: lesson.description,
      metadata: lesson.metadata,
      content: englishContent,
      interactions: lesson.interactions,
      locale: normalizedLocale,
      supportedLocales: availableLocales,
      localized: {
        title: localizedLesson.title || lesson.title,
        description: localizedLesson.description || lesson.description,
        metadata: localizedLesson.metadata || lesson.metadata,
        content: localizedContent,
        interactions: localizedLesson.interactions || lesson.interactions
      }
    };
  }

  /**
   * Get precomputed localization bundle for a lesson
   */
  async getLocalizationBundle(lessonId) {
    if (this.localizationCache.has(lessonId)) {
      return this.localizationCache.get(lessonId);
    }

    await this.loadLesson(lessonId);
    return this.localizationCache.get(lessonId);
  }

  /**
   * Get today's lesson for specific age
   */
  async getTodaysLessonForAge(age, locale = DEFAULT_FALLBACK_LANGUAGE) {
    const todaysLesson = await this.getTodaysLesson();
    return await this.getLessonForAge(todaysLesson.id, age, locale);
  }

  /**
   * Get day of year (1-365)
   */
  getDayOfYear() {
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 0);
    const diff = now - start;
    const oneDay = 1000 * 60 * 60 * 24;
    return Math.floor(diff / oneDay);
  }

  /**
   * Validate lesson against schema
   */
  async validateLesson(lesson) {
    try {
      const schemaContent = await fs.readFile(this.schemaPath, 'utf-8');
      const schema = JSON.parse(schemaContent);

      // Basic validation (in production, use Ajv or similar)
      const required = ['id', 'title', 'description', 'ageVariants', 'interactions', 'metadata'];
      for (const field of required) {
        if (!lesson[field]) {
          return {
            valid: false,
            errors: [`Missing required field: ${field}`]
          };
        }
      }

      // Check age variants
      const requiredAges = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
      for (const age of requiredAges) {
        if (!lesson.ageVariants[age]) {
          return {
            valid: false,
            errors: [`Missing age variant: ${age}`]
          };
        }
      }

      return {
        valid: true,
        errors: []
      };
    } catch (error) {
      return {
        valid: false,
        errors: [`Validation error: ${error.message}`]
      };
    }
  }

  /**
   * Get lesson statistics
   */
  async getStats() {
    const lessons = await this.getAllLessons();
    const todaysLesson = await this.getTodaysLesson();

    return {
      totalLessons: lessons.length,
      todaysLessonId: todaysLesson.id,
      todaysLessonTitle: todaysLesson.title,
      dayOfYear: this.getDayOfYear(),
      targetTotal: 365,
      percentComplete: ((lessons.length / 365) * 100).toFixed(1)
    };
  }
}

module.exports = LessonService;











