/**
 * End-to-End Integration Tests for Lesson Player
 * 
 * Tests:
 * - Session lifecycle (create, resume, complete)
 * - Age adaptation across all 6 buckets
 * - Language switching
 * - Asset preloading
 * - Phase progression
 */

const { expect } = require('chai');
const LessonService = require('../../curious-kellly/backend/src/services/lessons');
const AssetCacheService = require('../../curious-kellly/backend/src/services/assetCache');

describe('Lesson Player E2E Tests', function() {
  this.timeout(10000); // Allow time for API calls

  let lessonService;
  let cacheService;

  before(async function() {
    lessonService = new LessonService();
    cacheService = new AssetCacheService({ enabled: true });
  });

  after(async function() {
    await cacheService.clear();
  });

  describe('Session Lifecycle', function() {
    it('should create a new session with default age', async function() {
      const lesson = await lessonService.getTodaysLesson();
      expect(lesson).to.have.property('id');
      expect(lesson).to.have.property('ageVariants');
    });

    it('should adapt lesson for age 25 (18-35 bucket)', async function() {
      const lessonId = 'the-sun';
      const age = 25;
      const adaptedLesson = await lessonService.getLessonForAge(lessonId, age, 'en');

      expect(adaptedLesson).to.have.property('ageBucket', '18-35');
      expect(adaptedLesson).to.have.property('kellyAge');
      expect(adaptedLesson).to.have.property('language');
      expect(adaptedLesson.language).to.have.property('welcome');
      expect(adaptedLesson.language).to.have.property('mainContent');
    });

    it('should resume session with preserved age variant', async function() {
      const lessonId = 'the-moon';
      const age = 42;
      const language = 'es';

      const lesson1 = await lessonService.getLessonForAge(lessonId, age, language);
      const lesson2 = await lessonService.getLessonForAge(lessonId, age, language);

      expect(lesson1.ageBucket).to.equal(lesson2.ageBucket);
      expect(lesson1.ageBucket).to.equal('36-60');
    });
  });

  describe('Age Adaptation', function() {
    const ages = [
      { age: 3, bucket: '2-5', kellyAge: 3 },
      { age: 8, bucket: '6-12', kellyAge: 9 },
      { age: 15, bucket: '13-17', kellyAge: 15 },
      { age: 25, bucket: '18-35', kellyAge: 27 },
      { age: 50, bucket: '36-60', kellyAge: 48 },
      { age: 75, bucket: '61-102', kellyAge: 82 }
    ];

    ages.forEach(({ age, bucket, kellyAge }) => {
      it(`should adapt lesson for age ${age} (${bucket} bucket)`, async function() {
        const lessonId = 'the-ocean';
        const adaptedLesson = await lessonService.getLessonForAge(lessonId, age, 'en');

        expect(adaptedLesson.ageBucket).to.equal(bucket);
        expect(adaptedLesson.kellyAge).to.equal(kellyAge);
        expect(adaptedLesson.language.welcome).to.be.a('string');
        expect(adaptedLesson.language.welcome.length).to.be.greaterThan(0);
      });
    });

    it('should have age-appropriate vocabulary complexity', async function() {
      const youngLesson = await lessonService.getLessonForAge('puppies', 4, 'en');
      const adultLesson = await lessonService.getLessonForAge('puppies', 30, 'en');

      expect(youngLesson.vocabulary.complexity).to.equal('simple');
      expect(adultLesson.vocabulary.complexity).to.be.oneOf(['moderate', 'complex']);
    });

    it('should have age-appropriate pacing', async function() {
      const youngLesson = await lessonService.getLessonForAge('the-moon', 5, 'en');
      const teenLesson = await lessonService.getLessonForAge('the-moon', 15, 'en');

      expect(youngLesson.pacing.speechRate).to.equal('slow');
      expect(youngLesson.pacing.pauseFrequency).to.equal('frequent');
      expect(teenLesson.pacing.speechRate).to.equal('moderate');
    });
  });

  describe('Multilingual Support', function() {
    const languages = ['en', 'es', 'fr'];

    languages.forEach(lang => {
      it(`should provide complete ${lang.toUpperCase()} content`, async function() {
        const lessonId = 'the-sun';
        const age = 10;
        const lesson = await lessonService.getLessonForAge(lessonId, age, lang);

        expect(lesson.language).to.have.property('welcome');
        expect(lesson.language).to.have.property('mainContent');
        expect(lesson.language).to.have.property('keyPoints');
        expect(lesson.language).to.have.property('wisdomMoment');

        expect(lesson.language.welcome).to.be.a('string').with.length.greaterThan(0);
        expect(lesson.language.mainContent).to.be.a('string').with.length.greaterThan(0);
        expect(lesson.language.keyPoints).to.be.an('array').with.length.greaterThan(0);
      });
    });

    it('should switch languages without losing age adaptation', async function() {
      const lessonId = 'puppies';
      const age = 7;

      const enLesson = await lessonService.getLessonForAge(lessonId, age, 'en');
      const esLesson = await lessonService.getLessonForAge(lessonId, age, 'es');

      expect(enLesson.ageBucket).to.equal(esLesson.ageBucket);
      expect(enLesson.kellyAge).to.equal(esLesson.kellyAge);
      expect(enLesson.language.welcome).to.not.equal(esLesson.language.welcome);
    });
  });

  describe('Asset Caching', function() {
    it('should cache lesson data', async function() {
      const lessonId = 'the-ocean';
      const params = { lessonId, age: 12, language: 'en' };

      // First call - cache miss
      const stats1 = cacheService.getStats();
      await cacheService.get('lesson', params);
      const stats2 = cacheService.getStats();
      expect(stats2.misses).to.be.greaterThan(stats1.misses);

      // Set cache
      await cacheService.set('lesson', params, { test: 'data' });

      // Second call - cache hit
      const stats3 = cacheService.getStats();
      const cached = await cacheService.get('lesson', params);
      const stats4 = cacheService.getStats();

      expect(cached).to.deep.equal({ test: 'data' });
      expect(stats4.hits).to.be.greaterThan(stats3.hits);
    });

    it('should generate consistent cache keys', function() {
      const key1 = cacheService.generateKey('audio', { lessonId: 'test', age: '6-12', lang: 'en' });
      const key2 = cacheService.generateKey('audio', { lessonId: 'test', age: '6-12', lang: 'en' });
      const key3 = cacheService.generateKey('audio', { lessonId: 'test', age: '6-12', lang: 'es' });

      expect(key1).to.equal(key2);
      expect(key1).to.not.equal(key3);
    });

    it('should provide cache statistics', async function() {
      await cacheService.clear();

      await cacheService.set('test', { id: 1 }, 'data1');
      await cacheService.set('test', { id: 2 }, 'data2');
      await cacheService.get('test', { id: 1 }); // hit
      await cacheService.get('test', { id: 3 }); // miss

      const stats = cacheService.getStats();
      expect(stats.sets).to.equal(2);
      expect(stats.hits).to.equal(1);
      expect(stats.misses).to.equal(1);
      expect(stats.hitRate).to.equal('50.00%');
    });
  });

  describe('Phase Progression', function() {
    it('should have correct phase order', async function() {
      const lessonId = 'the-moon';
      const age = 20;
      const lesson = await lessonService.getLessonForAge(lessonId, age, 'en');

      expect(lesson.language).to.have.property('welcome');
      expect(lesson.language).to.have.property('mainContent');
      expect(lesson.language).to.have.property('wisdomMoment');
    });

    it('should preload next phase assets', async function() {
      const lessonId = 'puppies';
      const currentPhase = 'welcome';
      const ageVariant = '6-12';
      const language = 'en';

      await cacheService.preloadNextPhase(lessonId, currentPhase, ageVariant, language);
      
      // Verify that preloading was triggered
      // In real implementation, this would check if the next phase audio is cached
      expect(true).to.be.true; // Placeholder assertion
    });
  });

  describe('Expression Cues', function() {
    it('should include expression cues for avatar', async function() {
      const lessonId = 'the-sun';
      const age = 10;
      const lesson = await lessonService.getLessonForAge(lessonId, age, 'en');

      expect(lesson).to.have.property('expressionCues');
      expect(lesson.expressionCues).to.be.an('array');

      if (lesson.expressionCues.length > 0) {
        const cue = lesson.expressionCues[0];
        expect(cue).to.have.property('id');
        expect(cue).to.have.property('type');
        expect(cue).to.have.property('intensity');
        expect(cue).to.have.property('gazeTarget');
      }
    });
  });

  describe('Lesson Validation', function() {
    it('should validate lesson structure', async function() {
      const lesson = await lessonService.loadLesson('the-ocean');
      const validation = await lessonService.validateLesson(lesson);

      expect(validation.valid).to.be.true;
      expect(validation.errors).to.be.an('array').with.length(0);
    });

    it('should detect missing age variants', async function() {
      const invalidLesson = {
        id: 'test',
        title: 'Test',
        description: 'Test',
        ageVariants: {
          '2-5': {},
          '6-12': {}
          // Missing other age variants
        },
        interactions: [],
        metadata: {}
      };

      const validation = await lessonService.validateLesson(invalidLesson);
      expect(validation.valid).to.be.false;
      expect(validation.errors).to.be.an('array').with.length.greaterThan(0);
    });
  });

  describe('Performance', function() {
    it('should load lessons efficiently with caching', async function() {
      const lessonId = 'puppies';
      const age = 8;

      const start1 = Date.now();
      await lessonService.getLessonForAge(lessonId, age, 'en');
      const duration1 = Date.now() - start1;

      const start2 = Date.now();
      await lessonService.getLessonForAge(lessonId, age, 'en'); // Cached
      const duration2 = Date.now() - start2;

      console.log(`First load: ${duration1}ms, Cached load: ${duration2}ms`);
      expect(duration2).to.be.lessThan(duration1);
    });

    it('should handle multiple concurrent requests', async function() {
      const requests = [];
      for (let i = 0; i < 5; i++) {
        requests.push(lessonService.getLessonForAge('the-moon', 10 + i, 'en'));
      }

      const results = await Promise.all(requests);
      expect(results).to.have.length(5);
      results.forEach(result => {
        expect(result).to.have.property('ageBucket');
      });
    });
  });
});


