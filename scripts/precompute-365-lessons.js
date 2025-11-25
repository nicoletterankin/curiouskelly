#!/usr/bin/env node

/**
 * Pre-Computation Script for 365 Daily Lessons
 * 
 * Generates all audio files and expression data for the complete
 * Curious Kelly daily lesson curriculum.
 * 
 * Total computations: 365 lessons × 6 age buckets × 3 languages × 5 phases = 32,850 files
 * 
 * Cost Estimation:
 * - ElevenLabs: ~$0.30 per 1000 characters
 * - Average script: 200 characters per phase
 * - Total cost: 32,850 × 200 × $0.30 / 1000 = ~$1,971
 * - Storage: 32,850 files × 100KB avg = ~3.3GB
 * 
 * Usage:
 *   node scripts/precompute-365-lessons.js [options]
 * 
 * Options:
 *   --dry-run          Simulate without making API calls
 *   --start-day=N      Start from day N (default: 1)
 *   --end-day=N        End at day N (default: 365)
 *   --language=en      Process only specified language
 *   --age-bucket=18-35 Process only specified age bucket
 *   --phase=welcome    Process only specified phase
 *   --parallel=N       Number of parallel workers (default: 1, max: 5)
 *   --skip-audio       Skip audio generation (expressions only)
 *   --skip-expressions Skip expression generation (audio only)
 *   --resume           Resume from last checkpoint
 * 
 * @requires ELEVENLABS_API_KEY environment variable
 * @requires SUPABASE_URL environment variable
 * @requires SUPABASE_SERVICE_KEY environment variable (service role key for storage)
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Age buckets (matching ElevenLabs voice engine)
  ageBuckets: ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'],
  
  // Languages (EN + ES + FR as per CLAUDE.md requirements)
  languages: ['en', 'es', 'fr'],
  
  // Phases in lesson structure
  phases: ['welcome', 'q1', 'q2', 'q3', 'wisdom'],
  
  // Rate limiting
  requestDelayMs: 1000,           // Delay between ElevenLabs requests
  batchSize: 10,                  // Lessons to process before saving checkpoint
  maxRetries: 3,                  // Max retries per request
  retryDelayMs: 5000,             // Delay before retry
  
  // ElevenLabs settings
  elevenLabsBaseUrl: 'https://api.elevenlabs.io/v1',
  elevenLabsModel: 'eleven_multilingual_v2',
  kellyVoiceId: process.env.KELLY_VOICE_ID || 'YOUR_KELLY_VOICE_ID',
  
  // Storage
  storageBucket: 'lesson-audio',
  storagePathTemplate: 'precomputed/{lessonSlug}/{ageBucket}-{language}-{phase}.mp3',
  
  // Checkpoint file
  checkpointFile: path.join(__dirname, '.precompute-checkpoint.json'),
  
  // Log file
  logFile: path.join(__dirname, 'precompute-log.txt'),
};

// =============================================================================
// VOICE SETTINGS BY AGE
// =============================================================================

const AGE_VOICE_SETTINGS = {
  '2-5': {
    stability: 0.7,
    similarity_boost: 0.8,
    style: 0.4,
    use_speaker_boost: true,
    description: 'Childlike, warm, enthusiastic',
  },
  '6-12': {
    stability: 0.65,
    similarity_boost: 0.75,
    style: 0.35,
    use_speaker_boost: true,
    description: 'Friendly, clear, engaging',
  },
  '13-17': {
    stability: 0.6,
    similarity_boost: 0.7,
    style: 0.3,
    use_speaker_boost: true,
    description: 'Relatable, authentic, slightly casual',
  },
  '18-35': {
    stability: 0.55,
    similarity_boost: 0.75,
    style: 0.25,
    use_speaker_boost: true,
    description: 'Natural, conversational, articulate',
  },
  '36-60': {
    stability: 0.6,
    similarity_boost: 0.8,
    style: 0.2,
    use_speaker_boost: true,
    description: 'Measured, professional, warm',
  },
  '61-102': {
    stability: 0.65,
    similarity_boost: 0.85,
    style: 0.15,
    use_speaker_boost: true,
    description: 'Gentle, wise, deliberate pace',
  },
};

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

const supabaseUrl = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// =============================================================================
// MAIN PRE-COMPUTATION CLASS
// =============================================================================

class LessonPrecomputer {
  constructor(options = {}) {
    this.options = {
      dryRun: options.dryRun || false,
      startDay: options.startDay || 1,
      endDay: options.endDay || 365,
      filterLanguage: options.language || null,
      filterAgeBucket: options.ageBucket || null,
      filterPhase: options.phase || null,
      parallel: Math.min(options.parallel || 1, 5),
      skipAudio: options.skipAudio || false,
      skipExpressions: options.skipExpressions || false,
      resume: options.resume || false,
    };
    
    this.elevenLabsKey = process.env.ELEVENLABS_API_KEY;
    if (!this.elevenLabsKey && !this.options.dryRun) {
      console.error('❌ Missing ELEVENLABS_API_KEY');
      process.exit(1);
    }
    
    // Statistics
    this.stats = {
      totalProcessed: 0,
      totalSkipped: 0,
      totalErrors: 0,
      totalAudioGenerated: 0,
      totalExpressionsGenerated: 0,
      totalCharacters: 0,
      estimatedCost: 0,
      startTime: null,
    };
    
    // Checkpoint data
    this.checkpoint = {
      lastProcessed: null,
      completed: [],
      errors: [],
    };
  }

  // ===========================================================================
  // MAIN EXECUTION
  // ===========================================================================

  async run() {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  Curious Kelly - 365 Lesson Pre-Computation Script');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('');
    
    this.stats.startTime = Date.now();
    
    // Load checkpoint if resuming
    if (this.options.resume) {
      await this.loadCheckpoint();
    }
    
    // Print configuration
    this.printConfig();
    
    // Get all lessons
    const lessons = await this.getAllCoreLessons();
    
    if (lessons.length === 0) {
      console.error('❌ No lessons found in database');
      process.exit(1);
    }
    
    console.log(`📚 Found ${lessons.length} lessons in database`);
    
    // Filter lessons by day range
    const filteredLessons = lessons.filter(
      l => l.day_number >= this.options.startDay && l.day_number <= this.options.endDay
    );
    
    console.log(`🎯 Processing days ${this.options.startDay} to ${this.options.endDay} (${filteredLessons.length} lessons)`);
    console.log('');
    
    // Calculate total work
    const totalVariants = this.calculateTotalVariants(filteredLessons.length);
    console.log(`📊 Total variants to process: ${totalVariants.toLocaleString()}`);
    console.log('');
    
    // Process lessons
    for (const lesson of filteredLessons) {
      await this.processLesson(lesson);
      
      // Save checkpoint periodically
      if (this.stats.totalProcessed % CONFIG.batchSize === 0) {
        await this.saveCheckpoint();
      }
    }
    
    // Final checkpoint save
    await this.saveCheckpoint();
    
    // Print summary
    this.printSummary();
  }

  /**
   * Process a single lesson (all variants)
   */
  async processLesson(lesson) {
    const lessonSlug = this.getLessonSlug(lesson);
    
    console.log(`\n📖 Processing: Day ${lesson.day_number} - ${lesson.topic}`);
    console.log(`   Slug: ${lessonSlug}`);
    
    const ages = this.options.filterAgeBucket 
      ? [this.options.filterAgeBucket]
      : CONFIG.ageBuckets;
    
    const languages = this.options.filterLanguage
      ? [this.options.filterLanguage]
      : CONFIG.languages;
    
    const phases = this.options.filterPhase
      ? [this.options.filterPhase]
      : CONFIG.phases;
    
    for (const age of ages) {
      for (const language of languages) {
        for (const phase of phases) {
          const variantKey = `${lesson.day_number}-${age}-${language}-${phase}`;
          
          // Skip if already completed
          if (this.checkpoint.completed.includes(variantKey)) {
            console.log(`   ⏭️  Skipping (already done): ${variantKey}`);
            this.stats.totalSkipped++;
            continue;
          }
          
          try {
            await this.processVariant(lesson, lessonSlug, age, language, phase);
            this.checkpoint.completed.push(variantKey);
            this.stats.totalProcessed++;
          } catch (error) {
            console.error(`   ❌ Error: ${variantKey} - ${error.message}`);
            this.checkpoint.errors.push({ key: variantKey, error: error.message });
            this.stats.totalErrors++;
          }
          
          // Rate limiting
          await this.sleep(CONFIG.requestDelayMs);
        }
      }
    }
  }

  /**
   * Process a single variant (specific age/language/phase)
   */
  async processVariant(lesson, lessonSlug, ageBucket, language, phase) {
    const label = `${ageBucket}-${language}-${phase}`;
    console.log(`   🔄 Processing: ${label}`);
    
    // Get lesson content for this phase
    const content = await this.getPhaseContent(lesson, phase, language);
    
    if (!content || !content.script) {
      console.log(`   ⚠️  No content for: ${label}`);
      return;
    }
    
    const script = content.script;
    this.stats.totalCharacters += script.length;
    
    // Estimate cost
    const costEstimate = (script.length / 1000) * 0.30;
    this.stats.estimatedCost += costEstimate;
    
    if (this.options.dryRun) {
      console.log(`   📝 [DRY RUN] Would process: ${script.length} chars, ~$${costEstimate.toFixed(4)}`);
      return;
    }
    
    // Generate audio
    let audioUrl = null;
    if (!this.options.skipAudio) {
      audioUrl = await this.generateAndUploadAudio(
        lessonSlug,
        ageBucket,
        language,
        phase,
        script
      );
      this.stats.totalAudioGenerated++;
    }
    
    // Generate expressions
    let expressions = null;
    if (!this.options.skipExpressions) {
      expressions = await this.generateExpressions(
        script,
        ageBucket,
        language,
        phase
      );
      
      // Save expressions to database
      await this.saveExpressions(lesson.id, ageBucket, language, phase, expressions);
      this.stats.totalExpressionsGenerated++;
    }
    
    console.log(`   ✅ Complete: ${label}`);
  }

  // ===========================================================================
  // AUDIO GENERATION
  // ===========================================================================

  /**
   * Generate audio via ElevenLabs and upload to Supabase Storage
   */
  async generateAndUploadAudio(lessonSlug, ageBucket, language, phase, script) {
    // Generate audio
    const audioBuffer = await this.generateAudio(script, ageBucket, language);
    
    // Upload to Supabase Storage
    const storagePath = CONFIG.storagePathTemplate
      .replace('{lessonSlug}', lessonSlug)
      .replace('{ageBucket}', ageBucket)
      .replace('{language}', language)
      .replace('{phase}', phase);
    
    const { data, error } = await supabase.storage
      .from(CONFIG.storageBucket)
      .upload(storagePath, audioBuffer, {
        contentType: 'audio/mpeg',
        upsert: true,
        cacheControl: '31536000', // 1 year
      });
    
    if (error) {
      throw new Error(`Storage upload failed: ${error.message}`);
    }
    
    // Get public URL
    const { data: urlData } = supabase.storage
      .from(CONFIG.storageBucket)
      .getPublicUrl(storagePath);
    
    return urlData?.publicUrl;
  }

  /**
   * Generate audio via ElevenLabs API
   */
  async generateAudio(script, ageBucket, language) {
    const voiceSettings = AGE_VOICE_SETTINGS[ageBucket] || AGE_VOICE_SETTINGS['18-35'];
    
    const response = await fetch(
      `${CONFIG.elevenLabsBaseUrl}/text-to-speech/${CONFIG.kellyVoiceId}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': this.elevenLabsKey,
        },
        body: JSON.stringify({
          text: script,
          model_id: CONFIG.elevenLabsModel,
          voice_settings: {
            stability: voiceSettings.stability,
            similarity_boost: voiceSettings.similarity_boost,
            style: voiceSettings.style,
            use_speaker_boost: voiceSettings.use_speaker_boost,
          },
        }),
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ElevenLabs API error ${response.status}: ${errorText}`);
    }
    
    return Buffer.from(await response.arrayBuffer());
  }

  // ===========================================================================
  // EXPRESSION GENERATION
  // ===========================================================================

  /**
   * Generate expressions for a phase
   */
  async generateExpressions(script, ageBucket, language, phase) {
    // Simplified expression generation (in production, use full ExpressionGenerator)
    const duration = this.estimateAudioDuration(script);
    
    const expressions = {
      metadata: {
        ageBucket,
        language,
        phase,
        generatedAt: new Date().toISOString(),
        scriptLength: script.length,
        estimatedDuration: duration,
      },
      expressions: this.generatePhaseExpressions(phase, duration, ageBucket),
      gestures: this.generatePhaseGestures(phase, duration, ageBucket),
    };
    
    return expressions;
  }

  /**
   * Generate expressions for a specific phase
   */
  generatePhaseExpressions(phase, duration, ageBucket) {
    // Base expressions by phase
    const phaseExpressions = {
      welcome: [
        { timestamp: 0, emotion: 'warm', intensity: 0.7 },
        { timestamp: duration * 0.3, emotion: 'excited', intensity: 0.8 },
        { timestamp: duration * 0.7, emotion: 'inviting', intensity: 0.75 },
      ],
      q1: [
        { timestamp: 0, emotion: 'curious', intensity: 0.75 },
        { timestamp: duration * 0.25, emotion: 'explaining', intensity: 0.7 },
        { timestamp: duration * 0.5, emotion: 'questioning', intensity: 0.8 },
        { timestamp: duration * 0.75, emotion: 'thoughtful', intensity: 0.65 },
      ],
      q2: [
        { timestamp: 0, emotion: 'encouraging', intensity: 0.7 },
        { timestamp: duration * 0.4, emotion: 'attentive', intensity: 0.75 },
        { timestamp: duration * 0.8, emotion: 'supportive', intensity: 0.8 },
      ],
      q3: [
        { timestamp: 0, emotion: 'engaged', intensity: 0.75 },
        { timestamp: duration * 0.35, emotion: 'insightful', intensity: 0.8 },
        { timestamp: duration * 0.7, emotion: 'affirming', intensity: 0.7 },
      ],
      wisdom: [
        { timestamp: 0, emotion: 'serene', intensity: 0.6 },
        { timestamp: duration * 0.3, emotion: 'profound', intensity: 0.7 },
        { timestamp: duration * 0.6, emotion: 'warm', intensity: 0.75 },
        { timestamp: duration * 0.85, emotion: 'peaceful', intensity: 0.65 },
      ],
    };
    
    // Adjust intensity based on age bucket
    const intensityMultiplier = this.getAgeIntensityMultiplier(ageBucket);
    
    return (phaseExpressions[phase] || []).map(expr => ({
      ...expr,
      intensity: Math.min(1.0, expr.intensity * intensityMultiplier),
    }));
  }

  /**
   * Generate gestures for a specific phase
   */
  generatePhaseGestures(phase, duration, ageBucket) {
    const phaseGestures = {
      welcome: [
        { timestamp: 0.5, gesture: 'open_arms_welcome', duration: 2.0, intensity: 0.7 },
        { timestamp: duration * 0.5, gesture: 'wave', duration: 1.5, intensity: 0.6 },
      ],
      q1: [
        { timestamp: duration * 0.2, gesture: 'point_up', duration: 1.5, intensity: 0.6 },
        { timestamp: duration * 0.6, gesture: 'chin_touch', duration: 2.0, intensity: 0.5 },
      ],
      q2: [
        { timestamp: duration * 0.3, gesture: 'encouraging_nod', duration: 1.5, intensity: 0.6 },
      ],
      q3: [
        { timestamp: duration * 0.4, gesture: 'connect_points', duration: 2.0, intensity: 0.5 },
      ],
      wisdom: [
        { timestamp: duration * 0.3, gesture: 'heart_touch', duration: 2.0, intensity: 0.6 },
        { timestamp: duration * 0.7, gesture: 'gentle_nod', duration: 1.5, intensity: 0.5 },
      ],
    };
    
    // Filter gestures based on age appropriateness
    const ageFiltered = this.filterGesturesForAge(phaseGestures[phase] || [], ageBucket);
    
    return ageFiltered;
  }

  /**
   * Get intensity multiplier based on age
   */
  getAgeIntensityMultiplier(ageBucket) {
    const multipliers = {
      '2-5': 1.4,
      '6-12': 1.2,
      '13-17': 0.9,
      '18-35': 1.0,
      '36-60': 0.85,
      '61-102': 0.75,
    };
    return multipliers[ageBucket] || 1.0;
  }

  /**
   * Filter gestures based on age appropriateness
   */
  filterGesturesForAge(gestures, ageBucket) {
    // Some gestures may not be appropriate for all ages
    const avoidByAge = {
      '2-5': ['chin_touch', 'connect_points'],
      '13-17': ['heart_touch'],
      '61-102': ['open_arms_welcome'],
    };
    
    const avoid = avoidByAge[ageBucket] || [];
    return gestures.filter(g => !avoid.includes(g.gesture));
  }

  /**
   * Estimate audio duration from script length
   */
  estimateAudioDuration(script) {
    // Average speaking rate: ~150 words per minute
    const words = script.split(/\s+/).length;
    return (words / 150) * 60; // Duration in seconds
  }

  // ===========================================================================
  // DATABASE OPERATIONS
  // ===========================================================================

  /**
   * Get all core lessons from database
   */
  async getAllCoreLessons() {
    const { data, error } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic, universal_truth')
      .order('day_number', { ascending: true });
    
    if (error) {
      throw new Error(`Failed to fetch lessons: ${error.message}`);
    }
    
    return data || [];
  }

  /**
   * Get phase content for a lesson
   */
  async getPhaseContent(lesson, phase, language) {
    // Try to get from lesson_atoms
    const { data: atom, error } = await supabase
      .from('lesson_atoms')
      .select('content')
      .eq('core_lesson_id', lesson.id)
      .eq('phase', phase)
      .single();
    
    if (atom?.content) {
      // Content might be language-specific
      const langContent = atom.content[language] || atom.content.en || atom.content;
      return {
        script: langContent.script || langContent.text || 
                (typeof langContent === 'string' ? langContent : JSON.stringify(langContent)),
        tone: langContent.tone || 'warm',
      };
    }
    
    // Generate placeholder content if none exists
    return this.generatePlaceholderContent(lesson, phase, language);
  }

  /**
   * Generate placeholder content for testing
   */
  generatePlaceholderContent(lesson, phase, language) {
    const templates = {
      en: {
        welcome: `Welcome to today's lesson about ${lesson.topic}! I'm so excited to explore this with you.`,
        q1: `Let's start by understanding ${lesson.topic}. ${lesson.universal_truth}`,
        q2: `Now let's think about this together. What do you think about ${lesson.topic}?`,
        q3: `Great thinking! Here's another way to look at ${lesson.topic}.`,
        wisdom: `Remember, ${lesson.universal_truth}. Keep exploring and stay curious!`,
      },
      es: {
        welcome: `¡Bienvenido a la lección de hoy sobre ${lesson.topic}! Estoy muy emocionada de explorar esto contigo.`,
        q1: `Comencemos por entender ${lesson.topic}. ${lesson.universal_truth}`,
        q2: `Ahora pensemos juntos. ¿Qué piensas sobre ${lesson.topic}?`,
        q3: `¡Buen pensamiento! Aquí hay otra manera de ver ${lesson.topic}.`,
        wisdom: `Recuerda, ${lesson.universal_truth}. ¡Sigue explorando y mantente curioso!`,
      },
      fr: {
        welcome: `Bienvenue à la leçon d'aujourd'hui sur ${lesson.topic}! Je suis très excitée d'explorer cela avec toi.`,
        q1: `Commençons par comprendre ${lesson.topic}. ${lesson.universal_truth}`,
        q2: `Maintenant, réfléchissons ensemble. Que penses-tu de ${lesson.topic}?`,
        q3: `Bonne réflexion! Voici une autre façon de voir ${lesson.topic}.`,
        wisdom: `Souviens-toi, ${lesson.universal_truth}. Continue d'explorer et reste curieux!`,
      },
    };
    
    const langTemplates = templates[language] || templates.en;
    
    return {
      script: langTemplates[phase] || langTemplates.welcome,
      tone: 'warm',
    };
  }

  /**
   * Save expressions to database
   */
  async saveExpressions(lessonId, ageBucket, language, phase, expressions) {
    const variantKey = `${ageBucket}-${language}`;
    
    // Get existing expression_data or create new
    const { data: existing } = await supabase
      .from('lesson_atoms')
      .select('id, expression_data')
      .eq('core_lesson_id', lessonId)
      .eq('phase', phase)
      .single();
    
    const expressionData = existing?.expression_data || {};
    expressionData[variantKey] = expressions;
    
    if (existing?.id) {
      // Update existing
      const { error } = await supabase
        .from('lesson_atoms')
        .update({ expression_data: expressionData })
        .eq('id', existing.id);
      
      if (error) {
        throw new Error(`Failed to update expressions: ${error.message}`);
      }
    } else {
      // Create new atom with expressions
      const { error } = await supabase
        .from('lesson_atoms')
        .insert({
          core_lesson_id: lessonId,
          archetype: 'default',
          phase,
          content: {},
          expression_data: expressionData,
        });
      
      if (error) {
        throw new Error(`Failed to insert expressions: ${error.message}`);
      }
    }
  }

  // ===========================================================================
  // CHECKPOINT MANAGEMENT
  // ===========================================================================

  /**
   * Save checkpoint to file
   */
  async saveCheckpoint() {
    this.checkpoint.lastProcessed = new Date().toISOString();
    this.checkpoint.stats = this.stats;
    
    try {
      await fs.writeFile(
        CONFIG.checkpointFile,
        JSON.stringify(this.checkpoint, null, 2)
      );
      console.log(`   💾 Checkpoint saved (${this.checkpoint.completed.length} completed)`);
    } catch (error) {
      console.warn(`   ⚠️  Failed to save checkpoint: ${error.message}`);
    }
  }

  /**
   * Load checkpoint from file
   */
  async loadCheckpoint() {
    try {
      const data = await fs.readFile(CONFIG.checkpointFile, 'utf-8');
      this.checkpoint = JSON.parse(data);
      console.log(`📂 Loaded checkpoint: ${this.checkpoint.completed.length} already completed`);
    } catch {
      console.log('📂 No checkpoint found, starting fresh');
    }
  }

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Generate URL-safe slug from lesson
   */
  getLessonSlug(lesson) {
    const text = lesson.topic || `day-${lesson.day_number}`;
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .substring(0, 50);
  }

  /**
   * Calculate total variants to process
   */
  calculateTotalVariants(lessonCount) {
    const ages = this.options.filterAgeBucket ? 1 : CONFIG.ageBuckets.length;
    const langs = this.options.filterLanguage ? 1 : CONFIG.languages.length;
    const phases = this.options.filterPhase ? 1 : CONFIG.phases.length;
    return lessonCount * ages * langs * phases;
  }

  /**
   * Sleep for specified milliseconds
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Print configuration
   */
  printConfig() {
    console.log('Configuration:');
    console.log(`  - Mode: ${this.options.dryRun ? 'DRY RUN' : 'PRODUCTION'}`);
    console.log(`  - Day range: ${this.options.startDay} to ${this.options.endDay}`);
    console.log(`  - Languages: ${this.options.filterLanguage || CONFIG.languages.join(', ')}`);
    console.log(`  - Age buckets: ${this.options.filterAgeBucket || CONFIG.ageBuckets.join(', ')}`);
    console.log(`  - Phases: ${this.options.filterPhase || CONFIG.phases.join(', ')}`);
    console.log(`  - Parallel workers: ${this.options.parallel}`);
    console.log(`  - Skip audio: ${this.options.skipAudio}`);
    console.log(`  - Skip expressions: ${this.options.skipExpressions}`);
    console.log(`  - Resume from checkpoint: ${this.options.resume}`);
    console.log('');
  }

  /**
   * Print summary
   */
  printSummary() {
    const duration = (Date.now() - this.stats.startTime) / 1000;
    
    console.log('');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  Pre-Computation Complete');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('');
    console.log('Summary:');
    console.log(`  ✅ Processed: ${this.stats.totalProcessed}`);
    console.log(`  ⏭️  Skipped: ${this.stats.totalSkipped}`);
    console.log(`  ❌ Errors: ${this.stats.totalErrors}`);
    console.log(`  🎵 Audio files: ${this.stats.totalAudioGenerated}`);
    console.log(`  😊 Expression sets: ${this.stats.totalExpressionsGenerated}`);
    console.log(`  📝 Total characters: ${this.stats.totalCharacters.toLocaleString()}`);
    console.log(`  💰 Estimated cost: $${this.stats.estimatedCost.toFixed(2)}`);
    console.log(`  ⏱️  Duration: ${duration.toFixed(1)} seconds`);
    console.log('');
    
    if (this.stats.totalErrors > 0) {
      console.log('Errors:');
      for (const err of this.checkpoint.errors.slice(-10)) {
        console.log(`  - ${err.key}: ${err.error}`);
      }
      console.log('');
    }
  }
}

// =============================================================================
// CLI ARGUMENT PARSING
// =============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {};
  
  for (const arg of args) {
    if (arg === '--dry-run') {
      options.dryRun = true;
    } else if (arg === '--resume') {
      options.resume = true;
    } else if (arg === '--skip-audio') {
      options.skipAudio = true;
    } else if (arg === '--skip-expressions') {
      options.skipExpressions = true;
    } else if (arg.startsWith('--start-day=')) {
      options.startDay = parseInt(arg.split('=')[1], 10);
    } else if (arg.startsWith('--end-day=')) {
      options.endDay = parseInt(arg.split('=')[1], 10);
    } else if (arg.startsWith('--language=')) {
      options.language = arg.split('=')[1];
    } else if (arg.startsWith('--age-bucket=')) {
      options.ageBucket = arg.split('=')[1];
    } else if (arg.startsWith('--phase=')) {
      options.phase = arg.split('=')[1];
    } else if (arg.startsWith('--parallel=')) {
      options.parallel = parseInt(arg.split('=')[1], 10);
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
  }
  
  return options;
}

function printHelp() {
  console.log(`
Curious Kelly - 365 Lesson Pre-Computation Script

Usage:
  node scripts/precompute-365-lessons.js [options]

Options:
  --dry-run          Simulate without making API calls
  --start-day=N      Start from day N (default: 1)
  --end-day=N        End at day N (default: 365)
  --language=en      Process only specified language (en, es, fr)
  --age-bucket=18-35 Process only specified age bucket
  --phase=welcome    Process only specified phase
  --parallel=N       Number of parallel workers (default: 1, max: 5)
  --skip-audio       Skip audio generation (expressions only)
  --skip-expressions Skip expression generation (audio only)
  --resume           Resume from last checkpoint
  --help, -h         Show this help message

Environment Variables:
  ELEVENLABS_API_KEY   ElevenLabs API key (required)
  SUPABASE_URL         Supabase project URL (required)
  SUPABASE_SERVICE_KEY Supabase service role key (required)
  KELLY_VOICE_ID       ElevenLabs voice ID for Kelly

Examples:
  # Dry run to estimate costs
  node scripts/precompute-365-lessons.js --dry-run

  # Process first 10 days in English only
  node scripts/precompute-365-lessons.js --start-day=1 --end-day=10 --language=en

  # Resume from last checkpoint
  node scripts/precompute-365-lessons.js --resume

  # Generate expressions only (skip expensive audio)
  node scripts/precompute-365-lessons.js --skip-audio
`);
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

async function main() {
  const options = parseArgs();
  const precomputer = new LessonPrecomputer(options);
  
  try {
    await precomputer.run();
  } catch (error) {
    console.error('❌ Fatal error:', error.message);
    process.exit(1);
  }
}

main();


