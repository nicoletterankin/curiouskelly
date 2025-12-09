/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * GOLDEN V2 - AUDIO GENERATOR
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Generates high-quality voice audio for all lessons using ElevenLabs API.
 * Creates audio for each phase of each lesson across all age buckets.
 * 
 * Features:
 * - Age-adaptive voice settings (pitch, speed, warmth)
 * - Batch processing with rate limiting
 * - Automatic retry on failure
 * - Audio manifest generation
 * - Progress tracking and resume capability
 * 
 * @version 2.0.0 - Golden V2
 */

import fs from 'fs';
import path from 'path';

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
  // ElevenLabs API
  apiKey: process.env.ELEVENLABS_API_KEY,
  apiUrl: 'https://api.elevenlabs.io/v1',
  
  // Kelly's voice ID (use your trained voice ID)
  voiceId: process.env.KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0', // Default Kelly voice
  
  // Model settings
  modelId: 'eleven_multilingual_v2', // Best quality multilingual
  
  // Output settings
  outputDir: './generated/audio',
  lessonsDir: './generated/lessons',
  
  // Rate limiting
  requestsPerMinute: 20,
  batchSize: 5,
  retryAttempts: 3,
  retryDelay: 5000,
  
  // Audio format
  outputFormat: 'mp3_44100_128',
  
  // Age bucket voice settings
  voiceSettings: {
    '2-5': {
      stability: 0.4,        // More expressive
      similarity_boost: 0.8,
      style: 0.7,           // Playful style
      use_speaker_boost: true,
      speed: 0.95            // Slightly slower for kids
    },
    '6-12': {
      stability: 0.5,
      similarity_boost: 0.75,
      style: 0.6,
      use_speaker_boost: true,
      speed: 1.0
    },
    '13-17': {
      stability: 0.6,
      similarity_boost: 0.7,
      style: 0.4,           // More direct
      use_speaker_boost: true,
      speed: 1.05
    },
    '18-35': {
      stability: 0.65,
      similarity_boost: 0.75,
      style: 0.35,
      use_speaker_boost: true,
      speed: 1.0
    },
    '36-60': {
      stability: 0.7,
      similarity_boost: 0.8,
      style: 0.3,
      use_speaker_boost: true,
      speed: 0.95
    },
    '61-102': {
      stability: 0.75,       // More stable/calm
      similarity_boost: 0.85,
      style: 0.25,          // Warm, gentle
      use_speaker_boost: true,
      speed: 0.9            // Slower, thoughtful
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// AUDIO GENERATOR CLASS
// ═══════════════════════════════════════════════════════════════════════════════

class AudioGenerator {
  constructor(config = {}) {
    this.config = { ...CONFIG, ...config };
    this.requestCount = 0;
    this.lastRequestTime = 0;
    this.progress = this.loadProgress();
  }
  
  /**
   * Load progress file to resume generation
   */
  loadProgress() {
    const progressFile = path.join(this.config.outputDir, '.progress.json');
    if (fs.existsSync(progressFile)) {
      return JSON.parse(fs.readFileSync(progressFile, 'utf-8'));
    }
    return { completed: [], failed: [], lastDay: 0 };
  }
  
  /**
   * Save progress for resume capability
   */
  saveProgress() {
    const progressFile = path.join(this.config.outputDir, '.progress.json');
    fs.writeFileSync(progressFile, JSON.stringify(this.progress, null, 2));
  }
  
  /**
   * Rate limiter to respect API limits
   */
  async rateLimiter() {
    const now = Date.now();
    const minInterval = 60000 / this.config.requestsPerMinute;
    const elapsed = now - this.lastRequestTime;
    
    if (elapsed < minInterval) {
      await this.sleep(minInterval - elapsed);
    }
    
    this.lastRequestTime = Date.now();
  }
  
  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Generate audio for a single text segment
   */
  async generateAudio(text, ageBucket, retryCount = 0) {
    await this.rateLimiter();
    
    const voiceSettings = this.config.voiceSettings[ageBucket] || this.config.voiceSettings['18-35'];
    
    try {
      const response = await fetch(
        `${this.config.apiUrl}/text-to-speech/${this.config.voiceId}`,
        {
          method: 'POST',
          headers: {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': this.config.apiKey
          },
          body: JSON.stringify({
            text: text,
            model_id: this.config.modelId,
            voice_settings: {
              stability: voiceSettings.stability,
              similarity_boost: voiceSettings.similarity_boost,
              style: voiceSettings.style,
              use_speaker_boost: voiceSettings.use_speaker_boost
            }
          })
        }
      );
      
      if (!response.ok) {
        const error = await response.text();
        throw new Error(`ElevenLabs API error: ${response.status} - ${error}`);
      }
      
      const audioBuffer = await response.arrayBuffer();
      return Buffer.from(audioBuffer);
      
    } catch (error) {
      if (retryCount < this.config.retryAttempts) {
        console.log(`  ⚠️ Retry ${retryCount + 1}/${this.config.retryAttempts}: ${error.message}`);
        await this.sleep(this.config.retryDelay);
        return this.generateAudio(text, ageBucket, retryCount + 1);
      }
      throw error;
    }
  }
  
  /**
   * Generate all audio for a single lesson
   */
  async generateLessonAudio(lessonDNA) {
    const day = lessonDNA.meta.day;
    const paddedDay = String(day).padStart(3, '0');
    const lessonDir = path.join(this.config.outputDir, `day-${paddedDay}`);
    
    // Create lesson directory
    if (!fs.existsSync(lessonDir)) {
      fs.mkdirSync(lessonDir, { recursive: true });
    }
    
    const audioManifest = {
      day: day,
      topic: lessonDNA.meta.topic,
      generatedAt: new Date().toISOString(),
      files: {}
    };
    
    // Generate audio for each age bucket
    for (const [bucketId, variant] of Object.entries(lessonDNA.ageVariants)) {
      const bucketDir = path.join(lessonDir, bucketId);
      if (!fs.existsSync(bucketDir)) {
        fs.mkdirSync(bucketDir, { recursive: true });
      }
      
      audioManifest.files[bucketId] = {};
      
      // Generate audio for each phase
      for (const [phase, text] of Object.entries(variant.phases)) {
        const filename = `${phase}.mp3`;
        const filePath = path.join(bucketDir, filename);
        
        // Skip if already exists
        if (fs.existsSync(filePath)) {
          audioManifest.files[bucketId][phase] = {
            file: filename,
            duration: variant.durations[phase],
            skipped: true
          };
          continue;
        }
        
        try {
          console.log(`    📢 Generating ${bucketId}/${phase}...`);
          const audioBuffer = await this.generateAudio(text, bucketId);
          fs.writeFileSync(filePath, audioBuffer);
          
          audioManifest.files[bucketId][phase] = {
            file: filename,
            duration: variant.durations[phase],
            bytes: audioBuffer.length
          };
          
        } catch (error) {
          console.error(`    ❌ Failed ${bucketId}/${phase}: ${error.message}`);
          this.progress.failed.push({ day, bucket: bucketId, phase, error: error.message });
        }
      }
    }
    
    // Save audio manifest
    fs.writeFileSync(
      path.join(lessonDir, 'audio-manifest.json'),
      JSON.stringify(audioManifest, null, 2)
    );
    
    return audioManifest;
  }
  
  /**
   * Generate audio for all lessons
   */
  async generateAllAudio(startDay = 1, endDay = 365) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  GOLDEN V2 - AUDIO GENERATOR');
    console.log(`  Generating audio for days ${startDay}-${endDay}`);
    console.log('═══════════════════════════════════════════════════════════════');
    
    // Ensure output directory exists
    if (!fs.existsSync(this.config.outputDir)) {
      fs.mkdirSync(this.config.outputDir, { recursive: true });
    }
    
    const results = [];
    
    for (let day = startDay; day <= endDay; day++) {
      const paddedDay = String(day).padStart(3, '0');
      const lessonFile = path.join(this.config.lessonsDir, `day-${paddedDay}.json`);
      
      if (!fs.existsSync(lessonFile)) {
        console.log(`  ⚠️ Skipping day ${day}: lesson file not found`);
        continue;
      }
      
      // Skip if already completed
      if (this.progress.completed.includes(day)) {
        console.log(`  ⏭️ Day ${day} already complete, skipping...`);
        continue;
      }
      
      console.log(`\n  📅 Day ${day}: ${paddedDay}`);
      
      const lessonDNA = JSON.parse(fs.readFileSync(lessonFile, 'utf-8'));
      const manifest = await this.generateLessonAudio(lessonDNA);
      results.push(manifest);
      
      this.progress.completed.push(day);
      this.progress.lastDay = day;
      this.saveProgress();
      
      console.log(`  ✓ Day ${day} complete (${Math.round((day - startDay + 1) / (endDay - startDay + 1) * 100)}%)`);
    }
    
    // Write master audio manifest
    const masterManifest = {
      version: '2.0.0-golden',
      generatedAt: new Date().toISOString(),
      totalDays: results.length,
      totalFiles: results.reduce((sum, m) => {
        return sum + Object.keys(m.files).length * 5; // 5 phases per bucket
      }, 0),
      days: results
    };
    
    fs.writeFileSync(
      path.join(this.config.outputDir, 'master-manifest.json'),
      JSON.stringify(masterManifest, null, 2)
    );
    
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log(`  ✅ COMPLETE: ${results.length} days of audio generated`);
    console.log(`  📁 Output: ${this.config.outputDir}`);
    if (this.progress.failed.length > 0) {
      console.log(`  ⚠️ ${this.progress.failed.length} items failed (see progress file)`);
    }
    console.log('═══════════════════════════════════════════════════════════════');
    
    return masterManifest;
  }
  
  /**
   * Generate audio for a single day (useful for testing)
   */
  async generateDayAudio(dayNumber) {
    const paddedDay = String(dayNumber).padStart(3, '0');
    const lessonFile = path.join(this.config.lessonsDir, `day-${paddedDay}.json`);
    
    if (!fs.existsSync(lessonFile)) {
      throw new Error(`Lesson file not found: ${lessonFile}`);
    }
    
    const lessonDNA = JSON.parse(fs.readFileSync(lessonFile, 'utf-8'));
    return this.generateLessonAudio(lessonDNA);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK AUDIO GENERATOR (for testing without API)
// ═══════════════════════════════════════════════════════════════════════════════

class MockAudioGenerator extends AudioGenerator {
  /**
   * Generate mock audio (creates placeholder files)
   */
  async generateAudio(text, ageBucket) {
    // Create a simple audio placeholder
    const duration = Math.ceil(text.split(/\s+/).length / 2.5); // ~2.5 words/sec
    
    // Generate a simple tone or use silence
    // For now, return a minimal valid MP3 header
    const mp3Header = Buffer.from([
      0xFF, 0xFB, 0x90, 0x00, // MP3 frame header
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00
    ]);
    
    // Simulate processing time
    await this.sleep(100);
    
    return mp3Header;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

export { AudioGenerator, MockAudioGenerator, CONFIG };
export default AudioGenerator;

// CLI execution
if (process.argv[1]?.includes('audio-generator')) {
  const useMock = process.argv.includes('--mock');
  const Generator = useMock ? MockAudioGenerator : AudioGenerator;
  
  const startDay = parseInt(process.argv.find(a => a.startsWith('--start='))?.split('=')[1]) || 1;
  const endDay = parseInt(process.argv.find(a => a.startsWith('--end='))?.split('=')[1]) || 30;
  
  const generator = new Generator();
  generator.generateAllAudio(startDay, endDay).catch(console.error);
}

