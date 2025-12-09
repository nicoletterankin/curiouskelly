/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * GOLDEN V2 - LIP-SYNC GENERATOR
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Generates lip-sync viseme data from audio files using text-to-phoneme mapping.
 * Creates frame-accurate viseme sequences for Kelly's avatar animations.
 * 
 * Viseme Categories (Oculus Standard):
 * - viseme_sil: Silence / neutral mouth
 * - viseme_PP: P, B, M (lips pressed)
 * - viseme_FF: F, V (lower lip to teeth)
 * - viseme_TH: Th (tongue to teeth)
 * - viseme_DD: D, T, N (tongue to roof)
 * - viseme_KK: K, G, Ng (back of tongue)
 * - viseme_CH: Ch, J, Sh (wide rounded)
 * - viseme_SS: S, Z (teeth together)
 * - viseme_NN: N (nasal)
 * - viseme_RR: R (curled)
 * - viseme_AA: A, Ah, Uh (open)
 * - viseme_E: E, Eh (slight smile)
 * - viseme_I: Ee, I (smile)
 * - viseme_O: O, Oh, W (rounded)
 * - viseme_U: Oo, U (tight rounded)
 * 
 * @version 2.0.0 - Golden V2
 */

import fs from 'fs';
import path from 'path';

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
  audioDir: './generated/audio',
  lessonsDir: './generated/lessons',
  outputDir: './generated/lipsync',
  
  // Frame rate for viseme data
  frameRate: 60,
  
  // Viseme timing parameters
  visemeDuration: 0.08,      // Base duration per viseme (seconds)
  transitionTime: 0.03,      // Blend time between visemes
  silenceThreshold: 0.1,     // Duration to trigger silence viseme
  
  // Emphasis multipliers
  emphasisMultiplier: 1.3,   // For emphasized syllables
  questionMultiplier: 1.1    // For question marks
};

// ═══════════════════════════════════════════════════════════════════════════════
// PHONEME TO VISEME MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

const PHONEME_TO_VISEME = {
  // Silence
  ' ': 'viseme_sil',
  '.': 'viseme_sil',
  ',': 'viseme_sil',
  '!': 'viseme_sil',
  '?': 'viseme_sil',
  
  // Bilabial (lips pressed)
  'P': 'viseme_PP', 'B': 'viseme_PP', 'M': 'viseme_PP',
  'p': 'viseme_PP', 'b': 'viseme_PP', 'm': 'viseme_PP',
  
  // Labiodental (lip to teeth)
  'F': 'viseme_FF', 'V': 'viseme_FF',
  'f': 'viseme_FF', 'v': 'viseme_FF',
  
  // Dental (tongue to teeth)
  'TH': 'viseme_TH', 'th': 'viseme_TH',
  
  // Alveolar (tongue to ridge)
  'D': 'viseme_DD', 'T': 'viseme_DD', 'N': 'viseme_NN', 'L': 'viseme_DD',
  'd': 'viseme_DD', 't': 'viseme_DD', 'n': 'viseme_NN', 'l': 'viseme_DD',
  
  // Velar (back of tongue)
  'K': 'viseme_KK', 'G': 'viseme_KK', 'NG': 'viseme_KK',
  'k': 'viseme_KK', 'g': 'viseme_KK', 'ng': 'viseme_KK',
  
  // Postalveolar (palate)
  'CH': 'viseme_CH', 'J': 'viseme_CH', 'SH': 'viseme_CH', 'ZH': 'viseme_CH',
  'ch': 'viseme_CH', 'j': 'viseme_CH', 'sh': 'viseme_CH',
  
  // Sibilant (teeth together)
  'S': 'viseme_SS', 'Z': 'viseme_SS',
  's': 'viseme_SS', 'z': 'viseme_SS',
  
  // Rhotic (curled)
  'R': 'viseme_RR', 'r': 'viseme_RR',
  
  // Vowels
  'A': 'viseme_AA', 'a': 'viseme_AA',
  'AH': 'viseme_AA', 'ah': 'viseme_AA',
  'UH': 'viseme_AA', 'uh': 'viseme_AA',
  
  'E': 'viseme_E', 'e': 'viseme_E',
  'EH': 'viseme_E', 'eh': 'viseme_E',
  
  'I': 'viseme_I', 'i': 'viseme_I',
  'EE': 'viseme_I', 'ee': 'viseme_I',
  'Y': 'viseme_I', 'y': 'viseme_I',
  
  'O': 'viseme_O', 'o': 'viseme_O',
  'OH': 'viseme_O', 'oh': 'viseme_O',
  'W': 'viseme_O', 'w': 'viseme_O',
  
  'U': 'viseme_U', 'u': 'viseme_U',
  'OO': 'viseme_U', 'oo': 'viseme_U',
  
  // Glottal (throat)
  'H': 'viseme_sil', 'h': 'viseme_sil'
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEXT TO PHONEME CONVERTER
// ═══════════════════════════════════════════════════════════════════════════════

class TextToPhoneme {
  constructor() {
    // Simple grapheme-to-phoneme rules for English
    this.rules = [
      // Digraphs first (order matters)
      { pattern: /th/gi, phonemes: ['TH'] },
      { pattern: /ch/gi, phonemes: ['CH'] },
      { pattern: /sh/gi, phonemes: ['SH'] },
      { pattern: /ng/gi, phonemes: ['NG'] },
      { pattern: /wh/gi, phonemes: ['W'] },
      { pattern: /ck/gi, phonemes: ['K'] },
      { pattern: /qu/gi, phonemes: ['K', 'W'] },
      { pattern: /tion/gi, phonemes: ['SH', 'U', 'N'] },
      { pattern: /sion/gi, phonemes: ['ZH', 'U', 'N'] },
      
      // Vowel combinations
      { pattern: /ea/gi, phonemes: ['EE'] },
      { pattern: /ee/gi, phonemes: ['EE'] },
      { pattern: /ai/gi, phonemes: ['AY'] },
      { pattern: /ay/gi, phonemes: ['AY'] },
      { pattern: /oa/gi, phonemes: ['OH'] },
      { pattern: /oo/gi, phonemes: ['OO'] },
      { pattern: /ou/gi, phonemes: ['OW'] },
      { pattern: /ow/gi, phonemes: ['OW'] },
      { pattern: /oi/gi, phonemes: ['OY'] },
      { pattern: /oy/gi, phonemes: ['OY'] },
      { pattern: /au/gi, phonemes: ['AW'] },
      { pattern: /aw/gi, phonemes: ['AW'] },
      
      // Silent e pattern (simplified)
      { pattern: /([bcdfghjklmnpqrstvwxyz])e\b/gi, phonemes: ['$1'] },
      
      // Single consonants
      { pattern: /x/gi, phonemes: ['K', 'S'] },
      
      // Default single letter mappings handled in convert()
    ];
  }
  
  /**
   * Convert text to phoneme sequence
   */
  convert(text) {
    const phonemes = [];
    let remaining = text.toLowerCase();
    
    // Apply rules in order
    for (const rule of this.rules) {
      remaining = remaining.replace(rule.pattern, () => {
        phonemes.push(...rule.phonemes);
        return '';
      });
    }
    
    // Handle remaining characters
    for (const char of remaining) {
      if (/[a-z]/i.test(char)) {
        phonemes.push(char.toUpperCase());
      } else if (/[\s.,!?;:]/.test(char)) {
        phonemes.push(' '); // Silence marker
      }
    }
    
    return phonemes;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIP-SYNC GENERATOR CLASS
// ═══════════════════════════════════════════════════════════════════════════════

class LipSyncGenerator {
  constructor(config = {}) {
    this.config = { ...CONFIG, ...config };
    this.textToPhoneme = new TextToPhoneme();
  }
  
  /**
   * Generate viseme sequence from text
   */
  generateVisemeSequence(text, totalDuration) {
    const phonemes = this.textToPhoneme.convert(text);
    const visemes = [];
    
    // Calculate timing
    const wordsPerSecond = 2.5; // Average speaking rate
    const words = text.split(/\s+/).length;
    const expectedDuration = totalDuration || (words / wordsPerSecond);
    
    // Time per phoneme (with adjustments)
    const baseTime = expectedDuration / Math.max(phonemes.length, 1);
    
    let currentTime = 0;
    let lastViseme = 'viseme_sil';
    
    for (let i = 0; i < phonemes.length; i++) {
      const phoneme = phonemes[i];
      const viseme = PHONEME_TO_VISEME[phoneme] || PHONEME_TO_VISEME[phoneme.charAt(0)] || 'viseme_sil';
      
      // Calculate duration with variation
      let duration = baseTime;
      
      // Longer duration for vowels
      if (['viseme_AA', 'viseme_E', 'viseme_I', 'viseme_O', 'viseme_U'].includes(viseme)) {
        duration *= 1.2;
      }
      
      // Shorter duration for stops
      if (['viseme_PP', 'viseme_DD', 'viseme_KK'].includes(viseme)) {
        duration *= 0.8;
      }
      
      // Add silence gaps
      if (viseme === 'viseme_sil' && lastViseme !== 'viseme_sil') {
        duration = Math.max(duration, this.config.silenceThreshold);
      }
      
      visemes.push({
        time: currentTime,
        duration: duration,
        viseme: viseme,
        phoneme: phoneme,
        blend: this.calculateBlend(lastViseme, viseme)
      });
      
      currentTime += duration;
      lastViseme = viseme;
    }
    
    // Add final silence
    visemes.push({
      time: currentTime,
      duration: 0.2,
      viseme: 'viseme_sil',
      phoneme: ' ',
      blend: this.calculateBlend(lastViseme, 'viseme_sil')
    });
    
    return visemes;
  }
  
  /**
   * Calculate blend weights between visemes
   */
  calculateBlend(fromViseme, toViseme) {
    // Return blend shape targets
    return {
      from: fromViseme,
      to: toViseme,
      transitionTime: this.config.transitionTime
    };
  }
  
  /**
   * Convert viseme sequence to frame data
   */
  visemesToFrames(visemes, frameRate = this.config.frameRate) {
    const frames = [];
    let totalDuration = 0;
    
    if (visemes.length > 0) {
      const lastViseme = visemes[visemes.length - 1];
      totalDuration = lastViseme.time + lastViseme.duration;
    }
    
    const totalFrames = Math.ceil(totalDuration * frameRate);
    
    for (let frame = 0; frame < totalFrames; frame++) {
      const time = frame / frameRate;
      
      // Find current and next viseme
      let currentViseme = visemes[0];
      let nextViseme = visemes[1] || visemes[0];
      
      for (let i = 0; i < visemes.length - 1; i++) {
        if (visemes[i].time <= time && visemes[i + 1].time > time) {
          currentViseme = visemes[i];
          nextViseme = visemes[i + 1];
          break;
        }
        if (i === visemes.length - 2) {
          currentViseme = visemes[visemes.length - 1];
          nextViseme = currentViseme;
        }
      }
      
      // Calculate blend factor for smooth transitions
      const visemeDuration = currentViseme.duration;
      const elapsed = time - currentViseme.time;
      const blendFactor = Math.min(1, elapsed / visemeDuration);
      
      frames.push({
        frame: frame,
        time: time,
        viseme: currentViseme.viseme,
        nextViseme: nextViseme.viseme,
        blendFactor: blendFactor,
        blendShapes: this.visemeToBlendShapes(currentViseme.viseme, nextViseme.viseme, blendFactor)
      });
    }
    
    return frames;
  }
  
  /**
   * Convert viseme to blend shape values
   */
  visemeToBlendShapes(currentViseme, nextViseme, blendFactor) {
    const blendShapes = {
      jawOpen: 0,
      mouthClose: 0,
      mouthFunnel: 0,
      mouthPucker: 0,
      mouthSmile: 0,
      mouthStretch: 0,
      tongueOut: 0
    };
    
    // Current viseme contribution
    const current = this.getVisemeBlendShapes(currentViseme);
    // Next viseme contribution (for smooth blending)
    const next = this.getVisemeBlendShapes(nextViseme);
    
    // Interpolate
    for (const [key, value] of Object.entries(current)) {
      const nextValue = next[key] || 0;
      blendShapes[key] = value * (1 - blendFactor * 0.3) + nextValue * (blendFactor * 0.3);
    }
    
    return blendShapes;
  }
  
  /**
   * Get blend shape values for a specific viseme
   */
  getVisemeBlendShapes(viseme) {
    const shapes = {
      'viseme_sil': { jawOpen: 0, mouthClose: 1, mouthSmile: 0.1 },
      'viseme_PP': { jawOpen: 0, mouthClose: 1, mouthPucker: 0.5 },
      'viseme_FF': { jawOpen: 0.1, mouthClose: 0.8, mouthStretch: 0.3 },
      'viseme_TH': { jawOpen: 0.15, mouthClose: 0.7, tongueOut: 0.4 },
      'viseme_DD': { jawOpen: 0.2, mouthClose: 0.6 },
      'viseme_KK': { jawOpen: 0.25, mouthClose: 0.5 },
      'viseme_CH': { jawOpen: 0.3, mouthPucker: 0.4, mouthFunnel: 0.3 },
      'viseme_SS': { jawOpen: 0.15, mouthSmile: 0.3, mouthStretch: 0.2 },
      'viseme_NN': { jawOpen: 0.2, mouthClose: 0.6 },
      'viseme_RR': { jawOpen: 0.25, mouthPucker: 0.3, mouthFunnel: 0.2 },
      'viseme_AA': { jawOpen: 0.8, mouthOpen: 1, mouthStretch: 0.2 },
      'viseme_E': { jawOpen: 0.5, mouthSmile: 0.4, mouthStretch: 0.3 },
      'viseme_I': { jawOpen: 0.3, mouthSmile: 0.6, mouthStretch: 0.4 },
      'viseme_O': { jawOpen: 0.6, mouthFunnel: 0.6, mouthPucker: 0.3 },
      'viseme_U': { jawOpen: 0.25, mouthPucker: 0.8, mouthFunnel: 0.5 }
    };
    
    return shapes[viseme] || shapes['viseme_sil'];
  }
  
  /**
   * Generate lip-sync data for a lesson phase
   */
  generatePhaseLipSync(text, estimatedDuration) {
    const visemes = this.generateVisemeSequence(text, estimatedDuration);
    const frames = this.visemesToFrames(visemes);
    
    return {
      text: text,
      duration: estimatedDuration,
      frameRate: this.config.frameRate,
      totalFrames: frames.length,
      visemes: visemes,
      frames: frames
    };
  }
  
  /**
   * Generate lip-sync data for an entire lesson
   */
  async generateLessonLipSync(lessonDNA) {
    const day = lessonDNA.meta.day;
    const paddedDay = String(day).padStart(3, '0');
    const lessonDir = path.join(this.config.outputDir, `day-${paddedDay}`);
    
    // Create output directory
    if (!fs.existsSync(lessonDir)) {
      fs.mkdirSync(lessonDir, { recursive: true });
    }
    
    const lipSyncManifest = {
      day: day,
      topic: lessonDNA.meta.topic,
      generatedAt: new Date().toISOString(),
      frameRate: this.config.frameRate,
      files: {}
    };
    
    // Generate for each age bucket
    for (const [bucketId, variant] of Object.entries(lessonDNA.ageVariants)) {
      const bucketDir = path.join(lessonDir, bucketId);
      if (!fs.existsSync(bucketDir)) {
        fs.mkdirSync(bucketDir, { recursive: true });
      }
      
      lipSyncManifest.files[bucketId] = {};
      
      // Generate for each phase
      for (const [phase, text] of Object.entries(variant.phases)) {
        const duration = variant.durations[phase];
        const lipSyncData = this.generatePhaseLipSync(text, duration);
        
        // Save lip-sync JSON
        const filename = `${phase}-lipsync.json`;
        fs.writeFileSync(
          path.join(bucketDir, filename),
          JSON.stringify(lipSyncData, null, 2)
        );
        
        lipSyncManifest.files[bucketId][phase] = {
          file: filename,
          frames: lipSyncData.totalFrames,
          duration: duration
        };
        
        console.log(`    👄 Generated ${bucketId}/${phase}: ${lipSyncData.totalFrames} frames`);
      }
    }
    
    // Save manifest
    fs.writeFileSync(
      path.join(lessonDir, 'lipsync-manifest.json'),
      JSON.stringify(lipSyncManifest, null, 2)
    );
    
    return lipSyncManifest;
  }
  
  /**
   * Generate lip-sync data for all lessons
   */
  async generateAllLipSync(startDay = 1, endDay = 365) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  GOLDEN V2 - LIP-SYNC GENERATOR');
    console.log(`  Generating lip-sync data for days ${startDay}-${endDay}`);
    console.log('═══════════════════════════════════════════════════════════════');
    
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
      
      console.log(`\n  📅 Day ${day}`);
      
      const lessonDNA = JSON.parse(fs.readFileSync(lessonFile, 'utf-8'));
      const manifest = await this.generateLessonLipSync(lessonDNA);
      results.push(manifest);
      
      console.log(`  ✓ Day ${day} complete`);
    }
    
    // Write master manifest
    const masterManifest = {
      version: '2.0.0-golden',
      generatedAt: new Date().toISOString(),
      frameRate: this.config.frameRate,
      totalDays: results.length,
      days: results
    };
    
    fs.writeFileSync(
      path.join(this.config.outputDir, 'master-manifest.json'),
      JSON.stringify(masterManifest, null, 2)
    );
    
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log(`  ✅ COMPLETE: ${results.length} days of lip-sync generated`);
    console.log(`  📁 Output: ${this.config.outputDir}`);
    console.log('═══════════════════════════════════════════════════════════════');
    
    return masterManifest;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

export { LipSyncGenerator, TextToPhoneme, PHONEME_TO_VISEME, CONFIG };
export default LipSyncGenerator;

// CLI execution
if (process.argv[1]?.includes('lipsync-generator')) {
  const startDay = parseInt(process.argv.find(a => a.startsWith('--start='))?.split('=')[1]) || 1;
  const endDay = parseInt(process.argv.find(a => a.startsWith('--end='))?.split('=')[1]) || 30;
  
  const generator = new LipSyncGenerator();
  generator.generateAllLipSync(startDay, endDay).catch(console.error);
}

