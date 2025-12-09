/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * GOLDEN V2 - VISUAL GENERATOR
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Generates visual prompts and manifests for Kelly's phase images.
 * Creates detailed image generation prompts for each lesson phase.
 * 
 * Visual Styles:
 * - Consistent Kelly character design
 * - Age-adaptive expressions and energy
 * - Topic-relevant backgrounds and props
 * - Professional studio lighting
 * 
 * @version 2.0.0 - Golden V2
 */

import fs from 'fs';
import path from 'path';

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
  lessonsDir: './generated/lessons',
  outputDir: './generated/visuals',
  
  // Kelly's base appearance
  kellyBase: {
    age: '27 years old',
    ethnicity: 'racially ambiguous, universal appeal',
    hair: 'warm brown, shoulder length, soft waves',
    eyes: 'warm brown, intelligent, kind',
    skin: 'medium warm tone, healthy glow',
    style: 'smart casual, approachable teacher',
    clothing: 'comfortable professional, earth tones'
  },
  
  // Expression mappings per phase
  phaseExpressions: {
    hook: {
      expression: 'warm welcoming smile, eyes bright with excitement',
      gesture: 'open arms or hands together in greeting',
      energy: 'inviting, eager to share'
    },
    q1: {
      expression: 'curious, eyebrows slightly raised, engaged smile',
      gesture: 'pointing or gesturing to explain',
      energy: 'enthusiastic, teaching mode'
    },
    q2: {
      expression: 'animated, eyes wide with discovery',
      gesture: 'hands moving expressively',
      energy: 'building momentum, exciting'
    },
    q3: {
      expression: 'confident smile, knowing look',
      gesture: 'palm up, sharing knowledge',
      energy: 'revealing, climactic'
    },
    wisdom: {
      expression: 'warm, thoughtful smile, gentle eyes',
      gesture: 'hands at heart or in contemplative pose',
      energy: 'reflective, meaningful'
    }
  },
  
  // Image specifications
  imageSpec: {
    style: 'photorealistic digital art, studio quality',
    lighting: 'soft studio lighting, rim light, subtle fill',
    background: 'clean gradient or subtle bokeh',
    camera: 'medium close-up, slight angle, eye level',
    quality: '8K, highly detailed, professional photography'
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// VISUAL PROMPT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

class VisualGenerator {
  constructor(config = {}) {
    this.config = { ...CONFIG, ...config };
  }
  
  /**
   * Generate a detailed image prompt for a lesson phase
   */
  generatePhasePrompt(topic, phase, ageBucket) {
    const kelly = this.config.kellyBase;
    const phaseStyle = this.config.phaseExpressions[phase];
    const imageSpec = this.config.imageSpec;
    const ageStyle = this.getAgeStyle(ageBucket);
    
    // Build the prompt
    const prompt = [
      // Character
      `Portrait of Kelly, a ${kelly.age} ${kelly.ethnicity} female educator,`,
      `${kelly.hair} hair, ${kelly.eyes} eyes, ${kelly.skin} skin,`,
      `wearing ${kelly.clothing},`,
      
      // Expression and pose
      `${phaseStyle.expression},`,
      `${phaseStyle.gesture},`,
      `${ageStyle.energyModifier} energy,`,
      
      // Context
      `teaching about "${topic}",`,
      `${phase === 'hook' ? 'welcoming the learner' : 
         phase === 'wisdom' ? 'sharing wisdom and reflection' :
         'explaining an interesting concept'},`,
      
      // Style
      `${imageSpec.style},`,
      `${imageSpec.lighting},`,
      `${imageSpec.background},`,
      `${imageSpec.camera},`,
      `${imageSpec.quality}`
    ].join(' ');
    
    // Negative prompt to avoid common issues
    const negativePrompt = [
      'cartoon, anime, illustration, painting',
      'distorted features, asymmetric face',
      'extra fingers, extra limbs',
      'blurry, low quality, pixelated',
      'text, watermark, logo',
      'nsfw, inappropriate'
    ].join(', ');
    
    return {
      prompt,
      negativePrompt,
      guidance: 7.5,
      steps: 50
    };
  }
  
  /**
   * Get age-adaptive style modifications
   */
  getAgeStyle(ageBucket) {
    const styles = {
      '2-5': {
        energyModifier: 'playful, animated, high',
        backgroundStyle: 'bright, colorful, fun',
        clothingNote: 'bright colors, friendly'
      },
      '6-12': {
        energyModifier: 'enthusiastic, cool, engaging',
        backgroundStyle: 'vibrant, dynamic',
        clothingNote: 'trendy but appropriate'
      },
      '13-17': {
        energyModifier: 'confident, relatable, direct',
        backgroundStyle: 'modern, clean',
        clothingNote: 'stylish, professional'
      },
      '18-35': {
        energyModifier: 'professional, conversational',
        backgroundStyle: 'clean, sophisticated',
        clothingNote: 'smart casual'
      },
      '36-60': {
        energyModifier: 'measured, authoritative, warm',
        backgroundStyle: 'elegant, understated',
        clothingNote: 'refined professional'
      },
      '61-102': {
        energyModifier: 'gentle, warm, thoughtful',
        backgroundStyle: 'soft, calming',
        clothingNote: 'comfortable, dignified'
      }
    };
    
    return styles[ageBucket] || styles['18-35'];
  }
  
  /**
   * Generate visual manifest for a single lesson
   */
  generateLessonVisuals(lessonDNA) {
    const day = lessonDNA.meta.day;
    const topic = lessonDNA.meta.topic;
    const paddedDay = String(day).padStart(3, '0');
    const lessonDir = path.join(this.config.outputDir, `day-${paddedDay}`);
    
    // Create output directory
    if (!fs.existsSync(lessonDir)) {
      fs.mkdirSync(lessonDir, { recursive: true });
    }
    
    const visualManifest = {
      day: day,
      topic: topic,
      generatedAt: new Date().toISOString(),
      version: '2.0.0-golden',
      phases: {}
    };
    
    // Generate prompts for each phase
    const phases = ['hook', 'q1', 'q2', 'q3', 'wisdom'];
    
    for (const phase of phases) {
      visualManifest.phases[phase] = {
        basePrompt: this.generatePhasePrompt(topic, phase, '18-35'),
        ageVariants: {}
      };
      
      // Generate age-specific variants
      for (const bucket of Object.keys(lessonDNA.ageVariants)) {
        visualManifest.phases[phase].ageVariants[bucket] = 
          this.generatePhasePrompt(topic, phase, bucket);
      }
    }
    
    // Save visual manifest
    fs.writeFileSync(
      path.join(lessonDir, 'visual-manifest.json'),
      JSON.stringify(visualManifest, null, 2)
    );
    
    // Generate a simplified prompt file for easy batch processing
    const promptsFile = phases.map(phase => {
      const prompt = visualManifest.phases[phase].basePrompt;
      return `=== ${phase.toUpperCase()} ===\n${prompt.prompt}\n\nNegative: ${prompt.negativePrompt}\n`;
    }).join('\n');
    
    fs.writeFileSync(
      path.join(lessonDir, 'prompts.txt'),
      promptsFile
    );
    
    return visualManifest;
  }
  
  /**
   * Generate HTML preview page for visual prompts
   */
  generatePreviewPage(lessonDNA, visualManifest) {
    const day = lessonDNA.meta.day;
    const paddedDay = String(day).padStart(3, '0');
    const lessonDir = path.join(this.config.outputDir, `day-${paddedDay}`);
    
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${day} Visual Prompts - ${lessonDNA.meta.topic}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: system-ui, -apple-system, sans-serif;
      background: #0a0a0a; color: #e4e4e7;
      padding: 40px;
    }
    h1 { margin-bottom: 8px; color: #fff; }
    .topic { color: #3b82f6; font-size: 1.2rem; margin-bottom: 32px; }
    .phase { 
      background: #18181b;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .phase-header { 
      font-size: 1.1rem;
      font-weight: 600;
      color: #22c55e;
      margin-bottom: 16px;
      text-transform: uppercase;
    }
    .prompt { 
      background: #27272a;
      padding: 16px;
      border-radius: 8px;
      font-size: 0.95rem;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .negative { 
      margin-top: 12px;
      font-size: 0.85rem;
      color: #71717a;
    }
    .image-placeholder {
      width: 100%;
      height: 300px;
      background: linear-gradient(135deg, #27272a, #18181b);
      border-radius: 8px;
      margin-top: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #52525b;
    }
  </style>
</head>
<body>
  <h1>Day ${day} Visual Prompts</h1>
  <div class="topic">${lessonDNA.meta.topic}</div>
  
  ${Object.entries(visualManifest.phases).map(([phase, data]) => `
    <div class="phase">
      <div class="phase-header">${phase}</div>
      <div class="prompt">${data.basePrompt.prompt}</div>
      <div class="negative">Negative: ${data.basePrompt.negativePrompt}</div>
      <div class="image-placeholder">📸 Image placeholder - ${phase}</div>
    </div>
  `).join('')}
  
</body>
</html>
    `;
    
    fs.writeFileSync(
      path.join(lessonDir, 'preview.html'),
      html
    );
  }
  
  /**
   * Generate visuals for all lessons
   */
  async generateAllVisuals(startDay = 1, endDay = 365) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  GOLDEN V2 - VISUAL GENERATOR');
    console.log(`  Generating visual prompts for days ${startDay}-${endDay}`);
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
      
      const lessonDNA = JSON.parse(fs.readFileSync(lessonFile, 'utf-8'));
      const manifest = this.generateLessonVisuals(lessonDNA);
      this.generatePreviewPage(lessonDNA, manifest);
      
      results.push(manifest);
      
      if (day % 30 === 0) {
        console.log(`  ✓ Generated days 1-${day} (${Math.round((day - startDay + 1) / (endDay - startDay + 1) * 100)}%)`);
      }
    }
    
    // Write master manifest
    const masterManifest = {
      version: '2.0.0-golden',
      generatedAt: new Date().toISOString(),
      totalDays: results.length,
      kellyBase: this.config.kellyBase,
      imageSpec: this.config.imageSpec,
      days: results.map(r => ({
        day: r.day,
        topic: r.topic
      }))
    };
    
    fs.writeFileSync(
      path.join(this.config.outputDir, 'master-manifest.json'),
      JSON.stringify(masterManifest, null, 2)
    );
    
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log(`  ✅ COMPLETE: ${results.length} days of visual prompts generated`);
    console.log(`  📁 Output: ${this.config.outputDir}`);
    console.log('═══════════════════════════════════════════════════════════════');
    
    return masterManifest;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

export { VisualGenerator, CONFIG };
export default VisualGenerator;

// CLI execution
if (process.argv[1]?.includes('visual-generator')) {
  const startDay = parseInt(process.argv.find(a => a.startsWith('--start='))?.split('=')[1]) || 1;
  const endDay = parseInt(process.argv.find(a => a.startsWith('--end='))?.split('=')[1]) || 30;
  
  const generator = new VisualGenerator();
  generator.generateAllVisuals(startDay, endDay).catch(console.error);
}

