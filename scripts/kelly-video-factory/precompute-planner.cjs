/**
 * Kelly Video Pre-computation Planner
 * 
 * Analyzes what content needs to be generated and estimates costs/time.
 */

const config = require('./config.cjs');

// Lesson structure
const LESSON_STRUCTURE = {
  totalDays: 365,
  phases: ['hook', 'q1', 'q2', 'q3', 'wisdom'],
  ageGroups: ['4-5', '6-8', '9-11', '12-14', '15-17', '18+'],
  languages: ['en', 'es', 'fr'],
};

// Cost estimates (Replicate)
const COSTS = {
  imageGeneration: 0.003,
  animation_svd: 0.05,
  animation_svd_xt: 0.08,
  lipsync_wav2lip: 0.02,
  lipsync_sadtalker: 0.05,
  upscale_4k: 0.10,
  elevenlabs_per_char: 0.00003,
};

class PrecomputePlanner {
  constructor(options = {}) {
    this.days = options.days || LESSON_STRUCTURE.totalDays;
    this.phases = options.phases || LESSON_STRUCTURE.phases;
    this.ageGroups = options.ageGroups || LESSON_STRUCTURE.ageGroups;
    this.languages = options.languages || LESSON_STRUCTURE.languages;
    this.quality = options.quality || 'standard';
  }
  
  calculateScope() {
    const totalVideos = this.days * this.phases.length * this.ageGroups.length * this.languages.length;
    const uniqueImages = this.days * this.phases.length;
    const uniqueAudios = totalVideos;
    
    return {
      totalVideos,
      uniqueImages,
      uniqueAudios,
      breakdown: { days: this.days, phases: this.phases.length, ageGroups: this.ageGroups.length, languages: this.languages.length },
    };
  }
  
  estimateCosts() {
    const scope = this.calculateScope();
    const qualitySettings = config.quality[this.quality];
    
    const imageCost = scope.uniqueImages * COSTS.imageGeneration;
    const animationCost = scope.uniqueImages * COSTS.animation_svd;
    const audioCost = scope.uniqueAudios * 80 * COSTS.elevenlabs_per_char;
    const lipsyncCost = scope.totalVideos * COSTS.lipsync_wav2lip;
    const upscaleCost = qualitySettings.upscale ? scope.totalVideos * COSTS.upscale_4k : 0;
    const totalCost = imageCost + animationCost + audioCost + lipsyncCost + upscaleCost;
    
    return { imageCost, animationCost, audioCost, lipsyncCost, upscaleCost, totalCost };
  }
  
  printReport() {
    const scope = this.calculateScope();
    const costs = this.estimateCosts();
    
    console.log('KELLY VIDEO PRE-COMPUTATION PLAN');
    console.log('================================');
    console.log(`Days: ${scope.breakdown.days} | Phases: ${scope.breakdown.phases} | Ages: ${scope.breakdown.ageGroups} | Langs: ${scope.breakdown.languages}`);
    console.log(`Total Videos: ${scope.totalVideos.toLocaleString()}`);
    console.log(`Unique Images: ${scope.uniqueImages.toLocaleString()} (reusable)`);
    console.log('');
    console.log('COST ESTIMATE:');
    console.log(`  Total: $${costs.totalCost.toFixed(2)}`);
  }
}

if (require.main === module) {
  const planner = new PrecomputePlanner({ days: 30 }); // Start with 30 days
  planner.printReport();
}

module.exports = PrecomputePlanner;

