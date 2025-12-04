/**
 * Check available lip-sync models on Replicate
 */

import 'dotenv/config';
import Replicate from 'replicate';

const MODELS_TO_CHECK = [
  // Talking Head / Portrait Animation
  'lucataco/hallo2',
  'zsxkib/echomimic',
  'fofr/live-portrait',
  'cjwbw/video-retalking',
  'yoyo-nb/thin-plate-spline-motion-model',
  'wangfuyun/animateanyone',
  'chenxwh/musetalk',
  'pengbo-learn/latentsync',
  
  // Known working
  'cjwbw/sadtalker',
  'devxpy/wav2lip',
  
  // Full body
  'cuuupid/idm-vton',
  'zsxkib/instant-id',
];

async function main() {
  const replicate = new Replicate();
  
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🔍 REPLICATE LIP-SYNC MODEL SURVEY');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  const available: string[] = [];
  const unavailable: string[] = [];
  
  for (const model of MODELS_TO_CHECK) {
    const [owner, name] = model.split('/');
    try {
      const m = await replicate.models.get(owner, name);
      available.push(model);
      console.log(`✅ ${model}`);
      if (m.description) {
        console.log(`   ${m.description.substring(0, 100)}...`);
      }
      console.log('');
    } catch (e: any) {
      unavailable.push(model);
      console.log(`❌ ${model} - ${e.message?.substring(0, 50) || 'Not found'}`);
      console.log('');
    }
  }
  
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`   Summary: ${available.length} available, ${unavailable.length} unavailable`);
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  console.log('Available for use:');
  available.forEach(m => console.log(`  - ${m}`));
}

main().catch(console.error);

